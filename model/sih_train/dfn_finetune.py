"""DeepFilterNet3 fine-tuning on SIH26052 manifests (replaces Downloads/finetune_dfn.py; see docs/finetune_design.md).

Fixed design:
- Start: pretrained DeepFilterNet3 (model_120.ckpt.best) loaded with its own config.ini via df.enhance.init_df.
- Input/target: pre-mixed noisy/clean pairs from a manifest (role + use_for_training), 48 kHz, no resampling,
  no re-mixing, no level change. Random crop of `segment_sec` with the SAME offset for noisy and clean;
  shorter files are zero-padded at the end.
- Features: DFN3's own libdf STFT + erb_norm/unit_norm (identical code path to inference, bit-matched).
- Loss: df.loss.Loss exactly as df.train.setup_losses builds it from the checkpoint config
  (MultiResSpecLoss 500/500, gamma 0.3, FFT 256..2048 + LocalSnrLoss 1e-3).
- BatchNorm: running statistics frozen by default (small fine-tuning batches); `--bn-mode train` restores
  the official behaviour.
- Determinism/resume: epoch permutation from rng([seed, epoch]); crop offset from rng([seed, epoch, position]);
  the global step fully determines the data order, so resume reproduces an uninterrupted run.
- Selection: validation loss (official criterion: loss, min) on the `val_select` manifest rows, full files;
  SNR/STOI/PESQ from sih_train.evaluation logged alongside.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from df.enhance import get_model_basedir, init_df
from df.loss import Istft, Loss
from df.model import ModelParams
from df.utils import as_real

from sih_train.data import AudioSource
from sih_model.dfn3 import enhance_waveform, features_from_spectrum
from sih_train.evaluation import metrics_from_arrays

SAMPLE_RATE = 48000


@dataclass
class TrainConfig:
    manifest: str
    val_manifest: str
    run_dir: str
    roots: list[str] = field(default_factory=list)
    zips: list[str] = field(default_factory=list)
    train_role: str = "train"
    val_role: str = "val_select"
    val_max_files: int = 0            # 0 = all val_select rows
    steps: int = 1000
    batch_size: int = 16
    segment_sec: float = 3.0          # = config.ini max_sample_len_s
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 100
    grad_clip: float = 1.0            # = official
    bn_mode: str = "frozen"           # "frozen" | "train"
    seed: int = 26052
    val_every: int = 500
    checkpoint_every: int = 100
    max_nan_steps: int = 10
    init_model_dir: str | None = None  # None = pretrained DeepFilterNet3
    row_filter: str = "use_for_training"  # or "metadata_no_clip" (no audio check)
    validate: bool = True              # False: no validation passes; export_best = final model


def read_rows(path: str, role: str, flag: str) -> list[dict]:
    """flag = a manifest boolean column, or "metadata_no_clip": rows whose metadata did not FAIL and are not
    clip-flagged (used when the audio check was skipped; audio problems then surface only at read time)."""
    def keep(r: dict) -> bool:
        if flag == "metadata_no_clip":
            return r["meta_status"] != "FAIL" and r.get("clipping_flag_meta") != "True"
        return r[flag] == "True"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["role"] == role and keep(r)]
    if not rows:
        raise ValueError(f"No rows with role={role} and {flag}=True in {path}")
    return sorted(rows, key=lambda r: r["mixture_id"])


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_state() -> dict:
    """Global torch RNG states for a checkpoint: CPU always, every CUDA device when CUDA is available."""
    return {"torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def as_rng_tensor(state, name: str) -> torch.Tensor:
    """torch.set_rng_state / torch.cuda.set_rng_state accept only a CPU uint8 (Byte) tensor. Tensors on any device,
    any integer dtype, lists and numpy arrays of byte values are converted; anything else raises TypeError."""
    if isinstance(state, torch.Tensor):
        t = state.detach().to("cpu")
    elif isinstance(state, (list, tuple, np.ndarray)):
        t = torch.as_tensor(np.asarray(state))
    else:
        raise TypeError(f"checkpoint {name} has unsupported type {type(state).__name__}")
    if t.dtype != torch.uint8:
        if t.dtype.is_floating_point or t.numel() == 0 or int(t.min()) < 0 or int(t.max()) > 255:
            raise TypeError(f"checkpoint {name} is not byte-valued: {t.dtype} {tuple(t.shape)}")
        t = t.to(torch.uint8)
    return t.contiguous()


def restore_rng_state(blob: dict) -> str:
    """Best-effort RNG restore. Never raises: returns "FULL" or "RNG_RESTORE_PARTIAL: <reasons>" and prints the
    latter. Safe because training does not depend on torch's global RNG (data order/crops are seeded numpy per step,
    DFN3 has no dropout); model/optimizer/step are restored before and independently of this."""
    problems = []
    try:
        torch.set_rng_state(as_rng_tensor(blob["torch_rng"], "torch_rng"))
    except Exception as error:
        problems.append(f"cpu: {type(error).__name__}: {error}")
    cuda_states = blob.get("cuda_rng")  # absent in checkpoints written before it was saved
    if torch.cuda.is_available():
        if cuda_states is None:
            problems.append("cuda: no CUDA RNG state in checkpoint")
        elif len(cuda_states) != torch.cuda.device_count():
            problems.append(f"cuda: checkpoint has {len(cuda_states)} device state(s), runtime has {torch.cuda.device_count()}")
        else:
            for index, state in enumerate(cuda_states):
                try:
                    torch.cuda.set_rng_state(as_rng_tensor(state, f"cuda_rng[{index}]"), index)
                except Exception as error:
                    problems.append(f"cuda[{index}]: {type(error).__name__}: {error}")
    elif cuda_states is not None:
        problems.append("cuda: checkpoint has CUDA RNG state but CUDA is not available")
    if problems:
        status = "RNG_RESTORE_PARTIAL: " + "; ".join(problems)
        print(status, flush=True)
        return status
    return "FULL"


class PairBatcher:
    def __init__(self, rows: list[dict], source: AudioSource, segment_sec: float, batch_size: int, seed: int):
        self.rows, self.source, self.batch_size, self.seed = rows, source, batch_size, seed
        self.segment = int(round(segment_sec * SAMPLE_RATE))

    def indices_for_step(self, step: int) -> list[tuple[int, int, int]]:
        """(row index, epoch, position) for each item of the batch at `step` (deterministic)."""
        items, n = [], len(self.rows)
        for k in range(self.batch_size):
            position = step * self.batch_size + k
            epoch, within = divmod(position, n)
            order = np.random.default_rng([self.seed, epoch]).permutation(n)
            items.append((int(order[within]), epoch, position))
        return items

    def load(self, row: dict, epoch: int, position: int) -> tuple[np.ndarray, np.ndarray]:
        _, sr, total = self.source.read(row["noisy_relpath"], 0, 1)
        start = 0
        if total > self.segment:
            start = int(np.random.default_rng([self.seed, epoch, position]).integers(0, total - self.segment + 1))
        noisy, sr_n, _ = self.source.read(row["noisy_relpath"], start, self.segment)
        clean, sr_c, _ = self.source.read(row["clean_relpath"], start, self.segment)
        if not (sr == sr_n == sr_c == SAMPLE_RATE) or noisy.shape != clean.shape:
            raise ValueError(f"{row['mixture_id']}: not an aligned 48 kHz pair")
        noisy, clean = noisy[:, 0], clean[:, 0]
        if len(noisy) < self.segment:
            pad = self.segment - len(noisy)
            noisy, clean = np.pad(noisy, (0, pad)), np.pad(clean, (0, pad))
        return noisy, clean

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        noisy, clean, snr, ids = [], [], [], []
        for index, epoch, position in self.indices_for_step(step):
            row = self.rows[index]
            n, c = self.load(row, epoch, position)
            noisy.append(n); clean.append(c); snr.append(float(row["target_snr_db"])); ids.append(row["mixture_id"])
        return np.stack(noisy), np.stack(clean), np.asarray(snr, dtype=np.float32), ids


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.model, self.state, _ = init_df(cfg.init_model_dir, log_level="ERROR", log_file=None, epoch="best")
        self.model_base_dir = get_model_basedir(cfg.init_model_dir)
        self.device = next(self.model.parameters()).device
        p = ModelParams()
        self.nb_df = p.nb_df
        istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(self.state.fft_window().copy())).to(self.device)
        self.loss_fn = Loss(self.state, istft).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.source = AudioSource([Path(r) for r in cfg.roots], [Path(z) for z in cfg.zips])
        self.train_rows = read_rows(cfg.manifest, cfg.train_role, cfg.row_filter)
        val_flag = "use_for_selection" if cfg.val_role == "val_select" else "use_for_training"
        self.val_rows = read_rows(cfg.val_manifest, cfg.val_role, val_flag) if cfg.validate else []
        if cfg.val_max_files:
            self.val_rows = self.val_rows[: cfg.val_max_files]
        overlap = {r["mixture_id"] for r in self.train_rows} & {r["mixture_id"] for r in self.val_rows}
        if overlap:
            raise ValueError(f"Train/validation overlap: {sorted(overlap)[:5]}")
        self.batcher = PairBatcher(self.train_rows, self.source, cfg.segment_sec, cfg.batch_size, cfg.seed)
        self.step, self.best_val, self.nan_steps = 0, math.inf, 0

    # ------------------------------------------------------------------ core
    def set_train_mode(self) -> None:
        self.model.train()
        if self.cfg.bn_mode == "frozen":
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def spectra(self, audio: np.ndarray):
        spec = self.state.analysis(np.ascontiguousarray(audio, dtype=np.float32))
        spec_t, erb_feat, spec_feat = features_from_spectrum(spec, self.state, self.nb_df)
        complex_spec = torch.as_tensor(spec).unsqueeze(1).to(self.device)
        return complex_spec, spec_t.to(self.device), erb_feat.to(self.device), spec_feat.to(self.device)

    def loss_on(self, noisy: np.ndarray, clean: np.ndarray, snr: np.ndarray) -> torch.Tensor:
        noisy_c, noisy_r, erb_feat, spec_feat = self.spectra(noisy)
        clean_c = torch.as_tensor(self.state.analysis(np.ascontiguousarray(clean, dtype=np.float32))).unsqueeze(1).to(self.device)
        enh, m, lsnr, _ = self.model(noisy_r, erb_feat, spec_feat)
        return self.loss_fn(clean_c, noisy_c, enh, m, lsnr, snrs=torch.as_tensor(snr, device=self.device))

    def lr_at(self, step: int) -> float:
        return self.cfg.lr * min(1.0, (step + 1) / max(self.cfg.warmup_steps, 1))

    def train_step(self) -> dict:
        noisy, clean, snr, ids = self.batcher.batch(self.step)
        self.set_train_mode()
        for group in self.opt.param_groups:
            group["lr"] = self.lr_at(self.step)
        started = time.perf_counter()
        loss = self.loss_on(noisy, clean, snr)
        self.opt.zero_grad(set_to_none=True)
        record = {"step": self.step, "lr": self.lr_at(self.step), "loss": float(loss.detach()), "ids": ";".join(ids)}
        if not torch.isfinite(loss):
            self.nan_steps += 1
            record["skipped"] = "non-finite loss"
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            if torch.isfinite(grad_norm):
                self.opt.step()
                record["grad_norm"] = float(grad_norm)
            else:
                self.nan_steps += 1
                record["skipped"] = "non-finite gradient"
        if self.nan_steps > self.cfg.max_nan_steps:
            raise RuntimeError(f"Too many non-finite steps ({self.nan_steps})")
        record["step_sec"] = time.perf_counter() - started
        self.step += 1
        return record

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        losses, metrics, peaks, started, audio_sec = [], [], [], time.perf_counter(), 0.0
        for row in self.val_rows:
            noisy, _, _ = self.source.read(row["noisy_relpath"])
            clean, _, _ = self.source.read(row["clean_relpath"])
            noisy, clean = noisy[:, 0].astype(np.float64), clean[:, 0].astype(np.float64)
            losses.append(float(self.loss_on(noisy[None], clean[None], np.asarray([float(row["target_snr_db"])]))))
            output = enhance_waveform(self.model, self.state, noisy)
            if not np.isfinite(output).all():
                raise RuntimeError(f"Non-finite validation output for {row['mixture_id']}")
            metrics.append(metrics_from_arrays(clean, noisy, output, SAMPLE_RATE))
            peaks.append(float(np.max(np.abs(output))))
            audio_sec += len(noisy) / SAMPLE_RATE
        pesq_values = [m["pesq"] for m in metrics if m["pesq"] is not None]
        return {"step": self.step, "val_loss": float(np.mean(losses)), "val_files": len(self.val_rows),
                "val_snr_improvement_db": float(np.mean([m["snr_improvement_db"] for m in metrics])),
                "val_stoi": float(np.mean([m["stoi"] for m in metrics])),
                "val_pesq": float(np.mean(pesq_values)) if pesq_values else None,
                "val_output_peak_max": float(np.max(peaks)),
                "val_clipping_samples": int(sum(m["output_clipping_samples"] for m in metrics)),
                "val_sec": time.perf_counter() - started, "val_audio_sec": audio_sec}

    # ------------------------------------------------------------------ persistence
    def save_latest(self) -> None:
        """Write latest.pt via a temp file; the previous checkpoint is kept as latest_prev.pt (disconnect safety)."""
        path = self.run_dir / "checkpoints" / "latest.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "optimizer": self.opt.state_dict(), "step": self.step,
                    "best_val": self.best_val, "nan_steps": self.nan_steps, "config": asdict(self.cfg),
                    **rng_state()}, tmp)
        if path.exists():
            path.replace(path.with_name("latest_prev.pt"))
        tmp.replace(path)

    def resume(self) -> bool:
        """Load latest.pt (falling back to latest_prev.pt if unreadable). Refuses to resume a different experiment."""
        ckpt_dir = self.run_dir / "checkpoints"
        candidates = [p for p in (ckpt_dir / "latest.pt", ckpt_dir / "latest_prev.pt") if p.exists()]
        if not candidates:
            return False
        self.check_same_experiment()
        blob, errors = None, []
        for path in candidates:
            try:
                # load on CPU: load_state_dict copies weights/optimizer state to their devices; RNG states stay on CPU
                blob = torch.load(path, map_location="cpu")
                break
            except Exception as error:  # truncated write after a disconnect
                errors.append(f"{path.name}: {type(error).__name__}: {error}")
        if blob is None:
            raise RuntimeError(f"No readable checkpoint in {ckpt_dir}: {errors}")
        if errors:
            print(f"WARNING: fell back to {path.name}; {errors}", flush=True)
        self.model.load_state_dict(blob["model"])
        self.opt.load_state_dict(blob["optimizer"])
        self.step, self.best_val, self.nan_steps = blob["step"], blob["best_val"], blob["nan_steps"]
        rng_status = restore_rng_state(blob)
        self.truncate_logs(self.step)
        self.append_csv(self.run_dir / "resume_log.csv", {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "resumed_step": self.step, "checkpoint": path.name, "device": str(self.device),
                        "rng_restore": rng_status})
        return True

    def check_same_experiment(self) -> None:
        """A resume must continue the identical experiment: same hyperparameters and same manifests."""
        path = self.run_dir / "run_config.json"
        if not path.exists():
            raise RuntimeError(f"{self.run_dir} has a checkpoint but no run_config.json; refusing to resume")
        saved = json.loads(path.read_text(encoding="utf-8"))
        ignore = {"roots", "zips", "manifest", "val_manifest", "run_dir"}  # paths may differ between sessions
        now = {k: v for k, v in asdict(self.cfg).items() if k not in ignore}
        before = {k: v for k, v in saved["config"].items() if k not in ignore}
        diffs = {k: (before.get(k), now.get(k)) for k in set(now) | set(before) if before.get(k) != now.get(k)}
        if diffs:
            raise RuntimeError(f"Refusing to resume: configuration differs from the original run: {diffs}")
        checks = [("train_manifest_sha256", self.cfg.manifest)]
        if self.cfg.validate:
            checks.append(("val_manifest_sha256", self.cfg.val_manifest))
        for key, file in checks:
            if saved[key] != sha256_file(file):
                raise RuntimeError(f"Refusing to resume: {key} differs from the original run ({file})")

    def truncate_logs(self, step: int) -> None:
        """Drop log rows written after the checkpoint we resume from (they will be recomputed)."""
        for name, keep in (("train_log.csv", lambda s: s < step), ("val_log.csv", lambda s: s <= step)):
            path = self.run_dir / name
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            kept = [r for r in rows if keep(int(r["step"]))]
            if len(kept) != len(rows):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(kept)
                print(f"{name}: removed {len(rows) - len(kept)} rows written after step {step}", flush=True)

    def archive_incomplete_attempt(self) -> None:
        """Fresh start in a folder with logs but no checkpoint: move the old files aside instead of mixing them."""
        leftovers = [p for p in self.run_dir.iterdir() if p.name != "checkpoints"] + list((self.run_dir / "checkpoints").iterdir())
        leftovers = [p for p in leftovers if p.name not in {"stdout.log", "pid"}]
        if not leftovers:
            return
        archive = self.run_dir / f"aborted_attempt_{time.strftime('%Y%m%d_%H%M%S')}"
        archive.mkdir()
        for p in leftovers:
            shutil.move(str(p), archive / p.name)
        print(f"No checkpoint found; previous incomplete attempt moved to {archive}", flush=True)

    def export(self, name: str) -> Path:
        """DFN-compatible model dir loadable with df.enhance.init_df(model_base_dir=...). Written to a temp dir
        first and swapped in, so a disconnect never leaves the previous export deleted."""
        export_dir = self.run_dir / name
        tmp = self.run_dir / f"{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "checkpoints").mkdir(parents=True)
        shutil.copy(Path(self.model_base_dir) / "config.ini", tmp / "config.ini")
        torch.save(self.model.state_dict(), tmp / "checkpoints" / f"model_{self.step}.ckpt.best")
        (tmp / "export_info.json").write_text(json.dumps({"step": self.step, "best_val_loss": self.best_val,
                                                          "source_run": str(self.run_dir)}, indent=2))
        old = self.run_dir / f"{name}.old"
        if old.exists():
            shutil.rmtree(old)
        if export_dir.exists():
            export_dir.rename(old)
        tmp.rename(export_dir)
        if old.exists():
            shutil.rmtree(old)
        return export_dir

    def write_run_config(self) -> None:
        pretrained_cp = next((Path(self.model_base_dir) / "checkpoints").glob("*.best"))
        info = {"config": asdict(self.cfg), "design": __doc__,
                "train_manifest_sha256": sha256_file(self.cfg.manifest),
                "val_manifest_sha256": sha256_file(self.cfg.val_manifest) if self.cfg.validate else None,
                "train_rows": len(self.train_rows), "val_rows": [r["mixture_id"] for r in self.val_rows],
                "init_checkpoint": str(pretrained_cp), "init_checkpoint_sha256": sha256_file(pretrained_cp),
                "code_sha256": {p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("data.py"),
                                                                 Path(__file__).parents[1] / "sih_model" / "dfn3.py",
                                                                 Path(__file__).with_name("evaluation.py"))},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "device": str(self.device), "platform": platform.platform()}}
        (self.run_dir / "run_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    @staticmethod
    def append_csv(path: Path, record: dict) -> None:
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record.keys()), extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(record)

    def run(self, resume: bool = True) -> None:
        resumed = resume and self.resume()
        if not resumed:
            if resume:
                self.archive_incomplete_attempt()
            self.write_run_config()
            if self.cfg.validate:
                baseline = self.validate()
                baseline.update(best=True, note="initial checkpoint (before fine-tuning)")
                self.append_csv(self.run_dir / "val_log.csv", baseline)
                self.best_val = baseline["val_loss"]
                print(f"[step 0] val_loss {baseline['val_loss']:.5f} dSNR {baseline['val_snr_improvement_db']:.2f} "
                      f"STOI {baseline['val_stoi']:.3f} PESQ {baseline['val_pesq']}", flush=True)
            else:
                print("validation disabled: no val passes; export_best will be the final model", flush=True)
        else:
            print(f"resumed at step {self.step} (best val_loss {self.best_val:.5f})"
                  + (" - run already complete" if self.step >= self.cfg.steps else ""), flush=True)
        while self.step < self.cfg.steps:
            record = self.train_step()
            self.append_csv(self.run_dir / "train_log.csv", {k: record.get(k) for k in
                            ("step", "lr", "loss", "grad_norm", "step_sec", "skipped", "ids")})
            if self.step % 10 == 0:
                print(f"step {self.step} loss {record['loss']:.5f} ({record['step_sec']:.2f} s/step)", flush=True)
            if self.cfg.validate and (self.step % self.cfg.val_every == 0 or self.step == self.cfg.steps):
                val = self.validate()
                improved = val["val_loss"] < self.best_val
                val.update(best=improved, note="")
                self.append_csv(self.run_dir / "val_log.csv", val)
                if improved:
                    self.best_val = val["val_loss"]
                    self.export("export_best")
                print(f"[step {self.step}] val_loss {val['val_loss']:.5f}{' *best' if improved else ''} "
                      f"dSNR {val['val_snr_improvement_db']:.2f} STOI {val['val_stoi']:.3f} PESQ {val['val_pesq']}", flush=True)
            if self.step % self.cfg.checkpoint_every == 0 or self.step == self.cfg.steps:
                self.save_latest()
        self.export("export_last")
        if not self.cfg.validate:
            self.export("export_best")  # no selection possible: best == final step
