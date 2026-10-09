"""Train the per-ERB-band expert FusionGate over two frozen DFN3 experts (default E0 pretrained + E1; v2: --e0 E1 --e1 E2).

v2 changes: --e0 picks the first expert; the gate receives the causal impulse features (sih_model/impulse.py);
--tier-c-weight N repeats tier-C (impulsive) gate-train rows N times so the impulse expert is seen often enough.

Data   gate-train rows = val manifest role `val_pool` (verified, not clipped): disjoint from E1's training data
       (train_fast), from the 42 `val_select` selection files, the frozen evaluation files, calibration and test.
       ~30 % of batches are remixed to -10..0 dB with noise = noisy - clean (deterministic per step).
Model  ExpertBank (ONE shared DFN3 STFT of the noisy input) -> FusionGate -> fused spectrum. Only the gate trains.
Loss   DFN3's own MultiResSpecLoss (checkpoint config: 500/500, gamma 0.3, FFT 256..2048) on iSTFT(fused) vs
       iSTFT(clean spectrum) + lambda * temporal smoothness of the band weights. The clean STFT is a training target
       only; inference uses the noisy input alone.
Select validation loss on the fixed 42-file val_select set (full files); best gate -> <run>/best_gate.pt.
Resume automatic from <run>/checkpoints/latest.pt (refuses if the configuration changed).

  python scripts/train_gate.py --e1 <FT-v1 export_best dir | .pt> --run-dir D:\\SIH26052\\experiments\\gate\\gate_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from df.loss import Istft, Loss
from df.model import ModelParams

from sih_train.data import AudioSource
from sih_train.dfn_finetune import PairBatcher
from sih_train.evaluation import metrics_from_arrays
from sih_model.experts import ExpertBank, load_expert
from sih_model.fusion_gate import FusionGate
from sih_model.impulse import impulse_features

SR = 48000


@dataclass
class GateConfig:
    run_dir: str
    e0: str = "pretrained"                 # first expert: "pretrained" or an export dir / .pt (v2: E1)
    tier_c_weight: int = 1                  # repeat tier-C gate-train rows this many times
    e1: str = ""                                 # FT-v1 export_best dir or .pt; "pretrained" = E0 copy (smoke only)
    manifest: str = r"D:\SIH26052\manifests\val_manifest.csv"
    roots: list[str] = field(default_factory=list)
    zips: list[str] = field(default_factory=list)
    train_role: str = "val_pool"
    val_role: str = "val_select"
    val_max_files: int = 0
    steps: int = 2000
    batch_size: int = 8
    segment_sec: float = 3.0
    lr: float = 1e-3
    lambda_smooth: float = 0.01
    remix_prob: float = 0.3
    remix_snr_low: float = -10.0
    remix_snr_high: float = 0.0
    val_every: int = 250
    checkpoint_every: int = 100
    seed: int = 26052


def gate_rows(path: str, role: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as h:
        rows = [r for r in csv.DictReader(h) if r["role"] == role and r["audio_status"] in {"PASS", "WARN"}
                and "clipped" not in (r.get("audio_flags") or "")]
    if not rows:
        raise ValueError(f"no usable rows with role={role} in {path}")
    if any(r["noisy_relpath"].startswith("mixtures/train_fast") for r in rows):
        raise ValueError("gate rows overlap E1 training data (train_fast)")
    return sorted(rows, key=lambda r: r["mixture_id"])


def remix(noisy: np.ndarray, clean: np.ndarray, rng: np.random.Generator, low: float, high: float):
    """Per item: noisy' = clean + g * (noisy - clean) at a target SNR in [low, high] dB; joint peak safety."""
    out_n, out_c = noisy.copy(), clean.copy()
    for i in range(len(noisy)):
        noise = noisy[i] - clean[i]
        ps, pn = float(np.mean(clean[i] ** 2)), float(np.mean(noise ** 2))
        if ps < 1e-10 or pn < 1e-12:
            continue                         # silent crop: keep as is
        g = np.sqrt(ps / (pn * 10 ** (rng.uniform(low, high) / 10)))
        n2 = clean[i] + g * noise
        peak = float(np.max(np.abs(n2)))
        scale = 0.99 / peak if peak > 0.99 else 1.0
        out_n[i], out_c[i] = n2 * scale, clean[i] * scale
    return out_n.astype(np.float32), out_c.astype(np.float32)


class GateTrainer:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        e0 = load_expert(None) if cfg.e0 == "pretrained" else load_expert(cfg.e0)
        e1 = load_expert(None) if cfg.e1 == "pretrained" else load_expert(cfg.e1)
        self.bank = ExpertBank([e0, e1])
        self.device = self.bank.device
        self.gate = FusionGate(2, self.bank.erb_inv_fb).to(self.device)
        self.opt = torch.optim.Adam(self.gate.parameters(), lr=cfg.lr)
        p = ModelParams()
        self.istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(self.bank.state.fft_window().copy())).to(self.device)
        self.mrsl = Loss(self.bank.state, self.istft).to(self.device).mrsl   # DFN3's own MultiResSpecLoss (checkpoint config)
        if self.mrsl is None:
            raise RuntimeError("checkpoint config has no MultiResSpecLoss")
        source = AudioSource([Path(r) for r in cfg.roots], [Path(z) for z in cfg.zips])
        self.source = source
        rows = gate_rows(cfg.manifest, cfg.train_role)
        self.train_rows = rows + [r for r in rows if r.get("tier") == "C"] * (cfg.tier_c_weight - 1)
        print(f"gate-train rows: {len(rows)} ({sum(r.get('tier') == 'C' for r in rows)} tier C) -> {len(self.train_rows)} after weighting", flush=True)
        with open(cfg.manifest, newline="", encoding="utf-8") as h:
            self.val_rows = sorted((r for r in csv.DictReader(h) if r["role"] == cfg.val_role and r["use_for_selection"] == "True"),
                                   key=lambda r: r["mixture_id"])
        if cfg.val_max_files:
            self.val_rows = self.val_rows[: cfg.val_max_files]
        assert not {r["mixture_id"] for r in self.train_rows} & {r["mixture_id"] for r in self.val_rows}
        self.batcher = PairBatcher(self.train_rows, source, cfg.segment_sec, cfg.batch_size, cfg.seed)
        self.step, self.best_val = 0, float("inf")

    # ---------------------------------------------------------------- core
    def clean_spec(self, clean: np.ndarray) -> torch.Tensor:
        x = np.atleast_2d(clean).astype(np.float32)
        return torch.as_tensor(self.bank.state.analysis(np.ascontiguousarray(np.pad(x, ((0, 0), (0, self.bank.state.fft_size())))))).to(self.device)

    def loss(self, fused: torch.Tensor, clean_spec: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        enh_td = self.istft(torch.view_as_real(fused).unsqueeze(1))
        clean_td = self.istft(torch.view_as_real(clean_spec).unsqueeze(1))
        spec_loss = self.mrsl(enh_td, clean_td)
        smooth = (w[:, 1:] - w[:, :-1]).pow(2).mean()
        return spec_loss + self.cfg.lambda_smooth * smooth, float(spec_loss), float(smooth)

    def train_step(self) -> dict:
        noisy, clean, _, ids = self.batcher.batch(self.step)
        rng = np.random.default_rng([self.cfg.seed, self.step, 7])
        remixed = bool(rng.random() < self.cfg.remix_prob)
        if remixed:
            noisy, clean = remix(noisy, clean, rng, self.cfg.remix_snr_low, self.cfg.remix_snr_high)
        started = time.perf_counter()
        self.gate.train()
        outs = self.bank(noisy)                                   # frozen experts, no_grad, one STFT
        fused, w, _ = self.gate(outs, impulse=impulse_features(outs.noisy_spec))
        loss, spec_loss, smooth = self.loss(fused, self.clean_spec(clean), w)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gate.parameters(), 1.0)
        self.opt.step()
        self.step += 1
        wm = w.detach().mean(dim=(0, 1, 2)).cpu().numpy()
        return {"step": self.step - 1, "loss": float(loss), "spec_loss": spec_loss, "smooth": smooth, "remixed": remixed,
                "w_e0": float(wm[0]), "w_e1": float(wm[1]), "step_sec": time.perf_counter() - started, "ids": ";".join(ids)}

    @torch.no_grad()
    def validate(self) -> dict:
        self.gate.eval()
        losses, mets, w_sum = [], [], np.zeros(2)
        for row in self.val_rows:
            noisy, _, _ = self.source.read(row["noisy_relpath"])
            clean, _, _ = self.source.read(row["clean_relpath"])
            noisy, clean = noisy[:, 0], clean[:, 0]
            outs = self.bank(noisy)
            fused, w, _ = self.gate(outs, impulse=impulse_features(outs.noisy_spec))
            losses.append(self.loss(fused, self.clean_spec(clean), w)[1])
            y = self.bank.synthesize(fused, outs.orig_len)[0].astype(np.float64)
            if not np.isfinite(y).all():
                raise RuntimeError(f"non-finite validation output {row['mixture_id']}")
            mets.append(metrics_from_arrays(clean.astype(np.float64), noisy.astype(np.float64), y, SR))
            w_sum += w.mean(dim=(0, 1, 2)).cpu().numpy()
        pesq = [m["pesq"] for m in mets if m["pesq"] is not None]
        return {"step": self.step, "val_loss": float(np.mean(losses)), "val_files": len(self.val_rows),
                "val_dsnr_db": float(np.mean([m["snr_improvement_db"] for m in mets])),
                "val_stoi": float(np.mean([m["stoi"] for m in mets])), "val_pesq": float(np.mean(pesq)) if pesq else None,
                "val_w_e0": float(w_sum[0] / len(self.val_rows)), "val_w_e1": float(w_sum[1] / len(self.val_rows))}

    # ---------------------------------------------------------------- persistence
    def _state(self) -> dict:
        return {"gate": self.gate.state_dict(), "optimizer": self.opt.state_dict(), "step": self.step,
                "best_val": self.best_val, "config": asdict(self.cfg)}

    def save_latest(self) -> None:
        path = self.run_dir / "checkpoints" / "latest.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(self._state(), tmp)
        if path.exists():
            path.replace(path.with_name("latest_prev.pt"))
        tmp.replace(path)

    def resume(self) -> bool:
        path = self.run_dir / "checkpoints" / "latest.pt"
        if not path.exists():
            return False
        blob = torch.load(path, map_location="cpu")
        ignore = {"roots", "zips", "manifest", "run_dir", "e0", "e1"}
        diff = {k: (blob["config"].get(k), v) for k, v in asdict(self.cfg).items() if k not in ignore and blob["config"].get(k) != v}
        if diff:
            raise RuntimeError(f"refusing to resume: configuration changed {diff}")
        self.gate.load_state_dict(blob["gate"])
        self.opt.load_state_dict(blob["optimizer"])
        self.step, self.best_val = blob["step"], blob["best_val"]
        for name, keep in (("train_log.csv", lambda s: s < self.step), ("val_log.csv", lambda s: s <= self.step)):
            p = self.run_dir / name
            if p.exists():
                rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
                kept = [r for r in rows if keep(int(r["step"]))]
                if len(kept) != len(rows):
                    with p.open("w", newline="", encoding="utf-8") as h:
                        wr = csv.DictWriter(h, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(kept)
        return True

    @staticmethod
    def append(path: Path, row: dict) -> None:
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(row)); (w.writeheader() if new else None); w.writerow(row)

    def write_run_config(self) -> None:
        e1_file = Path(self.cfg.e1)
        if e1_file.is_dir():
            e1_file = next((e1_file / "checkpoints").glob("*.ckpt.best"))
        info = {"config": asdict(self.cfg), "design": __doc__, "experts": self.bank.names,
                "e1_sha256": hashlib.sha256(e1_file.read_bytes()).hexdigest() if e1_file.is_file() else self.cfg.e1,
                "train_rows": len(self.train_rows), "val_rows": [r["mixture_id"] for r in self.val_rows],
                "device": str(self.device), "torch": torch.__version__}
        (self.run_dir / "run_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    def run(self) -> None:
        if self.resume():
            print(f"resumed at step {self.step} (best val_loss {self.best_val:.5f})", flush=True)
        else:
            self.write_run_config()
            v = self.validate(); v["best"] = True
            self.best_val = v["val_loss"]
            self.append(self.run_dir / "val_log.csv", v)
            torch.save({"gate": self.gate.state_dict(), "step": 0, "val": v, "config": asdict(self.cfg)}, self.run_dir / "best_gate.pt")
            print(f"[step 0 uniform gate] {v}", flush=True)
        while self.step < self.cfg.steps:
            r = self.train_step()
            self.append(self.run_dir / "train_log.csv", r)
            if self.step % 10 == 0:
                print(f"step {self.step} loss {r['loss']:.4f} w_e1 {r['w_e1']:.3f} ({r['step_sec']:.2f} s)", flush=True)
            if self.step % self.cfg.val_every == 0 or self.step == self.cfg.steps:
                v = self.validate()
                v["best"] = v["val_loss"] < self.best_val
                self.append(self.run_dir / "val_log.csv", v)
                if v["best"]:
                    self.best_val = v["val_loss"]
                    torch.save({"gate": self.gate.state_dict(), "step": self.step, "val": v, "config": asdict(self.cfg)},
                               self.run_dir / "best_gate.pt")
                print(f"[val step {self.step}] {v}", flush=True)
            if self.step % self.cfg.checkpoint_every == 0 or self.step == self.cfg.steps:
                self.save_latest()


def main() -> None:
    d = GateConfig(run_dir="", e1="")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--e0", default="pretrained", help="first expert: 'pretrained' or export dir / .pt (v2: E1)")
    p.add_argument("--e1", required=True, help="second expert: export dir or .pt (v2: E2 impulse); 'pretrained' = smoke")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--manifest", default=d.manifest)
    p.add_argument("--root", action="append", default=[])
    p.add_argument("--zip", action="append", default=[])
    for name in ("steps", "batch_size", "val_every", "checkpoint_every", "seed", "val_max_files", "tier_c_weight"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(d, name))
    for name in ("segment_sec", "lr", "lambda_smooth", "remix_prob"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(d, name))
    a = p.parse_args()
    cfg = GateConfig(run_dir=a.run_dir, e0=a.e0, tier_c_weight=a.tier_c_weight, e1=a.e1, manifest=a.manifest, roots=a.root, zips=a.zip, steps=a.steps,
                     batch_size=a.batch_size, val_every=a.val_every, checkpoint_every=a.checkpoint_every, seed=a.seed,
                     val_max_files=a.val_max_files, segment_sec=a.segment_sec, lr=a.lr, lambda_smooth=a.lambda_smooth,
                     remix_prob=a.remix_prob)
    GateTrainer(cfg).run()


if __name__ == "__main__":
    main()
