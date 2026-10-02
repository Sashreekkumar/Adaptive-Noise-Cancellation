"""B0 (pretrained DFN3) vs B1 (fine-tuned DFN3) on the frozen evaluation pairs.

Metrics per file: input/output SNR, dSNR, STOI, PESQ-WB (16 kHz metric copy), SI-SDR (+ SI-SDR improvement).
Summary per suite and per input-SNR bin (<0, 0-5, 5-10, 10-15, 15-20, >=20 dB, measured input SNR).
  B0 only (baseline check):  python scripts/eval_b0_b1.py --tag b0_check
  B0 vs B1:                  python scripts/eval_b0_b1.py --ft <export_best dir | best.pt | latest.pt> --tag e1_ft_v1
Output: D:\\SIH26052\\experiments\\eval_b0_b1\\<tag>\\{per_file.csv, summary.md} (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sih_model.dfn3 import enhance_waveform, load_dfn3
from sih_train.evaluation import metrics_from_arrays

SUITES = {"transient": Path(r"D:\SIH26052\validation\alignment_eligible_manifest.csv"),
          "continuous": Path(r"D:\SIH26052\validation\continuous_eval_v1_manifest.csv")}
BINS = [(-np.inf, 0, "<0"), (0, 5, "0-5"), (5, 10, "5-10"), (10, 15, "10-15"), (15, 20, "15-20"), (20, np.inf, ">=20")]


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = reference - reference.mean(), estimate - estimate.mean()
    target = np.dot(estimate, reference) / max(np.dot(reference, reference), 1e-20) * reference
    return float(10 * np.log10(max(np.dot(target, target), 1e-20) / max(np.dot(estimate - target, estimate - target), 1e-20)))


def load_ft(path: Path):
    """export dir (init_df layout) or a .pt file: bare state_dict (model_*.ckpt.best / best.pt) or trainer latest.pt."""
    if path.is_dir():
        return load_dfn3(str(path))
    model, state, _ = load_dfn3()
    blob = torch.load(path, map_location="cpu")
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(sd)  # strict: must be a full DFN3 state_dict
    return model, state, path.name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ft", type=Path, default=None, help="B1 model: export_best dir or .pt file (omit for B0 only)")
    p.add_argument("--tag", required=True)
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    p.add_argument("--output-root", type=Path, default=Path(r"D:\SIH26052\experiments\eval_b0_b1"))
    a = p.parse_args()
    out = a.output_root / a.tag
    if out.exists():
        raise FileExistsError(f"{out} exists; use a new --tag")
    systems = {"B0_pretrained": load_dfn3()}
    if a.ft is not None:
        systems["B1_ft"] = load_ft(a.ft)
    rows = []
    for suite in a.suites:
        with SUITES[suite].open(newline="", encoding="utf-8") as h:
            pairs = list(csv.DictReader(h))
        for pair in pairs:
            clean, sr = sf.read(pair["local_clean_path"], dtype="float64")
            noisy, sr_n = sf.read(pair["local_noisy_path"], dtype="float64")
            assert sr == sr_n == 48000 and len(clean) == len(noisy)
            base = {"suite": suite, "mixture_id": pair["mixture_id"], "category": pair["category"]}
            for name, (model, state, _) in systems.items():
                y = enhance_waveform(model, state, noisy)
                m = metrics_from_arrays(clean, noisy, y, 48000)
                rows.append({**base, "system": name, "input_snr_db": m["input_snr_db"], "output_snr_db": m["output_snr_db"],
                             "dsnr_db": m["snr_improvement_db"], "stoi": m["stoi"], "pesq_wb": m["pesq"],
                             "si_sdr_db": si_sdr(clean, y), "si_sdr_in_db": si_sdr(clean, noisy),
                             "output_peak": m["output_peak"]})
                r = rows[-1]
                print(f"{suite:10s} {pair['mixture_id']:24s} {name:14s} in {r['input_snr_db']:6.2f} out {r['output_snr_db']:6.2f} "
                      f"dSNR {r['dsnr_db']:+6.2f} STOI {r['stoi']:.3f} PESQ {r['pesq_wb']:.3f} SI-SDR {r['si_sdr_db']:6.2f}")
    for r in rows:
        r["si_sdr_improvement_db"] = r["si_sdr_db"] - r["si_sdr_in_db"]
        r["snr_bin"] = next(label for lo, hi, label in BINS if lo <= r["input_snr_db"] < hi)
    out.mkdir(parents=True)
    with (out / "per_file.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def table(title: str, key) -> list[str]:
        lines = [f"## {title}", "", "| group | system | n | out SNR | dSNR | STOI | PESQ-WB | SI-SDR | dSI-SDR |", "|---|---|---|---|---|---|---|---|---|"]
        groups = sorted({key(r) for r in rows}, key=lambda g: [b[2] for b in BINS].index(g) if g in [b[2] for b in BINS] else g)
        for g in groups:
            for name in systems:
                s = [r for r in rows if key(r) == g and r["system"] == name]
                f = lambda k: np.mean([r[k] for r in s if r[k] is not None])
                lines.append(f"| {g} | {name} | {len(s)} | {f('output_snr_db'):.2f} | {f('dsnr_db'):.2f} | {f('stoi'):.3f} | "
                             f"{f('pesq_wb'):.3f} | {f('si_sdr_db'):.2f} | {f('si_sdr_improvement_db'):.2f} |")
        return lines + [""]

    ft_info = ""
    if a.ft is not None:
        f = a.ft if a.ft.is_file() else next((a.ft / "checkpoints").glob("*.ckpt.best"))
        ft_info = f"B1 = {a.ft} ({f.name}, sha256 {hashlib.sha256(f.read_bytes()).hexdigest()[:16]})"
    text = [f"# {a.tag}: B0 pretrained DFN3" + (" vs B1 fine-tuned" if a.ft else ""), "", ft_info, "",
            *table("By suite", lambda r: r["suite"]), *table("By measured input SNR (dB)", lambda r: r["snr_bin"]),
            "Bins are small (12 frozen pairs); treat per-bin differences as indicative."]
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
