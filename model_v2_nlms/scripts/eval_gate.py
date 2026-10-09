"""B0 pretrained / B1 FT-v1 / B2 uniform 50-50 fusion / B3 trained FusionGate / B4 impulse rule on the frozen evaluation pairs.

Same pairs, SNR bins, SI-SDR and metrics_from_arrays as scripts/eval_b0_b1.py (imported from it). B2/B3 use
ExpertBank's single shared STFT and FusionGate (B2 = zero-initialised gate -> exactly uniform weights).
  python scripts/eval_gate.py --e1 <E1 export_best> --gate <best_gate.pt> --tag c2_gate_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_b0_b1 import BINS, SUITES, si_sdr  # same pairs, bins and SI-SDR as B0/B1 evaluation
from sih_model.dfn3 import enhance_waveform
from sih_train.evaluation import metrics_from_arrays
from sih_model.experts import ExpertBank, load_expert
from sih_model.fusion_gate import FusionGate
from sih_model.impulse import impulse_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--e0", default="pretrained", help="first expert (v2: E1 export dir); B0 = this expert alone")
    p.add_argument("--e1", type=Path, required=True, help="second expert (v2: E2 export dir); B1 = this expert alone")
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-root", type=Path, default=Path(r"D:\SIH26052\experiments\eval_b0_b1"))
    a = p.parse_args()
    out = a.output_root / a.tag
    if out.exists():
        raise FileExistsError(f"{out} exists; use a new --tag")
    e0 = load_expert(None) if a.e0 == "pretrained" else load_expert(a.e0)
    e1 = load_expert(a.e1)
    bank = ExpertBank([e0, e1])
    uniform = FusionGate(2, bank.erb_inv_fb).to(bank.device).eval()
    trained = FusionGate(2, bank.erb_inv_fb).to(bank.device).eval()
    blob = torch.load(a.gate, map_location="cpu")
    trained.load_state_dict(blob["gate"])
    rows, issues = [], []
    for suite, manifest in SUITES.items():
        pairs = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
        for pair in pairs:
            clean, sr = sf.read(pair["local_clean_path"], dtype="float64")
            noisy, _ = sf.read(pair["local_noisy_path"], dtype="float64")
            outs = bank(noisy)
            ys = {"B0_pretrained": (enhance_waveform(e0[0], e0[1], noisy), None),
                  "B1_ft_v1": (enhance_waveform(e1[0], e1[1], noisy), None)}
            with torch.no_grad():
                impulse = impulse_features(outs.noisy_spec)                       # [B,T,2] (flag, severity)
                for name, gate in (("B2_uniform", uniform), ("B3_gate", trained)):
                    fused, w, _ = gate(outs, impulse=impulse)
                    if not (torch.isfinite(w).all() and torch.allclose(w.sum(-1), torch.ones(1), atol=1e-5)):
                        issues.append(f"{pair['mixture_id']} {name}: weights non-finite or not summing to 1")
                    ys[name] = (bank.synthesize(fused, outs.orig_len)[0].astype(np.float64), w[0].cpu().numpy())
                # B4 rule: second expert (v2: E2 impulse) in all 32 bands where the impulse flag is 1, first expert elsewhere
                flag = impulse[..., 0][..., None].expand(-1, -1, 32)                  # [B,T,32]
                w = torch.stack([1.0 - flag, flag], dim=-1)                            # [B,T,32,2]
                fused = uniform.fuse(outs.spec, w)
                ys["B4_rule"] = (bank.synthesize(fused, outs.orig_len)[0].astype(np.float64), w[0].cpu().numpy())
            for name, (y, w) in ys.items():
                if not np.isfinite(y).all():
                    issues.append(f"{pair['mixture_id']} {name}: non-finite output")
                m = metrics_from_arrays(clean, noisy, y, sr)
                r = {"suite": suite, "mixture_id": pair["mixture_id"], "category": pair["category"], "system": name,
                     "input_snr_db": m["input_snr_db"], "output_snr_db": m["output_snr_db"], "dsnr_db": m["snr_improvement_db"],
                     "stoi": m["stoi"], "pesq_wb": m["pesq"], "pesq_nb": m["pesq_nb"], "si_sdr_db": si_sdr(clean, y),
                     "si_sdr_improvement_db": si_sdr(clean, y) - si_sdr(clean, noisy), "output_peak": m["output_peak"],
                     "w_e0": float(w[..., 0].mean()) if w is not None else None, "w_e1": float(w[..., 1].mean()) if w is not None else None}
                r["snr_bin"] = next(label for lo, hi, label in BINS if lo <= r["input_snr_db"] < hi)
                if r["output_peak"] >= 0.999:
                    issues.append(f"{pair['mixture_id']} {name}: output peak {r['output_peak']:.3f}")
                rows.append(r)
            print(pair["mixture_id"], " ".join(f"{n}:{r['dsnr_db']:+.2f}" for n, r in zip(ys, rows[-len(ys):])), flush=True)
    out.mkdir(parents=True)
    with (out / "per_file.csv").open("w", newline="", encoding="utf-8") as h:
        wr = csv.DictWriter(h, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    systems = ["B0_pretrained", "B1_ft_v1", "B2_uniform", "B3_gate", "B4_rule"]
    order = [b[2] for b in BINS]

    def mean(sel, k):  # PESQ can be None for files the library rejects
        v = [r[k] for r in sel if r[k] is not None]
        return np.mean(v) if v else float("nan")

    def table(title, key):
        lines = [f"## {title}", "", "| group | system | n | out SNR | dSNR | STOI | PESQ-WB | PESQ-NB | SI-SDR | dSI-SDR | w_E0 | w_E1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in sorted({key(r) for r in rows}, key=lambda g: order.index(g) if g in order else -1 if g == "ALL" else 0):
            for s in systems:
                sel = [r for r in rows if key(r) == g and r["system"] == s]
                f = lambda k: mean(sel, k)
                wv = (f"{f('w_e0'):.3f} | {f('w_e1'):.3f}" if sel[0]["w_e0"] is not None else "- | -")
                lines.append(f"| {g} | {s} | {len(sel)} | {f('output_snr_db'):.2f} | {f('dsnr_db'):.2f} | {f('stoi'):.4f} | "
                             f"{f('pesq_wb'):.3f} | {f('pesq_nb'):.3f} | {f('si_sdr_db'):.2f} | {f('si_sdr_improvement_db'):.2f} | {wv} |")
        return lines + [""]

    per_file = ["## Per file", "", "| mixture | input SNR | " + " | ".join(f"{s} dSNR / PESQ-WB / PESQ-NB / STOI" for s in systems) + " |",
                "|---|---|" + "---|" * len(systems)]
    for mid in dict.fromkeys(r["mixture_id"] for r in rows):
        sel = {r["system"]: r for r in rows if r["mixture_id"] == mid}
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        per_file.append(f"| {mid} | {sel['B0_pretrained']['input_snr_db']:.2f} | " +
                        " | ".join(f"{sel[s]['dsnr_db']:+.2f} / {fmt(sel[s]['pesq_wb'])} / {fmt(sel[s]['pesq_nb'])} / {sel[s]['stoi']:.3f}"
                                   for s in systems) + " |")
    gate_file = hashlib.sha256(a.gate.read_bytes()).hexdigest()[:16]
    text = [f"# {a.tag}: B0 / B1 / B2 uniform / B3 trained gate / B4 impulse rule (frozen evaluation pairs)", "",
            f"E0 = {a.e0}; E1 = {a.e1}; gate = {a.gate} (sha256 {gate_file}, trained step {blob.get('step')})", "",
            "B0 = first expert (--e0) alone, B1 = second expert (--e1) alone; B4_rule = second expert in all 32 bands where "
            "the causal impulse flag (sih_model/impulse.py) is 1, first expert elsewhere.", "",
            *table("Overall", lambda r: "ALL"), *table("By suite", lambda r: r["suite"]),
            *table("By measured input SNR (dB)", lambda r: r["snr_bin"]), *per_file, "",
            "## Numerical issues", "", *(issues or ["none (weights finite and summing to 1, outputs finite, peaks < 0.999)"]),
            "", "FusionGate has no guards; the residual-canceller guards are not in this path."]
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"e0": str(a.e0), "e1": str(a.e1), "gate": str(a.gate), "gate_sha256": gate_file,
                                                 "suites": {k: str(v) for k, v in SUITES.items()}}, indent=2))
    print("\n".join(text))


if __name__ == "__main__":
    main()
