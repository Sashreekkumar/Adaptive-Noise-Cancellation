"""
evaluate_metrics.py

Computes SNR, PESQ, and STOI between clean reference files and
DeepFilterNet3-enhanced files (matched by filename stem). Optionally
also scores the original noisy files against clean, as a baseline, so
you can see the improvement DeepFilterNet3 provides.

Usage:
    python evaluate_metrics.py \
        --clean_dir ./clean \
        --enhanced_dir ./enhanced \
        --noisy_dir ./noisy \
        --output_csv results.csv

Assumptions:
- clean/{stem}.mp3 (or .wav) corresponds to enhanced/{stem}.wav
  (the enhance script writes .wav outputs with the same stem as the
  input mp3). If your noisy/clean filenames don't match exactly,
  rename them first or adjust the matching logic below.
- PESQ is computed in wideband mode ('wb'), which requires 16 kHz
  audio; all signals are resampled to 16 kHz for PESQ and STOI.
- SNR is a simple time-domain global SNR: 10*log10(sum(clean^2) /
  sum((clean-estimate)^2)) after trimming to equal length. This
  assumes clean and enhanced/noisy signals are time-aligned (which
  they should be, since DeepFilterNet3 and mp3 decoding do not
  introduce significant delay/drift). For stricter alignment you can
  cross-correlate first; see align_signals() below (off by default).
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

EVAL_SR = 16000  # sample rate used for PESQ (wb) and STOI


def align_signals(ref: np.ndarray, deg: np.ndarray, max_shift: int = 4000) -> np.ndarray:
    """Optionally align deg to ref using cross-correlation, searching
    +/- max_shift samples. Returns the shifted/truncated deg signal.
    Not called by default; enable with --align if you see suspiciously
    low PESQ/STOI scores that suggest misalignment."""
    n = min(len(ref), len(deg))
    ref_seg = ref[:n]
    deg_seg = deg[:n]
    corr = np.correlate(deg_seg - deg_seg.mean(), ref_seg - ref_seg.mean(), mode="full")
    lag = np.argmax(corr) - (n - 1)
    lag = int(np.clip(lag, -max_shift, max_shift))
    if lag > 0:
        deg = deg[lag:]
    elif lag < 0:
        deg = np.concatenate([np.zeros(-lag, dtype=deg.dtype), deg])
    return deg


def compute_snr(clean: np.ndarray, est: np.ndarray) -> float:
    n = min(len(clean), len(est))
    clean = clean[:n]
    est = est[:n]
    noise = clean - est
    num = np.sum(clean.astype(np.float64) ** 2)
    den = np.sum(noise.astype(np.float64) ** 2) + 1e-12
    return 10 * np.log10(num / den)


def load_16k(path: Path) -> np.ndarray:
    audio, sr = librosa.load(str(path), sr=EVAL_SR, mono=True)
    return audio


def evaluate_pair(clean_path: Path, deg_path: Path, do_align: bool = False):
    clean = load_16k(clean_path)
    deg = load_16k(deg_path)

    if do_align:
        deg = align_signals(clean, deg)

    n = min(len(clean), len(deg))
    clean = clean[:n]
    deg = deg[:n]

    snr_val = compute_snr(clean, deg)
    pesq_val = pesq(EVAL_SR, clean, deg, "wb")
    stoi_val = stoi(clean, deg, EVAL_SR, extended=False)
    return snr_val, pesq_val, stoi_val


def find_matching_file(folder: Path, stem: str):
    for ext in (".wav", ".mp3"):
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute SNR/PESQ/STOI for enhanced (and optionally noisy) audio")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--enhanced_dir", required=True)
    parser.add_argument("--noisy_dir", default=None, help="Optional: also score noisy-vs-clean as a baseline")
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--align", action="store_true", help="Cross-correlate to fix small time misalignment")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    enhanced_dir = Path(args.enhanced_dir)
    noisy_dir = Path(args.noisy_dir) if args.noisy_dir else None

    clean_files = sorted(list(clean_dir.glob("*.mp3")) + list(clean_dir.glob("*.wav")))
    if not clean_files:
        raise SystemExit(f"No .mp3/.wav files found in {clean_dir}")

    rows = []
    for cf in tqdm(clean_files, desc="Evaluating"):
        stem = cf.stem
        enh_path = find_matching_file(enhanced_dir, stem)
        if enh_path is None:
            print(f"[SKIP] No enhanced file found for '{stem}' in {enhanced_dir}")
            continue

        row = {"file": stem}
        try:
            snr_e, pesq_e, stoi_e = evaluate_pair(cf, enh_path, do_align=args.align)
            row.update(snr_enhanced=snr_e, pesq_enhanced=pesq_e, stoi_enhanced=stoi_e)
        except Exception as e:
            print(f"[ERROR] enhanced eval failed for '{stem}': {e}")
            continue

        if noisy_dir is not None:
            noisy_path = find_matching_file(noisy_dir, stem)
            if noisy_path is not None:
                try:
                    snr_n, pesq_n, stoi_n = evaluate_pair(cf, noisy_path, do_align=args.align)
                    row.update(snr_noisy=snr_n, pesq_noisy=pesq_n, stoi_noisy=stoi_n)
                except Exception as e:
                    print(f"[ERROR] noisy baseline eval failed for '{stem}': {e}")
            else:
                print(f"[SKIP] No noisy file found for '{stem}' in {noisy_dir}")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print("\n=== Per-file results saved to:", args.output_csv, "===")
    print("\n=== Mean scores ===")
    print(df.mean(numeric_only=True).round(4))

    if noisy_dir is not None and "pesq_noisy" in df.columns:
        print("\n=== Mean improvement (enhanced - noisy) ===")
        improvement = pd.DataFrame({
            "snr_improvement": df["snr_enhanced"] - df["snr_noisy"],
            "pesq_improvement": df["pesq_enhanced"] - df["pesq_noisy"],
            "stoi_improvement": df["stoi_enhanced"] - df["stoi_noisy"],
        })
        print(improvement.mean().round(4))


if __name__ == "__main__":
    main()
