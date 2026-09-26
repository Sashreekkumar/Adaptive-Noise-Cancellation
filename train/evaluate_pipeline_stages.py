"""
evaluate_pipeline_stages.py

Runs the same SNR / SI-SDR / PESQ / STOI evaluation used by
evaluate_metrics.py against EVERY stage of the pipeline in one pass,
so you can see exactly what each stage does to the metrics rather than
only comparing the final output to noisy.

Typical pipeline this is meant for:

    noisy  ->  preprocess_noisy.py       ->  noisy_preprocessed
    noisy_preprocessed -> enhance_with_deepfilternet.py -> enhanced
    enhanced -> postprocess_enhanced.py  ->  enhanced_postprocessed

Each of those four folders (noisy, noisy_preprocessed, enhanced,
enhanced_postprocessed) is scored against the same clean_dir, using the
same matching-by-stem logic as the other scripts. That gives you one
row per file per stage and a per-stage mean, so the deltas between
consecutive stages tell you what each processing step is actually
buying you (or costing you).

Fairness note on alignment/gain:
    Raw noisy audio and the raw enhanced (pre-postprocess) audio were
    never explicitly delay-aligned or gain-matched to clean the way
    postprocess_enhanced.py's output is. If we scored them without any
    alignment, part of the SNR/PESQ/STOI difference between stages
    would just be "postprocessing already aligns, the earlier stages
    don't" -- a bookkeeping difference, not a signal-quality one. To
    keep the stage-to-stage comparison about the actual audio content,
    this script applies the same GCC-PHAT alignment (align_utils.py)
    and the same least-squares gain match at evaluation time to every
    stage, every time, before computing metrics. Use --no_align /
    --no_gain_match to turn either off if you specifically want to see
    the raw, unaligned numbers (e.g. to confirm postprocessing's own
    alignment is doing what it claims).

Usage:
    python evaluate_pipeline_stages.py \
        --clean_dir ./clean \
        --stage noisy=./noisy \
        --stage preprocessed=./noisy_preprocessed \
        --stage enhanced=./enhanced \
        --stage postprocessed=./enhanced_postprocessed \
        --output_prefix pipeline_results

    Produces:
        pipeline_results_per_file.csv   (one row per file per stage)
        pipeline_results_summary.csv    (mean per stage + stage-over-stage delta)

Requires align_utils.py (from this same pipeline) to be importable,
i.e. run this script from the same directory, or otherwise have it on
PYTHONPATH.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

from align_utils import estimate_delay, apply_delay, si_sdr

EVAL_SR = 16000  # sample rate used for PESQ (wb) and STOI, matches evaluate_metrics.py
MAX_SHIFT_SAMPLES = int(0.1 * EVAL_SR)  # 100 ms search window at 16kHz


def find_matching_file(folder: Path, stem: str):
    for ext in (".wav", ".mp3"):
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def load_16k(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=EVAL_SR, mono=True)
    return audio


def optimal_gain(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    ref_seg, deg_seg = ref[:n], deg[:n]
    denom = np.sum(deg_seg.astype(np.float64) ** 2) + 1e-12
    return float(np.sum(ref_seg.astype(np.float64) * deg_seg.astype(np.float64)) / denom)


def compute_snr(clean: np.ndarray, est: np.ndarray) -> float:
    n = min(len(clean), len(est))
    clean, est = clean[:n], est[:n]
    noise = clean - est
    num = np.sum(clean.astype(np.float64) ** 2)
    den = np.sum(noise.astype(np.float64) ** 2) + 1e-12
    return 10 * np.log10(num / den)


def evaluate_pair(clean: np.ndarray, deg: np.ndarray, do_align: bool, do_gain_match: bool):
    lag = None
    gain = None

    if do_align:
        lag = estimate_delay(clean, deg, MAX_SHIFT_SAMPLES)
        deg = apply_delay(deg, lag)

    n = min(len(clean), len(deg))
    clean_c, deg_c = clean[:n], deg[:n]

    if do_gain_match:
        gain = optimal_gain(clean_c, deg_c)
        deg_c = (deg_c * gain).astype(np.float64)

    n = min(len(clean_c), len(deg_c))
    clean_c, deg_c = clean_c[:n], deg_c[:n]

    snr_val = compute_snr(clean_c, deg_c)
    sisdr_val = si_sdr(clean_c, deg_c)
    pesq_val = pesq(EVAL_SR, clean_c, deg_c, "wb")
    stoi_val = stoi(clean_c, deg_c, EVAL_SR, extended=False)
    return snr_val, sisdr_val, pesq_val, stoi_val, lag, gain


def parse_stage_arg(raw: str):
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--stage must be NAME=DIR, got '{raw}'"
        )
    name, _, path = raw.partition("=")
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"--stage must be NAME=DIR, got '{raw}'"
        )
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Score SNR/SI-SDR/PESQ/STOI at every stage of the pipeline against clean."
    )
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        type=parse_stage_arg,
        metavar="NAME=DIR",
        help="Repeatable. One per pipeline stage, in pipeline order, "
             "e.g. --stage noisy=./noisy --stage enhanced=./enhanced",
    )
    parser.add_argument("--output_prefix", default="pipeline_results")
    parser.add_argument("--align", dest="align", action="store_true", default=True)
    parser.add_argument("--no_align", dest="align", action="store_false")
    parser.add_argument("--gain_match", dest="gain_match", action="store_true", default=True)
    parser.add_argument("--no_gain_match", dest="gain_match", action="store_false")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    stages = args.stage  # list of (name, Path), in the order given on the CLI

    clean_files = sorted(list(clean_dir.glob("*.mp3")) + list(clean_dir.glob("*.wav")))
    if not clean_files:
        raise SystemExit(f"No .mp3/.wav files found in {clean_dir}")

    # Find, per stem, the matching file in every stage. Only stems present
    # in clean AND every single stage go into the comparison, so the
    # stage-over-stage deltas are computed on an identical set of files
    # throughout -- otherwise an uneven file set could itself explain part
    # of an apparent improvement or regression.
    stem_paths = {}  # stem -> {stage_name: path}
    for cf in clean_files:
        stem = cf.stem
        per_stage = {}
        missing = []
        for name, folder in stages:
            p = find_matching_file(folder, stem)
            if p is None:
                missing.append(name)
            else:
                per_stage[name] = p
        if missing:
            print(f"[SKIP] '{stem}': missing in stage(s) {missing}")
            continue
        stem_paths[stem] = per_stage

    if not stem_paths:
        raise SystemExit("No file has a match in every stage -- nothing to evaluate.")

    print(f"Evaluating {len(stem_paths)} files across {len(stages)} stages "
          f"(align={args.align}, gain_match={args.gain_match})...")

    rows = []
    for stem, per_stage in tqdm(stem_paths.items(), desc="Files"):
        clean = load_16k(clean_dir / next(f.name for f in clean_files if f.stem == stem))
        for name, _ in stages:
            deg_path = per_stage[name]
            try:
                snr_v, sisdr_v, pesq_v, stoi_v, lag, gain = evaluate_pair(
                    clean, load_16k(deg_path), do_align=args.align, do_gain_match=args.gain_match
                )
                rows.append({
                    "file": stem,
                    "stage": name,
                    "snr": snr_v,
                    "si_sdr": sisdr_v,
                    "pesq": pesq_v,
                    "stoi": stoi_v,
                    "lag_samples_16k": lag,
                    "gain": gain,
                })
            except Exception as e:
                print(f"[ERROR] stage '{name}', file '{stem}': {e}")

    per_file_df = pd.DataFrame(rows)
    per_file_csv = f"{args.output_prefix}_per_file.csv"
    per_file_df.to_csv(per_file_csv, index=False)

    stage_order = [name for name, _ in stages]
    summary = (
        per_file_df.groupby("stage")[["snr", "si_sdr", "pesq", "stoi"]]
        .mean()
        .reindex(stage_order)
    )
    delta = summary.diff()
    delta.columns = [f"d_{c}" for c in delta.columns]
    summary_out = pd.concat([summary, delta], axis=1)
    summary_csv = f"{args.output_prefix}_summary.csv"
    summary_out.to_csv(summary_csv)

    print(f"\nPer-file results: {per_file_csv}")
    print(f"Stage summary + deltas: {summary_csv}\n")
    print("=== Mean per stage ===")
    print(summary.round(4))
    print("\n=== Change from previous stage ===")
    print(delta.round(4))


if __name__ == "__main__":
    main()