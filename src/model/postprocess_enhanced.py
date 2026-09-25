"""
postprocess_enhanced.py (parallelized)

Run this AFTER enhance_with_deepfilternet.py, before scoring.

For each enhanced file, matched against its clean reference:
  1. Delay-aligns the enhanced signal to the reference (cross-correlation).
  2. Gain-matches the enhanced signal to the reference (least-squares
     optimal scalar) -- removes level-mismatch penalty from raw SNR.
  3. Band-limits to the speech range (default 80 Hz - 8000 Hz) to strip
     residual broadband noise/artifacts outside the band that matters.
  4. Optionally blends a fraction of the original noisy signal back in
     (--blend_wet, default 1.0 = no blending) to soften musical-noise /
     over-suppression artifacts that hurt PESQ.

Files are processed in parallel across worker processes (one file per
worker task), since each file is independent.

Usage:
    python postprocess_enhanced.py \
        --clean_dir ./clean \
        --enhanced_dir ./enhanced \
        --noisy_dir ./noisy \
        --output_dir ./enhanced_postprocessed \
        --blend_wet 0.9 \
        --workers 8
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, correlate
from tqdm import tqdm

SR = 48000
MAX_SHIFT_SAMPLES = 4000  # ~83 ms at 48kHz, generous for DF's algorithmic delay


def find_matching_file(folder: Path, stem: str):
    for ext in (".wav", ".mp3"):
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def estimate_delay(ref: np.ndarray, deg: np.ndarray, max_shift: int = MAX_SHIFT_SAMPLES) -> int:
    n = min(len(ref), len(deg))
    ref_seg = ref[:n] - ref[:n].mean()
    deg_seg = deg[:n] - deg[:n].mean()
    # FFT-based correlation (O(n log n)) instead of np.correlate's direct
    # O(n^2) computation -- direct correlation on full-length audio at
    # 48kHz is what was causing ~20s/file.
    corr = correlate(deg_seg, ref_seg, mode="full", method="fft")
    lag = np.argmax(corr) - (n - 1)
    return int(np.clip(lag, -max_shift, max_shift))


def apply_delay(deg: np.ndarray, lag: int) -> np.ndarray:
    if lag > 0:
        return deg[lag:]
    elif lag < 0:
        return np.concatenate([np.zeros(-lag, dtype=deg.dtype), deg])
    return deg


def optimal_gain(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    ref_seg, deg_seg = ref[:n], deg[:n]
    denom = np.sum(deg_seg.astype(np.float64) ** 2) + 1e-12
    return float(np.sum(ref_seg.astype(np.float64) * deg_seg.astype(np.float64)) / denom)


def bandlimit(audio: np.ndarray, sr: int, low_hz: float = 80, high_hz: float = 8000) -> np.ndarray:
    sos = butter(N=4, Wn=[low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def process_pair(clean_path: Path, enh_path: Path, noisy_path, blend_wet: float):
    clean, _ = librosa.load(str(clean_path), sr=SR, mono=True)
    enh, _ = librosa.load(str(enh_path), sr=SR, mono=True)

    # 1. Delay alignment
    lag = estimate_delay(clean, enh)
    enh = apply_delay(enh, lag)

    n = min(len(clean), len(enh))
    clean, enh = clean[:n], enh[:n]

    # 2. Gain match to clean reference
    g = optimal_gain(clean, enh)
    enh = (enh * g).astype(np.float32)

    # 3. Optional dry/wet blend with original noisy (softens artifacts)
    if blend_wet < 1.0 and noisy_path is not None:
        noisy, _ = librosa.load(str(noisy_path), sr=SR, mono=True)
        noisy = apply_delay(noisy, estimate_delay(clean, noisy))
        m = min(len(enh), len(noisy))
        enh, noisy = enh[:m], noisy[:m]
        enh = blend_wet * enh + (1 - blend_wet) * noisy

    # 4. Band-limit to speech range
    enh = bandlimit(enh, SR)

    # avoid clipping
    peak = np.max(np.abs(enh)) + 1e-12
    if peak > 0.99:
        enh = enh * (0.99 / peak)

    return enh.astype(np.float32)


def _worker(args_tuple):
    """Top-level function so it can be pickled for ProcessPoolExecutor.
    Does the file loading, processing, and writing for a single file,
    so worker processes don't have to ship large arrays back to the
    main process."""
    stem, clean_path, enh_path, noisy_path, blend_wet, output_dir = args_tuple
    try:
        out_audio = process_pair(Path(clean_path), Path(enh_path), noisy_path, blend_wet)
        out_path = Path(output_dir) / (stem + ".wav")
        sf.write(str(out_path), out_audio, SR)
        return (stem, True, None)
    except Exception as e:
        return (stem, False, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--enhanced_dir", required=True)
    parser.add_argument("--noisy_dir", default=None, help="Needed only if --blend_wet < 1.0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blend_wet", type=float, default=1.0, help="1.0 = pure enhanced, e.g. 0.9 = 90% enhanced + 10% noisy")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: os.cpu_count())")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    enhanced_dir = Path(args.enhanced_dir)
    noisy_dir = Path(args.noisy_dir) if args.noisy_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(list(clean_dir.glob("*.mp3")) + list(clean_dir.glob("*.wav")))
    if not clean_files:
        raise SystemExit(f"No .mp3/.wav files found in {clean_dir}")

    # Build the task list up front (cheap: just path lookups, no audio loaded yet)
    tasks = []
    for cf in clean_files:
        stem = cf.stem
        enh_path = find_matching_file(enhanced_dir, stem)
        if enh_path is None:
            print(f"[SKIP] no enhanced file for '{stem}'")
            continue
        noisy_path = find_matching_file(noisy_dir, stem) if noisy_dir else None
        tasks.append((stem, str(cf), str(enh_path), str(noisy_path) if noisy_path else None, args.blend_wet, str(output_dir)))

    if not tasks:
        raise SystemExit("No matching clean/enhanced pairs found.")

    print(f"Processing {len(tasks)} files with {args.workers or 'all available'} workers...")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Postprocessing"):
            stem, ok, err = fut.result()
            if not ok:
                failures.append((stem, err))
                print(f"[ERROR] '{stem}': {err}")

    print(f"\nDone. {len(tasks) - len(failures)}/{len(tasks)} files written to {output_dir}")
    if failures:
        print("Failures:")
        for stem, err in failures:
            print(f"  - {stem}: {err}")


if __name__ == "__main__":
    main()
