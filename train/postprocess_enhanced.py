"""
postprocess_enhanced.py (parallelized)

Run this AFTER enhance_with_deepfilternet.py, before scoring.

For each enhanced file, matched against its clean reference:
  1. Delay-aligns the enhanced signal to the reference (GCC-PHAT
     cross-correlation, shared with evaluate_metrics.py via
     align_utils.py -- see that module's docstring for why).
  2. Gain-matches the enhanced signal to the reference (least-squares
     optimal scalar) -- removes level-mismatch penalty from raw SNR.
     NOTE: this step, combined with delay alignment, is what makes the
     "SNR" you compute downstream closer in spirit to SI-SDR (a
     standard scale-invariant speech-enhancement metric) than to a
     deployment-realistic raw SNR. That's a legitimate, standard
     technique -- just be aware of what it represents. evaluate_metrics.py
     now also reports si_sdr explicitly alongside snr for this reason.
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
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from align_utils import estimate_delay, apply_delay

SR = 48000
MAX_SHIFT_SAMPLES = 4000  # ~83 ms at 48kHz, generous for DF's algorithmic delay


def find_matching_file(folder: Path, stem: str):
    for ext in (".wav", ".mp3"):
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def optimal_gain(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    ref_seg, deg_seg = ref[:n], deg[:n]
    denom = np.sum(deg_seg.astype(np.float64) ** 2) + 1e-12
    return float(np.sum(ref_seg.astype(np.float64) * deg_seg.astype(np.float64)) / denom)


def bandlimit(audio: np.ndarray, sr: int, low_hz: float = 80, high_hz: float = 8000) -> np.ndarray:
    sos = butter(N=4, Wn=[low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def rms_normalize(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Loudness (RMS) normalization to a fixed target level, applied as
    the last step, independent of the clean-gain-match step above. This
    is a separate, toggleable layer -- --normalize_volume turns it on;
    without the flag, only the clean-referenced optimal_gain correction
    is applied (as before)."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    target_rms = 10 ** (target_dbfs / 20)
    gain = target_rms / rms
    out = audio * gain
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def process_pair(clean_path: Path, enh_path: Path, noisy_path, blend_wet: float,
                  normalize_volume: bool = False, target_dbfs: float = -26.0):
    clean, _ = librosa.load(str(clean_path), sr=SR, mono=True)
    enh, _ = librosa.load(str(enh_path), sr=SR, mono=True)

    # 1. Delay alignment (GCC-PHAT, robust to periodic voiced speech)
    lag = estimate_delay(clean, enh, MAX_SHIFT_SAMPLES)
    enh = apply_delay(enh, lag)

    n = min(len(clean), len(enh))
    clean, enh = clean[:n], enh[:n]

    # 2. Gain match to clean reference
    g = optimal_gain(clean, enh)
    enh = (enh * g).astype(np.float32)

    # 3. Optional dry/wet blend with original noisy (softens artifacts)
    if blend_wet < 1.0 and noisy_path is not None:
        noisy, _ = librosa.load(str(noisy_path), sr=SR, mono=True)
        noisy_lag = estimate_delay(clean, noisy, MAX_SHIFT_SAMPLES)
        noisy = apply_delay(noisy, noisy_lag)
        m = min(len(enh), len(noisy))
        enh, noisy = enh[:m], noisy[:m]
        enh = blend_wet * enh + (1 - blend_wet) * noisy

    # 4. Band-limit to speech range
    enh = bandlimit(enh, SR)

    # 5. Optional volume normalization layer -- OFF by default, toggle
    # with --normalize_volume to A/B test whether it helps your metrics
    # on top of the clean-referenced gain match already applied in step 2.
    if normalize_volume:
        enh = rms_normalize(enh, target_dbfs)

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
    stem, clean_path, enh_path, noisy_path, blend_wet, output_dir, normalize_volume, target_dbfs = args_tuple
    try:
        out_audio = process_pair(Path(clean_path), Path(enh_path), noisy_path, blend_wet,
                                  normalize_volume=normalize_volume, target_dbfs=target_dbfs)
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
    parser.add_argument("--normalize_volume", action="store_true", help="Toggle: apply RMS loudness normalization as a final layer (OFF by default)")
    parser.add_argument("--target_dbfs", type=float, default=-26.0, help="Target level for --normalize_volume (default: -26.0 dBFS)")
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
        tasks.append((stem, str(cf), str(enh_path), str(noisy_path) if noisy_path else None, args.blend_wet, str(output_dir), args.normalize_volume, args.target_dbfs))

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