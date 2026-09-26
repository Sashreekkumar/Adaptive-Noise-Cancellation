"""
preprocess_noisy.py

Run this BEFORE enhance_with_deepfilternet.py.

Applies to each noisy mp3:
  1. High-pass filter (~70 Hz) to strip rumble/DC offset that isn't speech.
  2. RMS loudness normalization to a fixed target level.

A stable, consistent input level and no sub-speech-band junk tends to
give DeepFilterNet3 a more consistent, less artifact-prone signal to
work with, which helps downstream PESQ/SNR.

Usage:
    python preprocess_noisy.py --noisy_dir ./noisy --output_dir ./noisy_preprocessed
    (then point enhance_with_deepfilternet.py's --noisy_dir at noisy_preprocessed)
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

TARGET_SR = 48000  # match DeepFilterNet3's native rate
HPF_CUTOFF_HZ = 70
TARGET_DBFS = -26.0


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Subtracts the mean to remove DC bias. Distinct from (and applied
    before) the high-pass filter: this directly zeroes the true DC
    component, whereas the high-pass filter only attenuates very low
    frequencies including but not limited to DC."""
    return (audio - np.mean(audio)).astype(np.float32)


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float = HPF_CUTOFF_HZ) -> np.ndarray:
    sos = butter(N=4, Wn=cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def rms_normalize(audio: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    target_rms = 10 ** (target_dbfs / 20)
    gain = target_rms / rms
    out = audio * gain
    # avoid clipping
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.mp3")
    parser.add_argument("--skip_dc_offset", action="store_true")
    parser.add_argument("--skip_highpass", action="store_true")
    parser.add_argument("--skip_normalize", action="store_true")
    args = parser.parse_args()

    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(noisy_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching '{args.pattern}' in {noisy_dir}")

    for f in tqdm(files, desc="Preprocessing"):
        audio, sr = librosa.load(str(f), sr=TARGET_SR, mono=True)

        if not args.skip_dc_offset:
            audio = remove_dc_offset(audio)

        if not args.skip_highpass:
            audio = highpass(audio, sr)

        if not args.skip_normalize:
            audio = rms_normalize(audio)

        out_path = output_dir / (f.stem + ".wav")
        sf.write(str(out_path), audio, sr)

    print(f"Wrote {len(files)} preprocessed files to {output_dir}")


if __name__ == "__main__":
    main()