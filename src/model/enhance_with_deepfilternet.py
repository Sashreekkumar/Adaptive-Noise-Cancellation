"""
enhance_with_deepfilternet.py

Runs DeepFilterNet3 over every mp3 file in a folder and writes enhanced
(denoised) versions as .wav files to an output folder.

Usage:
    python enhance_with_deepfilternet.py \
        --noisy_dir ./noisy \
        --output_dir ./enhanced

Notes:
- DeepFilterNet3 internally operates at 48 kHz mono. Input audio is
  resampled to 48 kHz on load; output wav files are written at 48 kHz.
- Output is written as .wav (not .mp3) to avoid lossy re-encoding before
  metric computation. Filenames match the input stem, e.g. "file01.mp3"
  -> "file01.wav".
- mp3 decoding uses librosa (via audioread/ffmpeg), so ffmpeg must be
  installed and on PATH (see install commands in README).
"""

import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from df.enhance import enhance, init_df


def process_file(model, df_state, in_path: Path, out_path: Path) -> None:
    sr_model = df_state.sr()

    # Load and resample to the model's expected sample rate, mono.
    audio, _ = librosa.load(str(in_path), sr=sr_model, mono=True)

    # DeepFilterNet expects a torch tensor shaped (channels, samples).
    audio_t = torch.from_numpy(audio).unsqueeze(0).float()

    enhanced_t = enhance(model, df_state, audio_t)
    enhanced_np = enhanced_t.squeeze(0).cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), enhanced_np, sr_model)


def main():
    parser = argparse.ArgumentParser(description="Enhance mp3 files with DeepFilterNet3")
    parser.add_argument("--noisy_dir", required=True, help="Folder containing noisy .mp3 files")
    parser.add_argument("--output_dir", required=True, help="Folder to write enhanced .wav files")
    parser.add_argument(
        "--pattern",
        default="*.mp3",
        help="Glob pattern for input files inside noisy_dir (default: *.mp3)",
    )
    args = parser.parse_args()

    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not noisy_dir.is_dir():
        raise SystemExit(f"noisy_dir does not exist: {noisy_dir}")

    files = sorted(noisy_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching '{args.pattern}' found in {noisy_dir}")

    print(f"Found {len(files)} noisy files. Loading DeepFilterNet3 model...")
    # init_df() with no model_base_dir downloads/loads the default checkpoint,
    # which is DeepFilterNet3 in current releases of the `deepfilternet` package.
    model, df_state, _ = init_df()

    failures = []
    for f in tqdm(files, desc="Enhancing"):
        out_path = output_dir / (f.stem + ".wav")
        try:
            process_file(model, df_state, f, out_path)
        except Exception as e:
            failures.append((f.name, str(e)))
            print(f"[FAILED] {f.name}: {e}")

    print(f"\nDone. {len(files) - len(failures)}/{len(files)} files enhanced successfully.")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
