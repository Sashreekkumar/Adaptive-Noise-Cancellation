"""
cli.py

Command-line entry point for the standalone ONNX pipeline: single
file or folder (batch) in, clean mp3s out. Mirrors audio_cleaner/cli.py
but always runs through ONNX Runtime -- there's no --backend switch
here since this whole folder only knows the ONNX path (see the sibling
audio_cleaner package for the PyTorch version).
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from . import config
from .model import get_model
from .pipeline import clean_audio_file

_AUDIO_EXTS = ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg")


def _iter_input_files(folder: Path):
    files = []
    for e in _AUDIO_EXTS:
        files.extend(folder.glob(e))
    return sorted(files)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Clean a noisy audio file (or folder) -> clean mp3, via ONNX Runtime")
    parser.add_argument("input", nargs="?", help="Noisy input file or folder")
    parser.add_argument("-o", "--output", help="Output mp3 file, or output folder for batch input")
    parser.add_argument("--onnx_path", required=True,
                         help="Path to the .onnx file from export_to_onnx.py (its .meta.json sidecar must sit next to it)")
    parser.add_argument("--atten_lim_db", type=float, default=None,
                         help="Limit DeepFilterNet3 attenuation (dB); lower = less over-suppression")
    parser.add_argument("--blend_wet", type=float, default=1.0,
                         help="1.0 = pure enhanced, e.g. 0.9 = 90%% enhanced + 10%% pre-enhancement audio")
    parser.add_argument("--normalize_volume", action="store_true", help="Apply final RMS loudness normalization")
    parser.add_argument("--target_dbfs", type=float, default=config.TARGET_DBFS)
    parser.add_argument("--hpf_cutoff", type=float, default=config.HPF_CUTOFF_HZ)
    parser.add_argument("--band_low", type=float, default=config.BAND_LOW_HZ)
    parser.add_argument("--band_high", type=float, default=config.BAND_HIGH_HZ)
    return parser


def _kwargs_from_args(args):
    return dict(
        onnx_path=args.onnx_path, atten_lim_db=args.atten_lim_db,
        blend_wet=args.blend_wet, normalize_volume=args.normalize_volume,
        target_dbfs=args.target_dbfs, hpf_cutoff=args.hpf_cutoff,
        band_low=args.band_low, band_high=args.band_high,
    )


def main():
    args = _build_arg_parser().parse_args()

    if not args.input:
        _build_arg_parser().error("input is required")

    in_path = Path(args.input)
    kwargs = _kwargs_from_args(args)

    if in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path.parent / (in_path.name + "_clean_onnx")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = _iter_input_files(in_path)
        if not files:
            raise SystemExit(f"No audio files found in {in_path}")

        print(f"Loading ONNX model from {args.onnx_path}...")
        get_model(args.onnx_path)  # warm up once, reused for every file below

        failures = []
        latencies = []
        rtfs = []
        for f in tqdm(files, desc="Cleaning"):
            out_path = out_dir / (f.stem + ".mp3")
            try:
                result = clean_audio_file(f, out_path, **kwargs)
                latencies.append(result.latency_seconds)
                rtfs.append(result.rtf)
            except Exception as e:
                failures.append((f.name, str(e)))
                print(f"[FAILED] {f.name}: {e}")
        print(f"\nDone. {len(files) - len(failures)}/{len(files)} files written to {out_dir}")
        if latencies:
            print(f"Latency: avg {sum(latencies) / len(latencies):.2f}s, "
                  f"min {min(latencies):.2f}s, max {max(latencies):.2f}s")
            print(f"RTF: avg {sum(rtfs) / len(rtfs):.2f} (below 1.0 = faster than real time)")
    else:
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_clean_onnx.mp3")
        result = clean_audio_file(in_path, out_path, **kwargs)
        breakdown = ", ".join(f"{k}={v:.2f}s" for k, v in result.stage_seconds.items())
        print(f"Wrote {result.output_path}")
        print(f"Latency: {result.latency_seconds:.2f}s for {result.audio_duration_seconds:.2f}s of audio "
              f"(RTF={result.rtf:.2f}, {breakdown})")


if __name__ == "__main__":
    main()
