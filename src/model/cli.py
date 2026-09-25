"""
cli.py

Command-line entry point: single file, folder (batch), or --serve to
launch the drag-and-drop web UI (see server.py).
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
    parser = argparse.ArgumentParser(description="Clean a noisy audio file (or folder) -> clean mp3")
    parser.add_argument("input", nargs="?", help="Noisy input file or folder")
    parser.add_argument("-o", "--output", help="Output mp3 file, or output folder for batch input")
    parser.add_argument("--atten_lim_db", type=float, default=None,
                         help="Limit DeepFilterNet3 attenuation (dB); lower = less over-suppression")
    parser.add_argument("--no_pad", action="store_true", help="Disable DeepFilterNet3's built-in delay compensation")
    parser.add_argument("--blend_wet", type=float, default=1.0,
                         help="1.0 = pure enhanced, e.g. 0.9 = 90%% enhanced + 10%% pre-enhancement audio")
    parser.add_argument("--normalize_volume", action="store_true", help="Apply final RMS loudness normalization")
    parser.add_argument("--target_dbfs", type=float, default=config.TARGET_DBFS)
    parser.add_argument("--hpf_cutoff", type=float, default=config.HPF_CUTOFF_HZ)
    parser.add_argument("--band_low", type=float, default=config.BAND_LOW_HZ)
    parser.add_argument("--band_high", type=float, default=config.BAND_HIGH_HZ)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto-detect)")
    parser.add_argument("--serve", action="store_true", help="Launch a drag-and-drop web UI instead of CLI mode")
    parser.add_argument("--port", type=int, default=7860)
    return parser


def _kwargs_from_args(args):
    return dict(
        device=args.device, pad=not args.no_pad, atten_lim_db=args.atten_lim_db,
        blend_wet=args.blend_wet, normalize_volume=args.normalize_volume,
        target_dbfs=args.target_dbfs, hpf_cutoff=args.hpf_cutoff,
        band_low=args.band_low, band_high=args.band_high,
    )


def main():
    args = _build_arg_parser().parse_args()

    if args.serve:
        from .server import serve
        serve(args)
        return

    if not args.input:
        _build_arg_parser().error("input is required unless --serve is used")

    in_path = Path(args.input)
    kwargs = _kwargs_from_args(args)

    if in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path.parent / (in_path.name + "_clean")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = _iter_input_files(in_path)
        if not files:
            raise SystemExit(f"No audio files found in {in_path}")

        print("Loading DeepFilterNet3 model...")
        get_model(args.device)  # warm up once, reused for every file below

        failures = []
        for f in tqdm(files, desc="Cleaning"):
            out_path = out_dir / (f.stem + ".mp3")
            try:
                clean_audio_file(f, out_path, **kwargs)
            except Exception as e:
                failures.append((f.name, str(e)))
                print(f"[FAILED] {f.name}: {e}")
        print(f"\nDone. {len(files) - len(failures)}/{len(files)} files written to {out_dir}")
    else:
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_clean.mp3")
        clean_audio_file(in_path, out_path, **kwargs)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
