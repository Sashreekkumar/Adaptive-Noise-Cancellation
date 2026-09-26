"""
loader.py

Fast audio decoding via a single direct ffmpeg call.

librosa.load() decodes mp3 through audioread, which shells out to
ffmpeg and then reads the output back frame-by-frame in Python -- slow,
and it was the dominant chunk of per-file latency (~3.7s out of ~5.5s
total on a 6s clip). This does the same decode + resample + downmix in
one ffmpeg invocation, piping raw float32 PCM straight back to us, cutting
that stage to a fraction of a second.

Duplicated verbatim from the torch audio_cleaner package.
"""

import subprocess

import numpy as np

from . import config


def load_audio(path, sr: int = config.MODEL_SR) -> np.ndarray:
    """Decode any ffmpeg-readable audio file to mono float32 PCM at `sr`."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {result.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()
