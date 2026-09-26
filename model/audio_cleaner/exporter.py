"""
exporter.py

Writes the final float32 waveform out as an mp3. Pipes raw PCM
directly into ffmpeg's stdin and lets it write the mp3 straight to
disk -- no intermediate temp .wav file, so there's no extra disk
write+read round trip between the model output and the final mp3.
"""

import subprocess

import numpy as np
from pathlib import Path

from . import config


def export_mp3(audio: np.ndarray, sr: int, out_path, bitrate: str = config.MP3_BITRATE) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_pcm = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", bitrate, str(out_path),
    ]
    result = subprocess.run(cmd, input=raw_pcm, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 export failed: {result.stderr.decode(errors='replace')}")