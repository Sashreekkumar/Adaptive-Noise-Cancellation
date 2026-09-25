"""
exporter.py

Writes the final float32 waveform out as an mp3. Goes via a temp wav
+ ffmpeg (libmp3lame) rather than adding a new Python mp3-encoding
dependency, since ffmpeg is already required for mp3 decoding
upstream (librosa/audioread).
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config


def export_mp3(audio: np.ndarray, sr: int, out_path, bitrate: str = config.MP3_BITRATE) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        sf.write(tmp_wav, audio, sr)
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav,
             "-codec:a", "libmp3lame", "-b:a", bitrate, str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mp3 export failed: {result.stderr}")
    finally:
        Path(tmp_wav).unlink(missing_ok=True)
