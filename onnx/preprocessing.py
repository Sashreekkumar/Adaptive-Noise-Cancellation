"""
preprocessing.py

Stage 1: reference-free cleanup applied BEFORE DeepFilterNet3 sees the
audio -- DC offset removal, high-pass filter (strips rumble that
isn't speech), RMS loudness normalization.

Duplicated verbatim from the torch audio_cleaner package.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    return (audio - np.mean(audio)).astype(np.float32)


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float = config.HPF_CUTOFF_HZ) -> np.ndarray:
    sos = butter(N=4, Wn=cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def rms_normalize(audio: np.ndarray, target_dbfs: float = config.TARGET_DBFS) -> np.ndarray:
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    target_rms = 10 ** (target_dbfs / 20)
    out = audio * (target_rms / rms)
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def preprocess(audio: np.ndarray, sr: int, skip_dc: bool = False, skip_hpf: bool = False,
               skip_norm: bool = False, hpf_cutoff: float = config.HPF_CUTOFF_HZ,
               target_dbfs: float = config.TARGET_DBFS) -> np.ndarray:
    if not skip_dc:
        audio = remove_dc_offset(audio)
    if not skip_hpf:
        audio = highpass(audio, sr, hpf_cutoff)
    if not skip_norm:
        audio = rms_normalize(audio, target_dbfs)
    return audio
