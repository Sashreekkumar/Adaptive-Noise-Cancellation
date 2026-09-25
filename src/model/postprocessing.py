"""
postprocessing.py

Stage 3: reference-free cleanup applied AFTER DeepFilterNet3.

This is the stage that changed most from the R&D version. The old
postprocess_enhanced.py needed a clean reference file to (a) estimate
delay via GCC-PHAT and (b) compute an optimal gain match. In
deployment there is no clean reference -- only one noisy file -- so
both of those steps are gone. What's left:

  - optional dry/wet blend with the PRE-ENHANCEMENT (preprocessed)
    audio, to soften musical-noise / over-suppression artifacts.
    `enhanced` and `pre_enhance_audio` come from the same source file
    and DeepFilterNet's pad=True already keeps them sample-aligned,
    so no delay estimation is needed to blend them.
  - band-limit to the speech range, to strip residual broadband
    noise/artifacts outside the band that matters.
  - optional final RMS loudness normalization.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config
from .preprocessing import rms_normalize


def bandlimit(audio: np.ndarray, sr: int, low_hz: float = config.BAND_LOW_HZ,
              high_hz: float = config.BAND_HIGH_HZ) -> np.ndarray:
    sos = butter(N=4, Wn=[low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def postprocess(enhanced: np.ndarray, pre_enhance_audio: np.ndarray, blend_wet: float = 1.0,
                 band_low: float = config.BAND_LOW_HZ, band_high: float = config.BAND_HIGH_HZ,
                 normalize_volume: bool = False, target_dbfs: float = config.TARGET_DBFS,
                 sr: int = config.MODEL_SR) -> np.ndarray:
    out = enhanced
    if blend_wet < 1.0:
        n = min(len(out), len(pre_enhance_audio))
        out = blend_wet * out[:n] + (1 - blend_wet) * pre_enhance_audio[:n]
        out = out.astype(np.float32)

    out = bandlimit(out, sr, band_low, band_high)

    if normalize_volume:
        out = rms_normalize(out, target_dbfs)

    peak = np.max(np.abs(out)) + 1e-12
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)
