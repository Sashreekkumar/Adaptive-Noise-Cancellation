"""
Steps

1. mono, float32, NaN/Inf cleanup       
2. declip (native rate, only if clipping)   
3. resample to the model rate (48 kHz DFN)
4. DC offset removal
5. zero-phase high-pass (rumble)            
6. level normalization (one gain, peak-safe) 
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from preprocess_utils import rms_dbfs, resample
from de_clip import declip
from audio_io import load_audio, save_audio


EPS = 1e-12

@dataclass
class PreprocessConfig:
    target_sr: int = 48000 # sampling rate
    declip: bool = True # enable declipping
    declip_trigger_frac: float = 5e-4  # clipping threshold 
    declip_min_run: int = 3 # minimum size of run
    declip_max_run: int = 30   # max run size
    declip_max_frac: float = 0.05  # >5% clipped: local repair makes it worse, so it is skipped
    dc_remove: bool = True # dc remove
    highpass: bool = True # high pass filter for attenuation
    hp_cutoff_hz: float = 60.0 # cuff of hpf
    hp_order: int = 4 # order of hfp
    normalize: bool = True # normalization
    target_dbfs: float = -25.0  # RMS level of the model input
    max_gain_db: float = 30.0 # maximum allowed gain
    min_input_dbfs: float = -80.0  # quieter than this = digital silence, no gain applied
    peak_limit: float = 0.98 # peak safety


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig | None = None):
        self.cfg = cfg or PreprocessConfig()
    """
    Although currently -inf and +inf are converted to infinitely small number and big number, it could add noise, we could just replace it with 0

    """
    def process(self, x, sr):
        """x: 1-D or (n, ch) array. Returns (float32 mono @ target_sr, target_sr, meta)."""
        c = self.cfg
        x = np.nan_to_num(np.asarray(x, np.float32)) # NaN = 0 and -inf and +inf. 
        if x.ndim == 2:
            x = x.mean(axis=1)

        meta = {"in_sr": sr, "in_dbfs": round(rms_dbfs(x), 2) if len(x) else None, "gain": 1.0, "gain_db": 0.0,
                "clipped_frac": 0.0, "declipped_samples": 0, "silent": False}

        if len(x) == 0:
            return x, c.target_sr, meta

        if c.declip:
            x, meta["declipped_samples"], meta["clipped_frac"] = declip(
                x, c.declip_trigger_frac, c.declip_min_run, c.declip_max_run, max_frac=c.declip_max_frac)

        x = resample(x, sr, c.target_sr)

        if c.dc_remove:
            x = x - x.mean()

        if c.highpass and len(x) > 200:
            sos = signal.butter(c.hp_order, c.hp_cutoff_hz, "highpass", fs=c.target_sr, output="sos")
            x = signal.sosfiltfilt(sos, x).astype(np.float32)

        if c.normalize:
            cur = rms_dbfs(x)
            if cur <= c.min_input_dbfs:
                meta["silent"] = True
            else:
                g = 10 ** (float(np.clip(c.target_dbfs - cur, -c.max_gain_db, c.max_gain_db)) / 20)
                g = min(g, c.peak_limit / (float(np.max(np.abs(x))) + EPS))   # peak-safe: never clips
                x = (x * g).astype(np.float32)
                meta["gain"], meta["gain_db"] = g, round(20 * np.log10(g), 2)

        meta["out_dbfs"] = round(rms_dbfs(x), 2)
        return x, c.target_sr, meta


def undo_gain(y, meta):
    return (np.asarray(y, np.float32) / meta["gain"]).astype(np.float32)


def preprocess(input_path, output_path=None, cfg: PreprocessConfig | None = None):
    x, sr = load_audio(input_path)
    y, sr_out, meta = Preprocessor(cfg).process(x, sr)

    if output_path is not None:
        save_audio(output_path, y, sr_out)

    return y, sr_out, meta


if __name__ == "__main__":
    input_path = "filepath"
    output = preprocess(input_path)