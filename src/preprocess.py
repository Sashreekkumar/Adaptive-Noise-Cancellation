"""
  x_out, sr_out, meta = Preprocessor().process(x, sr)      # feed x_out to the model

  y = undo_gain(model_output, meta)                        # optional: restore the original level afterwards
"""

"""
Steps (in order), each with a switch:

1. mono, float32, NaN/Inf cleanup           - model input format
2. declip (native rate, only if clipping)   - resampling a clipped signal causes ringing, so it goes first
3. resample to the model rate (48 kHz DFN)
4. DC offset removal
5. zero-phase high-pass (rumble)            - zero-phase => no delay vs any reference
6. level normalization (one gain, peak-safe) - models are level-sensitive; too quiet => speech treated as noise
"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

from preprocess_utils import rms_dbfs, resample
from audio_io import load_audio, save_audio
from de_clip import _runs, declip

EPS = 1e-12

AUDIO_EXT = {".wav", ".flac", ".ogg"}


@dataclass
class PreprocessConfig:
    """
noisy_preprocess.py - preprocessing ONLY, for a single noisy recording, before it goes to DeepFilterNet (or any
enhancement model). No model, no clean/noise reference needed. Self-contained: numpy + scipy (+ soundfile for files).

  python noisy_preprocess.py noisy.wav prepped.wav

  python noisy_preprocess.py noisy_folder/ prepped_folder/ --target-sr 48000 --target-dbfs -25

  from noisy_preprocess import Preprocessor

  x_out, sr_out, meta = Preprocessor().process(x, sr)
  y = undo_gain(model_output, meta)

Steps (in order), each with a switch:

1. mono, float32, NaN/Inf cleanup
2. declip (native rate, only if clipping)
3. resample to the model rate (48 kHz DFN)
4. DC offset removal
5. zero-phase high-pass (rumble)
6. level normalization (one gain, peak-safe)

Deliberately NOT included: spectral subtraction, Wiener/noise gating, EQ, compression.
"""

import argparse

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy import signal

from preprocess_utils import rms_dbfs, resample
from de_clip import declip

EPS = 1e-12
AUDIO_EXT = {".wav", ".flac", ".ogg"}


@dataclass
class PreprocessConfig:
    target_sr: int = 48000
    declip: bool = True
    declip_trigger_frac: float = 5e-4
    declip_min_run: int = 3
    declip_max_run: int = 40
    declip_max_frac: float = 0.05
    dc_remove: bool = True
    highpass: bool = True
    hp_cutoff_hz: float = 60.0
    hp_order: int = 4
    normalize: bool = True
    target_dbfs: float = -25.0
    max_gain_db: float = 30.0
    min_input_dbfs: float = -80.0
    peak_limit: float = 0.98


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig | None = None):
        self.cfg = cfg or PreprocessConfig()

    def process(self, x, sr):
        """x: 1-D or (n, ch) array. Returns (float32 mono @ target_sr, target_sr, meta)."""
        c = self.cfg
        x = np.nan_to_num(np.asarray(x, np.float32))

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
                g = min(g, c.peak_limit / (float(np.max(np.abs(x))) + EPS))
                x = (x * g).astype(np.float32)
                meta["gain"], meta["gain_db"] = g, round(20 * np.log10(g), 2)

        meta["out_dbfs"] = round(rms_dbfs(x), 2)
        return x, c.target_sr, meta


def undo_gain(y, meta):
    """Optional: put the model output back at the original level."""
    return (np.asarray(y, np.float32) / meta["gain"]).astype(np.float32)


# ----------------------------------------------------------------------------- files / CLI
def main():
    import soundfile as sf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--target-sr", type=int)
    p.add_argument("--target-dbfs", type=float)
    p.add_argument("--hp-cutoff-hz", type=float)

    for s in ("declip", "dc-remove", "highpass", "normalize"):
        p.add_argument(f"--no-{s}", action="store_true")

    a = p.parse_args()
    cfg = PreprocessConfig()

    for f in fields(cfg):
        v = getattr(a, f.name, None)

        if v is not None and not isinstance(v, bool):
            setattr(cfg, f.name, v)

    for s in ("declip", "dc_remove", "highpass", "normalize"):
        if getattr(a, f"no_{s}"):
            setattr(cfg, s, False)

    pre = Preprocessor(cfg)
    src, dst = Path(a.input), Path(a.output)

    jobs = ([(f, dst / f.relative_to(src)) for f in sorted(src.rglob("*")) if f.suffix.lower() in AUDIO_EXT]
            if src.is_dir() else [(src, dst)])

    for fin, fout in jobs:
        x, sr = sf.read(str(fin), dtype="float32", always_2d=True)
        y, sr_out, meta = pre.process(x, sr)

        fout.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(fout), y, sr_out, subtype="FLOAT")
        print(fin.name, meta)


if __name__ == "__main__":
    main()target_sr: int = 48000            # DeepFilterNet = 48000; set 16000 etc. for other models
    declip: bool = True
    declip_trigger_frac: float = 5e-4  # only act if >=0.05% of samples sit in flat clipped runs
    declip_min_run: int = 3
    declip_max_run: int = 40           # longer runs are not reliably recoverable: left alone
    declip_max_frac: float = 0.05      # >5% clipped: local repair makes it worse, so it is skipped
    dc_remove: bool = True
    highpass: bool = True
    hp_cutoff_hz: float = 60.0
    hp_order: int = 4
    normalize: bool = True
    target_dbfs: float = -25.0         # RMS level of the model input
    max_gain_db: float = 30.0
    min_input_dbfs: float = -80.0      # quieter than this = digital silence, no gain applied
    peak_limit: float = 0.98           # gain is reduced so the output never exceeds this


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig | None = None):
        self.cfg = cfg or PreprocessConfig()

    def process(self, x, sr):
        """x: 1-D or (n, ch) array. Returns (float32 mono @ target_sr, target_sr, meta)."""
        c = self.cfg
        x = np.nan_to_num(np.asarray(x, np.float32))
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
    """Optional: put the model output back at the original level."""
    return (np.asarray(y, np.float32) / meta["gain"]).astype(np.float32)


# ----------------------------------------------------------------------------- files / CLI
def main():
    import soundfile as sf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--target-sr", type=int)
    p.add_argument("--target-dbfs", type=float)
    p.add_argument("--hp-cutoff-hz", type=float)

    for s in ("declip", "dc-remove", "highpass", "normalize"):
        p.add_argument(f"--no-{s}", action="store_true")

    a = p.parse_args()
    cfg = PreprocessConfig()

    for f in fields(cfg):
        v = getattr(a, f.name, None)
        if v is not None and not isinstance(v, bool):
            setattr(cfg, f.name, v)

    for s in ("declip", "dc_remove", "highpass", "normalize"):
        if getattr(a, f"no_{s}"):
            setattr(cfg, s, False)

    pre = Preprocessor(cfg)
    src, dst = Path(a.input), Path(a.output)

    jobs = ([(f, dst / f.relative_to(src)) for f in sorted(src.rglob("*")) if f.suffix.lower() in AUDIO_EXT]
            if src.is_dir() else [(src, dst)])

    for fin, fout in jobs:
        x, sr = sf.read(str(fin), dtype="float32", always_2d=True)
        y, sr_out, meta = pre.process(x, sr)
        fout.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(fout), y, sr_out, subtype="FLOAT")        # 32-bit float wav: no quantization
        print(fin.name, meta)


if __name__ == "__main__":
    main()