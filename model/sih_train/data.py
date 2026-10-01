"""Dataset access + pair validation shared by manifest building and training.

Manifests store paths relative to the SIH26052_DATASET root ("mixtures/<split>/..."), so the same manifest works on
Windows (extracted folders and/or the downloaded Drive ZIPs, whose members omit the "mixtures/" prefix) and on Colab
(/content/drive/MyDrive/SIH26052_DATASET). Nothing is resampled, cropped or repaired here.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

DRIVE_PREFIX = "/content/drive/MyDrive/SIH26052_DATASET/"
SAMPLE_RATE = 48000
TIER_CATEGORIES = {"A": {"engine", "wind"}, "B": {"vehicle", "helicopter", "siren"},
                   "C": {"gunfire", "artillery", "explosion"}}  # from Dataset_v2_CLEAN_SIH26052.ipynb
CATEGORY_TIER = {cat: tier for tier, cats in TIER_CATEGORIES.items() for cat in cats}


def relpath_from_drive(path: str) -> str:
    if not path.startswith(DRIVE_PREFIX):
        raise ValueError(f"Not a SIH26052_DATASET path: {path}")
    return path.removeprefix(DRIVE_PREFIX)


class AudioSource:
    """Resolve dataset-relative paths against directory roots first, then ZIP archives."""

    def __init__(self, roots: list[Path] | None = None, zips: list[Path] | None = None):
        self.roots = [Path(root) for root in roots or []]
        self.archives = [zipfile.ZipFile(path) for path in zips or []]
        self.members = {}
        for archive in self.archives:
            for name in archive.namelist():
                self.members.setdefault(name, archive)

    def locate(self, relpath: str) -> str | None:
        for root in self.roots:
            if (root / relpath).is_file():
                return str(root / relpath)
        member = relpath.removeprefix("mixtures/")
        return f"zip:{member}" if member in self.members else None

    def read(self, relpath: str, start: int = 0, frames: int = -1, retries: int = 3) -> tuple[np.ndarray, int, int]:
        """Returns (float32 [T, C] audio, sample rate, total frames in file). Transient I/O errors (e.g. the Colab
        Drive FUSE mount) are retried with backoff; the same bytes are read, nothing is substituted."""
        for attempt in range(retries + 1):
            try:
                return self._read(relpath, start, frames)
            except FileNotFoundError:
                raise
            except OSError:
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def _read(self, relpath: str, start: int = 0, frames: int = -1) -> tuple[np.ndarray, int, int]:
        for root in self.roots:
            path = root / relpath
            if path.is_file():
                info = sf.info(str(path))
                audio, rate = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
                return audio, rate, info.frames
        member = relpath.removeprefix("mixtures/")
        if member in self.members:
            data = io.BytesIO(self.members[member].read(member))
            info = sf.info(data)
            data.seek(0)
            audio, rate = sf.read(data, start=start, frames=frames, dtype="float32", always_2d=True)
            return audio, rate, info.frames
        raise FileNotFoundError(relpath)


def alignment_lag(clean: np.ndarray, noisy: np.ndarray, max_lag: int = 2400) -> int:
    limit = min(len(clean), len(noisy), SAMPLE_RATE * 10)
    corr = correlate(noisy[:limit], clean[:limit], mode="full", method="fft")
    lags = correlation_lags(limit, limit, mode="full")
    usable = np.abs(lags) <= max_lag
    return int(lags[usable][np.argmax(np.abs(corr[usable]))])


def validate_pair(source: AudioSource, noisy_rel: str, clean_rel: str, meta_duration: float | None = None,
                  meta_snr: float | None = None) -> dict:
    """Audio-level checks. status: PASS / WARN / FAIL; flags list every problem found (nothing is repaired)."""
    result: dict = {"audio_status": "FAIL", "audio_flags": ""}
    flags: list[str] = []
    if source.locate(noisy_rel) is None or source.locate(clean_rel) is None:
        result["audio_status"] = "UNAVAILABLE"
        result["audio_flags"] = "noisy missing" if source.locate(noisy_rel) is None else "clean missing"
        return result
    try:
        noisy, sr_n, _ = source.read(noisy_rel)
        clean, sr_c, _ = source.read(clean_rel)
    except Exception as error:  # corrupt/truncated file: flag, never repair
        result["audio_flags"] = f"unreadable: {type(error).__name__}: {error}"[:200]
        return result
    result.update(sample_rate=sr_n, clean_sample_rate=sr_c, channels=noisy.shape[1], clean_channels=clean.shape[1],
                  frames=len(noisy), clean_frames=len(clean), duration_sec=len(noisy) / sr_n)
    fatal = []
    if sr_n != SAMPLE_RATE or sr_c != SAMPLE_RATE:
        fatal.append("not 48 kHz")
    if noisy.shape[1] != 1 or clean.shape[1] != 1:
        fatal.append("not mono")
    if len(noisy) != len(clean):
        fatal.append("length mismatch")
    if not (np.isfinite(noisy).all() and np.isfinite(clean).all()):
        fatal.append("non-finite samples")
    if fatal:
        result["audio_flags"] = ";".join(fatal)
        return result
    n, c = noisy[:, 0].astype(np.float64), clean[:, 0].astype(np.float64)
    residual = n - c
    rms_c, rms_r = np.sqrt(np.mean(c ** 2)), np.sqrt(np.mean(residual ** 2))
    lag = alignment_lag(c, n)
    result.update(peak_noisy=float(np.max(np.abs(n))), peak_clean=float(np.max(np.abs(c))), alignment_lag=lag,
                  measured_snr_db_audio=float(20 * np.log10(max(rms_c, 1e-12) / max(rms_r, 1e-12))))
    if lag != 0:
        fatal.append(f"misaligned ({lag} samples)")
    if rms_r <= 1e-8:
        fatal.append("no residual noise")
    if rms_c <= 1e-6:
        fatal.append("silent clean")
    if result["peak_noisy"] >= 0.999:
        flags.append("noisy clipped")
    if result["peak_clean"] >= 0.999:
        flags.append("clean clipped")
    if meta_duration is not None and abs(result["duration_sec"] - meta_duration) > 0.002:
        flags.append("duration != metadata")
    if meta_snr is not None and abs(result["measured_snr_db_audio"] - meta_snr) > 1.0:
        flags.append("SNR differs from metadata by > 1 dB")
    result["audio_flags"] = ";".join(fatal + flags)
    result["audio_status"] = "FAIL" if fatal else ("WARN" if flags else "PASS")
    return result
