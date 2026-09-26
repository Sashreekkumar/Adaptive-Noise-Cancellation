"""Audio I/O: 48 kHz mono standardisation, on-disk cache, slicing, activity detection, PCM encode/decode."""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .constants import SR
from .errors import SampleRejected
from .sources import Source
from .util import log

_AUDIO = {"enabled": False, "dir": None, "min_samples": 0, "mem_max": 16}
_MEM: "OrderedDict[str, np.ndarray]" = OrderedDict()

READ_DTYPE = {"PCM_16": "int16", "PCM_24": "int32", "FLOAT": "float32"}


def configure_audio(cfg: dict, cache_dir: Path) -> None:
    _AUDIO.update(enabled=bool(cfg["standardized_cache"]), dir=Path(cache_dir),
                  min_samples=int(cfg["cache_min_seconds"] * SR), mem_max=int(cfg["cache_size"]))
    _MEM.clear()
    if _AUDIO["enabled"]:
        _AUDIO["dir"].mkdir(parents=True, exist_ok=True)


def std_length(sample_rate: int, frames: int) -> int:
    """Length in samples of the standardized (48 kHz) signal, without decoding anything."""
    if sample_rate == SR:
        return frames
    g = gcd(SR, sample_rate)
    up, down = SR // g, sample_rate // g
    return -(-frames * up // down)  # ceil, identical to scipy.signal.resample_poly's output length


def _downmix(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 1:
        return np.ascontiguousarray(data[:, 0])
    return data.astype(np.float64).mean(axis=1).astype(np.float32)  # channel mean


def _standardize_full(src: Source) -> np.ndarray:
    data, sr = sf.read(src.abs_path, dtype="float32", always_2d=True)
    mono = _downmix(data)
    del data
    if not np.all(np.isfinite(mono)):
        raise SampleRejected(f"non-finite samples in {src.rel}")
    if sr != SR:
        g = gcd(SR, sr)
        mono = resample_poly(mono, SR // g, sr // g).astype(np.float32)
    if len(mono) != std_length(src.sample_rate, src.frames):
        raise RuntimeError(f"unexpected standardized length for {src.rel}")
    return mono


def _cached_full(src: Source) -> np.ndarray:
    key = hashlib.sha1(f"{src.rel}|{src.size}|{src.mtime_ns}|{src.frames}|{src.sample_rate}".encode()).hexdigest()[:24]
    path = _AUDIO["dir"] / f"{key}.npy"
    expected = std_length(src.sample_rate, src.frames)
    if path.exists():
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.shape == (expected,):
                return arr
        except Exception:  # noqa: BLE001 - corrupt cache entry: rebuild below
            pass
    lock = _AUDIO["dir"] / f"{key}.lock"
    while True:  # one worker resamples a given file; the others wait for it (saves CPU and memory)
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            for _ in range(600):  # wait up to ~5 min, then re-check / steal a stale lock
                time.sleep(0.5)
                if path.exists() or not lock.exists():
                    break
            else:
                try:
                    if time.time() - lock.stat().st_mtime > 1800:
                        lock.unlink()
                except OSError:
                    pass
            if path.exists():
                try:
                    arr = np.load(path, mmap_mode="r")
                    if arr.shape == (expected,):
                        return arr
                except Exception:  # noqa: BLE001
                    pass
    try:
        log(f"  [resample] {src.rel} ({src.frames / src.sample_rate:.0f}s @ {src.sample_rate} Hz) -> cached 48 kHz mono")
        arr = _standardize_full(src)
        tmp = _AUDIO["dir"] / f"{key}.{os.getpid()}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    return np.load(path, mmap_mode="r")


def _get_full(src: Source) -> np.ndarray:
    if _AUDIO["enabled"] and std_length(src.sample_rate, src.frames) >= _AUDIO["min_samples"]:
        return _cached_full(src)
    a = _MEM.get(src.rel)
    if a is not None:
        _MEM.move_to_end(src.rel)
        return a
    a = _standardize_full(src)
    a.setflags(write=False)
    _MEM[src.rel] = a
    while len(_MEM) > max(1, _AUDIO["mem_max"]):
        _MEM.popitem(last=False)
    return a


def get_slice(src: Source, start: int, stop: int) -> np.ndarray:
    """Samples [start, stop) of the standardized 48 kHz mono signal (float32), reading only what is needed."""
    if src.sample_rate == SR:
        data, _ = sf.read(src.abs_path, start=start, stop=stop, dtype="float32", always_2d=True)
        out = _downmix(data)
        if not np.all(np.isfinite(out)):
            raise SampleRejected(f"non-finite samples in {src.rel}")
    else:
        out = np.array(_get_full(src)[start:stop])
    if len(out) != stop - start:
        raise RuntimeError(f"short read from {src.rel}: wanted {stop - start}, got {len(out)}")
    return out


def detect_active_regions(x: np.ndarray, ad: dict) -> list[tuple[int, int]]:
    n = len(x)
    fl = max(1, int(round(ad["frame_ms"] * SR / 1000)))
    hop = max(1, int(round(ad["hop_ms"] * SR / 1000)))
    x = x.astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(x * x)))  # frame energies via cumulative sums: O(n), no big temporaries
    if n < fl:
        fl, starts = n, np.array([0])
    else:
        starts = np.arange(0, n - fl + 1, hop)
    energy = np.maximum(csum[starts + fl] - csum[starts], 0.0) / fl
    db = 10 * np.log10(energy + 1e-24)
    thr = max(np.percentile(db, 95) - ad["threshold_db_below_p95"], ad["absolute_floor_dbfs"])
    idx = np.flatnonzero(db > thr)
    if idx.size == 0:
        return []
    # contiguous runs of active frames -> sample regions
    breaks = np.flatnonzero(np.diff(idx) > 1)
    run_starts = np.r_[idx[0], idx[breaks + 1]]
    run_ends = np.r_[idx[breaks], idx[-1]]
    regions = [[int(starts[a]), int(min(n, starts[b] + fl))] for a, b in zip(run_starts, run_ends)]
    gap = int(ad["min_gap_ms"] * SR / 1000)
    merged = [regions[0]]
    for a, b in regions[1:]:
        if a - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    min_len = int(ad["min_region_ms"] * SR / 1000)
    return [(a, b) for a, b in merged if b - a >= min_len]


def encode(x: np.ndarray, subtype: str) -> np.ndarray:
    if subtype == "PCM_16":
        return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    if subtype == "PCM_24":
        return (np.clip(np.rint(x * 8388608.0), -8388608, 8388607).astype(np.int32)) * 256
    return x.astype(np.float32)


def decode(a: np.ndarray, subtype: str) -> np.ndarray:
    if subtype == "PCM_16":
        return a.astype(np.float64) / 32768.0
    if subtype == "PCM_24":
        return a.astype(np.float64) / 2147483648.0
    return a.astype(np.float64)


def read_decoded(path: Path, subtype: str) -> np.ndarray:
    a, sr = sf.read(str(path), dtype=READ_DTYPE[subtype], always_2d=False)
    if sr != SR or a.ndim != 1:
        raise RuntimeError(f"unexpected format in {path}: sr={sr}, ndim={a.ndim}")
    return decode(a, subtype)


def write_mp3(path: Path, pcm_i16: np.ndarray, compression_level: float) -> None:
    """Encode a PCM_16 array to mp3. libsndfile/LAME here is deterministic (verified: repeated encodes of
    the same input at the same compression_level are byte-identical), which is what makes --reconstruct
    possible for mp3 output."""
    sf.write(str(path), pcm_i16, SR, format="MP3", subtype="MPEG_LAYER_III", compression_level=compression_level)


def read_mp3_peak(path: Path) -> float:
    """Decode an mp3 file and return its peak absolute amplitude (for a post-compression clipping check)."""
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR or a.ndim != 1:
        raise RuntimeError(f"unexpected format in {path}: sr={sr}, ndim={a.ndim}")
    return float(np.max(np.abs(a))) if len(a) else 0.0


def mask_from_regions(n: int, regions: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in regions:
        m[s:e] = True
    return m


def apply_fade(x: np.ndarray, fade_samples: int) -> np.ndarray:
    """Linear fade-in/out at both ends of a short event segment, so splicing it into a timeline (as
    impulsive noise events are) doesn't produce an audible click. Returns a new array; `x` is untouched."""
    n = len(x)
    f = max(0, min(int(fade_samples), n // 2))
    if f == 0:
        return x
    out = x.copy()
    ramp = np.linspace(0.0, 1.0, f, endpoint=False)
    out[:f] *= ramp
    out[-f:] *= ramp[::-1]
    return out
