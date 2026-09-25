#!/usr/bin/env python3
"""
Reproducible, provenance-preserving synthetic noisy-speech dataset generator.

Layout expected under --root (default "."):

    wav/    clean speech, any depth            (recursively scanned)
    noise/  noise recordings, any depth        (recursively scanned; categories are auto-discovered)

Layout produced under --output (default <root>/output):

    <split>/clean/000001.wav        clean reference (48 kHz, mono)
    <split>/noisy/000001.wav        clean + scaled composite noise
    metadata/dataset_metadata.jsonl exactly one JSON record per generated sample
    metadata/dataset_summary.json   statistics computed from the JSONL
    metadata/run_config.json        effective config, hashes, source inventory hash
    metadata/source_inventory.json  every discovered source with inferred attributes
    metadata/failed_attempts.jsonl  every rejected/failed attempt (never silent)

Guarantees
  * A sample's audio is only moved into place *before* its metadata line is appended, and any
    audio without a metadata record is removed on the next start (crash recovery), so the final
    state is always: #records == #clean files == #noisy files, with matching IDs.
  * Sample indices in metadata (source_start_sample, chunks, ...) refer to the *standardized*
    48 kHz mono version of the source (mean-downmix, scipy.signal.resample_poly). Sources are
    never modified.
  * measured_snr_db is measured on the decoded output files (after quantisation and any
    clipping-prevention gain), over the recorded active regions.
  * Each sample has its own random_seed. Given the same sources, config and code version,
    the seed alone determines every random choice for that sample.

Composite noise reconstruction (what the metadata makes possible):
    seg_i   = concat(source_i[c.source_start_sample:c.source_end_sample] for c in chunks_i)
    comp    = sum_i  place(seg_i / normalization_rms_i * 10^(relative_gain_db_i/20), start_sample_i)
    noisy   = output_gain * (clean + composite_noise_scale * comp)
    clean_output = output_gain * clean
Run `--reconstruct 000001` to verify this from the metadata alone.

Performance design (v1.1)
  * Multi-process generation (default: min(cpu_count, 8) workers; --workers N to change).
  * Only the audio actually needed is read: sources already at 48 kHz are read with partial reads;
    long non-48 kHz sources (>= cache_min_seconds) are resampled ONCE into a float32 .npy cache
    (<output>/.cache/std48k, safe to delete) and memory-mapped afterwards.
  * Source files are probed lazily (no need to open every file at start-up); --probe-all validates
    everything up-front instead.
  * Progress is logged during scanning, generation, and while workers are busy.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import Counter, OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly

__version__ = "1.2.0"
SR = 48000
CHANNELS = 1
EPS = 1e-12

DEFAULT_CONFIG: dict[str, Any] = {
    "base_seed": 12345,
    # split name -> number of samples. IDs are global and sequential across splits, in order.
    "splits": {"train": 50000},
    "clean_dir": "wav",
    "noise_dir": "noise",
    "audio_extensions": [".wav", ".flac"],
    "output_subtype": "PCM_16",  # PCM_16 (smallest) | PCM_24 | FLOAT

    # ---- noise selection -------------------------------------------------------------
    "num_noises": [1, 3],  # TOTAL noises per sample (including required ones); int, or [min, max] inclusive
    # Categories that must appear in EVERY sample (one recording from each), e.g. ["gunshot"]. The remaining
    # noises are drawn only from the other categories. Needs num_noises min > len(required_noise_categories).
    "required_noise_categories": [],
    "category_mode": "prefer_different",  # any | prefer_different | require_different (applies to the non-required noises)
    "noise_category_level": "parent",  # parent (immediate parent dir) | top (first dir below noise/)
    "short_noise_policy": "loop",  # loop | reject
    "noise_placement": {"mode": "full", "max_start_fraction": 0.5},  # full | random_start
    # relative gain of every noise after the first (first noise is the 0 dB reference)
    "relative_gain_db": {"type": "uniform", "min": -10.0, "max": 0.0},
    "min_noise_rms": 1e-6,

    # ---- SNR ---------------------------------------------------------------------------
    # {"type":"fixed","value":5} | {"type":"choice","values":[-5,0,5]} | {"type":"uniform","min":-5,"max":20}
    "snr_db": {"type": "uniform", "min": 5.0, "max": 20.0},
    "snr_method": "active_rms",  # active_rms | full_rms
    "activity_detection": {
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "threshold_db_below_p95": 25.0,  # frame is active if within this many dB of the 95th-percentile frame level
        "absolute_floor_dbfs": -70.0,  # ...and above this absolute level
        "min_gap_ms": 200.0,  # merge active regions separated by less than this
        "min_region_ms": 100.0,  # drop active regions shorter than this
    },

    # ---- clean speech ----------------------------------------------------------------------
    "clean_crop": {"min_duration_seconds": 1.0, "max_duration_seconds": None},

    # ---- clipping ----------------------------------------------------------------------------
    "peak_limit": 0.99,  # linear full-scale
    "clipping_policy": "scale",  # scale: apply one common gain to clean AND noisy | reject

    # ---- provenance attributes (only ever set from explicit information; never guessed) ------------
    "clean_dataset_name": None,
    "noise_dataset_name": None,
    # e.g. "{clean_language}/{clean_speaker_id}/{file}" - path below wav/, whole-component placeholders only.
    "clean_path_template": None,
    # optional CSV with a `path` column (relative to root, e.g. wav/x.wav) plus any of:
    #   clean_dataset, clean_speaker_id, clean_language, clean_accent | noise_dataset, noise_category
    "clean_metadata_csv": None,
    "noise_metadata_csv": None,

    "max_attempts_per_sample": 20,
    "summary_snr_bin_db": 5.0,

    # ---- performance ---------------------------------------------------------------------
    "probe_sources": "lazy",  # lazy: open a file only when first used | upfront: validate every file at start
    "standardized_cache": True,  # cache resampled long sources on disk (float32 .npy), memory-mapped on use
    "cache_min_seconds": 20.0,  # only non-48 kHz sources at least this long are disk-cached
    "cache_dir": None,  # default <output>/.cache/std48k
    "cache_size": 16,  # in-memory LRU entries for shorter non-48 kHz sources
}
# keys that do not influence the generated audio/metadata; excluded from the resume-compatibility hash
NON_SEMANTIC_KEYS = {"splits", "summary_snr_bin_db", "standardized_cache", "cache_min_seconds", "cache_dir", "cache_size"}

CLEAN_FIELDS = ("clean_dataset", "clean_speaker_id", "clean_language", "clean_accent")
NOISE_FIELDS = ("noise_dataset", "noise_category")


class ConfigError(Exception):
    pass


class SampleRejected(Exception):
    """An attempt that cannot yield a valid sample (logged, then retried with a new seed)."""


# ======================================================================================
# Configuration
# ======================================================================================
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "splits":
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def num_noises_range(cfg: dict) -> tuple[int, int]:
    nn = cfg["num_noises"]
    lo, hi = (nn, nn) if isinstance(nn, int) else (int(nn[0]), int(nn[1]))
    return lo, hi


def validate_config(c: dict) -> None:
    lo, hi = num_noises_range(c)
    if lo < 1 or hi < lo:
        raise ConfigError("num_noises must be an int >= 1 or [min, max] with 1 <= min <= max")
    req = c["required_noise_categories"]
    if not isinstance(req, list) or not all(isinstance(x, str) for x in req) or len(set(req)) != len(req):
        raise ConfigError("required_noise_categories must be a list of distinct category names")
    if req and lo < len(req) + 1:
        raise ConfigError(f"required_noise_categories={req} needs num_noises >= {len(req) + 1} "
                          f"(required noise(s) plus at least one other); got min {lo}")
    if c["category_mode"] not in ("any", "prefer_different", "require_different"):
        raise ConfigError("category_mode must be any | prefer_different | require_different")
    if c["output_subtype"] not in ("PCM_16", "PCM_24", "FLOAT"):
        raise ConfigError("output_subtype must be PCM_16 | PCM_24 | FLOAT")
    if c["snr_method"] not in ("active_rms", "full_rms"):
        raise ConfigError("snr_method must be active_rms | full_rms")
    if c["short_noise_policy"] not in ("loop", "reject"):
        raise ConfigError("short_noise_policy must be loop | reject")
    if c["clipping_policy"] not in ("scale", "reject"):
        raise ConfigError("clipping_policy must be scale | reject")
    if c["noise_category_level"] not in ("parent", "top"):
        raise ConfigError("noise_category_level must be parent | top")
    if not (0 < c["peak_limit"] <= 1.0):
        raise ConfigError("peak_limit must be in (0, 1]")
    p = c["noise_placement"]
    if p["mode"] not in ("full", "random_start") or not (0 <= p["max_start_fraction"] < 1):
        raise ConfigError("noise_placement: mode full|random_start, max_start_fraction in [0,1)")
    for name in ("snr_db", "relative_gain_db"):
        s = c[name]
        ok = (s.get("type") == "fixed" and "value" in s) or \
             (s.get("type") == "choice" and s.get("values")) or \
             (s.get("type") == "uniform" and s["min"] <= s["max"])
        if not ok:
            raise ConfigError(f"invalid distribution spec for {name}: {s}")
    if not c["splits"] or any(int(n) < 0 for n in c["splits"].values()):
        raise ConfigError("splits must map split names to non-negative counts")
    if int(c["base_seed"]) < 0:
        raise ConfigError("base_seed must be >= 0")
    if c["probe_sources"] not in ("lazy", "upfront"):
        raise ConfigError("probe_sources must be lazy | upfront")


def draw_dist(spec: dict, rng: np.random.Generator) -> float:
    t = spec["type"]
    if t == "fixed":
        return float(spec["value"])
    if t == "choice":
        vals = spec["values"]
        return float(vals[int(rng.integers(len(vals)))])
    return round(float(rng.uniform(spec["min"], spec["max"])), 2)  # uniform, recorded value == used value


def config_hash(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if k not in NON_SEMANTIC_KEYS}  # e.g. allow growing splits on --resume
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


# ======================================================================================
# Source discovery
# ======================================================================================
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Ticker:
    def __init__(self, interval: float):
        self.interval, self.last = interval, time.time()

    def due(self) -> bool:
        now = time.time()
        if now - self.last >= self.interval:
            self.last = now
            return True
        return False


@dataclass
class Source:
    rel: str  # posix path relative to root, e.g. "wav/hindi/speaker_001/001.wav"
    abs_path: str
    size: int
    mtime_ns: int = 0
    sample_rate: int = 0  # 0/0/0 until probed (lazily on first use, or up-front with --probe-all)
    channels: int = 0
    frames: int = 0
    meta: dict = field(default_factory=dict)  # provenance attributes named like the record fields


def ensure_probed(src: Source) -> Source:
    if src.frames <= 0:
        info = sf.info(src.abs_path)
        if info.frames <= 0:
            raise SampleRejected(f"zero-length audio: {src.rel}")
        src.sample_rate, src.channels, src.frames = int(info.samplerate), int(info.channels), int(info.frames)
    return src


def source_from_path(root: Path, rel: str) -> Source:
    p = root / rel
    st = p.stat()
    return ensure_probed(Source(rel, str(p), st.st_size, st.st_mtime_ns))


def infer_from_template(parts: list[str], template: Optional[str]) -> dict:
    """Map path components (below wav/) onto whole-component placeholders. Never guesses."""
    if not template:
        return {}
    tp = template.strip("/").split("/")
    if len(tp) != len(parts):
        return {}
    found = {}
    for t, p in zip(tp, parts):
        if t in ("*", "{file}"):
            continue
        if t.startswith("{") and t.endswith("}"):
            found[t[1:-1]] = p
        elif t != p:
            return {}
    return {k: v for k, v in found.items() if k in CLEAN_FIELDS}


def load_sidecar(path: Optional[str], root: Path) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    p = p if p.is_absolute() else root / p
    out: dict[str, dict] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = (row.get("path") or "").strip().replace("\\", "/")
            if rel:
                out[rel] = {k: (v.strip() or None) for k, v in row.items() if k != "path" and v is not None}
    return out


def _scan_tree(base: Path, exts: set, label: str) -> list[tuple[str, int, int]]:
    """Iterative recursive scan using scandir (file size/mtime come free with the directory listing)."""
    found, stack, tick = [], [str(base)], Ticker(2.0)
    while stack:
        d = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError as e:
            log(f"  warning: cannot read directory {d}: {e}")
            continue
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file() and os.path.splitext(e.name)[1].lower() in exts:
                    st = e.stat()
                    found.append((e.path, st.st_size, st.st_mtime_ns))
            except OSError as ex:
                log(f"  warning: cannot stat {e.path}: {ex}")
        if tick.due():
            log(f"  scanning {label}/ ... {len(found)} audio files found so far")
    return found


def _probe(path: str):
    try:
        info = sf.info(path)
        if info.frames <= 0:
            raise ValueError("zero frames")
        return (int(info.samplerate), int(info.channels), int(info.frames)), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def discover_sources(root: Path, cfg: dict, kind: str, threads: int) -> tuple[list[Source], list[dict]]:
    sub = cfg["clean_dir" if kind == "clean" else "noise_dir"]
    base = root / sub
    if not base.is_dir():
        raise FileNotFoundError(f"{kind} directory not found: {base}")
    exts = {e.lower() for e in cfg["audio_extensions"]}
    sidecar = load_sidecar(cfg[f"{kind}_metadata_csv"], root)
    found = _scan_tree(base, exts, sub)
    found.sort(key=lambda t: Path(t[0]).relative_to(root).as_posix())  # OS-independent deterministic order
    log(f"  {kind}: {len(found)} audio files found")

    probes: list = [None] * len(found)
    if cfg["probe_sources"] == "upfront":
        tick = Ticker(2.0)
        with ThreadPoolExecutor(max_workers=threads) as tp:
            for i, res in enumerate(tp.map(_probe, [f[0] for f in found])):
                probes[i] = res
                if tick.due():
                    log(f"  validating {kind} files: {i + 1}/{len(found)}")

    sources, unreadable = [], []
    for (path, size, mtime), pr in zip(found, probes):
        p = Path(path)
        rel = p.relative_to(root).as_posix()
        sr = ch = fr = 0
        if pr is not None:
            info, err = pr
            if info is None:  # unreadable/corrupt: excluded but reported
                unreadable.append({"path": rel, "error": err})
                continue
            sr, ch, fr = info
        parts = list(p.relative_to(base).parts)
        meta: dict[str, Any]
        if kind == "clean":
            meta = {k: None for k in CLEAN_FIELDS}
            meta["clean_dataset"] = cfg["clean_dataset_name"]
            meta.update(infer_from_template(parts, cfg["clean_path_template"]))
            fields = CLEAN_FIELDS
        else:
            meta = {k: None for k in NOISE_FIELDS}
            meta["noise_dataset"] = cfg["noise_dataset_name"]
            dirs = parts[:-1]  # directories between noise/ and the file
            if dirs:
                meta["noise_category"] = dirs[0] if cfg["noise_category_level"] == "top" else dirs[-1]
            fields = NOISE_FIELDS
        for k, v in sidecar.get(rel, {}).items():
            if k in fields and v is not None:
                meta[k] = v
        sources.append(Source(rel, path, size, mtime, sr, ch, fr, meta))
    return sources, unreadable


def inventory_hash(cleans: list[Source], noises: list[Source]) -> str:
    h = hashlib.sha256()
    for s in cleans + noises:
        h.update(f"{s.rel}|{s.size}\n".encode())
    return h.hexdigest()


def group_noises(noises: list[Source]) -> tuple[list[Optional[str]], list[list[int]]]:
    """(category keys, file indices per category); unknown/None is one group; deterministic order."""
    groups: dict[Optional[str], list[int]] = {}
    for i, s in enumerate(noises):
        groups.setdefault(s.meta["noise_category"], []).append(i)
    keys = sorted(groups, key=lambda k: (k is None, k or ""))
    return keys, [groups[k] for k in keys]


# ======================================================================================
# Audio helpers
# ======================================================================================
_AUDIO = {"enabled": False, "dir": None, "min_samples": 0, "mem_max": 16}
_MEM: "OrderedDict[str, np.ndarray]" = OrderedDict()


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


_READ_DTYPE = {"PCM_16": "int16", "PCM_24": "int32", "FLOAT": "float32"}


def decode(a: np.ndarray, subtype: str) -> np.ndarray:
    if subtype == "PCM_16":
        return a.astype(np.float64) / 32768.0
    if subtype == "PCM_24":
        return a.astype(np.float64) / 2147483648.0
    return a.astype(np.float64)


def read_decoded(path: Path, subtype: str) -> np.ndarray:
    a, sr = sf.read(str(path), dtype=_READ_DTYPE[subtype], always_2d=False)
    if sr != SR or a.ndim != 1:
        raise RuntimeError(f"unexpected format in {path}: sr={sr}, ndim={a.ndim}")
    return decode(a, subtype)


def mask_from_regions(n: int, regions: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in regions:
        m[s:e] = True
    return m


# ======================================================================================
# Random choices
# ======================================================================================
def derive_seed(base_seed: int, k: int, attempt: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(k), int(attempt)]).generate_state(1, dtype=np.uint32)[0])


def _pick_from(groups: list[list[int]], n: int, mode: str, rng: np.random.Generator) -> list[int]:
    """Pick n distinct files from category groups according to category_mode."""
    files = sorted(i for g in groups for i in g)
    n = min(n, len(files))
    ng = len(groups)
    if n <= 0:
        return []
    if mode == "any":
        return [int(files[int(i)]) for i in rng.choice(len(files), size=n, replace=False)]
    if ng >= n:  # enough categories: one file from each of n distinct categories
        gi = rng.choice(ng, size=n, replace=False)
        return [int(groups[int(g)][int(rng.integers(len(groups[int(g)])))]) for g in gi]
    if mode == "require_different":
        raise SampleRejected("require_different but not enough categories")
    chosen = [int(g[int(rng.integers(len(g)))]) for g in groups]  # prefer_different: every category once...
    taken = set(chosen)
    rest = [i for i in files if i not in taken]  # ...then fill from the remaining files
    chosen += [int(rest[int(i)]) for i in rng.choice(len(rest), size=n - ng, replace=False)]
    return [chosen[int(i)] for i in rng.permutation(len(chosen))]


def select_noises(keys: list, groups: list[list[int]], n: int, mode: str, required: list[str],
                  rng: np.random.Generator) -> list[int]:
    if not required:
        return _pick_from(groups, n, mode, rng)
    req_idx = [i for i, k in enumerate(keys) if k in required]
    picks = [int(groups[i][int(rng.integers(len(groups[i])))]) for i in req_idx]  # one per required category
    others = [g for i, g in enumerate(groups) if i not in set(req_idx)]  # extras never come from required categories
    picks += _pick_from(others, n - len(picks), mode, rng)
    return [picks[int(i)] for i in rng.permutation(len(picks))]  # random order -> random 0 dB reference noise


def plan_noise_chunks(L: int, need: int, policy: str, rng: np.random.Generator):
    """Decide which source ranges cover `need` samples. Returns (chunks[(src_start, src_end, dest_offset)], looped)."""
    if L >= need:
        st = int(rng.integers(0, L - need + 1))
        return [(st, st + need, 0)], False
    if policy == "reject":
        raise SampleRejected("noise shorter than required and short_noise_policy=reject")
    cur = int(rng.integers(0, L))
    chunks, filled = [], 0
    while filled < need:
        take = min(L - cur, need - filled)
        chunks.append((cur, cur + take, filled))
        filled += take
        cur = 0
    return chunks, True


# ======================================================================================
# Sample generation
# ======================================================================================
@dataclass
class Context:
    cfg: dict
    cleans: list[Source]
    noises: list[Source]
    keys: list
    groups: list[list[int]]
    tmp_dir: Path


_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log(msg)


def make_sample(k: int, split: str, attempt: int, ctx: Context) -> dict:
    cfg = ctx.cfg
    sub = cfg["output_subtype"]
    seed = derive_seed(cfg["base_seed"], k, attempt)
    rng = np.random.default_rng(seed)
    sid = f"{k:06d}"

    # ---- clean speech ---------------------------------------------------------------
    cf = ensure_probed(ctx.cleans[int(rng.integers(len(ctx.cleans)))])
    L = std_length(cf.sample_rate, cf.frames)
    cc = cfg["clean_crop"]
    min_s = int(round(cc["min_duration_seconds"] * SR)) if cc["min_duration_seconds"] else 1
    if L < min_s:
        raise SampleRejected(f"clean shorter than min_duration_seconds ({cf.rel})")
    max_s = int(round(cc["max_duration_seconds"] * SR)) if cc["max_duration_seconds"] else None
    if max_s and L > max_s:
        cs = int(rng.integers(0, L - max_s + 1))
        ce, cropped = cs + max_s, True
    else:
        cs, ce, cropped = 0, L, False
    clean = get_slice(cf, cs, ce).astype(np.float64)
    N = len(clean)
    if not cropped and N > 300 * SR:
        _warn_once(f"long-clean:{cf.rel}", f"  warning: {cf.rel} is {N / SR:.0f}s long and clean_crop.max_duration_seconds "
                   f"is not set, so samples will be this long (slow, large output).")

    # ---- active regions ---------------------------------------------------------------
    if cfg["snr_method"] == "active_rms":
        regions = detect_active_regions(clean, cfg["activity_detection"])
        if not regions:
            raise SampleRejected(f"no active speech detected in {cf.rel}")
    else:
        regions = [(0, N)]
    mask = mask_from_regions(N, regions)

    # ---- noises ---------------------------------------------------------------------------
    lo, hi = num_noises_range(cfg)
    n_req = lo if lo == hi else int(rng.integers(lo, hi + 1))
    picks = select_noises(ctx.keys, ctx.groups, n_req, cfg["category_mode"], cfg["required_noise_categories"], rng)
    pl = cfg["noise_placement"]
    composite = np.zeros(N)
    noise_meta = []
    for j, ni in enumerate(picks):
        ns = ensure_probed(ctx.noises[ni])
        Ln = std_length(ns.sample_rate, ns.frames)
        start = 0 if pl["mode"] == "full" else int(rng.integers(0, int(pl["max_start_fraction"] * N) + 1))
        need = N - start
        chunks, looped = plan_noise_chunks(Ln, need, cfg["short_noise_policy"], rng)
        if looped:  # source shorter than needed, so it is cheap to load whole
            whole = get_slice(ns, 0, Ln)
            seg = np.concatenate([whole[a:b] for a, b, _ in chunks]).astype(np.float64)
        else:
            seg = get_slice(ns, chunks[0][0], chunks[0][1]).astype(np.float64)
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < cfg["min_noise_rms"]:
            raise SampleRejected(f"noise segment is (near) silent: {ns.rel}")
        gain_db = 0.0 if j == 0 else draw_dist(cfg["relative_gain_db"], rng)
        composite[start:] += seg / rms * 10 ** (gain_db / 20)
        nm = {
            "source": ns.rel,
            "noise_dataset": ns.meta["noise_dataset"],
            "noise_category": ns.meta["noise_category"],
            "relative_gain_db": gain_db,
            "normalization_rms": rms,
            "start_sample": start,
            "start_time_seconds": round(start / SR, 6),
            "end_sample": N,
            "looped": looped,
        }
        if not looped:
            nm["source_start_sample"], nm["source_end_sample"] = chunks[0][0], chunks[0][1]
        nm["chunks"] = [{"source_start_sample": a, "source_end_sample": b,
                         "destination_start_sample": start + d, "destination_end_sample": start + d + (b - a)}
                        for a, b, d in chunks]
        nm["source_original_sample_rate"] = ns.sample_rate
        nm["source_original_channels"] = ns.channels
        nm["source_standardized_num_samples"] = Ln
        noise_meta.append(nm)

    # ---- scale composite noise to target SNR (over active region) ---------------------------
    target = draw_dist(cfg["snr_db"], rng)
    Pc = float(np.mean(clean[mask] ** 2))
    Pn = float(np.mean(composite[mask] ** 2))
    if Pc < EPS or Pn < EPS:
        raise SampleRejected("zero clean or noise power in measurement region")
    scale = math.sqrt(Pc / (Pn * 10 ** (target / 10)))
    mix = clean + composite * scale

    # ---- clipping prevention: one common gain on clean AND noisy keeps the SNR relationship ---
    peak = float(np.max(np.abs(mix)))
    g = 1.0
    if peak > cfg["peak_limit"]:
        if cfg["clipping_policy"] == "reject":
            raise SampleRejected(f"mixture peak {peak:.3f} exceeds peak_limit")
        g = cfg["peak_limit"] / peak
    clean_out, noisy_out = clean * g, mix * g

    # ---- write, then measure on the decoded files ---------------------------------------------
    tmp_c, tmp_n = ctx.tmp_dir / f"{sid}_clean.wav", ctx.tmp_dir / f"{sid}_noisy.wav"
    sf.write(str(tmp_c), encode(clean_out, sub), SR, subtype=sub, format="WAV")
    sf.write(str(tmp_n), encode(noisy_out, sub), SR, subtype=sub, format="WAV")
    c_dec, n_dec = read_decoded(tmp_c, sub), read_decoded(tmp_n, sub)
    if len(c_dec) != N or len(n_dec) != N:
        raise RuntimeError("length mismatch after write")
    dec_peak = float(np.max(np.abs(n_dec)))
    if dec_peak >= 1.0:
        raise SampleRejected("clipping detected in written file")
    res = n_dec - c_dec
    Pc2, Pr2 = float(np.mean(c_dec[mask] ** 2)), float(np.mean(res[mask] ** 2))
    if Pc2 < EPS or Pr2 < EPS:
        raise SampleRejected("degenerate measured power")
    measured = 10 * math.log10(Pc2 / Pr2)

    record = {
        "sample_id": sid,
        "split": split,
        "clean_source": cf.rel,
        "clean_output": f"{split}/clean/{sid}.wav",
        **{f: cf.meta[f] for f in CLEAN_FIELDS},
        "clean_original_sample_rate": cf.sample_rate,
        "clean_original_channels": cf.channels,
        "clean_source_standardized_num_samples": L,
        "clean_cropped": cropped,
        "clean_start_sample": cs,
        "clean_end_sample": ce,
        "clean_start_time_seconds": round(cs / SR, 6),
        "clean_duration_seconds": round(N / SR, 6),
        "noisy_output": f"{split}/noisy/{sid}.wav",
        "noises": noise_meta,
        "num_noises": len(noise_meta),
        "category_mode": cfg["category_mode"],
        "required_noise_categories": cfg["required_noise_categories"],
        "target_snr_db": target,
        "measured_snr_db": round(measured, 4),
        "snr_error_db": round(measured - target, 4),
        "snr_measurement": {
            "method": cfg["snr_method"],
            "measured_on": "decoded_output_files",
            "active_regions": [{"start_sample": s, "end_sample": e} for s, e in regions],
            "active_fraction": round(float(mask.mean()), 4),
            "detector": cfg["activity_detection"] if cfg["snr_method"] == "active_rms" else None,
        },
        "composite_noise_scale": scale,
        "composite_noise_scale_db": 20 * math.log10(scale),
        "output_gain": g,
        "output_gain_db": 20 * math.log10(g),
        "clipping_prevented": g < 1.0,
        "peak_amplitude": round(dec_peak, 6),
        "duration_seconds": round(N / SR, 6),
        "sample_rate": SR,
        "channels": CHANNELS,
        "output_subtype": sub,
        "random_seed": seed,
        "generator_version": __version__,
    }
    return record


_CTX: Optional[Context] = None


def _init_worker(cfg, cleans, noises, tmp_dir, cache_dir):
    global _CTX
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the parent only
    except (ValueError, OSError):
        pass
    configure_audio(cfg, Path(cache_dir))
    _CTX = Context(cfg, cleans, noises, *group_noises(noises), Path(tmp_dir))


def _run_sample(task: tuple[int, str]) -> dict:
    k, split = task
    fails, t0 = [], time.time()
    for attempt in range(_CTX.cfg["max_attempts_per_sample"]):
        try:
            rec = make_sample(k, split, attempt, _CTX)
            return {"ok": True, "record": rec, "failures": fails, "seconds": time.time() - t0}
        except Exception as e:  # noqa: BLE001 - every failure is logged, never silent
            fails.append({"sample_id": f"{k:06d}", "attempt": attempt,
                          "random_seed": derive_seed(_CTX.cfg["base_seed"], k, attempt),
                          "error_type": type(e).__name__, "error": str(e)})
    return {"ok": False, "record": None, "failures": fails, "seconds": time.time() - t0}


# ======================================================================================
# Output-directory state, commit and verification
# ======================================================================================
def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_records(meta_path: Path, repair: bool = False) -> list[dict]:
    """Read JSONL. With repair=True a torn (partial) final line is truncated away."""
    if not meta_path.exists():
        return []
    recs, offset, good_end = [], 0, 0
    with open(meta_path, "rb") as f:
        lines = f.readlines()
    for i, raw in enumerate(lines):
        try:
            if not raw.endswith(b"\n"):
                raise ValueError("no trailing newline")
            recs.append(json.loads(raw))
            good_end = offset + len(raw)
        except ValueError:
            if i == len(lines) - 1 and repair:
                with open(meta_path, "r+b") as f:
                    f.truncate(good_end)
                print(f"[recover] truncated torn final metadata line", file=sys.stderr)
                break
            raise RuntimeError(f"corrupt metadata at line {i + 1} of {meta_path}")
        offset += len(raw)
    return recs


def audio_files_on_disk(out: Path) -> dict[str, set[str]]:
    res = {"clean": set(), "noisy": set()}
    for d in out.iterdir() if out.exists() else []:
        if d.is_dir() and d.name not in ("metadata", ".tmp"):
            for kind in res:
                kd = d / kind
                if kd.is_dir():
                    res[kind].update(f"{d.name}/{kind}/{p.name}" for p in kd.glob("*.wav"))
    return res


def recover_state(out: Path) -> dict[str, dict]:
    """Make the output dir consistent: torn line repaired, orphan audio removed, temp cleared."""
    shutil.rmtree(out / ".tmp", ignore_errors=True)
    (out / ".tmp").mkdir(parents=True, exist_ok=True)
    recs = read_records(out / "metadata" / "dataset_metadata.jsonl", repair=True)
    done: dict[str, dict] = {}
    for r in recs:
        if r["sample_id"] in done:
            raise RuntimeError(f"duplicate metadata record for sample {r['sample_id']}")
        done[r["sample_id"]] = r
    expected = set()
    for r in recs:
        for key in ("clean_output", "noisy_output"):
            expected.add(r[key])
            if not (out / r[key]).exists():
                raise RuntimeError(f"metadata record {r['sample_id']} references missing file {r[key]}")
    for kind, files in audio_files_on_disk(out).items():
        for f in sorted(files - expected):
            (out / f).unlink()
            print(f"[recover] removed audio without metadata record: {f}", file=sys.stderr)
    return done


def verify_output(out: Path) -> dict:
    recs = read_records(out / "metadata" / "dataset_metadata.jsonl")
    disk = audio_files_on_disk(out)
    problems = []
    ids = [r["sample_id"] for r in recs]
    if len(set(ids)) != len(ids):
        problems.append("duplicate sample_ids in metadata")
    clean_exp = {r["clean_output"] for r in recs}
    noisy_exp = {r["noisy_output"] for r in recs}
    for r in recs:
        if Path(r["clean_output"]).stem != r["sample_id"] or Path(r["noisy_output"]).stem != r["sample_id"]:
            problems.append(f"{r['sample_id']}: filename/ID mismatch")
    if clean_exp != disk["clean"]:
        problems.append(f"clean files vs records differ: {len(disk['clean'] ^ clean_exp)} mismatches")
    if noisy_exp != disk["noisy"]:
        problems.append(f"noisy files vs records differ: {len(disk['noisy'] ^ noisy_exp)} mismatches")
    return {"metadata_records": len(recs), "clean_files": len(disk["clean"]), "noisy_files": len(disk["noisy"]),
            "one_to_one": not problems and len(recs) == len(disk["clean"]) == len(disk["noisy"]),
            "problems": problems}


class Committer:
    """Moves finished audio into place, THEN appends the metadata line (single writer)."""

    def __init__(self, out: Path):
        self.out = out
        self.meta = open(out / "metadata" / "dataset_metadata.jsonl", "a", encoding="utf-8")
        self.fail = open(out / "metadata" / "failed_attempts.jsonl", "a", encoding="utf-8")
        self.last_sync = time.time()

    def log_failures(self, fails: list[dict]) -> None:
        for f in fails:
            self.fail.write(_dumps(f) + "\n")
        if fails:
            self.fail.flush()

    def commit(self, rec: dict) -> None:
        sid = rec["sample_id"]
        for kind, key in (("clean", "clean_output"), ("noisy", "noisy_output")):
            dst = self.out / rec[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.out / ".tmp" / f"{sid}_{kind}.wav", dst)
        self.meta.write(_dumps(rec) + "\n")
        self.meta.flush()
        if time.time() - self.last_sync > 2.0:  # batched fsync; crash recovery covers anything unsynced
            os.fsync(self.meta.fileno())
            self.last_sync = time.time()

    def close(self):
        self.meta.flush()
        os.fsync(self.meta.fileno())
        self.meta.close()
        self.fail.close()


# ======================================================================================
# Summary
# ======================================================================================
def build_summary(out: Path, cfg: dict, cleans: list[Source], noises: list[Source],
                  unreadable: dict, integrity: dict) -> dict:
    recs = read_records(out / "metadata" / "dataset_metadata.jsonl")
    clean_use, noise_use, cat_use, count_dist, split_ct = Counter(), Counter(), Counter(), Counter(), Counter()
    snr, tsnr, err, speakers, dur = [], [], [], set(), 0.0
    n_clip = n_loop = 0
    for r in recs:
        clean_use[r["clean_source"]] += 1
        split_ct[r["split"]] += 1
        count_dist[str(r["num_noises"])] += 1
        snr.append(r["measured_snr_db"]); tsnr.append(r["target_snr_db"]); err.append(r["snr_error_db"])
        dur += r["duration_seconds"]
        n_clip += bool(r["clipping_prevented"])
        if r["clean_speaker_id"] is not None:
            speakers.add(r["clean_speaker_id"])
        for nz in r["noises"]:
            noise_use[nz["source"]] += 1
            cat_use[nz["noise_category"] if nz["noise_category"] is not None else "null"] += 1
            n_loop += bool(nz["looped"])
    w = float(cfg["summary_snr_bin_db"])
    dist = Counter()
    for v in snr:
        lo = math.floor(v / w) * w
        dist[f"{lo:g}_to_{lo + w:g}"] += 1
    dist = dict(sorted(dist.items(), key=lambda kv: float(kv[0].split("_to_")[0])))
    cats = {s.meta["noise_category"] for s in noises if s.meta["noise_category"] is not None}
    a = np.array(snr) if snr else np.array([np.nan])
    t = np.array(tsnr) if tsnr else np.array([np.nan])
    e = np.array(err) if err else np.array([np.nan])
    f = lambda v: None if not snr else round(float(v), 4)  # noqa: E731
    return {
        "generator_version": __version__,
        "total_samples": len(recs),
        "samples_per_split": dict(split_ct),
        "total_duration_hours": round(dur / 3600, 4),
        "sample_rate": SR,
        "channels": CHANNELS,
        "output_subtype": cfg["output_subtype"],
        "number_of_clean_source_files": len(cleans),
        "number_of_clean_source_files_used": len(clean_use),
        "number_of_noise_source_files": len(noises),
        "number_of_noise_source_files_used": len(noise_use),
        "number_of_unique_clean_speakers": len(speakers) if speakers else None,
        "number_of_noise_categories": len(cats),
        "number_of_noise_categories_used": len([c for c in cat_use if c != "null"]),
        "snr_min": f(a.min()), "snr_max": f(a.max()), "snr_mean": f(a.mean()),
        "snr_basis": "measured_snr_db",
        "target_snr_min": f(t.min()), "target_snr_max": f(t.max()), "target_snr_mean": f(t.mean()),
        "snr_error_mean_db": f(e.mean()), "snr_error_max_abs_db": f(np.abs(e).max()),
        "snr_distribution": dist,
        "noise_count_distribution": dict(sorted(count_dist.items())),
        "clean_source_usage": dict(sorted(clean_use.items())),
        "noise_source_usage": dict(sorted(noise_use.items())),
        "noise_category_usage": dict(sorted(cat_use.items())),
        "samples_with_clipping_prevention_gain": n_clip,
        "noise_usages_with_looping": n_loop,
        "unreadable_source_files": unreadable,
        "integrity": integrity,
        "schema_notes": {
            "sample_indices": "source_start/end_sample refer to the standardized 48 kHz mono source signal",
            "destination_start/end_sample": "positions in the generated sample's timeline",
            "measured_snr_db": "measured on decoded output files over snr_measurement.active_regions",
            "output_gain": "single gain applied to both clean and noisy outputs when clipping had to be prevented",
        },
    }


# ======================================================================================
# Reconstruction check (proves the metadata is sufficient)
# ======================================================================================
def reconstruct_check(out: Path, root: Path, sample_id: str) -> int:
    recs = [r for r in read_records(out / "metadata" / "dataset_metadata.jsonl") if r["sample_id"] == sample_id]
    if not recs:
        print(f"no record for sample {sample_id}")
        return 1
    r = recs[0]
    sub = r["output_subtype"]
    clean = get_slice(source_from_path(root, r["clean_source"]), r["clean_start_sample"], r["clean_end_sample"]).astype(np.float64)
    comp = np.zeros(len(clean))
    for nz in r["noises"]:
        src = source_from_path(root, nz["source"])
        seg = np.concatenate([get_slice(src, c["source_start_sample"], c["source_end_sample"])
                              for c in nz["chunks"]]).astype(np.float64)
        comp[nz["start_sample"]:] += seg / nz["normalization_rms"] * 10 ** (nz["relative_gain_db"] / 20)
    g = r["output_gain"]
    rc = encode(clean * g, sub)
    rn = encode((clean + comp * r["composite_noise_scale"]) * g, sub)
    ac = sf.read(str(out / r["clean_output"]), dtype=_READ_DTYPE[sub])[0]
    an = sf.read(str(out / r["noisy_output"]), dtype=_READ_DTYPE[sub])[0]
    dc, dn = int(np.max(np.abs(rc.astype(np.int64) - ac))), int(np.max(np.abs(rn.astype(np.int64) - an)))
    print(f"sample {sample_id}: max |reconstructed - stored| clean={dc}, noisy={dn} (integer LSBs)")
    return 0 if dc <= 1 and dn <= 1 else 1


# ======================================================================================
# Main
# ======================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate a reproducible synthetic noisy-speech dataset.")
    ap.add_argument("--root", default=".", help="directory containing wav/ and noise/")
    ap.add_argument("--output", default=None, help="output directory (default: <root>/output)")
    ap.add_argument("--config", default=None, help="JSON config overriding the defaults")
    ap.add_argument("--num-samples", type=int, default=None, help="shortcut for splits={'train': N}")
    ap.add_argument("--seed", type=int, default=None, help="override base_seed")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (default: auto = min(cpu_count, 8))")
    ap.add_argument("--probe-all", action="store_true", help="validate every source file up-front (slower start)")
    ap.add_argument("--cache-dir", default=None, help="where resampled long sources are cached")
    ap.add_argument("--resume", action="store_true", help="continue an existing output directory")
    ap.add_argument("--overwrite", action="store_true", help="delete previously generated dataset first")
    ap.add_argument("--summary-only", action="store_true", help="only verify + rebuild dataset_summary.json")
    ap.add_argument("--reconstruct", metavar="SAMPLE_ID", help="rebuild a sample from metadata and compare")
    ap.add_argument("--print-default-config", action="store_true")
    args = ap.parse_args(argv)

    if args.print_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        return 0

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        cfg = deep_merge(cfg, json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.num_samples is not None:
        cfg["splits"] = {"train": args.num_samples}
    if args.seed is not None:
        cfg["base_seed"] = args.seed
    if args.probe_all:
        cfg["probe_sources"] = "upfront"
    try:
        validate_config(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    out = Path(args.output).resolve() if args.output else root / "output"
    cache_dir = Path(args.cache_dir or cfg["cache_dir"] or (out / ".cache" / "std48k")).resolve()
    workers = args.workers if args.workers > 0 else min(os.cpu_count() or 1, 8)
    configure_audio(cfg, cache_dir)
    for stale in (list(cache_dir.glob("*.tmp.npy")) + list(cache_dir.glob("*.lock"))) if cache_dir.is_dir() else []:
        try:
            stale.unlink()
        except OSError:
            pass
    if args.reconstruct:
        return reconstruct_check(out, root, args.reconstruct)

    # ---- discovery -------------------------------------------------------------------------
    t_start = time.time()
    log(f"scanning source folders under {root} ...")
    cleans, bad_c = discover_sources(root, cfg, "clean", workers * 4)
    noises, bad_n = discover_sources(root, cfg, "noise", workers * 4)
    unreadable = {"clean": bad_c, "noise": bad_n}
    if not cleans or not noises:
        print("error: no usable clean or noise files found", file=sys.stderr)
        return 2
    keys, groups = group_noises(noises)
    lo, hi = num_noises_range(cfg)
    req = cfg["required_noise_categories"]
    missing = [c for c in req if c not in keys]
    if missing:
        print(f"config error: required noise categories {missing} not found under {cfg['noise_dir']}/. "
              f"Discovered categories: {[k for k in keys if k]}", file=sys.stderr)
        return 2
    n_other = len([k for k in keys if k not in req])
    if req and n_other == 0:
        print("config error: every noise category is 'required', so there is nothing to add alongside them.", file=sys.stderr)
        return 2
    if cfg["category_mode"] == "require_different" and hi - len(req) > n_other:
        print(f"config error: require_different needs {hi - len(req)} non-required categories, found {n_other}", file=sys.stderr)
        return 2
    if hi > len(noises):
        print(f"warning: only {len(noises)} noise files exist; num_noises is capped per sample", file=sys.stderr)
    inv_hash, cfg_hash = inventory_hash(cleans, noises), config_hash(cfg)
    if req:
        log(f"every sample will contain: {req} + {lo - len(req)}-{hi - len(req)} noise(s) from the other {n_other} categories")
    log(f"found {len(cleans)} clean files, {len(noises)} noise files in {len(groups)} categories "
        f"({len(bad_c) + len(bad_n)} unreadable skipped) in {time.time() - t_start:.1f}s")

    # ---- output directory state -------------------------------------------------------------
    meta_dir = out / "metadata"
    if args.overwrite and out.exists():
        for child in list(out.iterdir()):
            if child.is_dir() and (child.name in ("metadata", ".tmp") or (child / "clean").is_dir() or (child / "noisy").is_dir()):
                shutil.rmtree(child)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if (meta_dir / "dataset_metadata.jsonl").exists() and (meta_dir / "dataset_metadata.jsonl").stat().st_size > 0 \
            and not (args.resume or args.summary_only):
        print("error: output already contains a dataset. Use --resume to continue or --overwrite to restart.", file=sys.stderr)
        return 2
    rc_path = meta_dir / "run_config.json"
    if rc_path.exists() and (args.resume or args.summary_only):
        prev = json.loads(rc_path.read_text())
        if prev.get("generator_version") != __version__:
            print(f"error: existing output was made by generator {prev.get('generator_version')} (this is {__version__}); "
                  f"use --overwrite to regenerate.", file=sys.stderr)
            return 2
        if prev["config_hash"] != cfg_hash or prev["source_inventory_hash"] != inv_hash:
            print("error: config or source files differ from the original run; resuming would break reproducibility.",
                  file=sys.stderr)
            return 2

    done = recover_state(out)
    plan, k = [], 0
    for split, cnt in cfg["splits"].items():
        for _ in range(int(cnt)):
            k += 1
            plan.append((k, split))
    for r in done.values():
        idx = int(r["sample_id"])
        if idx > len(plan) or plan[idx - 1][1] != r["split"]:
            print(f"error: existing sample {r['sample_id']} does not fit the configured splits", file=sys.stderr)
            return 2
    todo = [(k, s) for k, s in plan if f"{k:06d}" not in done]

    _write_json_atomic(rc_path, {"generator_version": __version__, "config": cfg, "config_hash": cfg_hash,
                                 "source_inventory_hash": inv_hash, "root": str(root),
                                 "number_of_clean_source_files": len(cleans), "number_of_noise_source_files": len(noises)})
    _write_json_atomic(meta_dir / "source_inventory.json", {
        "clean": [{"path": s.rel, "size_bytes": s.size, **s.meta} for s in cleans],
        "noise": [{"path": s.rel, "size_bytes": s.size, **s.meta} for s in noises],
        "unreadable": unreadable})

    # ---- generation ----------------------------------------------------------------------------
    status = 0
    if not args.summary_only and todo:
        committer = Committer(out)
        init_args = (cfg, cleans, noises, str(out / ".tmp"), str(cache_dir))
        total, n_done, t0 = len(todo), 0, time.time()
        log(f"generating {total} samples with {workers} worker process(es) ...")
        ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=init_args)
        pending, it, tick = deque(), iter(todo), Ticker(3.0)

        def fill():
            while len(pending) < workers * 3:
                task = next(it, None)
                if task is None:
                    break
                pending.append((task, ex.submit(_run_sample, task)))

        try:
            fill()
            while pending:
                task, fut = pending[0]
                try:
                    res = fut.result(timeout=2.0)
                except FutureTimeout:  # heartbeat while a worker is busy (e.g. first resample of a long file)
                    if tick.due():
                        log(f"  ... {n_done}/{total} done, still working on sample {task[0]:06d} "
                            f"({time.time() - t0:.0f}s elapsed)")
                    continue
                pending.popleft()
                committer.log_failures(res["failures"])
                if not res["ok"]:
                    last = res["failures"][-1] if res["failures"] else {}
                    log(f"error: could not generate a valid sample after {cfg['max_attempts_per_sample']} attempts "
                        f"(last: {last.get('error')}). See metadata/failed_attempts.jsonl.")
                    status = 1
                    break
                committer.commit(res["record"])
                n_done += 1
                if n_done == 1:
                    log(f"  first sample done: {res['record']['duration_seconds']:.1f}s of audio in {res['seconds']:.1f}s")
                if tick.due() or n_done == total:
                    rate = n_done / (time.time() - t0)
                    log(f"  {n_done}/{total} samples ({rate:.1f}/s, ETA {(total - n_done) / rate / 60:.1f} min)")
                fill()
        except BrokenProcessPool:
            log("error: a worker process died unexpectedly (most likely out of memory). Retry with fewer --workers "
                "and/or set clean_crop.max_duration_seconds.")
            status = 1
        except KeyboardInterrupt:
            log("interrupted; already-committed samples are kept. Re-run with --resume to continue.")
            status = 130
        finally:
            committer.close()
            procs = list((getattr(ex, "_processes", None) or {}).values())  # grab before shutdown clears them
            ex.shutdown(wait=(status == 0), cancel_futures=True)
            if status != 0:
                for proc in procs:  # do not wait for in-flight samples after an error/interrupt
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
            shutil.rmtree(out / ".tmp", ignore_errors=True)
        n_fail = sum(1 for _ in open(meta_dir / "failed_attempts.jsonl", encoding="utf-8")) \
            if (meta_dir / "failed_attempts.jsonl").exists() else 0
        if n_fail:
            log(f"note: {n_fail} failed attempts were logged in metadata/failed_attempts.jsonl (each was retried)")
        if status == 130:
            return status

    # ---- verification + summary ------------------------------------------------------------------------
    integrity = verify_output(out)
    _write_json_atomic(meta_dir / "dataset_summary.json", build_summary(out, cfg, cleans, noises, unreadable, integrity))
    log(f"records={integrity['metadata_records']} clean={integrity['clean_files']} noisy={integrity['noisy_files']} "
        f"one_to_one={integrity['one_to_one']}")
    for p in integrity["problems"]:
        log(f"  PROBLEM: {p}")
    return status or (0 if integrity["one_to_one"] else 1)


if __name__ == "__main__":
    sys.exit(main())