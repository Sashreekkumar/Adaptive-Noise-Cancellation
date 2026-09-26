"""Default configuration, validation, distribution sampling and config hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from .errors import ConfigError

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
    "noise_placement": {"mode": "full", "max_start_fraction": 0.5},  # full | random_start; applies only to
                                                                      # noises NOT in impulsive_noise_categories
    # relative gain of every noise after the first (first noise is the 0 dB reference)
    "relative_gain_db": {"type": "uniform", "min": -10.0, "max": 0.0},
    "min_noise_rms": 1e-6,

    # Noise categories placed as several short scattered events instead of one continuous segment running
    # through the whole sample - for impulsive sources like gunfire, artillery or explosions. A pick from
    # one of these categories ignores noise_placement/short_noise_policy entirely: instead, impulsive_events
    # is used to scatter short excerpts of it at random (possibly overlapping) times across the sample.
    "impulsive_noise_categories": [],
    "impulsive_events": {
        "count": [1, 4],  # events per impulsive noise pick; int, or [min, max] inclusive
        "event_duration_seconds": [0.15, 1.5],  # crop length drawn per event; clamped to the source's length
        "fade_ms": 5.0,  # linear fade in/out at each event's edges, so splicing it in doesn't click
    },

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

    # ---- output container ---------------------------------------------------------------
    "output_format": "wav",  # wav | mp3. mp3 is lossy: see README for the verification/reconstruction caveats.
    # libsndfile/LAME quality knob for mp3 output: 0.0 = highest quality/largest files .. <1.0 = smaller/lower
    # quality (VBR, so kbps is content-dependent). Only used when output_format == "mp3".
    "mp3_compression_level": 0.3,

    # ---- disk space management ------------------------------------------------------------
    # If true, each clean speech source file under clean_dir/ is deleted once every sample in THIS run
    # that was going to use it has been generated (success or permanent failure). Safe to use with --resume
    # (each invocation only tracks the sources its own remaining work needs). Two things to know:
    #   * --reconstruct cannot rebuild a sample whose clean source has since been deleted.
    #   * a clean source may occasionally be read once more after "its" samples are done, if a retry for a
    #     different sample happens to pick it; that read simply fails and is retried like any other rejection.
    "delete_clean_source_after_use": False,

    # ---- performance ---------------------------------------------------------------------
    "probe_sources": "lazy",  # lazy: open a file only when first used | upfront: validate every file at start
    "standardized_cache": True,  # cache resampled long sources on disk (float32 .npy), memory-mapped on use
    "cache_min_seconds": 20.0,  # only non-48 kHz sources at least this long are disk-cached
    "cache_dir": None,  # default <output>/.cache/std48k
    "cache_size": 16,  # in-memory LRU entries for shorter non-48 kHz sources
}
# keys that do not influence the generated audio/metadata; excluded from the resume-compatibility hash
NON_SEMANTIC_KEYS = {"splits", "summary_snr_bin_db", "standardized_cache", "cache_min_seconds", "cache_dir",
                     "cache_size", "delete_clean_source_after_use"}


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


def impulsive_count_range(cfg: dict) -> tuple[int, int]:
    cnt = cfg["impulsive_events"]["count"]
    lo, hi = (cnt, cnt) if isinstance(cnt, int) else (int(cnt[0]), int(cnt[1]))
    return lo, hi


def impulsive_duration_range_seconds(cfg: dict) -> tuple[float, float]:
    dur = cfg["impulsive_events"]["event_duration_seconds"]
    lo, hi = (dur, dur) if isinstance(dur, (int, float)) else (float(dur[0]), float(dur[1]))
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
    if c["output_format"] not in ("wav", "mp3"):
        raise ConfigError("output_format must be wav | mp3")
    if not (0.0 <= float(c["mp3_compression_level"]) < 1.0):
        raise ConfigError("mp3_compression_level must be in [0.0, 1.0)")
    if not isinstance(c["delete_clean_source_after_use"], bool):
        raise ConfigError("delete_clean_source_after_use must be a boolean")
    imp_cats = c["impulsive_noise_categories"]
    if not isinstance(imp_cats, list) or not all(isinstance(x, str) for x in imp_cats) or len(set(imp_cats)) != len(imp_cats):
        raise ConfigError("impulsive_noise_categories must be a list of distinct category names")
    lo_c, hi_c = impulsive_count_range(c)
    if lo_c < 1 or hi_c < lo_c:
        raise ConfigError("impulsive_events.count must be an int >= 1 or [min, max] with 1 <= min <= max")
    lo_d, hi_d = impulsive_duration_range_seconds(c)
    if lo_d <= 0 or hi_d < lo_d:
        raise ConfigError("impulsive_events.event_duration_seconds must be a number > 0, or [min, max] with 0 < min <= max")
    if float(c["impulsive_events"]["fade_ms"]) < 0:
        raise ConfigError("impulsive_events.fade_ms must be >= 0")


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
