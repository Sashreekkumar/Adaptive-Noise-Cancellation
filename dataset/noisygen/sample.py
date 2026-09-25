"""Builds one (clean, noisy) pair plus its provenance record."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import detect_active_regions, encode, get_slice, mask_from_regions, read_decoded, std_length
from .config import draw_dist, num_noises_range
from .constants import CHANNELS, CLEAN_FIELDS, EPS, SR, __version__
from .errors import SampleRejected
from .selection import derive_seed, plan_noise_chunks, select_noises
from .sources import Source, ensure_probed
from .util import log


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
