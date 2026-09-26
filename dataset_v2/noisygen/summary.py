"""dataset_summary.json: statistics computed from the JSONL records."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from .constants import CHANNELS, SR, __version__
from .sources import Source
from .storage import read_records


def build_summary(out: Path, cfg: dict, cleans: list[Source], noises: list[Source],
                  unreadable: dict, integrity: dict) -> dict:
    recs = read_records(out / "metadata" / "dataset_metadata.jsonl")
    clean_use, noise_use, cat_use, count_dist, split_ct = Counter(), Counter(), Counter(), Counter(), Counter()
    snr, tsnr, err, speakers, dur = [], [], [], set(), 0.0
    n_clip = n_loop = n_impulsive_events = 0
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
            n_loop += bool(nz.get("looped", False))
            n_impulsive_events += nz.get("n_events", 0)
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
        "output_container": cfg["output_format"],
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
        "impulsive_noise_events_total": n_impulsive_events,
        "unreadable_source_files": unreadable,
        "integrity": integrity,
        "schema_notes": {
            "sample_indices": "source_start/end_sample refer to the standardized 48 kHz mono source signal",
            "destination_start/end_sample": "positions in the generated sample's timeline",
            "measured_snr_db": "measured on decoded output files over snr_measurement.active_regions",
            "output_gain": "single gain applied to both clean and noisy outputs when clipping had to be prevented",
        },
    }
