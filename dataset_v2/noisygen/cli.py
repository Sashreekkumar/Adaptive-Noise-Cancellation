"""Command-line interface: argument parsing, config assembly, discovery, orchestration."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .audio import configure_audio
from .config import (DEFAULT_CONFIG, config_hash, deep_merge, num_noises_range, validate_config)
from .constants import __version__
from .errors import ConfigError
from .reconstruct import reconstruct_check
from .runner import run_generation
from .selection import predict_clean_source
from .sources import discover_sources, group_noises, inventory_hash, load_persisted_sources
from .storage import CleanSourceReaper, recover_state, verify_output, write_json_atomic
from .summary import build_summary
from .util import log


def build_parser() -> argparse.ArgumentParser:
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
    return ap


def build_config(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        cfg = deep_merge(cfg, json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.num_samples is not None:
        cfg["splits"] = {"train": args.num_samples}
    if args.seed is not None:
        cfg["base_seed"] = args.seed
    if args.probe_all:
        cfg["probe_sources"] = "upfront"
    if cfg["output_format"] == "mp3" and cfg["output_subtype"] != "PCM_16":
        log(f"note: output_format=mp3 needs PCM_16 as the intermediate/verification precision; "
            f"overriding output_subtype ({cfg['output_subtype']} -> PCM_16)")
        cfg["output_subtype"] = "PCM_16"
    return cfg


def check_noise_setup(cfg: dict, keys: list, noises: list) -> bool:
    """Print an error and return False if the noise library cannot satisfy the config."""
    lo, hi = num_noises_range(cfg)
    req = cfg["required_noise_categories"]
    missing = [c for c in req if c not in keys]
    if missing:
        print(f"config error: required noise categories {missing} not found under {cfg['noise_dir']}/. "
              f"Discovered categories: {[k for k in keys if k]}", file=sys.stderr)
        return False
    n_other = len([k for k in keys if k not in req])
    if req and n_other == 0:
        print("config error: every noise category is 'required', so there is nothing to add alongside them.", file=sys.stderr)
        return False
    if cfg["category_mode"] == "require_different" and hi - len(req) > n_other:
        print(f"config error: require_different needs {hi - len(req)} non-required categories, found {n_other}", file=sys.stderr)
        return False
    if hi > len(noises):
        print(f"warning: only {len(noises)} noise files exist; num_noises is capped per sample", file=sys.stderr)
    unmatched_impulsive = [c for c in cfg["impulsive_noise_categories"] if c not in keys]
    if unmatched_impulsive:
        print(f"warning: impulsive_noise_categories {unmatched_impulsive} not found under {cfg['noise_dir']}/ "
              f"(harmless if intentional; they just never get impulsive treatment). Discovered categories: "
              f"{[k for k in keys if k]}", file=sys.stderr)
    return True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.print_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        return 0

    cfg = build_config(args)
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
    inv_path = out / "metadata" / "source_inventory.json"
    # delete_clean_source_after_use can remove clean files as it goes. On --resume/--summary-only, rescanning
    # wav/ would then see a smaller (differently-indexed) list, which would silently change which clean file
    # every seed points to. So in that situation the exact clean list from the first invocation is replayed
    # from source_inventory.json instead of rescanned; a deleted file just isn't re-probed (see load_persisted_sources).
    reuse_persisted_cleans = cfg["delete_clean_source_after_use"] and (args.resume or args.summary_only) and inv_path.exists()
    if reuse_persisted_cleans:
        cleans = load_persisted_sources(inv_path, root, "clean")
        bad_c = json.loads(inv_path.read_text(encoding="utf-8")).get("unreadable", {}).get("clean", [])
        log(f"  clean: reusing {len(cleans)} source(s) recorded by the first run (delete_clean_source_after_use is on)")
    else:
        cleans, bad_c = discover_sources(root, cfg, "clean", workers * 4)
    noises, bad_n = discover_sources(root, cfg, "noise", workers * 4)
    unreadable = {"clean": bad_c, "noise": bad_n}
    if not cleans or not noises:
        print("error: no usable clean or noise files found", file=sys.stderr)
        return 2
    keys, groups = group_noises(noises)
    lo, hi = num_noises_range(cfg)
    req = cfg["required_noise_categories"]
    if not check_noise_setup(cfg, keys, noises):
        return 2
    n_other = len([k for k in keys if k not in req])
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

    reaper = None
    if cfg["delete_clean_source_after_use"] and todo:
        predicted = {f"{k:06d}": predict_clean_source(cfg["base_seed"], k, cleans) for k, _ in todo}
        reaper = CleanSourceReaper(root, predicted)

    write_json_atomic(rc_path, {"generator_version": __version__, "config": cfg, "config_hash": cfg_hash,
                                "source_inventory_hash": inv_hash, "root": str(root),
                                "number_of_clean_source_files": len(cleans), "number_of_noise_source_files": len(noises)})
    write_json_atomic(meta_dir / "source_inventory.json", {
        "clean": [{"path": s.rel, "size_bytes": s.size, **s.meta} for s in cleans],
        "noise": [{"path": s.rel, "size_bytes": s.size, **s.meta} for s in noises],
        "unreadable": unreadable})

    # ---- generation ----------------------------------------------------------------------------
    status = 0
    if not args.summary_only and todo:
        status = run_generation(todo, cfg, cleans, noises, out, cache_dir, workers, reaper=reaper)
        if status == 130:
            return status

    # ---- verification + summary ------------------------------------------------------------------------
    integrity = verify_output(out)
    write_json_atomic(meta_dir / "dataset_summary.json", build_summary(out, cfg, cleans, noises, unreadable, integrity))
    log(f"records={integrity['metadata_records']} clean={integrity['clean_files']} noisy={integrity['noisy_files']} "
        f"one_to_one={integrity['one_to_one']}")
    for p in integrity["problems"]:
        log(f"  PROBLEM: {p}")
    return status or (0 if integrity["one_to_one"] else 1)
