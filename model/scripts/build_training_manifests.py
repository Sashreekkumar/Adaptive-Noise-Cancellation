"""Build train/dev_train/val/test manifests for DFN3 fine-tuning from the metadata CSVs and the ZIP audits.

Every candidate row is kept; invalid rows are flagged, never repaired or dropped silently.
Roles:
  train        mixtures_train_fast.csv (Drive; audio verified on Colab with --check-audio)
  dev_train    mixtures_train_fast_test.csv (100 pairs, local; pipeline dry runs ONLY, provenance undocumented)
  eval_frozen  4 transient evaluation pairs + continuous_eval_v1 (never used for training or selection)
  calibration  calibration_transient_manifest (threshold calibration only)
  val_select   per (category, SNR) the lowest-index verified val pair not in eval/calibration (checkpoint selection)
  val_pool     remaining verified val pairs (reserve for later development sets)
  test         verified test pairs (final reporting only)
use_for_training  = role == train and metadata != FAIL and audio in {PASS, WARN} and no clipping flag (meta or audio)
use_for_selection = role == val_select and audio in {PASS, WARN} and no clipping flag
(dev_train rows follow the train rule under their own role; the trainer accepts --role dev_train for dry runs.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sih_train.data import CATEGORY_TIER, AudioSource, relpath_from_drive, validate_pair

FIELDS = ["mixture_id", "split", "role", "category", "tier", "target_snr_db", "measured_snr_db_meta", "duration_sec_meta",
          "n_events", "clipping_flag_meta", "noisy_relpath", "clean_relpath", "meta_status", "meta_flags",
          "audio_status", "audio_flags", "sample_rate", "channels", "frames", "duration_sec", "peak_noisy", "peak_clean",
          "alignment_lag", "measured_snr_db_audio", "use_for_training", "use_for_selection", "provenance"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metadata_checks(row: dict, expected_split: str) -> tuple[str, list[str]]:
    """FAIL: wrong split, category/tier mismatch, SNR out of range. WARN: clipping flag, measured != target SNR.
    Note only (status unaffected): empty measured_snr_db, which is the case for every tier-C transient row."""
    fatal, warns, notes = [], [], []
    if row.get("split") != expected_split:
        fatal.append(f"split={row.get('split')}")
    if CATEGORY_TIER.get(row["category"]) != row["tier"]:
        fatal.append("category/tier mismatch")
    snr = float(row["target_snr_db"])
    if not -5.0 <= snr <= 20.0:
        fatal.append("target SNR outside [-5, 20]")
    if row["clipping_flag"] == "True":
        warns.append("clipping_flag (noisy peak > 1.0 before PCM write)")
    if not row["measured_snr_db"]:
        notes.append("note: measured_snr_db empty")
    elif abs(float(row["measured_snr_db"]) - snr) > 0.5:
        warns.append("measured vs target SNR > 0.5 dB")
    return ("FAIL" if fatal else "WARN" if warns else "PASS"), fatal + warns + notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--validation-dir", type=Path, default=Path(r"D:\SIH26052\validation"))
    parser.add_argument("--root", type=Path, action="append", default=[], help="Dataset root dir(s) containing mixtures/")
    parser.add_argument("--zip", type=Path, action="append", default=[], help="Drive ZIP(s) (members without mixtures/)")
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\SIH26052\manifests"))
    args = parser.parse_args()
    source = AudioSource(args.root, args.zip)
    rows: dict[str, list[dict]] = {"train": [], "dev_train": [], "val": [], "test": []}

    for name, csv_name, split in (("train", "mixtures_train_fast.csv", "train"),
                                  ("dev_train", "mixtures_train_fast_test.csv", "train_fast_test")):
        for meta in read_csv(args.metadata_dir / csv_name):
            status, flags = metadata_checks(meta, split)
            rows[name].append({
                "mixture_id": meta["mixture_id"], "split": meta["split"], "role": name, "category": meta["category"],
                "tier": meta["tier"], "target_snr_db": meta["target_snr_db"], "measured_snr_db_meta": meta["measured_snr_db"],
                "duration_sec_meta": meta["duration_sec"], "n_events": meta["n_events"], "clipping_flag_meta": meta["clipping_flag"],
                "noisy_relpath": relpath_from_drive(meta["noisy_path"]), "clean_relpath": relpath_from_drive(meta["clean_path"]),
                "meta_status": status, "meta_flags": ";".join(flags), "audio_status": "UNVERIFIED",
                "provenance": csv_name + (" (generator code for this split not found)" if name == "dev_train" else
                                          " (Dataset_v2 notebook: train-split speech/noise only)"),
            })

    frozen = {r["mixture_id"] for r in read_csv(args.validation_dir / "alignment_eligible_manifest.csv")}
    frozen |= {r["mixture_id"] for r in read_csv(args.validation_dir / "continuous_eval_v1_manifest.csv")}
    calibration = {r["mixture_id"] for r in read_csv(args.validation_dir / "calibration_transient_manifest.csv")}
    for name, audit_file in (("val", "zip_pair_audit.csv"), ("test", "zip_pair_audit_test.csv")):
        audit = read_csv(args.validation_dir / audit_file)
        for a in audit:
            index = int(re.search(r"(\d+)$", a["mixture_id"]).group(1))
            role = ("invalid" if a["status"] != "PASS" else "test" if name == "test" else
                    "eval_frozen" if a["mixture_id"] in frozen else
                    "calibration" if a["mixture_id"] in calibration else "val_pool")
            rows[name].append({
                "mixture_id": a["mixture_id"], "split": name, "role": role, "category": a["category"],
                "tier": CATEGORY_TIER.get(a["category"], ""), "target_snr_db": a["snr_db"], "_index": index,
                "noisy_relpath": "mixtures/" + a["noisy_member"], "clean_relpath": "mixtures/" + a["clean_member"],
                "meta_status": "PASS" if a["status"] == "PASS" else "FAIL",
                "meta_flags": "" if a["status"] == "PASS" else f"zip audit: {a['reason']}",
                "audio_status": "UNVERIFIED", "provenance": f"{audit_file} (noise-triplet or same-name pairing)",
            })
    def eligible(r: dict) -> bool:
        clipped = r.get("clipping_flag_meta") == "True" or "clipped" in r.get("audio_flags", "")
        return r["meta_status"] != "FAIL" and r["audio_status"] in {"PASS", "WARN"} and not clipped

    for name, group in rows.items():
        for r in group:
            if args.check_audio and r["meta_status"] != "FAIL":
                meta_duration = float(r["duration_sec_meta"]) if r.get("duration_sec_meta") else None
                meta_snr = float(r["measured_snr_db_meta"]) if r.get("measured_snr_db_meta") else None
                r.update(validate_pair(source, r["noisy_relpath"], r["clean_relpath"], meta_duration, meta_snr))
        # Duplicate ids among non-FAIL rows are fatal (stale audit FAIL twins are already role=invalid).
        ids = Counter(r["mixture_id"] for r in group if r["meta_status"] != "FAIL")
        for r in group:
            if r["meta_status"] != "FAIL" and ids[r["mixture_id"]] > 1:
                r["meta_status"], r["meta_flags"] = "FAIL", (r["meta_flags"] + ";duplicate mixture_id").strip(";")
        if name == "val":  # selection AFTER validation: lowest index per (category, SNR) passing every check
            cells: dict[tuple, dict] = {}
            for r in group:
                if r["role"] == "val_pool" and eligible(r):
                    key = (r["category"], r["target_snr_db"])
                    if key not in cells or r["_index"] < cells[key]["_index"]:
                        cells[key] = r
            for r in cells.values():
                r["role"] = "val_select"
        for r in group:
            r.pop("_index", None)
            r["use_for_training"] = r["role"] in {"train", "dev_train"} and eligible(r)
            r["use_for_selection"] = r["role"] == "val_select" and eligible(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"sources": {"roots": [str(p) for p in args.root], "zips": [str(p) for p in args.zip]},
               "check_audio": args.check_audio, "rules": __doc__.strip(), "files": {}}
    for name, group in rows.items():
        path = args.output_dir / f"{name}_manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore"); writer.writeheader(); writer.writerows(group)
        summary["files"][name] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(group),
            "roles": dict(Counter(r["role"] for r in group)), "meta_status": dict(Counter(r["meta_status"] for r in group)),
            "audio_status": dict(Counter(r["audio_status"] for r in group)),
            "use_for_training": sum(bool(r["use_for_training"]) for r in group),
            "use_for_selection": sum(bool(r["use_for_selection"]) for r in group),
            "audio_flags": dict(Counter(f for r in group for f in r.get("audio_flags", "").split(";") if f)),
            "meta_flags": dict(Counter(f for r in group for f in r["meta_flags"].split(";") if f)),
        }
    (args.output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("rows", "roles", "audio_status", "use_for_training", "use_for_selection", "audio_flags", "meta_flags")}
                      for k, v in summary["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
