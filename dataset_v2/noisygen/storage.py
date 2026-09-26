"""Output-directory state: JSONL I/O, crash recovery, verification, and the single-writer Committer."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from .util import log


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json_atomic(path: Path, obj) -> None:
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
                print("[recover] truncated torn final metadata line", file=sys.stderr)
                break
            raise RuntimeError(f"corrupt metadata at line {i + 1} of {meta_path}")
        offset += len(raw)
    return recs


AUDIO_EXTS = ("*.wav", "*.mp3")  # output_format can be wav or mp3; a dataset only ever contains one, but
                                 # both are recognised so verification/recovery work regardless of mode.


def audio_files_on_disk(out: Path) -> dict[str, set[str]]:
    res = {"clean": set(), "noisy": set()}
    for d in out.iterdir() if out.exists() else []:
        if d.is_dir() and d.name not in ("metadata", ".tmp"):
            for kind in res:
                kd = d / kind
                if kd.is_dir():
                    for ext in AUDIO_EXTS:
                        res[kind].update(f"{d.name}/{kind}/{p.name}" for p in kd.glob(ext))
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
            os.replace(self.out / ".tmp" / f"{sid}_{kind}{dst.suffix}", dst)
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


class CleanSourceReaper:
    """Deletes each clean speech source once every sample in this run's plan that was predicted to use it
    (on its first attempt) has been resolved - committed or permanently failed - freeing disk space as
    generation goes rather than at the very end.

    A retry can occasionally pick a *different* clean source than the one predicted for that sample id;
    when that happens the predicted source's reservation is still released (that sample no longer needs
    it), but the source actually used is never counted as a reservation it wasn't given, so it is never
    deleted early - at worst it is deleted a little later than strictly necessary. The rare opposite case
    - a later retry needing a source already deleted because every sample that was *supposed* to need it
    is done - simply fails that read and is retried like any other rejection reason; it never corrupts
    output or crashes the run."""

    def __init__(self, root: Path, predicted_source_by_id: dict[str, str]):
        self.root = root
        self.predicted = predicted_source_by_id
        self.remaining = Counter(predicted_source_by_id.values())
        self._deleted: set[str] = set()

    def release(self, sample_id: str) -> None:
        rel = self.predicted.get(sample_id)
        if rel is None:
            return
        self.remaining[rel] -= 1
        if self.remaining[rel] <= 0:
            self._delete(rel)

    def _delete(self, rel: str) -> None:
        if rel in self._deleted:
            return
        self._deleted.add(rel)
        try:
            (self.root / rel).unlink()
            log(f"  [cleanup] deleted clean source (fully used): {rel}")
        except FileNotFoundError:
            pass
        except OSError as e:
            log(f"  warning: could not delete clean source {rel}: {e}")
