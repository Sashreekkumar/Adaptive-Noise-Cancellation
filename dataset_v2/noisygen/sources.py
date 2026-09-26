"""Source discovery: scanning, lazy probing, provenance attributes, noise grouping."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import soundfile as sf

from .constants import CLEAN_FIELDS, NOISE_FIELDS
from .errors import SampleRejected
from .util import Ticker, log


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


def load_persisted_sources(source_inventory_path: Path, root: Path, kind: str) -> list[Source]:
    """Rebuild the Source list for `kind` ("clean"|"noise") from a previous run's source_inventory.json,
    instead of rescanning disk. Used on --resume/--summary-only when delete_clean_source_after_use is on:
    once some clean sources have been deleted, a fresh scan would see a smaller list and shift every
    remaining source's index (breaking `rng.integers(len(cleans))` reproducibility), so the exact original
    list is replayed from disk instead. Missing files are fine here: they are only probed lazily, and by
    construction a deleted source is never referenced by the samples still left to generate."""
    inv = json.loads(source_inventory_path.read_text(encoding="utf-8"))
    fields = CLEAN_FIELDS if kind == "clean" else NOISE_FIELDS
    out = []
    for e in inv[kind]:
        rel = e["path"]
        meta = {k: e.get(k) for k in fields}
        out.append(Source(rel=rel, abs_path=str(root / rel), size=int(e["size_bytes"]), mtime_ns=0, meta=meta))
    return out


def group_noises(noises: list[Source]) -> tuple[list[Optional[str]], list[list[int]]]:
    """(category keys, file indices per category); unknown/None is one group; deterministic order."""
    groups: dict[Optional[str], list[int]] = {}
    for i, s in enumerate(noises):
        groups.setdefault(s.meta["noise_category"], []).append(i)
    keys = sorted(groups, key=lambda k: (k is None, k or ""))
    return keys, [groups[k] for k in keys]
