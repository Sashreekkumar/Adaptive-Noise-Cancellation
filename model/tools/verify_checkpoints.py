"""Verify the detached model checkpoints against model/checkpoints_manifest.json.
Run from model/:  python tools/verify_checkpoints.py            (check)
                  python tools/verify_checkpoints.py --write    (regenerate the manifest from model/checkpoints/)
The checkpoints are not stored in git; copy them into model/checkpoints/ (layout in the manifest)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST, CKPT = ROOT / "checkpoints_manifest.json", ROOT / "checkpoints"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    if a.write:
        old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
        files = {f.relative_to(CKPT).as_posix(): {"bytes": f.stat().st_size, "sha256": sha256(f)}
                 for f in sorted(CKPT.rglob("*")) if f.is_file()}
        MANIFEST.write_text(json.dumps({"provenance": old.get("provenance", {}), "files": files}, indent=2), encoding="utf-8")
        print(f"wrote {MANIFEST} ({len(files)} files)")
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for rel, info in manifest["files"].items():
        f = CKPT / rel
        status = "MISSING" if not f.exists() else ("OK" if sha256(f) == info["sha256"] else "HASH MISMATCH")
        bad += status != "OK"
        print(f"{status:14s} {rel}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
