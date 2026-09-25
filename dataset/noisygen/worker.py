"""Process-pool worker entry points (must live at module level so they are picklable by reference)."""
from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Optional

from .audio import configure_audio
from .sample import Context, make_sample
from .selection import derive_seed
from .sources import group_noises

_CTX: Optional[Context] = None


def init_worker(cfg, cleans, noises, tmp_dir, cache_dir):
    global _CTX
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the parent only
    except (ValueError, OSError):
        pass
    configure_audio(cfg, Path(cache_dir))
    _CTX = Context(cfg, cleans, noises, *group_noises(noises), Path(tmp_dir))


def run_sample(task: tuple[int, str]) -> dict:
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
