"""Multi-process generation loop: submit tasks, collect results in order, commit, report progress."""
from __future__ import annotations

import shutil
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .sources import Source
from .storage import Committer
from .util import Ticker, log
from .worker import init_worker, run_sample


def run_generation(todo: list[tuple[int, str]], cfg: dict, cleans: list[Source], noises: list[Source],
                   out: Path, cache_dir: Path, workers: int) -> int:
    """Generate every task in `todo`. Returns a process status: 0 ok, 1 error, 130 interrupted."""
    meta_dir = out / "metadata"
    status = 0
    committer = Committer(out)
    init_args = (cfg, cleans, noises, str(out / ".tmp"), str(cache_dir))
    total, n_done, t0 = len(todo), 0, time.time()
    log(f"generating {total} samples with {workers} worker process(es) ...")
    ex = ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=init_args)
    pending, it, tick = deque(), iter(todo), Ticker(3.0)

    def fill():
        while len(pending) < workers * 3:
            task = next(it, None)
            if task is None:
                break
            pending.append((task, ex.submit(run_sample, task)))

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
    return status
