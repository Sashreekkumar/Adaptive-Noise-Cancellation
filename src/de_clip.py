import numpy as np
from scipy.interpolate import CubicSpline
from numpy.typing import NDArray

def _runs(mask: NDArray[np.bool_], min_run: int) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(s, e) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if e - s >= min_run]

def declip(x: NDArray[np.float32], trigger_frac: float = 5e-4, min_run: int = 3, max_run: int = 40, ctx: int = 8, overshoot: float = 1.3, max_frac: float = 0.05) -> tuple[NDArray[np.float32], int, float]:
    found = []

    for sign in (1.0, -1.0):
        pk = float(np.max(sign * x))
        if pk < 1e-3:
            continue

        runs = _runs(
            sign * x >= pk - max(1e-4 * pk, 1.5 / 32768),
            min_run
        )
        if len(runs) >= 2:
            found += [(s, e, sign, pk) for s, e in runs]

    clipped = sum(e - s for s, e, _, _ in found) / len(x)
    if clipped < trigger_frac or clipped > max_frac:
        return x, 0, clipped

    y, fixed = x.copy(), 0

    for s, e, sign, pk in found:
        if e - s > max_run or s < ctx or e + ctx > len(x):
            continue

        idx = np.r_[np.arange(s - ctx, s), np.arange(e, e + ctx)]
        interp = sign * CubicSpline(idx, x[idx])(np.arange(s, e))
        y[s:e] = sign * np.clip(interp, pk, overshoot * pk)
        fixed += e - s

    return y.astype(np.float32), fixed, clipped