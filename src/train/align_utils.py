"""
align_utils.py

Shared delay-estimation code for postprocess_enhanced.py and
evaluate_metrics.py.
"""

import numpy as np


def estimate_delay(ref: np.ndarray, deg: np.ndarray, max_shift: int) -> int:
    n = min(len(ref), len(deg))
    ref_seg = (ref[:n] - ref[:n].mean()).astype(np.float64)
    deg_seg = (deg[:n] - deg[:n].mean()).astype(np.float64)

    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2

    REF = np.fft.rfft(ref_seg, n=n_fft)
    DEG = np.fft.rfft(deg_seg, n=n_fft)

    R = DEG * np.conj(REF)
    R /= (np.abs(R) + 1e-12)

    cc = np.fft.irfft(R, n=n_fft)

    max_shift = int(min(max_shift, n_fft // 2))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    lag = int(np.argmax(cc)) - max_shift
    return lag


def apply_delay(deg: np.ndarray, lag: int) -> np.ndarray:
    if lag > 0:
        return deg[lag:]
    elif lag < 0:
        return np.concatenate([np.zeros(-lag, dtype=deg.dtype), deg])
    return deg


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64)
    est = est[:n].astype(np.float64)

    ref_energy = np.sum(ref ** 2) + eps
    proj = (np.sum(est * ref) / ref_energy) * ref
    noise = est - proj

    num = np.sum(proj ** 2) + eps
    den = np.sum(noise ** 2) + eps
    return 10 * np.log10(num / den)