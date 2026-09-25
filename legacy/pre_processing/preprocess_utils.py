import numpy as np
from math import gcd
from scipy import signal

EPS = 1e-12


def rms_dbfs(x) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + EPS))


def resample(x, sr, target) -> np.ndarray:
    if sr == target:
        return x

    g = gcd(sr, target)
    return signal.resample_poly(x, target // g, sr // g).astype(np.float32)