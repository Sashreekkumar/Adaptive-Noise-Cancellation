"""
align_utils.py

Shared delay-estimation code for postprocess_enhanced.py and
evaluate_metrics.py.

Why this exists
----------------
Both scripts used to have their own separate cross-correlation delay
estimator (postprocess_enhanced.py used scipy's FFT correlate,
evaluate_metrics.py used plain np.correlate). Plain cross-correlation
on voiced speech is prone to locking onto a spurious peak spaced at a
multiple of the pitch period, rather than the true delay -- especially
over long, mostly-periodic segments. That's almost certainly why
turning on evaluate_metrics.py's --align on top of already
delay-aligned, postprocessed audio produced a different (worse)
result than the alignment postprocess_enhanced.py had already done:
the two independent implementations could pick different lags.

estimate_delay() here uses GCC-PHAT (Generalized Cross-Correlation
with Phase Transform): it whitens the cross-power spectrum before
correlating, which removes the magnitude information that causes
periodic false peaks and keeps only phase (timing) information. This
is the standard, well-established fix for robust time-delay
estimation on periodic/quasi-periodic signals such as voiced speech.

Both scripts now import this single implementation, so postprocessing
alignment and evaluation alignment can no longer disagree with each
other.

Sign convention (unchanged from the original scripts):
    lag = estimate_delay(ref, deg)
    if lag > 0:  deg is trimmed from the front by `lag` samples
    if lag < 0:  deg is zero-padded at the front by `-lag` samples
    (see apply_delay() below)
"""

import numpy as np


def estimate_delay(ref: np.ndarray, deg: np.ndarray, max_shift: int) -> int:
    """GCC-PHAT delay estimate, in samples, of deg relative to ref.

    max_shift: largest lag (in samples) considered in either direction.
    """
    n = min(len(ref), len(deg))
    ref_seg = (ref[:n] - ref[:n].mean()).astype(np.float64)
    deg_seg = (deg[:n] - deg[:n].mean()).astype(np.float64)

    # FFT size: at least 2n-1 (linear, not circular, cross-correlation),
    # rounded up to a power of two for speed.
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2

    REF = np.fft.rfft(ref_seg, n=n_fft)
    DEG = np.fft.rfft(deg_seg, n=n_fft)

    R = DEG * np.conj(REF)
    # PHAT whitening: keep only phase (timing) information, discard
    # magnitude. This is what makes the estimate robust to periodic
    # speech content that would otherwise create spurious correlation
    # peaks at pitch-period offsets.
    R /= (np.abs(R) + 1e-12)

    cc = np.fft.irfft(R, n=n_fft)

    max_shift = int(min(max_shift, n_fft // 2))
    # Reorder so index 0 corresponds to lag = -max_shift ... last index
    # corresponds to lag = +max_shift, with zero-lag in the middle.
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    lag = int(np.argmax(cc)) - max_shift
    return lag


def apply_delay(deg: np.ndarray, lag: int) -> np.ndarray:
    """Apply an estimated lag to deg (same convention as the original
    scripts): shift deg so it lines up with ref."""
    if lag > 0:
        return deg[lag:]
    elif lag < 0:
        return np.concatenate([np.zeros(-lag, dtype=deg.dtype), deg])
    return deg


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    """Scale-Invariant SDR (Le Roux et al., 2019), in dB.

    This is the metric that gain-matching + delay-alignment against a
    reference is *actually* computing, formally. Reporting it alongside
    plain time-domain SNR makes explicit that the "SNR" numbers in this
    pipeline reflect scale/delay-invariant reconstruction quality
    against a known reference, not a deployment-realistic SNR (where
    you won't have the clean signal to fit against).
    """
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64)
    est = est[:n].astype(np.float64)

    ref_energy = np.sum(ref ** 2) + eps
    proj = (np.sum(est * ref) / ref_energy) * ref
    noise = est - proj

    num = np.sum(proj ** 2) + eps
    den = np.sum(noise ** 2) + eps
    return 10 * np.log10(num / den)