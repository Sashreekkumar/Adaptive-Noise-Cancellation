"""Residual canceller unit tests (synthetic, DFN3 STFT). Run: dfn-env python tests/test_residual_canceller.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from libdf import DF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sih_model.residual_canceller import ResidualCanceller  # noqa: E402

SR = 48000
STFT = DF(sr=SR, fft_size=960, hop_size=480, nb_bands=32, min_nb_erb_freqs=2)
rng = np.random.default_rng(0)


def spec(x: np.ndarray) -> np.ndarray:
    return STFT.analysis(np.asarray(x, dtype=np.float32)[None])[0]          # [T, 481] complex64


def path(x: np.ndarray) -> np.ndarray:
    h = np.zeros(20); h[10] = 0.6; h[12] = -0.25; h[15] = 0.1               # delay + short coloured path
    return np.convolve(x, h)[: len(x)]


def speech_like(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)                      # 3 Hz on/off syllables
    return 0.3 * env * sum(np.sin(2 * np.pi * f0 * k * t) / k for f0 in (180,) for k in range(1, 12))


def run(rc, S, Y, XR, mask):
    out = []
    for t in range(len(Y)):
        z, _ = rc.process_frame(S[t], Y[t], XR[t], mask[t])
        out.append(z)
    return np.array(out)


def main() -> None:
    n = SR * 6
    u = rng.normal(scale=0.1, size=n)
    Xr, V = spec(u), spec(path(u))

    # 1. W_hat = 0 -> exact input (mask high -> no adaptation)
    rc = ResidualCanceller()
    Z = run(rc, V, V, Xr, np.ones((len(V), 481)))
    assert np.array_equal(Z, V) and not rc.w.any()
    print("PASS W_hat=0 -> output == input exactly")

    # 2. known synthetic filter converges (noise-only primary, mask 0 -> adapt everywhere)
    rc = ResidualCanceller()
    run(rc, V, V, Xr, np.zeros((len(V), 481)))
    H = np.fft.rfft(np.r_[np.zeros(10), [0.6, 0, -0.25, 0, 0, 0.1]], 960)
    rel = np.median(np.abs(rc.w[5:470] - H[5:470]) / np.abs(H[5:470]))
    assert rel < 0.05, rel
    print(f"PASS path estimate converges: median |W_hat - H| / |H| = {rel:.4f}")

    # 3. coherent noise reduction > 20 dB (noise-only, S_hat = Y, after 2 s convergence)
    rc = ResidualCanceller()
    Z = run(rc, V, V, Xr, np.zeros((len(V), 481)))
    k = 200
    nr = 10 * np.log10(np.sum(np.abs(V[k:]) ** 2) / np.sum(np.abs(Z[k:]) ** 2))
    assert nr > 20, nr
    print(f"PASS coherent noise reduction {nr:.1f} dB (> 20)")

    # 4. speech energy loss: primary = speech + noise, reference = noise + speech leakage;
    #    gate mask = oracle speech presence (adapt only where noise dominates).
    #    Asserted at the project's simulated-reference leakage (0.02 = -34 dB, reference_sim / NLMS configs);
    #    -20 dB leakage is reported for information (subtraction then removes up to ~|W| x 0.1 of the speech).
    s = speech_like(n)
    S = spec(s)
    mask = (np.abs(S) ** 2 / (np.abs(S) ** 2 + np.abs(V) ** 2 + 1e-12)).astype(np.float32)
    active = np.sum(np.abs(S) ** 2, axis=1) > 1e-3 * np.max(np.sum(np.abs(S) ** 2, axis=1))
    active[:k] = False
    for leak_gain, assert_it in ((0.02, True), (0.1, False)):
        leak = spec(leak_gain * s)
        Y, XR = S + V, Xr + leak
        rc = ResidualCanceller()
        speech_out, noise_out = [], []
        for t in range(len(Y)):
            z, info = rc.process_frame(Y[t], Y[t], XR[t], mask[t])
            wg = info["w_used"] * info["g"] * info["applied"]
            speech_out.append(S[t] - wg * leak[t])                           # speech component of Z
            noise_out.append(z - speech_out[-1])
        speech_out = np.array(speech_out)
        loss = 10 * np.log10(np.sum(np.abs(S[active]) ** 2) / np.sum(np.abs(speech_out[active]) ** 2))
        nr_mix = 10 * np.log10(np.sum(np.abs(V[k:]) ** 2) / np.sum(np.abs(np.array(noise_out)[k:]) ** 2))
        tag = f"leakage {20 * np.log10(leak_gain):.0f} dB: speech energy change {loss:+.3f} dB, noise reduction with speech {nr_mix:.1f} dB"
        if assert_it:
            assert abs(loss) < 0.5, loss
            print(f"PASS {tag} (|loss| < 0.5); guards {rc.counters}")
        else:
            print(f"INFO {tag}")

    # 5. reference clipping / dropout -> bypass (exact S_hat, no adaptation)
    rc = ResidualCanceller()
    run(rc, V[:100], V[:100], Xr[:100], np.zeros((100, 481)))
    w_before = rc.w.copy()
    z, info = rc.process_frame(V[100], V[100], Xr[100], np.zeros(481), ref_clipped=True)
    assert info["bypass"] == "ref_clipped" and np.array_equal(z, V[100]) and np.array_equal(rc.w, w_before)
    z, info = rc.process_frame(V[101], V[101], np.zeros(481, np.complex64), np.zeros(481))
    assert info["bypass"] == "ref_dead" and np.array_equal(z, V[101]) and np.array_equal(rc.w, w_before)
    assert rc.counters["bypass_ref_clipped"] == 1 and rc.counters["bypass_ref_dead"] == 1
    assert rc.w.dtype == np.complex64 and z.dtype == np.complex64
    print("PASS clipped / dead reference -> bypass (exact S_hat, W_hat unchanged, counted); complex64 throughout")


if __name__ == "__main__":
    main()
