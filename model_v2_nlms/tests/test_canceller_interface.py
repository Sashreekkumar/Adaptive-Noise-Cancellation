"""FusionGate -> ResidualCanceller contract smoke test (synthetic tensors, no DFN3 model).
Run: dfn-env python tests/test_canceller_interface.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from libdf import DF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sih_model.canceller_interface import (frame_reference_clipped, gate_mask_bins, reference_spectrum,  # noqa: E402
                                                 run_canceller)
from sih_model.experts import ExpertOutputs  # noqa: E402
from sih_model.fusion_gate import FusionGate  # noqa: E402
from sih_model.residual_canceller import ResidualCanceller  # noqa: E402


class CountingState:
    """libdf DF proxy that counts STFT analyses (to prove no second primary STFT)."""
    def __init__(self):
        self.df, self.analyses = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32, min_nb_erb_freqs=2), 0
    def analysis(self, x):
        self.analyses += 1
        return self.df.analysis(x)
    def __getattr__(self, name):
        return getattr(self.df, name)


def main() -> None:
    torch.manual_seed(0); rng = np.random.default_rng(0)
    state = CountingState()
    widths = state.erb_widths()
    fb = torch.zeros(32, 481)
    start = 0
    for band, w in enumerate(int(w) for w in widths):
        fb[band, start:start + w] = 1; start += w
    b, k, n = 1, 2, 48000
    x_p = rng.normal(scale=0.1, size=n).astype(np.float32)
    y = torch.as_tensor(state.analysis(np.pad(x_p, (0, 960))[None]))              # stands in for ExpertBank's one STFT
    t = y.shape[1]
    outs = ExpertOutputs(noisy_spec=y, feat_erb=torch.randn(b, t, 32),
                         spec=(y.unsqueeze(1) * torch.rand(b, k, t, 481)).to(torch.complex64),
                         mask=torch.rand(b, k, t, 32), lsnr=torch.randn(b, k, t), orig_len=n)
    gate = FusionGate(k, fb)
    fused, band_w, _ = gate(outs)
    mask = gate_mask_bins(band_w, outs.mask, fb)
    x_r = rng.normal(scale=0.1, size=n).astype(np.float32)
    analyses_before = state.analyses
    xr = reference_spectrum(state, x_r)
    clip = frame_reference_clipped(state, x_r, t)
    assert state.analyses == analyses_before + 1, "reference must be analysed exactly once"

    # shapes / dtypes
    assert fused.shape == y.shape == xr.shape == (b, t, 481) and mask.shape == (b, t, 481) and clip.shape == (b, t)
    assert fused.dtype == y.dtype == xr.dtype == torch.complex64 and mask.dtype == torch.float32
    assert float(mask.min()) >= 0 and float(mask.max()) <= 1
    print(f"PASS contract shapes/dtypes: S_hat/Y/X_r {tuple(fused.shape)} complex64, gate mask {tuple(mask.shape)} float32 in [0,1]")

    # W_hat = 0 identity: adaptation blocked (mask >= 0.3) -> Z == S_hat exactly; no STFT during cancellation
    rc = [ResidualCanceller()]
    z, tele = run_canceller(rc, fused, y, xr, torch.ones_like(mask), ref_clipped=clip)
    assert z.dtype == torch.complex64 and torch.equal(z, fused) and not rc[0].w.any()
    assert state.analyses == analyses_before + 1, "run_canceller must not perform an STFT"
    print("PASS W_hat=0 identity through the interface (Z == S_hat exactly), no extra STFT")

    # telemetry fields present and counting (real gate mask, adaptation allowed where mask < 0.3; one clipped frame)
    clip[0, 10] = True
    z, tele = run_canceller([ResidualCanceller()], fused, y, xr, mask, ref_clipped=clip)
    need = {"bin_guard", "frame_guard", "ref_bypass", "low_y_skip_bins", "adaptation_bins"}
    # 1 clipped frame + the final all-padding reference frame (ExpertBank pads 960 zeros) detected as dead
    assert need <= set(tele[0]) and tele[0]["ref_bypass_clipped"] == 1 and tele[0]["frames"] == t
    assert tele[0]["ref_bypass"] == tele[0]["ref_bypass_clipped"] + tele[0]["ref_bypass_dead"]
    assert torch.equal(z[0, 10], fused[0, 10])
    assert torch.isfinite(torch.view_as_real(z)).all()
    print(f"PASS telemetry {tele[0]}")


if __name__ == "__main__":
    main()
