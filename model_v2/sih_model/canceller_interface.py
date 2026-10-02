"""FusionGate -> ResidualCanceller integration contract (no pipeline wiring yet; see docs/residual_canceller.md).

All spectra are DFN3 STFT frames: 481 bins, 10 ms hop, 48 kHz. The primary is analysed ONCE by ExpertBank; the
reference channel is analysed once by `reference_spectrum` with the identical libdf state and padding.

Per-utterance tensors (B = batch, T = frames, F = 481, K = experts):
  fused        [B,T,F] complex64  S_hat from FusionGate.forward
  noisy_spec   [B,T,F] complex64  Y = ExpertOutputs.noisy_spec (shared STFT, not recomputed)
  ref_spec     [B,T,F] complex64  X_r from reference_spectrum()
  gate_mask    [B,T,F] float32    sum_k w_k(band) * m_k(band), expanded with erb_inv_fb (gate_mask_bins)
  impulse      [B,T]   bool       impulse flag            (default False: no detector wired yet)
  impulse_tail [B,T]   bool       post-impulse tail state (default False)
  erased       [B,T,F] bool       ERASE-erased bins       (default False)
  ref_clipped  [B,T]   bool       reference clipped in this frame (reference validity; dead reference is
                                  detected inside the canceller from X_r power)
Output: Z [B,T,F] complex64 on the input device + one telemetry dict per batch element.
"""
from __future__ import annotations

import numpy as np
import torch

from sih_model.residual_canceller import ResidualCanceller


def gate_mask_bins(band_w: torch.Tensor, masks: torch.Tensor, erb_inv_fb: torch.Tensor) -> torch.Tensor:
    """band_w [B,T,32,K] (FusionGate weights), masks [B,K,T,32] (expert ERB gains) -> [B,T,F] float32."""
    fused_band = (band_w * masks.permute(0, 2, 3, 1)).sum(-1)                  # [B,T,32]
    return (fused_band @ erb_inv_fb.to(fused_band)).to(torch.float32)          # [B,T,F]


def reference_spectrum(state, x_r: np.ndarray) -> torch.Tensor:
    """Reference audio [T] or [B,T] -> [B,T',F] complex64 with ExpertBank's exact analysis/padding."""
    x = np.atleast_2d(np.asarray(x_r, dtype=np.float32))
    return torch.as_tensor(state.analysis(np.ascontiguousarray(np.pad(x, ((0, 0), (0, state.fft_size()))))))


def frame_reference_clipped(state, x_r: np.ndarray, n_frames: int, threshold: float = 0.999) -> torch.Tensor:
    """[B,T] bool: any |x_r| >= threshold within a frame's new hop (same framing as reference_spectrum)."""
    x = np.atleast_2d(np.asarray(x_r, dtype=np.float32))
    hop = state.hop_size()
    x = np.pad(x, ((0, 0), (0, n_frames * hop - x.shape[1] if n_frames * hop > x.shape[1] else 0)))[:, : n_frames * hop]
    return torch.as_tensor(np.abs(x.reshape(x.shape[0], n_frames, hop)).max(-1) >= threshold)


@torch.no_grad()
def run_canceller(cancellers: list[ResidualCanceller], fused: torch.Tensor, noisy_spec: torch.Tensor,
                  ref_spec: torch.Tensor, gate_mask: torch.Tensor, impulse: torch.Tensor | None = None,
                  impulse_tail: torch.Tensor | None = None, erased: torch.Tensor | None = None,
                  ref_clipped: torch.Tensor | None = None) -> tuple[torch.Tensor, list[dict]]:
    """Frame-by-frame canceller over whole utterances (one stateful canceller per batch element)."""
    b, t, f = fused.shape
    for name, x, shape in (("noisy_spec", noisy_spec, (b, t, f)), ("ref_spec", ref_spec, (b, t, f)),
                           ("gate_mask", gate_mask, (b, t, f))):
        if tuple(x.shape) != shape:
            raise ValueError(f"{name} shape {tuple(x.shape)} != {shape}")
    if len(cancellers) != b or any(c.n_bins != f for c in cancellers):
        raise ValueError("need one ResidualCanceller per batch element with n_bins == F")
    np_ = lambda x: x.detach().cpu().numpy()
    fused_n, y_n, xr_n = (np_(x).astype(np.complex64) for x in (fused, noisy_spec, ref_spec))
    mask_n = np_(gate_mask).astype(np.float32)
    imp = np_(impulse).astype(bool) if impulse is not None else np.zeros((b, t), bool)
    tail = np_(impulse_tail).astype(bool) if impulse_tail is not None else np.zeros((b, t), bool)
    era = np_(erased).astype(bool) if erased is not None else np.zeros((b, t, f), bool)
    clip = np_(ref_clipped).astype(bool) if ref_clipped is not None else np.zeros((b, t), bool)
    out = np.empty((b, t, f), dtype=np.complex64)
    for i, rc in enumerate(cancellers):
        for k in range(t):
            out[i, k], _ = rc.process_frame(fused_n[i, k], y_n[i, k], xr_n[i, k], mask_n[i, k], impulse=bool(imp[i, k]),
                                            impulse_tail=bool(tail[i, k]), erased=era[i, k], ref_clipped=bool(clip[i, k]))
    return torch.as_tensor(out).to(fused.device), [rc.telemetry() for rc in cancellers]
