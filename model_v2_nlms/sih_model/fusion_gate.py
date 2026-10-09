"""Per-ERB-band expert fusion gate: Linear -> 2-layer GRU(64) -> Linear -> softmax over experts per band.

Input per frame (dim 32 + 32K + K + 2 + 32):
  noisy ERB features [32] | expert ERB masks [32*K] | expert local SNR [K] | impulse flag + severity [2] |
  mask disagreement [32] (std of the expert masks across experts; 0 for K = 1)
Output: band weights [B, T, 32, K]; expanded to 481 bins with DFN3's own ERB mapping (erb_inv_fb, 0/1, one band per
bin); fused complex spectrum S_hat = sum_k w_k * S_hat_k. The last Linear is zero-initialised -> uniform weights.
"""
from __future__ import annotations

import torch
from torch import nn

N_ERB = 32


def impulse_rule_weights(impulse: torch.Tensor) -> torch.Tensor:
    """Rule gate for two experts: impulse [B,T,2] (flag, severity) -> band weights [B,T,32,2] = [1 - flag, flag] in all
    32 bands (second expert where the impulse flag is 1, first expert elsewhere). eval_gate.py B4 / gate mode "rule"."""
    flag = impulse[..., 0][..., None].expand(-1, -1, N_ERB)                  # [B,T,32]
    return torch.stack([1.0 - flag, flag], dim=-1)


class FusionGate(nn.Module):
    def __init__(self, n_experts: int, erb_inv_fb: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.k = n_experts
        self.register_buffer("erb_inv_fb", erb_inv_fb.detach().clone().float())   # [32, F]
        in_dim = N_ERB + N_ERB * n_experts + n_experts + 2 + N_ERB
        self.inp = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden, N_ERB * n_experts)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def features(self, feat_erb: torch.Tensor, masks: torch.Tensor, lsnr: torch.Tensor,
                 impulse: torch.Tensor | None = None) -> torch.Tensor:
        """feat_erb [B,T,32], masks [B,K,T,32], lsnr [B,K,T], impulse [B,T,2] (flag, severity; zeros if None)."""
        b, k, t, _ = masks.shape
        if impulse is None:
            impulse = torch.zeros(b, t, 2, device=masks.device)
        disagreement = masks.std(dim=1, unbiased=False)                      # [B,T,32]
        return torch.cat([feat_erb, masks.permute(0, 2, 1, 3).reshape(b, t, k * N_ERB), lsnr.permute(0, 2, 1),
                          impulse, disagreement], dim=-1)

    def band_weights(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        y, h = self.gru(self.inp(x), h)
        logits = self.out(y).view(x.shape[0], x.shape[1], N_ERB, self.k)
        return torch.softmax(logits, dim=-1), h                               # [B,T,32,K]

    def fuse(self, spec: torch.Tensor, band_w: torch.Tensor) -> torch.Tensor:
        """spec [B,K,T,F] complex, band_w [B,T,32,K] -> fused [B,T,F] complex."""
        bin_w = torch.einsum("btek,ef->bkft", band_w, self.erb_inv_fb)        # [B,K,F,T]
        return (spec * bin_w.permute(0, 1, 3, 2)).sum(dim=1)

    def forward(self, outputs, impulse: torch.Tensor | None = None, h: torch.Tensor | None = None):
        """outputs: experts.ExpertOutputs. Returns (fused [B,T,F] complex, band weights [B,T,32,K], GRU state)."""
        x = self.features(outputs.feat_erb, outputs.mask, outputs.lsnr, impulse)
        band_w, h = self.band_weights(x, h)
        return self.fuse(outputs.spec, band_w), band_w, h
