"""Causal per-frame impulse features for the FusionGate: [flag, severity] per STFT frame.

flag      1.0 while a frame's energy is more than JUMP_DB above a slow EMA floor (held for HOLD frames), else 0.0
severity  min(jump_db / 40, 1) for frames above the floor, else 0.0
The floor is frozen while the flag is held, so a blast cannot raise it. Works on the shared noisy STFT, so it costs
one mean over bins per frame. Same detector in training (scripts/train_gate.py) and inference (pipeline.py).
"""
from __future__ import annotations

import numpy as np

JUMP_DB = 13.0
ALPHA = 0.98
HOLD = 3


def impulse_features_np(frame_energy: np.ndarray, jump_db: float = JUMP_DB, alpha: float = ALPHA,
                        hold: int = HOLD) -> np.ndarray:
    """frame_energy [T] -> [T, 2] float32 (flag, severity)."""
    out = np.zeros((len(frame_energy), 2), dtype=np.float32)
    floor, held = None, 0
    for t, e in enumerate(np.asarray(frame_energy, dtype=np.float64)):
        e = max(e, 1e-12)
        if floor is None:
            floor = e
        jump = 10.0 * np.log10(e / floor)
        held = hold if jump > jump_db else max(held - 1, 0)
        out[t, 0] = 1.0 if held > 0 else 0.0
        out[t, 1] = min(max(jump, 0.0) / 40.0, 1.0)
        if held == 0:
            floor = alpha * floor + (1.0 - alpha) * e
    return out


def impulse_features(noisy_spec):
    """noisy_spec torch [B, T, F] complex -> torch [B, T, 2] on the same device."""
    import torch
    energy = noisy_spec.abs().pow(2).mean(dim=-1).detach().cpu().numpy()          # [B, T]
    feats = np.stack([impulse_features_np(e) for e in energy])                      # [B, T, 2]
    return torch.as_tensor(feats, device=noisy_spec.device)
