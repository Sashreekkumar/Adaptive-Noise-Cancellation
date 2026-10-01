"""Post-fusion residual canceller in the DFN3 STFT domain (one complex path estimate W_hat(f) per bin).

Per frame t (all complex64 / float32, never fp16):
  adapt   e = Y - W_hat X_r ;  W_hat += mu conj(X_r) e / (|X_r|^2 + delta)
          only in bins where: prev-frame gate-weighted mask < 0.3, no impulse flag, no impulse tail, bin not erased,
          reference not clipped / not dead
  apply   G = S_hat / (Y + eps), |G| clamped to <= 1.5 (phase kept)
          Z = S_hat - W_hat (G X_r)   using W_hat from BEFORE this frame's update (causal)
          skipped where |S_hat| or |W_hat G X_r| is at or below the noise floor
  guards  |Z| > |S_hat| in a bin -> Z = S_hat and W_hat(f) reset to 0
          frame energy sum|Z|^2 > sum|S_hat|^2 -> whole frame output = S_hat
          dead or clipped reference -> bypass (Z = S_hat, no adaptation)
          every firing is counted in .counters
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C64, F32 = np.complex64, np.float32


def reference_clipped(block: np.ndarray, threshold: float = 0.999) -> bool:
    """Time-domain reference block clipping check (caller passes the result as `ref_clipped`)."""
    return bool(np.max(np.abs(block)) >= threshold) if len(block) else False


@dataclass
class ResidualCanceller:
    n_bins: int = 481
    mu: float = 0.5
    delta: float = 1e-8
    eps: float = 1e-8
    g_max: float = 1.5
    mask_threshold: float = 0.3
    noise_floor: float = 1e-6
    dead_power: float = 1e-12        # mean |X_r|^2 below this -> reference considered dead
    w: np.ndarray = field(init=False)
    prev_mask: np.ndarray = field(init=False)
    counters: dict = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.n_bins, dtype=C64)
        self.prev_mask = np.ones(self.n_bins, dtype=F32)   # frame 0: no previous mask -> no adaptation
        self.counters = {"frames": 0, "bypass_ref_clipped": 0, "bypass_ref_dead": 0, "bin_reset": 0,
                         "frame_energy_guard": 0, "floor_skip_bins": 0, "adapted_bins": 0}

    def telemetry(self) -> dict:
        """Integration telemetry (read-only view of .counters; no effect on processing).
        low_y_skip_bins = bins where subtraction was skipped at/below the noise floor (|S_hat| or |W G X_r|)."""
        c = self.counters
        return {"frames": c["frames"], "bin_guard": c["bin_reset"], "frame_guard": c["frame_energy_guard"],
                "ref_bypass": c["bypass_ref_clipped"] + c["bypass_ref_dead"],
                "ref_bypass_clipped": c["bypass_ref_clipped"], "ref_bypass_dead": c["bypass_ref_dead"],
                "low_y_skip_bins": c["floor_skip_bins"], "adaptation_bins": c["adapted_bins"]}

    def process_frame(self, s_hat: np.ndarray, y: np.ndarray, x_r: np.ndarray, mask: np.ndarray,
                      impulse: bool = False, impulse_tail: bool = False, erased: np.ndarray | None = None,
                      ref_clipped: bool = False) -> tuple[np.ndarray, dict]:
        """s_hat: fused enhanced spectrum, y: noisy primary, x_r: reference (all [F] complex); mask: current frame
        gate-weighted mask per bin [F] (used as 'previous-frame mask' at the next frame). Returns (Z, info)."""
        s_hat, y, x_r = (np.asarray(a, dtype=C64) for a in (s_hat, y, x_r))
        mask = np.asarray(mask, dtype=F32)
        erased = np.zeros(self.n_bins, dtype=bool) if erased is None else np.asarray(erased, dtype=bool)
        c = self.counters
        c["frames"] += 1
        prev_mask, self.prev_mask = self.prev_mask, mask.copy()
        info = {"bypass": None, "applied": np.zeros(self.n_bins, dtype=bool), "w_used": self.w.copy(),
                "g": np.zeros(self.n_bins, dtype=C64)}
        dead = float(np.mean(np.abs(x_r) ** 2)) < self.dead_power
        if ref_clipped or dead:
            c["bypass_ref_clipped" if ref_clipped else "bypass_ref_dead"] += 1
            info["bypass"] = "ref_clipped" if ref_clipped else "ref_dead"
            return s_hat.copy(), info

        # apply with the current (pre-update) estimate
        g = s_hat / (y + F32(self.eps))
        mag = np.abs(g)
        g = np.where(mag > self.g_max, g * (F32(self.g_max) / np.maximum(mag, F32(1e-30))), g).astype(C64)
        estimate = (self.w * (g * x_r)).astype(C64)
        applied = (np.abs(s_hat) > self.noise_floor) & (np.abs(estimate) > self.noise_floor)
        c["floor_skip_bins"] += int(np.sum(~applied))
        z = np.where(applied, s_hat - estimate, s_hat).astype(C64)
        grew = np.abs(z) > np.abs(s_hat)
        if grew.any():
            c["bin_reset"] += int(grew.sum())
            z[grew] = s_hat[grew]
            applied &= ~grew
        if float(np.sum(np.abs(z) ** 2)) > float(np.sum(np.abs(s_hat) ** 2)):
            c["frame_energy_guard"] += 1
            z, applied = s_hat.copy(), np.zeros_like(applied)
        info.update(applied=applied, g=g)

        # adapt (NLMS on the primary), then reset bins whose subtraction diverged
        adapt = (prev_mask < self.mask_threshold) & ~erased
        if not (impulse or impulse_tail) and adapt.any():
            e = y - self.w * x_r
            step = F32(self.mu) * np.conj(x_r) * e / (np.abs(x_r) ** 2 + F32(self.delta))
            self.w = np.where(adapt, self.w + step, self.w).astype(C64)
            c["adapted_bins"] += int(adapt.sum())
        self.w[grew] = 0
        return z, info
