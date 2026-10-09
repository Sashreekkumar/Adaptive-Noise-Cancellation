"""Model-side pipeline:  shared STFT -> experts in parallel -> GRU fusion gate -> residual canceller -> iSTFT.

  primary x_p ──► ExpertBank: ONE libdf STFT (960/480, 481 bins, 48 kHz) ──► E0..E(K-1) (frozen, no_grad)
                                   │ Y                                         │ S_k, m_k, lsnr_k
                                   │                                           ▼
                                   │                               FusionGate (per ERB band weights)
                                   │                                           │ S_hat, gate-weighted mask
  reference x_r ─► one STFT (same state/padding) ─► X_r ─► ResidualCanceller(S_hat, Y, X_r, mask) ─► Z
                                                                               │
                                                                     libdf iSTFT ─► output [T]
Stage switches (config): any expert, the gate, the canceller. With one expert the gate is bypassed. With the gate
disabled and several experts, weights are uniform. Without a reference (or canceller disabled) Z = S_hat.
ERASE is not part of this pipeline (impulse flags / erased bins are passed as False to the canceller).
"""
from __future__ import annotations

import numpy as np
import torch

from sih_model.canceller_interface import frame_reference_clipped, gate_mask_bins, reference_spectrum, run_canceller
from sih_model.config import ModelConfig
from sih_model.experts import ExpertBank, load_expert
from sih_model.fusion_gate import FusionGate
from sih_model.impulse import impulse_features
from sih_model.interfaces import ModelInput, ModelOutput
from sih_model.residual_canceller import ResidualCanceller


class ModelPipeline:
    def __init__(self, cfg: ModelConfig):
        cfg.validate()
        self.cfg = cfg
        active = cfg.active_experts
        self.expert_names = [e.name for e in active]
        self.bank = ExpertBank([load_expert(None if e.source == "pretrained" else e.source) for e in active])
        self.k = len(active)
        self.gate: FusionGate | None = None
        self.gate_status = "bypassed (single expert)" if self.k == 1 else "disabled (uniform weights)"
        if cfg.gate.enabled and self.k > 1:
            self.gate = FusionGate(self.k, self.bank.erb_inv_fb).to(self.bank.device).eval()
            if cfg.gate.checkpoint:
                state = torch.load(cfg.gate.checkpoint, map_location="cpu")
                state = state["gate"] if isinstance(state, dict) and "gate" in state else state
                expected = self.gate.state_dict()
                bad = [k for k in expected if k not in state or state[k].shape != expected[k].shape]
                if bad:
                    raise ValueError(f"gate checkpoint {cfg.gate.checkpoint} does not match {self.k} experts "
                                     f"(mismatching tensors: {bad[:3]}...)")
                self.gate.load_state_dict(state)
                self.gate_status = f"trained ({cfg.gate.checkpoint})"
            else:
                self.gate_status = "UNTRAINED (zero-init: uniform weights)"
            for p in self.gate.parameters():
                p.requires_grad_(False)

    def new_canceller(self) -> ResidualCanceller:
        return ResidualCanceller(**self.cfg.canceller.params)

    @torch.no_grad()
    def process(self, inp: ModelInput, canceller: ResidualCanceller | None = None) -> ModelOutput:
        """One utterance/block. Pass a persistent `canceller` to keep W_hat across calls; default = fresh state."""
        inp.validate()
        outs = self.bank(inp.primary)                                    # shared STFT + all experts
        b, k, t, f = outs.spec.shape
        if self.gate is not None:
            fused, band_w, _ = self.gate(outs, impulse=impulse_features(outs.noisy_spec))
        else:
            band_w = torch.full((b, t, 32, k), 1.0 / k, device=outs.spec.device)
            fused = outs.spec[:, 0] if k == 1 else outs.spec.mean(dim=1)
        telemetry = {"experts": self.expert_names, "gate": self.gate_status,
                     "gate_mean_weights": {n: float(band_w[..., i].mean()) for i, n in enumerate(self.expert_names)},
                     "frames": t, "canceller": "disabled"}
        z = fused
        if self.cfg.canceller.enabled:
            if inp.reference is None or not inp.ref_valid:
                telemetry["canceller"] = "bypassed (no valid reference)"
            else:
                mask = gate_mask_bins(band_w, outs.mask, self.bank.erb_inv_fb)
                x_r = reference_spectrum(self.bank.state, inp.reference).to(fused.device)
                clipped = frame_reference_clipped(self.bank.state, inp.reference, t)
                z, tele = run_canceller([canceller or self.new_canceller()], fused, outs.noisy_spec, x_r, mask, ref_clipped=clipped)
                telemetry["canceller"] = tele[0]
        audio = self.bank.synthesize(z, outs.orig_len)[0].astype(np.float64)
        if not np.isfinite(audio).all():
            raise FloatingPointError("non-finite samples in pipeline output")
        telemetry.update(output_peak=float(np.max(np.abs(audio))), output_clipped=bool(np.max(np.abs(audio)) >= 0.999))
        return ModelOutput(audio=audio, telemetry=telemetry)
