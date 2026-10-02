"""Frozen DFN3 experts sharing ONE STFT/feature computation (E0 = pretrained, E1 = fine-tuned).

The noisy input is analysed once with DFN3's own libdf STFT (exactly as df.enhance.enhance, pad=True); the same
spectrum and features feed every expert. Experts run in eval mode under torch.no_grad(); DFN3 is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from df.utils import as_complex

from sih_model.dfn3 import features_from_spectrum, load_dfn3


def load_expert(source: str | Path | None):
    """None -> pretrained DFN3 (E0); a directory -> init_df export (e.g. FT-v1 export_best); a .pt file -> DFN3 with a
    full state_dict (bare state_dict or trainer checkpoint with a 'model' key)."""
    if source is None:
        model, state, name = load_dfn3()
    elif Path(source).is_dir():
        model, state, name = load_dfn3(str(source))
    else:
        model, state, _ = load_dfn3()
        blob = torch.load(source, map_location="cpu")
        model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        name = Path(source).name
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, state, name


@dataclass
class ExpertOutputs:
    noisy_spec: torch.Tensor   # [B, T, F] complex64, shared STFT
    feat_erb: torch.Tensor     # [B, T, 32] normalized noisy ERB features (DFN3 input features)
    spec: torch.Tensor         # [B, K, T, F] complex64, enhanced spectrum per expert
    mask: torch.Tensor         # [B, K, T, 32] ERB gain per expert
    lsnr: torch.Tensor         # [B, K, T] local SNR estimate per expert
    orig_len: int


class ExpertBank:
    def __init__(self, experts: list[tuple]):
        """experts: list of (model, df_state, name) from load_expert; all must share the DFN3 STFT config."""
        self.models = [e[0] for e in experts]
        self.names = [e[2] for e in experts]
        self.state = experts[0][1]
        for _, st, _ in experts[1:]:
            assert (st.sr(), st.fft_size(), st.hop_size(), st.nb_erb()) == (self.state.sr(), self.state.fft_size(),
                                                                             self.state.hop_size(), self.state.nb_erb())
        self.erb_inv_fb = self.models[0].mask.erb_inv_fb.detach().cpu()  # [32, F] 0/1 band -> bin mapping
        self.device = next(self.models[0].parameters()).device

    @torch.no_grad()
    def __call__(self, audio: np.ndarray) -> ExpertOutputs:
        """audio: mono [T] (or [B, T]) float at 48 kHz."""
        x = np.atleast_2d(np.asarray(audio, dtype=np.float32))
        n_fft = self.state.fft_size()
        spec = self.state.analysis(np.ascontiguousarray(np.pad(x, ((0, 0), (0, n_fft)))))       # ONE STFT
        spec_t, erb_feat, spec_feat = features_from_spectrum(spec, self.state, self.models[0].nb_df)
        spec_t, erb_feat, spec_feat = spec_t.to(self.device), erb_feat.to(self.device), spec_feat.to(self.device)
        specs, masks, lsnrs = [], [], []
        for model in self.models:
            enh, m, lsnr, _ = model(spec_t.clone(), erb_feat, spec_feat)
            specs.append(as_complex(enh.squeeze(1)))       # [B, T, F]
            masks.append(m.squeeze(1))                     # [B, T, 32]
            lsnrs.append(lsnr.squeeze(-1))                 # [B, T]
        return ExpertOutputs(noisy_spec=torch.as_tensor(spec).to(self.device), feat_erb=erb_feat.squeeze(1),
                             spec=torch.stack(specs, 1), mask=torch.stack(masks, 1), lsnr=torch.stack(lsnrs, 1),
                             orig_len=x.shape[-1])

    def synthesize(self, spec: torch.Tensor, orig_len: int) -> np.ndarray:
        """[B, T, F] complex -> [B, orig_len] waveform, same delay compensation as df.enhance.enhance."""
        audio = self.state.synthesis(np.ascontiguousarray(spec.detach().cpu().numpy().astype(np.complex64)))
        d = self.state.fft_size() - self.state.hop_size()
        return audio[:, d: orig_len + d]
