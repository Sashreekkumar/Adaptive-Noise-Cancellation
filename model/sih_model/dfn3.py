"""DeepFilterNet3 inference shared by all experiments (pretrained or fine-tuned checkpoints).

`enhance_waveform` is `df.enhance.enhance` unchanged. `enhance_spectrum_in` reproduces it step by step but lets a
caller transform DFN3's own STFT frames before feature extraction (the ERASE spectrum-domain insertion point).
With `spectrum_transform=None` it must bit-match `enhance_waveform` (see `bit_match_test`).
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import torch
from df.enhance import enhance, init_df
from df.model import ModelParams
from df.utils import as_complex, as_real, get_norm_alpha
from libdf import erb, erb_norm, unit_norm

SAMPLE_RATE = 48000
SpectrumTransform = Callable[[np.ndarray], np.ndarray]  # complex [C, T, F] -> complex [C, T, F]


def load_dfn3(model_dir: str | None = None, epoch: str | int = "best"):
    """model_dir=None loads the pretrained DeepFilterNet3 (model_120.ckpt.best)."""
    model, state, suffix = init_df(model_dir, log_level="ERROR", log_file=None, epoch=epoch)
    if state.sr() != SAMPLE_RATE:
        raise ValueError(f"DFN state sample rate {state.sr()} != {SAMPLE_RATE}")
    model.eval()
    return model, state, suffix


def enhance_waveform(model, state, audio: np.ndarray) -> np.ndarray:
    """audio: mono float [T] at 48 kHz -> enhanced [T] (same length)."""
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
    return enhance(model, state, tensor, pad=True)[0].numpy().astype(np.float64)


def features_from_spectrum(spec: np.ndarray, state, nb_df: int):
    """`df.enhance.df_features` starting from an already computed complex spectrum [C, T, F]."""
    alpha = get_norm_alpha(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        erb_feat = torch.as_tensor(erb_norm(erb(spec, state.erb_widths()), alpha)).unsqueeze(1)
    spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., :nb_df], alpha)).unsqueeze(1))
    spec_t = as_real(torch.as_tensor(spec).unsqueeze(1))
    return spec_t, erb_feat, spec_feat


@torch.no_grad()
def enhance_spectrum_in(model, state, audio: np.ndarray, spectrum_transform: SpectrumTransform | None = None) -> np.ndarray:
    model.eval()
    n_fft, hop = state.fft_size(), state.hop_size()
    orig_len = len(audio)
    padded = np.pad(np.asarray(audio, dtype=np.float32), (0, n_fft))[None]
    spec = state.analysis(padded)
    if spectrum_transform is not None:
        spec = np.ascontiguousarray(spectrum_transform(spec), dtype=spec.dtype)
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    spec_t, erb_feat, spec_feat = features_from_spectrum(spec, state, nb_df)
    device = next(model.parameters()).device
    enhanced = model(spec_t.clone().to(device), erb_feat.to(device), spec_feat.to(device))[0].cpu()
    enhanced = as_complex(enhanced.squeeze(1))
    out = state.synthesis(enhanced.numpy())
    delay = n_fft - hop
    return out[0, delay: orig_len + delay].astype(np.float64)


def bit_match_test(model, state, seconds: float = 2.0, seed: int = 0) -> float:
    """Max |enhance_spectrum_in(identity) - enhance_waveform|; must be exactly 0."""
    audio = np.random.default_rng(seed).normal(scale=0.1, size=int(seconds * SAMPLE_RATE)).astype(np.float32)
    return float(np.max(np.abs(enhance_spectrum_in(model, state, audio) - enhance_waveform(model, state, audio))))
