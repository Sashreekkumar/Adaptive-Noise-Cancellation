"""
enhancement.py

Stage 2: runs DeepFilterNet3 over the preprocessed audio -- same job
as audio_cleaner/enhancement.py's enhance_audio(), but through the
ONNX graph from model.py instead of PyTorch + the `df` package.

The ONNX graph only knows how to process one 480-sample frame at a
time and needs its recurrent/normalization state threaded through by
the caller (ONNX graphs have no persistent state of their own), so
this replicates the chunking loop that TorchDFPipeline.forward() does
in PyTorch (see export_to_onnx.py / vendor/torch_df_streaming.py), in
numpy instead.
"""

import numpy as np

from .model import OnnxModel


def enhance_audio(audio: np.ndarray, model: OnnxModel, atten_lim_db=None) -> np.ndarray:
    """Signature deliberately close to the torch version's
    enhance_audio(audio, model, df_state, device, ...) so pipeline.py
    reads the same either way -- just fewer arguments, since an
    onnxruntime session carries no df_state/device of its own."""
    hop = model.frame_size
    fft = model.fft_size

    true_len = len(audio)
    pad_to_hop = (hop - true_len % hop) % hop
    padded_len = true_len + pad_to_hop  # multiple of hop

    # Same padding as the reference TorchDFPipeline: round up to a hop
    # multiple, then add fft_size more so the synthesis tail has
    # somewhere to land.
    padded = np.zeros(padded_len + fft, dtype=np.float32)
    padded[:true_len] = audio.astype(np.float32)

    states = np.zeros(model.states_full_len, dtype=np.float32)
    atten = np.asarray(atten_lim_db if atten_lim_db is not None else 0.0, dtype=np.float32)  # 0-d scalar

    n_frames = len(padded) // hop
    out_frames = []
    for i in range(n_frames):
        frame = padded[i * hop:(i + 1) * hop]
        enhanced_frame, states = model.run_frame(frame, states, atten)
        out_frames.append(enhanced_frame)

    enhanced = np.concatenate(out_frames)
    d = fft - hop
    enhanced = enhanced[d: padded_len + d]
    # Trim the rest of the way back to the exact input length (the
    # reference implementation stops at padded_len, a hop multiple, so
    # it can leave up to hop-1 samples of trailing padding-derived
    # audio -- this keeps output length == input length regardless).
    return enhanced[:true_len].astype(np.float32)
