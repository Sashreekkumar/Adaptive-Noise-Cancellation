"""
export_to_onnx.py

One-time conversion step: DeepFilterNet3's neural net -> a single ONNX
graph that processes one 480-sample (10ms) frame at a time, carrying
its recurrent/normalization state in an explicit tensor (ONNX graphs
have no persistent state of their own).

The upstream `df` package does its STFT / ERB-feature / deep-filter
DSP in a compiled Rust extension, which torch.onnx.export can't trace.
This instead builds vendor/torch_df_streaming.py's
ExportableStreamingTorchDF -- a pure-PyTorch reimplementation of that
DSP from grazder/DeepFilterNet, written specifically to be
ONNX-exportable -- and exports that.

Usage (run once, from outside this folder or with it on your PYTHONPATH):
    python -m onnx.export_to_onnx -o deepfilternet3.onnx

Requires, for this export step only: torch, torchaudio, onnx,
onnxruntime, and the `df` package (deep-filter) with a DeepFilterNet3
checkpoint available the same way df.enhance.init_df() normally finds
one. Nothing here is needed at inference time -- model.py/enhancement.py
in this folder only need onnxruntime + numpy.

Writes deepfilternet3.onnx plus a deepfilternet3.meta.json sidecar
(frame_size, fft_size, states_full_len, sample_rate) that model.py
reads to know how to drive the graph.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .vendor.torch_df_streaming import TorchDFPipeline


def export(output_path, model_base_dir: str = "DeepFilterNet3", opset: int = 17) -> None:
    output_path = Path(output_path)

    pipeline = TorchDFPipeline(model_base_dir=model_base_dir, device="cpu")
    streaming_model = pipeline.torch_streaming_model
    streaming_model.eval()

    frame_size = streaming_model.frame_size
    states_len = streaming_model.states_full_len

    dummy_frame = torch.zeros(frame_size, dtype=torch.float32)
    dummy_states = torch.zeros(states_len, dtype=torch.float32)
    dummy_atten = torch.tensor(0.0, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        streaming_model,
        (dummy_frame, dummy_states, dummy_atten),
        str(output_path),
        input_names=["input_frame", "states", "atten_lim_db"],
        output_names=["enhanced_frame", "new_states", "lsnr"],
        opset_version=opset,
        do_constant_folding=True,
    )

    meta = {
        "frame_size": frame_size,
        "fft_size": streaming_model.fft_size,
        "states_full_len": states_len,
        "sample_rate": pipeline.sample_rate,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {output_path} and {meta_path}")

    _validate(streaming_model, output_path, dummy_frame, dummy_states, dummy_atten)
    _validate_nonzero_frame(streaming_model, output_path, frame_size, states_len)


def _validate(streaming_model, onnx_path, frame, states, atten) -> None:
    """Sanity check on a silent frame: PyTorch output vs. the exported graph."""
    import onnxruntime as ort

    with torch.no_grad():
        torch_enhanced, _, _ = streaming_model(frame, states, atten)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_enhanced, *_ = sess.run(
        None,
        {
            "input_frame": frame.numpy(),
            "states": states.numpy(),
            "atten_lim_db": atten.numpy(),
        },
    )
    max_diff = np.max(np.abs(torch_enhanced.numpy() - onnx_enhanced))
    print(f"PyTorch vs ONNX max abs diff (silent frame): {max_diff:.2e}")


def _validate_nonzero_frame(streaming_model, onnx_path, frame_size, states_len) -> None:
    """Same check with random audio and a few frames of state already
    built up, so the silence short-circuit in forward() isn't the only
    path exercised."""
    import onnxruntime as ort

    torch.manual_seed(0)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    states_t = torch.zeros(states_len, dtype=torch.float32)
    states_np = states_t.numpy().copy()
    atten_t = torch.tensor(0.0, dtype=torch.float32)
    atten_np = atten_t.numpy()

    max_diff = 0.0
    with torch.no_grad():
        for _ in range(5):
            frame_t = torch.randn(frame_size, dtype=torch.float32) * 0.1
            torch_enhanced, states_t, _ = streaming_model(frame_t, states_t, atten_t)

            onnx_enhanced, states_np, _ = sess.run(
                None,
                {
                    "input_frame": frame_t.numpy(),
                    "states": states_np,
                    "atten_lim_db": atten_np,
                },
            )
            max_diff = max(max_diff, float(np.max(np.abs(torch_enhanced.numpy() - onnx_enhanced))))
    print(f"PyTorch vs ONNX max abs diff (5 random frames, state carried forward): {max_diff:.2e}")


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Export DeepFilterNet3 to a single streaming ONNX graph")
    p.add_argument("-o", "--output", default="deepfilternet3.onnx")
    p.add_argument("--model_base_dir", default="DeepFilterNet3",
                   help="Same model dir df.enhance.init_df() would normally use")
    p.add_argument("--opset", type=int, default=17)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    export(args.output, args.model_base_dir, args.opset)
