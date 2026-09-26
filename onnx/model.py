"""
model.py

Loads the exported ONNX graph exactly once per process and caches it
in _MODEL_CACHE, same pattern as audio_cleaner/model.py's get_model()
for the torch backend -- every other module here calls get_model()
rather than constructing an onnxruntime.InferenceSession directly, so
a batch run pays the session-load cost once, not per file.

Needs export_to_onnx.py to have been run first, producing:
    <name>.onnx
    <name>.meta.json   (frame_size, fft_size, states_full_len, sample_rate)
next to each other. Only onnxruntime + numpy are required here -- no
torch, no `df` package.
"""

import json
from pathlib import Path

import onnxruntime as ort

_MODEL_CACHE = {}


def get_model(onnx_path, providers=None):
    """Return a loaded OnnxModel for onnx_path, loading and caching on
    first call per resolved path."""
    key = str(Path(onnx_path).resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = OnnxModel(onnx_path, providers=providers)
        print(f"[onnx] loaded {onnx_path} "
              f"(providers={_MODEL_CACHE[key].session.get_providers()})")
    return _MODEL_CACHE[key]


class OnnxModel:
    def __init__(self, onnx_path, providers=None):
        onnx_path = Path(onnx_path)
        meta_path = onnx_path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing {meta_path} -- run export_to_onnx.py first; it writes this "
                "sidecar (frame_size, fft_size, states_full_len, sample_rate) "
                "alongside the .onnx file."
            )
        self.meta = json.loads(meta_path.read_text())
        self.frame_size = self.meta["frame_size"]
        self.fft_size = self.meta["fft_size"]
        self.states_full_len = self.meta["states_full_len"]
        self.sample_rate = self.meta["sample_rate"]

        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

    def run_frame(self, frame, states, atten_lim_db):
        enhanced_frame, new_states, _lsnr = self.session.run(
            self._output_names,
            {
                self._input_names[0]: frame,
                self._input_names[1]: states,
                self._input_names[2]: atten_lim_db,
            },
        )
        return enhanced_frame, new_states
