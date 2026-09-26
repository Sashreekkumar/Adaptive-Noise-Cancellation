"""
model.py

Loads DeepFilterNet3 exactly once per process and caches it in
_MODEL_CACHE. Every other module calls get_model() rather than
init_df() directly, so a batch run or a long-running server pays the
model-load cost once, not per file / per request.
"""

import numpy as np
import torch

from . import config

_MODEL_CACHE = {}


def get_model(device=None):
    """Return (model, df_state, resolved_device), loading and caching
    on first call.

    `device` is currently advisory only and not enforced: DeepFilterNet
    manages its own device placement internally (init_df() moves the
    model to GPU automatically if a CUDA-enabled PyTorch sees a GPU;
    enhance() then moves its internal feature tensors to match).
    Manually moving the model or the input tensor ourselves causes a
    mismatch -- e.g. a cuda tensor hitting a raw .numpy() call inside
    the library -- so we leave the model exactly as init_df() returns
    it.

    On first load this also: puts the model in eval mode, enables
    cudnn.benchmark for GPU runs, and runs one dummy inference to warm
    up CUDA kernel/algorithm selection so the first *real* request
    doesn't pay that cost.
    """
    key = "default"
    if key not in _MODEL_CACHE:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        model.eval()
        resolved_device = next(model.parameters()).device
        print(f"[audio_cleaner] DeepFilterNet3 loaded on device: {resolved_device}")
        if resolved_device.type != "cuda":
            print("[audio_cleaner] WARNING: running on CPU. If you have an NVIDIA GPU, "
                  "check that PyTorch was installed with CUDA support: "
                  'python -c "import torch; print(torch.cuda.is_available())" should print True.')
        # NOTE: torch.backends.cudnn.benchmark is intentionally NOT enabled
        # here. It only pays off when every input is the same fixed shape,
        # so cuDNN can cache the fastest algorithm for that shape. Audio
        # files vary in length, so benchmark mode ends up re-searching for
        # the best algorithm on close to every call instead of reusing
        # anything -- net slower, not faster, for this workload.

        _MODEL_CACHE[key] = (model, df_state, resolved_device)

        try:
            from .enhancement import enhance_audio
            dummy = np.zeros(config.MODEL_SR, dtype=np.float32)  # 1s of silence
            enhance_audio(dummy, model, df_state, resolved_device)
            print("[audio_cleaner] warm-up inference complete")
        except Exception as e:
            print(f"[audio_cleaner] warm-up inference skipped: {e}")

    return _MODEL_CACHE[key]