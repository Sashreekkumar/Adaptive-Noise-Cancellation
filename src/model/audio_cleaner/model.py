"""
model.py

Loads DeepFilterNet3 exactly once per process and caches it in
_MODEL_CACHE. Every other module calls get_model() rather than
init_df() directly, so a batch run or a long-running server pays the
model-load cost once, not per file / per request.
"""

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
    """
    key = "default"
    if key not in _MODEL_CACHE:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        resolved_device = next(model.parameters()).device
        print(f"[audio_cleaner] DeepFilterNet3 loaded on device: {resolved_device}")
        if resolved_device.type != "cuda":
            print("[audio_cleaner] WARNING: running on CPU. If you have an NVIDIA GPU, "
                  "check that PyTorch was installed with CUDA support: "
                  'python -c "import torch; print(torch.cuda.is_available())" should print True.')
        _MODEL_CACHE[key] = (model, df_state, resolved_device)
    return _MODEL_CACHE[key]