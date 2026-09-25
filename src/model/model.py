"""
model.py

Loads DeepFilterNet3 exactly once per process and caches it in
_MODEL_CACHE. Every other module calls get_model() rather than
init_df() directly, so a batch run or a long-running server pays the
model-load cost once, not per file / per request.
"""

import torch


_MODEL_CACHE = {}


def get_model(device=None):
    """Return (model, df_state, resolved_device), loading and caching
    on first call. `device` is "cpu", "cuda", or None to auto-detect."""
    key = device or "auto"
    if key not in _MODEL_CACHE:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device)
        _MODEL_CACHE[key] = (model, df_state, resolved_device)
    return _MODEL_CACHE[key]
