"""
enhancement.py

Stage 2: runs DeepFilterNet3 over the preprocessed audio. pad=True
tells DeepFilterNet3 to compensate for its own fixed algorithmic delay
internally, which is why nothing downstream needs to estimate delay
against a reference anymore.
"""

import numpy as np
import torch


def enhance_audio(audio: np.ndarray, model, df_state, device=None,
                   pad: bool = True, atten_lim_db=None) -> np.ndarray:
    """`device` is accepted for signature/logging compatibility but not
    used to move tensors -- see model.get_model for why."""
    from df.enhance import enhance
    audio_t = torch.from_numpy(audio).unsqueeze(0).float()
    with torch.no_grad():
        enhanced_t = enhance(model, df_state, audio_t, pad=pad, atten_lim_db=atten_lim_db)
    return enhanced_t.squeeze(0).cpu().numpy()