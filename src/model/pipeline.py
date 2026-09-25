"""
pipeline.py

Ties the stage modules together into one function: a noisy audio
file in, a clean mp3 out. This is the module every entry point
(cli.py, server.py) calls -- neither of them talks to the individual
stage modules directly.
"""

import librosa

from . import config
from .model import get_model
from .preprocessing import preprocess
from .enhancement import enhance_audio
from .postprocessing import postprocess
from .exporter import export_mp3


def clean_audio_file(input_path, output_path, *, device=None, pad: bool = True,
                      atten_lim_db=None, blend_wet: float = 1.0, normalize_volume: bool = False,
                      target_dbfs: float = config.TARGET_DBFS, hpf_cutoff: float = config.HPF_CUTOFF_HZ,
                      band_low: float = config.BAND_LOW_HZ, band_high: float = config.BAND_HIGH_HZ,
                      skip_dc: bool = False, skip_hpf: bool = False, skip_pre_norm: bool = False):
    """Run the full pipeline on a single file. Reuses the cached
    model (see model.get_model), so only the first call in a process
    pays model-load cost."""
    model, df_state, resolved_device = get_model(device)

    audio, _ = librosa.load(str(input_path), sr=config.MODEL_SR, mono=True)

    pre = preprocess(audio, config.MODEL_SR, skip_dc, skip_hpf, skip_pre_norm,
                      hpf_cutoff, target_dbfs)

    enhanced = enhance_audio(pre, model, df_state, resolved_device,
                              pad=pad, atten_lim_db=atten_lim_db)

    final = postprocess(enhanced, pre, blend_wet, band_low, band_high,
                         normalize_volume, target_dbfs, config.MODEL_SR)

    export_mp3(final, config.MODEL_SR, output_path)
    return output_path
