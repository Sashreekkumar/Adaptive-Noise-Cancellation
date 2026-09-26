"""
pipeline.py

Ties the stage modules together into one function: a noisy audio
file in, a clean mp3 out. This is the module every entry point
(cli.py, server.py) calls -- neither of them talks to the individual
stage modules directly.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .model import get_model
from .loader import load_audio
from .preprocessing import preprocess
from .enhancement import enhance_audio
from .postprocessing import postprocess
from .exporter import export_mp3


@dataclass
class CleanResult:
    output_path: Path
    latency_seconds: float          # input-in to output-out time; excludes one-time model load
    audio_duration_seconds: float   # length of the input audio
    rtf: float                      # real-time factor = latency / audio_duration; <1 is faster than real time
    stage_seconds: dict              # per-stage breakdown: load, preprocess, enhance, postprocess, export


def clean_audio_file(input_path, output_path, *, device=None, pad: bool = True,
                      atten_lim_db=None, blend_wet: float = 1.0, normalize_volume: bool = False,
                      target_dbfs: float = config.TARGET_DBFS, hpf_cutoff: float = config.HPF_CUTOFF_HZ,
                      band_low: float = config.BAND_LOW_HZ, band_high: float = config.BAND_HIGH_HZ,
                      skip_dc: bool = False, skip_hpf: bool = False, skip_pre_norm: bool = False) -> CleanResult:
    """Run the full pipeline on a single file. Reuses the cached
    model (see model.get_model), so only the first call in a process
    pays model-load cost. Returns a CleanResult with the output path,
    latency, real-time factor, and a per-stage timing breakdown so you
    can see exactly where time is going."""
    model, df_state, resolved_device = get_model(device)

    stage_seconds = {}
    start = time.perf_counter()

    t0 = time.perf_counter()
    audio = load_audio(input_path, sr=config.MODEL_SR)
    stage_seconds["load"] = time.perf_counter() - t0
    audio_duration = len(audio) / config.MODEL_SR

    t0 = time.perf_counter()
    pre = preprocess(audio, config.MODEL_SR, skip_dc, skip_hpf, skip_pre_norm,
                      hpf_cutoff, target_dbfs)
    stage_seconds["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    enhanced = enhance_audio(pre, model, df_state, resolved_device,
                              pad=pad, atten_lim_db=atten_lim_db)
    stage_seconds["enhance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    final = postprocess(enhanced, pre, blend_wet, band_low, band_high,
                         normalize_volume, target_dbfs, config.MODEL_SR)
    stage_seconds["postprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    export_mp3(final, config.MODEL_SR, output_path)
    stage_seconds["export"] = time.perf_counter() - t0

    latency = time.perf_counter() - start
    rtf = latency / audio_duration if audio_duration > 0 else float("nan")
    return CleanResult(output_path=Path(output_path), latency_seconds=latency,
                        audio_duration_seconds=audio_duration, rtf=rtf,
                        stage_seconds=stage_seconds)