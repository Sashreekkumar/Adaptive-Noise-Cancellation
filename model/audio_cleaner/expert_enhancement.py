"""
expert_enhancement.py

Stage 2 (alternative): the multi-expert model-side pipeline from sih_model
instead of single-model DeepFilterNet3 inference.

  shared STFT -> E0 / E1 / E2 (DeepFilterNet3 experts) -> GRU fusion gate
  -> residual canceller using the reference-microphone spectrum -> inverse STFT

Selected by passing `model_config` (a sih_model JSON config) to
pipeline.clean_audio_file; without it the pipeline behaves exactly as
before. The ModelPipeline (all experts) is loaded once per config path and
cached, like model.get_model.
"""

from pathlib import Path

import numpy as np

_PIPELINE_CACHE = {}


def get_expert_pipeline(model_config):
    key = str(Path(model_config).resolve())
    if key not in _PIPELINE_CACHE:
        from sih_model import ModelConfig, ModelPipeline
        pipe = ModelPipeline(ModelConfig.load(key))
        print(f"[audio_cleaner] expert pipeline loaded: experts={pipe.expert_names}, gate={pipe.gate_status}, "
              f"canceller={'on' if pipe.cfg.canceller.enabled else 'off'}, device={pipe.bank.device}")
        _PIPELINE_CACHE[key] = pipe
    return _PIPELINE_CACHE[key]


def enhance_audio_experts(audio: np.ndarray, model_config, reference: np.ndarray = None):
    """`audio`: preprocessed primary (mono float32, 48 kHz). `reference`: reference-microphone
    signal, same length and sample-synchronous with `audio`, or None (canceller bypassed).
    Returns (enhanced float32 array of the same length, telemetry dict)."""
    from sih_model import ModelInput
    pipe = get_expert_pipeline(model_config)
    out = pipe.process(ModelInput(primary=np.asarray(audio, dtype=np.float32),
                                  reference=None if reference is None else np.asarray(reference, dtype=np.float32)))
    return out.audio.astype(np.float32), out.telemetry
