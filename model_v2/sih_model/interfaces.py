"""Interface between the preprocessing/input stage (owned by another team member) and the model-side pipeline.

The model side expects, per utterance or stream block:
  primary    float32/float64 mono [T], 48 kHz, full scale +-1.0 : speech + noise microphone
  reference  same length/rate, sample-synchronous with `primary` (same clock), or None : noise reference microphone
  ref_valid  optional bool: False if the preprocessing stage already knows the reference is unusable
No resampling, gain normalisation, alignment or channel selection is done on the model side.
The model side returns the enhanced mono waveform [T] at 48 kHz plus a telemetry dict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 48000


@dataclass
class ModelInput:
    primary: np.ndarray
    reference: np.ndarray | None = None
    sample_rate: int = SAMPLE_RATE
    ref_valid: bool = True

    def validate(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"model side runs at {SAMPLE_RATE} Hz only, got {self.sample_rate}")
        if self.primary.ndim != 1 or not np.isfinite(self.primary).all():
            raise ValueError("primary must be a finite mono [T] array")
        if self.reference is not None:
            if self.reference.shape != self.primary.shape:
                raise ValueError(f"reference shape {self.reference.shape} != primary shape {self.primary.shape}")
            if not np.isfinite(self.reference).all():
                raise ValueError("reference contains non-finite samples")


@dataclass
class ModelOutput:
    audio: np.ndarray            # [T] float64, 48 kHz, same length as primary
    telemetry: dict
