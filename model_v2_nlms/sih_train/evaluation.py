"""Common, non-mutating metrics for aligned audio experiment outputs."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly


def load_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    return audio.mean(axis=1), sample_rate


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    error = estimate - clean
    signal_rms = max(float(np.sqrt(np.mean(clean**2))), 1e-12)
    error_rms = max(float(np.sqrt(np.mean(error**2))), 1e-12)
    return float(20 * math.log10(signal_rms / error_rms))


def pesq_wb_16k(clean: np.ndarray, estimate: np.ndarray, sample_rate: int) -> tuple[float | None, str]:
    """PESQ is calculated on a metric-only 16 kHz resample; WAVs remain untouched."""
    try:
        clean_16k = resample_poly(clean, 16000, sample_rate)
        estimate_16k = resample_poly(estimate, 16000, sample_rate)
        return float(pesq(16000, clean_16k, estimate_16k, "wb")), "16 kHz metric-only resample"
    except Exception as error:  # PESQ rejects some short/problematic signals.
        return None, f"not computed: {error}"


PROJECT_SAMPLE_RATE = 48000

# Common result schema shared by every experiment (DFN3, NLMS, ERASE, hybrids).
RESULT_FIELDS = (
    "experiment", "mixture_id", "category", "tier", "target_snr_db", "input_snr_db", "output_snr_db",
    "snr_improvement_db", "stoi", "pesq", "pesq_note", "sample_rate", "duration_sec", "processing_time_sec",
    "rtf", "output_peak", "output_clipping_samples", "clean_path", "noisy_path", "output_path",
)


def evaluate_triplet(experiment: str, pair: dict[str, str], output_path: str,
                     processing_time_sec: float | None = None) -> dict[str, Any]:
    """Score one output against its aligned clean/noisy pair using the common schema.

    `pair` is a manifest row (mixture_id, category, tier, snr_db, local_clean_path, local_noisy_path).
    Files are read only; nothing is resampled except PESQ's internal 16 kHz copy.
    """
    clean_path, noisy_path = pair["local_clean_path"], pair["local_noisy_path"]
    metrics = evaluate_aligned(clean_path, noisy_path, output_path)
    if metrics["input_sample_rate"] != PROJECT_SAMPLE_RATE:
        raise ValueError(f"{pair['mixture_id']}: expected {PROJECT_SAMPLE_RATE} Hz, got {metrics['input_sample_rate']}")
    clean, _ = load_mono(clean_path)
    output, _ = load_mono(output_path)
    output_snr = math.inf if np.array_equal(clean, output) else metrics["output_snr_db"]
    duration = metrics["input_duration_sec"]
    return {
        "experiment": experiment, "mixture_id": pair["mixture_id"], "category": pair["category"],
        "tier": pair.get("tier", ""), "target_snr_db": float(pair["snr_db"]),
        "input_snr_db": metrics["input_snr_db"], "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - metrics["input_snr_db"], "stoi": metrics["stoi"],
        "pesq": metrics["pesq"], "pesq_note": metrics["pesq_note"], "sample_rate": metrics["input_sample_rate"],
        "duration_sec": duration, "processing_time_sec": processing_time_sec,
        "rtf": None if processing_time_sec is None else processing_time_sec / duration,
        "output_peak": metrics["output_peak"], "output_clipping_samples": metrics["output_clipping_samples"],
        "clean_path": clean_path, "noisy_path": noisy_path, "output_path": output_path,
    }


def evaluate_aligned(clean_path: str, noisy_path: str, output_path: str) -> dict[str, Any]:
    clean, clean_sr = load_mono(clean_path)
    noisy, noisy_sr = load_mono(noisy_path)
    output, output_sr = load_mono(output_path)
    if clean_sr != noisy_sr or clean_sr != output_sr:
        raise ValueError(f"Sample-rate mismatch: clean={clean_sr}, noisy={noisy_sr}, output={output_sr}")
    return {"input_sample_rate": clean_sr, "output_sample_rate": output_sr,
            **metrics_from_arrays(clean, noisy, output, clean_sr)}


def metrics_from_arrays(clean: np.ndarray, noisy: np.ndarray, output: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Same metrics as `evaluate_aligned`, for in-memory mono float64 arrays (e.g. training validation)."""
    if len(clean) != len(noisy) or len(clean) != len(output):
        raise ValueError(f"Frame mismatch: clean={len(clean)}, noisy={len(noisy)}, output={len(output)}")
    input_snr = snr_db(clean, noisy)
    output_snr = snr_db(clean, output)
    pesq_score, pesq_note = pesq_wb_16k(clean, output, sample_rate)
    clean_sr = sample_rate
    return {
        "input_duration_sec": len(noisy) / sample_rate,
        "output_duration_sec": len(output) / sample_rate,
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
        "stoi": float(stoi(clean, output, clean_sr, extended=False)),
        "pesq": pesq_score,
        "pesq_note": pesq_note,
        "output_peak": float(np.max(np.abs(output))),
        "output_clipping_samples": int(np.count_nonzero(np.abs(output) >= 0.999)),
    }
