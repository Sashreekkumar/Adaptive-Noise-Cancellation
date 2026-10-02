"""clean_audio_file with the expert pipeline (ffmpeg decode/encode stubbed so no ffmpeg or audio files are needed).
Run from model/:  python tests/test_audio_cleaner_integration.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_cleaner import pipeline  # noqa: E402

SR = 48000
rng = np.random.default_rng(0)
t = np.arange(2 * SR) / SR
NOISE = rng.normal(scale=0.05, size=len(t)).astype(np.float32)
PRIMARY = (0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0) + np.convolve(NOISE, [0, 0, 0.6, -0.2], mode="same")).astype(np.float32)
FILES = {"primary.wav": PRIMARY, "reference.wav": NOISE, "short_reference.wav": NOISE[:-10]}


def main():
    exported = {}
    pipeline.load_audio = lambda path, sr=SR: FILES[Path(path).name].copy()
    pipeline.export_mp3 = lambda audio, sr, path: exported.__setitem__(str(path), np.asarray(audio))

    # default single-model mode is unchanged and reports no model telemetry
    r = pipeline.clean_audio_file("primary.wav", "out_default.mp3")
    assert r.model_telemetry is None and len(exported["out_default.mp3"]) == len(PRIMARY)
    print("PASS default single-model mode unchanged")

    try:
        pipeline.clean_audio_file("primary.wav", "x.mp3", reference_path="reference.wav")
        raise AssertionError("reference without model_config accepted")
    except ValueError:
        print("PASS reference without model_config rejected")

    cfg = ROOT / "configs" / "frozen_v1.json"
    if not (ROOT / "checkpoints" / "e1_ft_v1" / "config.ini").exists():
        print("SKIP expert-pipeline tests: place the exports in model/checkpoints/ (see SIH_MODEL.md)")
        return
    r = pipeline.clean_audio_file("primary.wav", "out_expert.mp3", model_config=cfg, reference_path="reference.wav")
    out = exported["out_expert.mp3"]
    tel = r.model_telemetry
    assert len(out) == len(PRIMARY) and np.isfinite(out).all() and tel["experts"] == ["E0", "E1", "E2"]
    assert isinstance(tel["canceller"], dict) and tel["canceller"]["frames"] > 0
    print(f"PASS expert pipeline through clean_audio_file: gate {tel['gate']}, canceller adapted bins {tel['canceller']['adaptation_bins']}, "
          f"stages {dict((k, round(v, 2)) for k, v in r.stage_seconds.items())}")

    r = pipeline.clean_audio_file("primary.wav", "out_noref.mp3", model_config=cfg)
    assert r.model_telemetry["canceller"].startswith("bypassed")
    print("PASS expert pipeline without reference: canceller bypassed")

    try:
        pipeline.clean_audio_file("primary.wav", "x.mp3", model_config=cfg, reference_path="short_reference.wav")
        raise AssertionError("length-mismatched reference accepted")
    except ValueError as e:
        print("PASS mismatched reference length rejected:", str(e)[:60])


if __name__ == "__main__":
    main()
