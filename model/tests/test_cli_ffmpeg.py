"""End-to-end CLI test with real ffmpeg decode / mp3 encode (no stubs).
Run from model/:  python tests/test_cli_ffmpeg.py
Skips when ffmpeg is not on PATH; expert-pipeline parts skip when model/checkpoints/ is absent."""
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SR = 48000


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "run.py", *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def check_mp3(path: Path, n_samples: int) -> np.ndarray:
    from audio_cleaner.loader import load_audio
    assert path.exists() and path.stat().st_size > 1000, f"{path} missing or empty"
    audio = load_audio(path)
    assert np.isfinite(audio).all() and np.sqrt(np.mean(audio ** 2)) > 1e-4, "decoded output silent or non-finite"
    assert abs(len(audio) - n_samples) < 0.1 * SR, f"decoded length {len(audio)} vs input {n_samples}"   # mp3 padding
    return audio


def main():
    if shutil.which("ffmpeg") is None:
        print("SKIP: ffmpeg not on PATH")
        return
    rng = np.random.default_rng(0)
    t = np.arange(3 * SR) / SR
    noise = rng.normal(scale=0.05, size=len(t))
    primary = 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0) + np.convolve(noise, [0, 0, 0.6, -0.2], mode="same")
    tmp = Path(tempfile.mkdtemp(prefix="cli_ffmpeg_"))
    try:
        write_wav(tmp / "primary.wav", primary); write_wav(tmp / "reference.wav", noise)
        write_wav(tmp / "short_reference.wav", noise[:-4800])

        r = run_cli(tmp / "primary.wav", "-o", tmp / "default.mp3")
        assert r.returncode == 0, r.stderr[-800:]
        check_mp3(tmp / "default.mp3", len(primary))
        print("PASS default single-model CLI: wav -> ffmpeg decode -> DFN3 -> mp3")

        cfg = ROOT / "configs" / "frozen_v1.json"
        if not (ROOT / "checkpoints" / "e1_ft_v1" / "config.ini").exists():
            print("SKIP expert-pipeline CLI tests: place the exports in model/checkpoints/ (see SIH_MODEL.md)")
            return
        r = run_cli(tmp / "primary.wav", "-o", tmp / "expert.mp3", "--model_config", cfg, "--reference", tmp / "reference.wav")
        assert r.returncode == 0, r.stderr[-800:]
        assert "expert pipeline loaded" in r.stdout and "E0" in r.stdout, r.stdout[-400:]
        expert = check_mp3(tmp / "expert.mp3", len(primary))
        print("PASS expert CLI with reference: " + next(l for l in r.stdout.splitlines() if "expert pipeline loaded" in l).strip())

        r = run_cli(tmp / "primary.wav", "-o", tmp / "expert_noref.mp3", "--model_config", cfg)
        assert r.returncode == 0, r.stderr[-800:]
        noref = check_mp3(tmp / "expert_noref.mp3", len(primary))
        n = min(len(expert), len(noref))
        assert not np.array_equal(expert[:n], noref[:n]), "reference had no effect on the output"
        print("PASS expert CLI without reference (canceller bypassed); output differs from the with-reference run")

        r = run_cli(tmp / "primary.wav", "-o", tmp / "bad.mp3", "--model_config", cfg, "--reference", tmp / "short_reference.wav")
        assert r.returncode != 0 and "reference length" in (r.stderr + r.stdout), "length-mismatched reference accepted"
        print("PASS mismatched reference length rejected by the CLI")

        (tmp / "in").mkdir(); (tmp / "ref").mkdir()
        for name in ("a.wav", "b.wav"):
            write_wav(tmp / "in" / name, primary); write_wav(tmp / "ref" / name, noise)
        r = run_cli(tmp / "in", "-o", tmp / "out", "--model_config", cfg, "--reference", tmp / "ref")
        assert r.returncode == 0 and "2/2 files written" in r.stdout, (r.stdout[-400:], r.stderr[-400:])
        for name in ("a.mp3", "b.mp3"):
            check_mp3(tmp / "out" / name, len(primary))
        print("PASS batch folder mode with a reference folder (2/2 files)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
