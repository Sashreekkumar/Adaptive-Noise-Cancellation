"""Model-side pipeline integration tests (synthetic audio; E0-only test needs no checkpoints; the rest needs model/checkpoints/).
Run from model/:  python tests/test_pipeline.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sih_model import ModelConfig, ModelInput, ModelPipeline  # noqa: E402
from sih_model.config import CancellerSpec, ExpertSpec, GateSpec  # noqa: E402
from sih_model.dfn3 import enhance_waveform  # noqa: E402

SR = 48000
rng = np.random.default_rng(0)


def audio(seconds: float = 2.0):
    t = np.arange(int(seconds * SR)) / SR
    speech = 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    noise = rng.normal(scale=0.05, size=len(t))
    primary = speech + np.convolve(noise, [0, 0, 0.6, -0.2], mode="same")
    return primary.astype(np.float32), noise.astype(np.float32)


def main() -> None:
    primary, reference = audio()

    # 1. E0 only, gate off, canceller off == DeepFilterNet3 enhance (bit-exact)
    cfg = ModelConfig([ExpertSpec("E0", "pretrained")], GateSpec(False), CancellerSpec(False))
    p0 = ModelPipeline(cfg)
    out = p0.process(ModelInput(primary))
    ref_out = enhance_waveform(p0.bank.models[0], p0.bank.state, primary).astype(np.float32)
    assert np.array_equal(out.audio.astype(np.float32), ref_out) and out.telemetry["gate"].startswith("bypassed")
    print("PASS E0-only pipeline == df.enhance.enhance bit-exactly")

    if not (ROOT / "checkpoints" / "e1_ft_v1" / "config.ini").exists():
        print("SKIP checkpoint-dependent tests: place the exports in model/checkpoints/ (see SIH_MODEL.md)")
        return

    # 2. frozen architecture: E0 + E1 + E2, 3-expert gate (untrained -> uniform), canceller with reference
    frozen = ModelPipeline(ModelConfig.load(ROOT / "configs" / "frozen_v1.json"))
    out = frozen.process(ModelInput(primary, reference))
    tel = out.telemetry
    assert len(out.audio) == len(primary) and np.isfinite(out.audio).all() and tel["experts"] == ["E0", "E1", "E2"]
    assert tel["gate"].startswith("UNTRAINED") and all(abs(w - 1 / 3) < 1e-6 for w in tel["gate_mean_weights"].values())
    assert {"bin_guard", "frame_guard", "ref_bypass", "low_y_skip_bins", "adaptation_bins"} <= set(tel["canceller"])
    print(f"PASS frozen_v1 (E0+E1+E2, gate {tel['gate']}): length/finite ok, weights {tel['gate_mean_weights']}, "
          f"canceller {tel['canceller']}")

    # 3. no reference -> canceller bypassed == canceller disabled (exact); dead reference -> same output
    no_ref = frozen.process(ModelInput(primary)).audio
    frozen.cfg.canceller.enabled = False
    disabled = frozen.process(ModelInput(primary, reference)).audio
    frozen.cfg.canceller.enabled = True
    dead = frozen.process(ModelInput(primary, np.zeros_like(reference)))
    assert np.array_equal(no_ref, disabled) and np.array_equal(dead.audio, no_ref)
    assert dead.telemetry["canceller"]["ref_bypass_dead"] == dead.telemetry["frames"]
    print("PASS reference None == canceller disabled == dead reference (exact); dead frames all bypassed")

    # 4. clipped reference frames are bypassed and counted
    clipped_ref = reference.copy(); clipped_ref[SR // 2: SR // 2 + 4800] = 1.0
    t = frozen.process(ModelInput(primary, clipped_ref)).telemetry["canceller"]
    assert t["ref_bypass_clipped"] >= 10, t
    print(f"PASS clipped reference bypass counted ({t['ref_bypass_clipped']} frames)")

    # 5. switching experts / gate: E2 disabled + trained 2-expert gate loads; 3 experts + 2-expert gate is rejected
    two = ModelPipeline(ModelConfig.load(ROOT / "configs" / "e0_e1_gate_v1.json"))
    t2 = two.process(ModelInput(primary, reference)).telemetry
    assert t2["experts"] == ["E0", "E1"] and t2["gate"].startswith("trained") and abs(sum(t2["gate_mean_weights"].values()) - 1) < 1e-5
    bad = ModelConfig.load(ROOT / "configs" / "frozen_v1.json")
    bad.gate.checkpoint = str(ROOT / "checkpoints" / "gate_v1_e0_e1.pt")
    try:
        ModelPipeline(bad)
        raise AssertionError("2-expert gate checkpoint accepted for 3 experts")
    except ValueError as error:
        print(f"PASS E0+E1 with trained gate_v1 weights {t2['gate_mean_weights']}; mismatched gate rejected: {str(error)[:70]}")

    # 6. interface validation: wrong sample rate / shape rejected
    for bad_input in (ModelInput(primary, sample_rate=16000), ModelInput(primary, reference[:-1])):
        try:
            frozen.process(bad_input)
            raise AssertionError("invalid input accepted")
        except ValueError:
            pass
    print("PASS input contract enforced (48 kHz only, reference shape == primary)")


if __name__ == "__main__":
    main()
