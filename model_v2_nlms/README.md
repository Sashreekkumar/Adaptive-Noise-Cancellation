# model_v2_nlms: dual-expert speech enhancement with an impulse-rule gate (SIH PS 26052)

The main model of this repository. It takes a noisy 48 kHz speech recording (optionally with a noise-reference
microphone) and returns the enhanced speech.

```
primary ──► shared STFT ──► impulse detector ──► E1 + E2 ──► impulse-gated fusion ──► residual NLMS ──► iSTFT ──► output
            (DFN3, 960/480)  (causal flag per    (frozen     (gate mode "rule")      (mask-gated
                              frame)              DFN3s)                              adaptation)
reference ─► STFT (same framing) ───────────────────────────────────────────────────────┘
```

| stage | what it does | code |
|---|---|---|
| shared STFT | one DeepFilterNet3 STFT of the primary (960 window, 480 hop, 481 bins), reused by every stage | `sih_model/experts.py` |
| impulse detector | causal per-frame flag: frame energy more than 13 dB above a slow floor, held 3 frames | `sih_model/impulse.py` |
| E1 + E2 | two frozen DeepFilterNet3 experts run on the same STFT: E1 = general fine-tune, E2 = impulse specialist | `sih_model/experts.py`, `sih_model/dfn3.py` |
| impulse-gated fusion | gate mode `rule`: E2 in all 32 ERB bands on flagged frames, E1 elsewhere. No trained weights | `sih_model/fusion_gate.py`, `sih_model/pipeline.py` |
| residual NLMS | per-bin complex NLMS against the reference microphone; adaptation is gated by the fused mask and guarded (clipped / dead reference, energy guards). Bypassed when there is no reference | `sih_model/residual_canceller.py`, `sih_model/canceller_interface.py` |
| iSTFT | DeepFilterNet3 synthesis back to 48 kHz audio | `sih_model/experts.py` |

Module-level details and the input contract are in [SIH_MODEL.md](SIH_MODEL.md); how the checkpoints were trained is
in [TRAINING.md](TRAINING.md); the `audio_cleaner` wrapper (decode, pre/post-processing, MP3 export, web UI) is in
[MODEL.md](MODEL.md).

## Results

Evaluation on the 12 frozen pairs, all systems, per-file numbers and caveats:
[docs/results/c3_gate_v2_e1_e2.md](../docs/results/c3_gate_v2_e1_e2.md).

## Configurations (`configs/`)

| config | experts | gate | status |
|---|---|---|---|
| `e1_e2_rule_v2.json` | E1 + E2 | mode `rule` (impulse flag) | **recommended** |
| `e1_only_v2.json` | E1 | bypassed (single expert) | baseline for comparisons and demos |
| `e1_e2_gate_v2.json` | E1 + E2 | mode `learned`, `gate_v2_e1_e2.pt`, impulse features on | kept for comparison; it did not beat the rule in the evaluation |
| `e0_e1_gate_v1.json` | E0 (pretrained) + E1 | learned, `gate_v1_e0_e1.pt` | **deprecated** (legacy; was not better than E1 alone) |
| `frozen_v1.json` | E0 + E1 + E2 | learned, 3 experts, **untrained** (uniform weights) | **deprecated** (legacy architecture v1) |

Gate options in the JSON: `"mode": "learned" | "rule"` (default `learned`), `"use_impulse": true | false` (default
`false`; must be `true` for `rule` and for gates trained with impulse features), `"checkpoint"` (learned mode only).
Every stage can be switched off with `"enabled": false`.

## Checkpoints

Checkpoints are **not in git**; they are shared separately. Put them in `model_v2_nlms/checkpoints/` (git-ignored):

| path | needed for |
|---|---|
| `checkpoints/e1_ft_v1/` (`config.ini`, `checkpoints/model_2000.ckpt.best`) | every expert config |
| `checkpoints/e2_impulse_e2a/` (`config.ini`, `checkpoints/model_1000.ckpt.best`) | every config that enables E2 |
| `checkpoints/gate_v2_e1_e2.pt` | learned mode only (`e1_e2_gate_v2.json`) |
| `checkpoints/gate_v1_e0_e1.pt` | deprecated `e0_e1_gate_v1.json` only |

Verify a local copy against the SHA-256 values in `checkpoints_manifest.json`:
```
python tools/verify_checkpoints.py
```
It lists every file in the manifest as `OK`, `MISSING` or `HASH MISMATCH`; files you did not copy because you do not
use their config are reported as `MISSING`.

## How to run

Environment: Python 3.11, `pip install -r requirements.txt` (verified with torch 2.0.1 and deepfilternet 0.5.6), and
`ffmpeg` on PATH. Evaluation and training also need `requirements-train.txt` (`pesq`, `pystoi`, `soundfile`).
Run everything from `model_v2_nlms/`.

Enhance a file (add `--reference` to use the residual canceller; without it the canceller is bypassed):
```
python run.py noisy.wav -o clean.mp3 --model_config configs/e1_e2_rule_v2.json
python run.py noisy.wav -o clean.mp3 --model_config configs/e1_e2_rule_v2.json --reference reference_mic.wav
```
From Python:
```python
from sih_model import ModelConfig, ModelInput, ModelPipeline
pipe = ModelPipeline(ModelConfig.load("configs/e1_e2_rule_v2.json"))
out = pipe.process(ModelInput(primary, reference))      # out.audio, out.telemetry
```

Evaluate E1 alone, E2 alone, uniform fusion, the learned gate and the impulse rule on the frozen pairs:
```
python scripts/eval_gate.py --e0 checkpoints/e1_ft_v1 --e1 checkpoints/e2_impulse_e2a --gate checkpoints/gate_v2_e1_e2.pt --tag <new tag> --output-root <dir>
```
The frozen pairs and the default output root are paths on the author's machine (`SUITES` in `scripts/eval_b0_b1.py`).

Tests (plain scripts; each prints `PASS` / `SKIP` lines and exits non-zero on failure):
```
python tests/test_pipeline.py
python tests/test_audio_cleaner_integration.py
python tests/test_cli_ffmpeg.py
python tests/test_canceller_interface.py
python tests/test_residual_canceller.py
```
The checkpoint-dependent parts skip when `checkpoints/` is empty, the CLI test skips without `ffmpeg`, and the
rule-vs-evaluation parity check in `test_pipeline.py` skips without the frozen pairs.

## Status

- **Recommended configuration:** `e1_e2_rule_v2.json`. The learned gate is trained and loadable but is not the default.
- **SPP estimator:** in integration; not part of this pipeline yet.
- **Jetson FP16 / INT8:** not yet measured. All numbers so far are from FP32 PyTorch on a desktop CPU.
- **Processing mode:** offline, whole-file processing today. The impulse detector is causal, but there is no
  streaming path.
- **Residual canceller:** tested with a simulated reference only; the evaluation in `docs/results/` runs without a
  reference, so the canceller is not part of those numbers.
