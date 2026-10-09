# Multi-expert model-side pipeline (`sih_model`)

Overview, configurations, checkpoints and run commands: [README.md](README.md). This file has the module layout,
the input contract and the integration details.

```
primary ─► shared STFT ─► impulse detector ─► E1 | E2 (DFN3 experts, parallel) ─► impulse-gated fusion (gate mode "rule") ─► residual NLMS
reference ─► STFT (same framing) ─────────────────────────────────────────────────────────────────────► canceller ─► iSTFT ─► output
```
48 kHz throughout; STFT = DeepFilterNet3's own (960 window, 480 hop, 481 bins), computed once for the primary and
once for the reference. ERASE is not part of this pipeline. The recommended configuration is
`configs/e1_e2_rule_v2.json`; the earlier three-expert architecture (`frozen_v1.json`: E0 + E1 + E2 with an untrained
3-expert gate) is deprecated but still loads.

## Layout (paths relative to `model_v2_nlms/`)
| path | content |
|---|---|
| `sih_model/dfn3.py` | DFN3 loading / features / inference glue (model unchanged) |
| `sih_model/experts.py` | `load_expert` (pretrained, export dir or `.pt`), `ExpertBank` (shared STFT, frozen experts) |
| `sih_model/impulse.py` | causal per-frame impulse features `[flag, severity]` from the shared noisy STFT |
| `sih_model/fusion_gate.py` | `FusionGate`: Linear → 2-layer GRU(64) → Linear, softmax per ERB band over K experts; `impulse_rule_weights` (gate mode `rule`) |
| `sih_model/residual_canceller.py` | per-bin complex NLMS canceller with guards and telemetry |
| `sih_model/canceller_interface.py` | gate-weighted mask, reference STFT, clipping flags, `run_canceller` |
| `sih_model/interfaces.py` | `ModelInput` / `ModelOutput`: contract with the preprocessing stage |
| `sih_model/config.py` | JSON configuration (experts, gate `mode` / `use_impulse` / `checkpoint`, canceller switches) |
| `sih_model/pipeline.py` | `ModelPipeline`: wires the stages |
| `configs/e1_e2_rule_v2.json` | **recommended**: E1 + E2, gate mode `rule` (impulse flag), canceller |
| `configs/e1_only_v2.json` | E1 alone (gate bypassed), canceller: baseline |
| `configs/e1_e2_gate_v2.json` | E1 + E2 with the learned 2-expert gate `gate_v2_e1_e2.pt` (impulse features on) |
| `configs/e0_e1_gate_v1.json` | deprecated: E0 + E1 with the learned 2-expert gate `gate_v1_e0_e1.pt` |
| `configs/frozen_v1.json` | deprecated: E0 + E1 + E2, untrained 3-expert gate, canceller |
| `checkpoints/` | **not in git** (ignored). Copy here: `e1_ft_v1/`, `e2_impulse_e2a/` (DeepFilterNet export dirs: `config.ini` + `checkpoints/model_*.ckpt.best`); `gate_v2_e1_e2.pt` only for learned mode, `gate_v1_e0_e1.pt` only for the deprecated config. Verify with `python tools/verify_checkpoints.py` |
| `tests/` | pipeline + audio_cleaner integration tests, canceller / interface unit tests (synthetic audio; checkpoint-dependent parts skip when `checkpoints/` is empty) |
| `audio_cleaner/expert_enhancement.py` | bridge: lets `clean_audio_file` use this pipeline as its enhancement stage |

## Input contract (preprocessing stage → model side)
`ModelInput(primary, reference=None, sample_rate=48000, ref_valid=True)`
- `primary`: mono float `[T]`, 48 kHz, full scale ±1.0 (speech + noise microphone)
- `reference`: same length, sample-synchronous with `primary`, or `None` (noise reference microphone)
- the model side does not resample, align, normalise or select channels
Output: `ModelOutput(audio [T] 48 kHz, telemetry)`.

## Usage
```python
from sih_model import ModelConfig, ModelInput, ModelPipeline
pipe = ModelPipeline(ModelConfig.load("configs/e1_e2_rule_v2.json"))
out = pipe.process(ModelInput(primary, reference))      # out.audio, out.telemetry
```
Pass a persistent `canceller=pipe.new_canceller()` to `process` to keep the path estimate across blocks.
Switching: set `"enabled": false` on any expert, the gate or the canceller in the JSON. One expert → gate bypassed;
gate disabled with several experts → uniform weights; no reference → canceller bypassed (output = fused spectrum).
Gate `"mode": "rule"` needs exactly two enabled experts, `"use_impulse": true` and no checkpoint: band weights are
`[1 - flag, flag]` in all 32 bands (second expert on impulse-flagged frames, first expert elsewhere). Gate
`"mode": "learned"` (default) uses the `FusionGate`; it receives the impulse features only when `"use_impulse": true`,
which must match how the gate checkpoint was trained (`gate_v1`: false, `gate_v2`: true).

## Status / known limitations
- Evaluation of E1, E2, uniform fusion, the learned gate and the impulse rule on the 12 frozen pairs:
  [docs/results/c3_gate_v2_e1_e2.md](../docs/results/c3_gate_v2_e1_e2.md). The learned 2-expert gate (`gate_v2`) did
  not beat the impulse rule, so the rule is the recommended gate. The two experts are close to each other (outputs
  31-33 dB SI-SDR apart), which limits what any fusion of them can gain.
- The impulse flag also fires on frames without impulsive noise (mean E2 weight 0.110 on the continuous suite).
- The 3-expert gate of the deprecated `frozen_v1.json` is untrained (zero-initialised = exactly uniform 1/3 weights);
  2-expert gate checkpoints cannot be loaded for three experts (the pipeline rejects them).
- The canceller has only been tested with a simulated reference; speech leaking into the reference removes speech
  (passes at -34 dB leakage, fails at -20 dB in simulation). Real-mic leakage is unmeasured. The evaluation above runs
  without a reference, so the canceller is not part of those numbers.
- Offline (whole-file) processing. Algorithmic latency of DFN3 is 30–40 ms; a streaming path is not included.
- SPP estimator: in integration, not part of this pipeline yet. Jetson FP16 / INT8: not yet measured.
- Two experts = 2× DFN3 compute (2.14 M parameters each); the rule gate adds no parameters.

## Tests
Run from `model_v2_nlms/`: `python tests/test_pipeline.py`, `python tests/test_audio_cleaner_integration.py`,
`python tests/test_cli_ffmpeg.py` (needs ffmpeg on PATH; `tqdm` is optional), `python tests/test_residual_canceller.py`,
`python tests/test_canceller_interface.py`
Environment: Python 3.11, see `requirements.txt` (verified with torch 2.0.1, deepfilternet 0.5.6).

## Using it through the existing `audio_cleaner` pipeline
The default behaviour of `audio_cleaner` is unchanged (single pretrained DeepFilterNet3). Opt in with a config:
```
python run.py noisy.wav -o clean.mp3 --model_config configs/e1_e2_rule_v2.json --reference reference_mic.wav
python run.py ./noisy_dir -o ./clean_dir --model_config configs/e1_e2_rule_v2.json --reference ./reference_dir
```
```python
from audio_cleaner import clean_audio_file
r = clean_audio_file("noisy.wav", "clean.mp3", model_config="configs/e1_e2_rule_v2.json", reference_path="reference_mic.wav")
print(r.model_telemetry)   # experts, gate status/weights, canceller guard counters
```
- `loader` and `preprocessing` are unchanged. The reference goes through the same DC-removal and high-pass filter
  as the primary but is **not** RMS-normalized (a dead reference must stay dead so the canceller can bypass it).
- The reference must be sample-synchronous with the input and decode to the same number of samples (otherwise an
  error is raised; nothing is cropped or aligned).
- Without `--reference` the canceller is bypassed. `--atten_lim_db` / `--no_pad` apply only to single-model mode.
- End-to-end with real ffmpeg decode / mp3 encode is covered by `tests/test_cli_ffmpeg.py` (default mode, expert
  mode with and without a reference, mismatched reference, batch folder with a reference folder).
  Not exercised: the Gradio web UI (`--serve`) with `--model_config`.
