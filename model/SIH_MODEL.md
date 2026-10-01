# Multi-expert model-side pipeline (`sih_model`, frozen architecture v1)

```
primary ─► shared STFT ─► E0 | E1 | E2 (DFN3 experts, parallel) ─► 3-expert GRU fusion gate ─► residual NLMS
reference ─► STFT (same framing) ───────────────────────────────────────────────────────────► canceller ─► iSTFT ─► output
```
48 kHz throughout; STFT = DeepFilterNet3's own (960 window, 480 hop, 481 bins), computed once for the primary and
once for the reference. ERASE is not part of this pipeline.

## Layout
| path | content |
|---|---|
| `model/sih_model/dfn3.py` | DFN3 loading / features / inference glue (model unchanged) |
| `model/sih_model/experts.py` | `load_expert` (pretrained, export dir or `.pt`), `ExpertBank` (shared STFT, frozen experts) |
| `model/sih_model/fusion_gate.py` | `FusionGate`: Linear → 2-layer GRU(64) → Linear, softmax per ERB band over K experts |
| `model/sih_model/residual_canceller.py` | per-bin complex NLMS canceller with guards and telemetry |
| `model/sih_model/canceller_interface.py` | gate-weighted mask, reference STFT, clipping flags, `run_canceller` |
| `model/sih_model/interfaces.py` | `ModelInput` / `ModelOutput`: contract with the preprocessing stage |
| `model/sih_model/config.py` | JSON configuration (experts, gate, canceller switches) |
| `model/sih_model/pipeline.py` | `ModelPipeline`: wires the stages |
| `model/configs/frozen_v1.json` | E0 + E1 + E2, 3-expert gate, canceller |
| `model/configs/e0_e1_gate_v1.json` | E0 + E1 with the trained 2-expert gate (E2 disabled) |
| `model/checkpoints/` | **not in git** (ignored). Copy here: `e1_ft_v1/`, `e2_impulse_e2a/` (DeepFilterNet export dirs: `config.ini` + `checkpoints/model_*.ckpt.best`) and `gate_v1_e0_e1.pt` |
| `model/tests/` | pipeline + audio_cleaner integration tests, canceller / interface unit tests (synthetic audio; checkpoint-dependent parts skip when `model/checkpoints/` is empty) |
| `model/audio_cleaner/expert_enhancement.py` | bridge: lets `clean_audio_file` use this pipeline as its enhancement stage |

## Input contract (preprocessing stage → model side)
`ModelInput(primary, reference=None, sample_rate=48000, ref_valid=True)`
- `primary`: mono float `[T]`, 48 kHz, full scale ±1.0 (speech + noise microphone)
- `reference`: same length, sample-synchronous with `primary`, or `None` (noise reference microphone)
- the model side does not resample, align, normalise or select channels
Output: `ModelOutput(audio [T] 48 kHz, telemetry)`.

## Usage
```python
from sih_model import ModelConfig, ModelInput, ModelPipeline
pipe = ModelPipeline(ModelConfig.load("configs/frozen_v1.json"))
out = pipe.process(ModelInput(primary, reference))      # out.audio, out.telemetry
```
Pass a persistent `canceller=pipe.new_canceller()` to `process` to keep the path estimate across blocks.
Switching: set `"enabled": false` on any expert, the gate or the canceller in the JSON. One expert → gate bypassed;
gate disabled with several experts → uniform weights; no reference → canceller bypassed (output = fused spectrum).

## Status / known limitations (not yet benchmarked as a whole)
- **The 3-expert gate is UNTRAINED** (zero-initialised = exactly uniform 1/3 weights). Only a 2-expert gate
  (`gate_v1`, E0+E1) has been trained; its weights cannot be loaded for three experts (the pipeline rejects it).
- E2 is the tier-C impulse specialist `e2a` (step 1000). In its own evaluation it did not meet its acceptance
  criteria vs E1 (+0.11 dB on impulse windows, -0.057 PESQ on continuous noise); it is integrated here as requested.
- The canceller has only been tested with a simulated reference; speech leaking into the reference removes speech
  (passes at -34 dB leakage, fails at -20 dB in simulation). Real-mic leakage is unmeasured.
- Offline (whole-utterance) processing. Algorithmic latency of DFN3 is 30–40 ms; a streaming path is not included.
- Three experts = 3× DFN3 compute (2.14 M parameters each) + gate (≈63 k for 2 experts).

## Tests
Run from `model/`: `python tests/test_pipeline.py`, `python tests/test_audio_cleaner_integration.py`,
`python tests/test_cli_ffmpeg.py` (needs ffmpeg on PATH and `tqdm`), `python tests/test_residual_canceller.py`,
`python tests/test_canceller_interface.py`
Environment: Python 3.11, see `requirements.txt` (verified with torch 2.0.1, deepfilternet 0.5.6).

## Using it through the existing `audio_cleaner` pipeline
The default behaviour of `audio_cleaner` is unchanged (single pretrained DeepFilterNet3). Opt in with a config:
```
python run.py noisy.wav -o clean.mp3 --model_config configs/frozen_v1.json --reference reference_mic.wav
python run.py ./noisy_dir -o ./clean_dir --model_config configs/frozen_v1.json --reference ./reference_dir
```
```python
from audio_cleaner import clean_audio_file
r = clean_audio_file("noisy.wav", "clean.mp3", model_config="configs/frozen_v1.json", reference_path="reference_mic.wav")
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
