# Speech Enhancement Model Pipeline - Usage Guide

A small pipeline that takes a noisy speech recording (mp3/wav/m4a/flac/ogg) and
produces a cleaned-up mp3, using [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet)
for the actual noise removal, wrapped in pre/post-processing.

It can be used as a Python library, a CLI tool (single file or batch folder),
or a drag-and-drop web UI (Gradio).

---

## Usage

```python
from audio_cleaner import clean_audio_file

result = clean_audio_file("noisy.mp3", "clean.mp3", normalize_volume=True)
print(result.latency_seconds, result.rtf, result.stage_seconds)
```

---

## How the pipeline works

```
input file
   │
   ▼
loader.py        decode to mono float32 PCM @ 48kHz (via ffmpeg)
   │
   ▼
preprocessing.py DC-offset removal → high-pass filter → RMS normalize
   │
   ▼
enhancement.py   DeepFilterNet3 inference (the actual denoising)
   │
   ▼
postprocessing.py  optional dry/wet blend → band-pass filter → optional
                    RMS normalize → peak safety clip
   │
   ▼
exporter.py      encode to mp3 (via ffmpeg), streamed straight from memory
```

Every stage is a plain function that takes/returns a numpy array, so the
whole thing is orchestrated by one function, `pipeline.clean_audio_file()`,
which every entry point (CLI, server) calls. No entry point talks to the
individual stage modules directly.

---

## File-by-file reference

### `config.py`
Central place for every tunable default (sample rate, filter cutoffs,
loudness target, mp3 bitrate). Other modules import from here rather than
hardcoding numbers, so tuning one value only means editing one line.

| Constant | Meaning |
|---|---|
| `MODEL_SR` | DeepFilterNet3's native sample rate (48000 Hz) — everything is resampled to this |
| `HPF_CUTOFF_HZ` | Pre-enhancement high-pass filter cutoff (70 Hz) — removes rumble |
| `TARGET_DBFS` | RMS loudness normalization target (-26.0 dBFS) |
| `BAND_LOW_HZ` / `BAND_HIGH_HZ` | Post-enhancement band-pass edges (80 Hz–8000 Hz) — keeps the speech band, strips residual broadband artifacts |
| `MP3_BITRATE` | Output mp3 bitrate (192k) |

### `loader.py`
Decodes any ffmpeg-readable audio file straight to mono float32 PCM at the
model's sample rate, using a single `ffmpeg` subprocess call that pipes raw
PCM back over stdout. This replaces `librosa.load()` (which shells out to
ffmpeg via `audioread` and then re-reads the output frame-by-frame in
Python) — decoding was the dominant chunk of per-file latency, and this cuts
it to a fraction of a second.

### `preprocessing.py`
Stage 1, reference-free cleanup applied **before** DeepFilterNet3 sees the
audio:
- `remove_dc_offset()` — subtracts the mean
- `highpass()` — 4th-order Butterworth high-pass (removes non-speech rumble)
- `rms_normalize()` — normalizes to a target RMS loudness, with a peak
  safety clamp
- `preprocess()` — runs the three above in order, each individually
  skippable

### `enhancement.py`
Stage 2, the actual denoising: runs DeepFilterNet3 (`df.enhance.enhance`)
over the preprocessed audio. `pad=True` tells DeepFilterNet3 to internally
compensate for its own fixed algorithmic delay, which is why nothing
downstream needs to estimate delay against a reference.

### `postprocessing.py`
Stage 3, reference-free cleanup applied **after** DeepFilterNet3. This is
the stage that differs most from an R&D/offline version that had access to
a clean reference file (for delay estimation via GCC-PHAT and gain
matching). In deployment there's only the one noisy file, so instead:
- optional dry/wet blend with the *pre-enhancement* audio (`blend_wet`),
  to soften musical-noise / over-suppression artifacts — no delay
  estimation needed since `pad=True` already keeps enhanced and
  pre-enhancement audio sample-aligned
- `bandlimit()` — band-passes to the speech range to strip residual
  broadband noise/artifacts
- optional final RMS loudness normalization
- a peak safety clamp

### `exporter.py`
Encodes the final waveform to mp3 by piping raw PCM directly into ffmpeg's
stdin, which writes the mp3 straight to disk. No intermediate temp `.wav`
file, so there's no extra disk write+read between model output and the
final mp3.

### `model.py`
Loads DeepFilterNet3 exactly once per process and caches it in-memory
(`get_model()`), so a batch run or long-running server pays the model-load
cost once, not per file / per request. Also notes that `device` is
currently **advisory only** — DeepFilterNet manages its own device
placement internally, and moving tensors manually causes device mismatches
inside the library. On first load it also does a one-time dummy inference
to warm up CUDA kernel selection.

### `pipeline.py`
Ties every stage together into `clean_audio_file(input_path, output_path, **options)`,
the single function every entry point calls. Returns a `CleanResult`
dataclass with the output path, total latency, audio duration, real-time
factor (RTF — latency / duration; below 1.0 is faster than real-time), and
a per-stage timing breakdown (`load`, `preprocess`, `enhance`,
`postprocess`, `export`).

### `cli.py`
Command-line entry point. Supports a single file, a folder (batch mode),
or `--serve` to launch the web UI. See **CLI usage** below.

### `server.py`
Drag-and-drop web UI built with Gradio. Loads the model once at startup
and reuses it for every upload, so per-request latency after the first is
just inference time. Launched via `cli.py --serve`.

### `run.py`
Thin top-level entry point: `python run.py ...` just calls `cli.main()`.

### `__init__.py`
Exposes `clean_audio_file` as the package's public API:
```python
from audio_cleaner import clean_audio_file
```

---

## CLI usage

All commands below work either as `python run.py ...` or, if installed as a
package (see `pyproject.toml`), as `audio-cleaner ...`.

### Clean a single file
```bash
python run.py noisy.mp3
```
Writes `noisy_clean.mp3` next to the input by default.

### Clean a single file with a specific output path
```bash
python run.py noisy.wav -o cleaned/output.mp3
```

### Batch-clean a folder
```bash
python run.py ./noisy_recordings -o ./cleaned_recordings
```
Processes every `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` file in the folder.
If `-o` is omitted, output goes to `<folder>_clean` next to the input
folder. Prints a progress bar, per-file failures (without stopping the
batch), and an average latency/RTF summary at the end.

### Launch the drag-and-drop web UI
```bash
python run.py --serve
python run.py --serve --port 8000
```

### All flags
| Flag | Default | Meaning |
|---|---|---|
| `input` | — | Noisy input file or folder (omit only with `--serve`) |
| `-o, --output` | derived from input | Output mp3 file, or output folder for batch input |
| `--atten_lim_db` | none | Limit DeepFilterNet3's attenuation in dB; lower = less over-suppression |
| `--no_pad` | off | Disable DeepFilterNet3's built-in delay compensation |
| `--blend_wet` | `1.0` | Dry/wet blend; `1.0` = pure enhanced, e.g. `0.9` = 90% enhanced + 10% pre-enhancement audio |
| `--normalize_volume` | off | Apply final RMS loudness normalization |
| `--target_dbfs` | `-26.0` | RMS loudness normalization target |
| `--hpf_cutoff` | `70` | Pre-enhancement high-pass cutoff (Hz) |
| `--band_low` | `80` | Post-enhancement band-pass low edge (Hz) |
| `--band_high` | `8000` | Post-enhancement band-pass high edge (Hz) |
| `--device` | none | Advisory only — DeepFilterNet controls its own device placement |
| `--serve` | off | Launch the web UI instead of CLI mode |
| `--port` | `7860` | Port for `--serve` |



---

