# Speech Enhancement R&D Pipeline - Usage Guide

This pipeline denoises speech with DeepFilterNet3, then post-processes and
scores the results against clean references. Run the stages in this order:


1. `preprocess_noisy.py`
2. `enchance_with_deepfilternet.py`
3. `post_process.py`
4. `evaluate_metrics.py`

`align_utils.py` is not a standalone script - it's a shared module
(`estimate_delay`, `apply_delay`, `si_sdr`) imported by the evaluation and
post-processing scripts so every stage uses the same GCC-PHAT delay
estimator and SI-SDR definition.

`evaluate_pipeline_stages.py` , evaluates every single step in the Speech Enhancement Pipeline and calculates SNR, SI-SDR, PESQ and STOI for each stage.

---

## 1. `preprocess_noisy.py`

Cleans up raw noisy mp3s before they go into DeepFilterNet3: removes DC
offset, high-pass filters rumble below ~70 Hz, and RMS-normalizes loudness
to a fixed target level (-26 dBFS). Writes `.wav` files.

```bash
python preprocess_noisy.py \
    --noisy_dir ./noisy \
    --output_dir ./noisy_preprocessed
```

| Flag | Default | Description |
|---|---|---|
| `--noisy_dir` | *required* | Folder of input noisy audio |
| `--output_dir` | *required* | Where preprocessed `.wav` files are written |
| `--pattern` | `*.mp3` | Glob pattern for input files |
| `--skip_dc_offset` | off | Skip DC-offset removal |
| `--skip_highpass` | off | Skip the 70 Hz 4th order high-pass butterworth filter |
| `--skip_normalize` | off | Skip RMS loudness normalization |

---

## 2. `enhance_with_deepfilternet.py`

Runs DeepFilterNet3 (48 kHz mono) over every input file and writes
enhanced `.wav` output (never re-encoded to mp3, to avoid lossy loss
before metrics).

```bash
python enhance_with_deepfilternet.py \
    --noisy_dir ./noisy_preprocessed \
    --output_dir ./enhanced \
    --pad \
    --atten_lim_db 20
```

| Flag | Default | Description |
|---|---|---|
| `--noisy_dir` | *required* | Folder of input audio (mp3 or wav) |
| `--output_dir` | *required* | Where enhanced `.wav` files are written |
| `--pattern` | `*.mp3` | Glob pattern for input files |
| `--pad` / `--no_pad` | `--pad` (on) | Use DeepFilterNet3's built-in algorithmic-delay compensation |
| `--atten_lim_db` | unset (model default) | Caps noise suppression in dB; lower values (12–24) reduce "musical noise" artifacts that can hurt PESQ |

Requires `ffmpeg` on `PATH` (mp3 decoding via librosa/audioread).

---

## 3. `postprocess_enhanced.py`

Run **after** enhancement, **before** scoring. For each enhanced file,
matched to its clean reference by filename stem, and processed in
parallel across worker processes:

1. Delay-aligns enhanced audio to clean (GCC-PHAT, via `align_utils.py`).
2. Gain-matches enhanced audio to clean (least-squares optimal scalar).
3. Optionally blends in a fraction of the original noisy signal
   (`--blend_wet`) to soften over-suppression artifacts.
4. Band-limits to the speech range (80 Hz–8000 Hz).
5. Optionally applies RMS loudness normalization as a final, independent
   layer (`--normalize_volume`).

```bash
python postprocess_enhanced.py \
    --clean_dir ./clean \
    --enhanced_dir ./enhanced \
    --noisy_dir ./noisy \
    --output_dir ./enhanced_postprocessed \
    --blend_wet 0.9 \
    --workers 8
```

| Flag | Default | Description |
|---|---|---|
| `--clean_dir` | *required* | Clean reference files |
| `--enhanced_dir` | *required* | DeepFilterNet3 output to post-process |
| `--noisy_dir` | none | Original noisy files; only needed if `--blend_wet < 1.0` |
| `--output_dir` | *required* | Where post-processed `.wav` files are written |
| `--blend_wet` | `1.0` | Wet/dry mix; `1.0` = pure enhanced, e.g. `0.9` = 90% enhanced + 10% noisy |
| `--workers` | `os.cpu_count()` | Number of parallel worker processes |
| `--normalize_volume` | off | Apply RMS loudness normalization as a final layer |
| `--target_dbfs` | `-26.0` | Target level for `--normalize_volume` |

---

## 4. `evaluate_metrics.py`

Scores SNR, SI-SDR, PESQ (wideband, 16 kHz), and STOI between clean and
enhanced files (matched by stem), with an optional noisy-vs-clean
baseline for comparison.

```bash
python evaluate_metrics.py \
    --clean_dir ./clean \
    --enhanced_dir ./enhanced_postprocessed \
    --noisy_dir ./noisy \
    --output_csv results.csv \
    --align
```

| Flag | Default | Description |
|---|---|---|
| `--clean_dir` | *required* | Clean reference files |
| `--enhanced_dir` | *required* | Files to score against clean |
| `--noisy_dir` | none | Optional: also scores noisy-vs-clean as a baseline |
| `--output_csv` | `results.csv` | Per-file output CSV path |
| `--align` | off | GCC-PHAT re-alignment at evaluation time (shares the estimator in `align_utils.py` with `postprocess_enhanced.py`) |

Prints per-file results, mean scores, and — if `--noisy_dir` is given —
mean improvement (enhanced − noisy) for each metric.

---

## 5. `evaluate_pipeline_stages.py`

Runs the same four metrics across **every named pipeline stage** in one
pass (e.g. noisy → preprocessed → enhanced → postprocessed), so you can
see what each stage buys or costs you. Only files present in `clean_dir`
**and every stage** are scored, keeping the file set identical across
stages. Applies the same GCC-PHAT alignment and least-squares gain match
to every stage before scoring (toggle off with `--no_align`/
`--no_gain_match` to see raw numbers).

```bash
python evaluate_pipeline_stages.py \
    --clean_dir ./clean \
    --stage noisy=./noisy \
    --stage preprocessed=./noisy_preprocessed \
    --stage enhanced=./enhanced \
    --stage postprocessed=./enhanced_postprocessed \
    --output_prefix pipeline_results
```

| Flag | Default | Description |
|---|---|---|
| `--clean_dir` | *required* | Clean reference files |
| `--stage NAME=DIR` | *required, repeatable* | One per pipeline stage, in order |
| `--output_prefix` | `pipeline_results` | Prefix for the two output CSVs |
| `--align` / `--no_align` | `--align` (on) | GCC-PHAT delay alignment before scoring |
| `--gain_match` / `--no_gain_match` | `--gain_match` (on) | Least-squares gain match before scoring |

Outputs:
- `{prefix}_per_file.csv` — one row per file per stage (includes estimated
  lag and gain)
- `{prefix}_summary.csv` — mean per stage + stage-over-stage delta

Must be run from the directory containing `align_utils.py` (or with it on
`PYTHONPATH`).

---

## `align_utils.py` (shared module, no CLI)

- `estimate_delay(ref, deg, max_shift)` - GCC-PHAT cross-correlation delay
  estimate between two signals, searching ±`max_shift` samples.
- `apply_delay(deg, lag)` - shifts/pads a signal by the estimated lag.
- `si_sdr(ref, est)` - scale-invariant SDR, the same underlying quantity as
  the gain-matched SNR computed elsewhere in the pipeline.

Imported by `evaluate_metrics.py`, `evaluate_pipeline_stages.py`, and
`postprocess_enhanced.py` so all three scripts agree on delay/SI-SDR
computation.

---
