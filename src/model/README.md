# audio_cleaner

Modular pipeline: one noisy audio file in -> one clean mp3 out.

## Layout

```
run.py                       # entry point
audio_cleaner/
    config.py                 # shared constants (sample rate, cutoffs, dBFS target...)
    model.py                  # loads & caches DeepFilterNet3 once per process
    preprocessing.py          # DC offset removal, high-pass, RMS normalize
    enhancement.py             # DeepFilterNet3 inference
    postprocessing.py         # dry/wet blend, band-limit, final normalize
    exporter.py                # writes final audio out as mp3 (via ffmpeg)
    pipeline.py                # clean_audio_file() -- wires the stages together
    cli.py                      # argparse CLI (single file / folder / --serve)
    server.py                  # Gradio drag-and-drop web UI
```

## Install

```
pip install deepfilternet librosa soundfile torch tqdm gradio
```
ffmpeg must be installed and on PATH.

## Usage

```
python run.py input.mp3 -o clean.mp3        # single file
python run.py ./noisy_dir -o ./clean_dir    # batch folder
python run.py --serve                       # drag-and-drop UI at localhost:7860
```

Or from Python:
```python
from audio_cleaner import clean_audio_file
clean_audio_file("input.mp3", "clean.mp3")
```

## What changed from the R&D scripts

- `align_utils.py`'s GCC-PHAT delay estimation and gain-matching are gone.
  Both required a clean reference file, which doesn't exist in production --
  there's just one noisy file. DeepFilterNet3's `pad=True` already keeps its
  output sample-aligned with its input, so no external alignment step is
  needed.
- `evaluate_metrics.py` (SNR/SI-SDR/PESQ/STOI) is dropped -- pipeline design
  is finalized, nothing to score against in production anyway.
- `postprocessing.py`'s dry/wet blend now mixes in the pre-enhancement
  (preprocessed) audio instead of a separate raw noisy file, since that's
  the only "noisy" version available in a single-file run.
- The model loads once per process and is cached (`model.get_model`), reused
  across every file in a batch run and every upload to the server.
