# Speech Denoising Monorepo

Adaptive noise cancellation for SIH PS 26052. The main model is in `model_v2_nlms/`; the dataset generator is in
`dataset_v2/`.

| Folder | Package | Purpose |
|---|---|---|
| `model_v2_nlms/` | `sih_model`, `audio_cleaner` | **Main model.** Dual-expert DeepFilterNet3 pipeline (E1 + E2) with impulse-rule fusion and a residual NLMS canceller, plus the `audio_cleaner` library / CLI / web UI around it. See [model_v2_nlms/README.md](model_v2_nlms/README.md) |
| `dataset_v2/` | `noisygen` | Modular generator for paired clean/noisy speech datasets with full metadata |
| `docs/` | | Documentation site; evaluation write-ups in `docs/results/` ([c3_gate_v2_e1_e2](docs/results/c3_gate_v2_e1_e2.md)) |
| `model/`, `model_v2/` | | Earlier versions of the model folder, superseded by `model_v2_nlms/` |
| `dataset/`, `train/`, `onnx/`, `legacy/` | | Earlier dataset generator, training / evaluation scripts, ONNX export and legacy code |

```
.
├── pyproject.toml        # installs both packages
├── requirements.txt      # union of both projects' dependencies
├── README.md
├── model_v2_nlms/        # main model: sih_model + audio_cleaner (see its own README for details)
├── dataset_v2/           # noisygen (see its own README for details)
└── docs/                 # documentation and results
```

## Install

```bash
pip install -r requirements.txt     # or: pip install -e .
```

`ffmpeg` is required (system binary): `sudo apt install ffmpeg` / `brew install ffmpeg`.
Python >= 3.9.

Each subproject can also be installed on its own from its folder (`pip install -e model_v2_nlms`, `pip install -e dataset_v2`).

---

## 1. audio_cleaner (`model_v2_nlms/`)

Takes a noisy recording (mp3/wav/m4a/flac/ogg) and writes a cleaned mp3. By default it runs a single pretrained
DeepFilterNet3; with `--model_config configs/e1_e2_rule_v2.json` the enhancement stage is the dual-expert pipeline
(model checkpoints are shared separately; see [model_v2_nlms/README.md](model_v2_nlms/README.md)).

```
load (ffmpeg, 48 kHz mono) → preprocess (DC removal, HPF, RMS norm)
  → DeepFilterNet3 → postprocess (optional dry/wet blend, band-pass, norm, clip)
  → export (ffmpeg → mp3)
```

Library:

```python
from audio_cleaner import clean_audio_file

result = clean_audio_file("noisy.mp3", "clean.mp3", normalize_volume=True)
print(result.latency_seconds, result.rtf, result.stage_seconds)
```

CLI:

```bash
python run.py noisy.mp3                          # -> noisy_clean.mp3
python run.py noisy.wav -o cleaned/output.mp3
python run.py ./noisy_recordings -o ./cleaned    # batch
python run.py --serve --port 8000                # web UI
python run.py noisy.wav -o clean.mp3 --model_config configs/e1_e2_rule_v2.json   # dual-expert pipeline
```

Key flags: `--atten_lim_db`, `--blend_wet`, `--normalize_volume`, `--target_dbfs`,
`--hpf_cutoff`, `--band_low`, `--band_high`, `--no_pad`.

Defaults live in `config.py` (48 kHz, 70 Hz HPF, -26 dBFS, 80–8000 Hz band, 192k mp3).

## 2. noisygen (`dataset_v2/`)

Generates reproducible noisy-speech datasets from a folder of clean speech and a folder of noise.

```bash
python generate_dataset.py --root /path/with/wav_and_noise --num-samples 1000 --workers 4
python -m noisygen --root ... --resume | --summary-only | --reconstruct 000001 | --print-default-config
```

Features:
- Config-driven SNR / noise selection with a config hash and per-sample metadata (JSONL, crash recovery, `--resume`)
- `output_format: "mp3"` with `mp3_compression_level` (needs libsndfile >= 1.1 with mp3 support)
- `delete_clean_source_after_use` to save disk space (disables `--reconstruct` for deleted sources)
- `impulsive_noise_categories` (gunfire, artillery, explosion, ...) scattered as short events with fades
- `--reconstruct` verifies a sample can be rebuilt byte-for-byte

Modules: `constants`, `errors`, `util`, `config`, `sources`, `audio`, `selection`, `sample`,
`worker`, `storage`, `runner`, `summary`, `reconstruct`, `cli`.

## Typical workflow

1. Build a dataset with `noisygen` (clean / noise / noisy + metadata).
2. Run `audio_cleaner` on the noisy files.
3. Compare against the clean references (e.g. PESQ) to tune preprocessing/postprocessing.

## License

MIT