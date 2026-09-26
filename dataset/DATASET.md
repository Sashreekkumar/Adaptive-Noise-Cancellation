# noisygen

`noisygen` is a reproducible, provenance-preserving synthetic noisy-speech dataset generator.

It takes a collection of clean speech recordings and a collection of noise recordings, automatically selects clean/noise sources, constructs composite noise, scales that noise to a requested SNR, prevents clipping, writes clean/noisy WAV pairs, and records enough metadata to reproduce and audit every generated sample.

The generator is designed around four principles:

1. **Reproducibility** — every sample and retry has a deterministic random seed.
2. **Provenance preservation** — every generated sample records its exact source files, source ranges, gains, SNR, crop, and configuration.
3. **Fault tolerance** — failed attempts are recorded rather than silently discarded, and interrupted runs can be resumed.
4. **Integrity** — audio is committed before metadata is appended, allowing the output directory to be recovered after crashes.

## Table of contents

- [Installation](#installation)
- [Input directory structure](#input-directory-structure)
- [Basic usage](#basic-usage)
- [CLI reference](#cli-reference)
- [Configuration reference](#configuration-reference)
- [Output structure](#output-structure)
- [Metadata](#metadata)
- [Reconstruction model](#reconstruction-model)
- [Reproducibility](#reproducibility)
- [Audio pipeline](#audio-pipeline)
- [Crash recovery & resume](#crash-recovery--resume)
- [Exit codes](#exit-codes)
- [Module responsibilities](#module-responsibilities)
- [Key design property](#key-design-property)

## Installation

The package requires Python 3 and the following major dependencies:

- `numpy`
- `scipy`
- `soundfile`

Install the project according to your project's packaging setup, then run either of the equivalent entry points:

```bash
python -m noisygen --help
# or
python generate_dataset.py --help
```

## Input directory structure

The default layout is:

```text
project/
├── wav/
│   ├── speech1.wav
│   ├── speech2.wav
│   └── hindi/
│       ├── speaker001/
│       │   ├── 001.wav
│       │   └── 002.wav
│       └── speaker002/
│           └── 001.wav
├── noise/
│   ├── gunshot/
│   │   ├── noise001.wav
│   │   └── noise002.wav
│   └── vehicle/
│       ├── vehicle001.wav
│       └── vehicle002.wav
└── generate_dataset.py
```

- `wav/` and `noise/` are both scanned **recursively** — the internal directory shape is otherwise unconstrained.
- Default audio extensions: `.wav`, `.flac` (extension matching is case-insensitive; more can be added via configuration).

## Basic usage

```bash
# Generate 1,000 training samples in the current directory
python -m noisygen --num-samples 1000

# Use a custom project root
python -m noisygen --root /path/to/project --num-samples 1000

# Specify input root and output directory separately
python -m noisygen --root /path/to/project --output /path/to/dataset --num-samples 1000

# Load a configuration file
python -m noisygen --config config.json --num-samples 10000

# Override the seed
python -m noisygen --seed 12345 --num-samples 10000

# Use a specific number of worker processes
python -m noisygen --workers 8 --num-samples 10000
```

`generate_dataset.py` is a standalone entry point equivalent to `python -m noisygen`.

## CLI reference

| Flag | Argument | Default | Purpose |
|---|---|---|---|
| `--root` | `PATH` | `.` | Root directory containing `wav/` and `noise/`. |
| `--output` | `PATH` | `<root>/output` | Directory where the generated dataset is written. |
| `--config` | `PATH` | — | JSON configuration file, recursively merged into `DEFAULT_CONFIG`. |
| `--num-samples` | `N` | — | Shortcut for `{"splits": {"train": N}}`. |
| `--seed` | `N` | — | Overrides `base_seed`, the root of all deterministic random selection. |
| `--workers` | `N` | `min(cpu_count, 8)` | Number of worker processes. |
| `--probe-all` | — | off | Validate every discovered source before generation starts. |
| `--cache-dir` | `PATH` | `<output>/.cache/std48k` | Location of the standardised-audio cache. |
| `--resume` | — | off | Continue a previously interrupted run. |
| `--overwrite` | — | off | Delete existing dataset output before generating. |
| `--summary-only` | — | off | Skip generation; verify existing output and rebuild the summary. |
| `--reconstruct` | `SAMPLE_ID` | — | Reconstruct a sample from its metadata and diff it against the stored WAVs. |
| `--print-default-config` | — | — | Print the full built-in configuration as JSON. |

CLI flags override the configuration file only where an explicit CLI argument exists. Currently that's `--num-samples`, `--seed`, and `--probe-all`.

### `--root`

Root directory containing the clean and noise directories. The generator expects `<root>/wav/` and `<root>/noise/` to exist.

```bash
python -m noisygen --root /data/noisy_dataset
```

### `--output`

Directory where the generated dataset is stored (default `<root>/output`).

```bash
python -m noisygen --root /data/noisy_dataset --output /data/generated_dataset
```

### `--config`

Loads a JSON configuration file and recursively merges it into `DEFAULT_CONFIG`.

```bash
python -m noisygen --config config.json
```

### `--num-samples`

Shorthand for setting the `train` split size:

```json
{ "splits": { "train": 50000 } }
```

```bash
python -m noisygen --num-samples 50000
```

produces globally sequential IDs:

```text
train/clean/000001.wav   train/noisy/000001.wav
...
train/clean/050000.wav   train/noisy/050000.wav
```

### `--seed`

Overrides `"base_seed"` in the configuration. The same source inventory, configuration, generator version, and seed always produce the same random decisions.

```bash
python -m noisygen --seed 98765 --num-samples 10000
```

### `--workers`

Number of worker processes (default `min(cpu_count, 8)`). The main process coordinates and commits metadata while workers generate samples, and the scheduler keeps roughly `workers × 3` tasks pending — e.g. 8 workers → up to 24 pending tasks.

```bash
python -m noisygen --workers 8 --num-samples 50000
```

### `--probe-all`

Validates every discovered audio source before generation starts, rather than probing lazily on first use. Slower to start, but surfaces bad sources up front.

```bash
python -m noisygen --probe-all
```

### `--cache-dir`

Where standardised (48 kHz, mono, NumPy) audio caches are stored for sufficiently long non-48 kHz sources. Defaults to `<output>/.cache/std48k`.

```bash
python -m noisygen --cache-dir /fast_ssd/noisygen_cache
```

### `--resume`

Continues a previously interrupted generation run.

```bash
python -m noisygen --resume --output /data/generated_dataset
```

Before resuming, the generator compares the previous run's `generator version`, `configuration hash`, and `source inventory hash` against the current ones. If any differ, resume is rejected, since continuing anyway could violate reproducibility.

### `--overwrite`

Deletes previously generated dataset content before starting a fresh run.

```bash
python -m noisygen --overwrite --num-samples 50000
```

### `--summary-only`

Skips generation entirely. Verifies the existing output and rebuilds `metadata/dataset_summary.json`.

```bash
python -m noisygen --summary-only --output /data/generated_dataset
```

### `--reconstruct`

Reconstructs a generated sample from its stored metadata and compares it against the WAV files on disk, using the recorded clean source/crop, noise sources/chunks/RMS/gain/placement, composite scale, and output gain/subtype.

```bash
python -m noisygen \
    --root /data/source \
    --output /data/generated_dataset \
    --reconstruct 000123
```

It reports the maximum difference between the reconstructed and stored encoded samples. A successful reconstruction should normally differ by no more than 1 integer LSB.

### `--print-default-config`

Prints the complete built-in configuration as JSON — a good starting point for your own `config.json`.

```bash
python -m noisygen --print-default-config
```

## Configuration reference

The default configuration lives in `config.py`. Each subsection below shows the relevant key(s) and their meaning.

### Dataset splits

```json
"splits": { "train": 50000 }
```

Multiple splits share one globally sequential ID space:

```json
"splits": { "train": 80000, "validation": 10000, "test": 10000 }
```

```text
train:      000001 – 080000
validation: 080001 – 090000
test:       090001 – 100000
```

### Input directories

```json
"clean_dir": "wav",
"noise_dir": "noise"
```

Interpreted relative to `--root`. For a project laid out as `root/speech/` and `root/noises/`, use:

```json
{ "clean_dir": "speech", "noise_dir": "noises" }
```

### Audio extensions

```json
"audio_extensions": [".wav", ".flac"]
```

The source scanner recursively searches for these extensions (case-insensitive).

### Output audio format

```json
"output_subtype": "PCM_16"
```

Supported values: `PCM_16`, `PCM_24`, `FLOAT`. All generated files are 48,000 Hz, mono WAV regardless of subtype; the subtype only controls the stored sample representation.

### Number of noises per sample

```json
"num_noises": [1, 3]
```

Each sample receives a random number of noise recordings in `[1, 3]`. Use a single integer (e.g. `"num_noises": 2`) to fix the count.

### Required noise categories

```json
"required_noise_categories": []
```

```json
"required_noise_categories": ["gunshot"]
```

Every sample must then include one recording from the `gunshot` category. The configuration requires `num_noises >= number_of_required_categories + 1`, so there's always room for at least one non-required noise. For example:

```json
"required_noise_categories": ["gunshot", "vehicle"],
"num_noises": [3, 5]
```

means every sample gets 1 gunshot + 1 vehicle + 1–3 additional noises, with the additional noises drawn only from non-required categories.

### Noise category mode

```json
"category_mode": "prefer_different"
```

| Value | Behavior |
|---|---|
| `any` | Noise files are chosen without requiring category diversity. |
| `prefer_different` | The generator tries different categories first, then fills remaining slots from any category if there aren't enough. |
| `require_different` | Every selected noise must come from a different category; the attempt is rejected if there aren't enough categories. |

### Noise category level

```json
"noise_category_level": "parent"
```

Given `noise/military/gunfire/recording.wav`:

- `"parent"` → category is `gunfire`
- `"top"` → category is `military`

### Short noise policy

```json
"short_noise_policy": "loop"
```

If a noise recording is shorter than the required duration:

- `"loop"` — repeat it until enough samples are available.
- `"reject"` — reject the attempt.

### Noise placement

```json
"noise_placement": {
    "mode": "full",
    "max_start_fraction": 0.5
}
```

- `"full"` — noise begins at sample 0.
- `"random_start"` — noise begins at a random point within `0 → max_start_fraction × sample_length`. E.g. `max_start_fraction: 0.5` allows the noise to start anywhere from the beginning to the halfway point.

### Relative noise gain

```json
"relative_gain_db": {
    "type": "uniform",
    "min": -10.0,
    "max": 0.0
}
```

The first selected noise is always the 0 dB reference; every subsequent noise gets a relative gain drawn from one of:

```json
{ "type": "fixed", "value": -5 }
{ "type": "choice", "values": [-10, -5, 0] }
{ "type": "uniform", "min": -10, "max": 0 }
```

### Minimum noise RMS

```json
"min_noise_rms": 1e-6
```

Noise segments with RMS below this threshold are treated as effectively silent, and the attempt is rejected — this keeps silent recordings from becoming meaningless noise sources.

### Target SNR

```json
"snr_db": {
    "type": "uniform",
    "min": 5.0,
    "max": 20.0
}
```

Same three distribution types as relative gain (`fixed`, `choice`, `uniform`). The chosen value is stored as `target_snr_db`; the SNR actually measured after encode/decode is stored as `measured_snr_db`.

### SNR measurement method

```json
"snr_method": "active_rms"
```

- `"active_rms"` — SNR computed over detected active speech regions. Recommended for speech, since it prevents long silences from dominating the measurement.
- `"full_rms"` — SNR computed over the entire sample.

### Activity detection

```json
"activity_detection": {
    "frame_ms": 25.0,
    "hop_ms": 10.0,
    "threshold_db_below_p95": 25.0,
    "absolute_floor_dbfs": -70.0,
    "min_gap_ms": 200.0,
    "min_region_ms": 100.0
}
```

Short-time frame energy is computed; a frame is "active" when it's close enough to the 95th-percentile frame level and above the absolute floor. Adjacent active regions closer than `min_gap_ms` are merged, and regions shorter than `min_region_ms` are discarded.

### Clean speech cropping

```json
"clean_crop": {
    "min_duration_seconds": 1.0,
    "max_duration_seconds": null
}
```

Every clean recording must be at least 1 second long. With `max_duration_seconds: null`, the full recording is used; setting it to e.g. `10.0` randomly crops longer recordings to 10 seconds. The crop location is deterministic, derived from the sample's random seed.

### Clipping policy

```json
"peak_limit": 0.99,
"clipping_policy": "scale"
```

If the mixture peak exceeds `peak_limit`:

- `"scale"` — apply a single gain to both clean and noisy outputs, preserving their SNR relationship.
- `"reject"` — reject the attempt.

### Provenance configuration

```json
"clean_dataset_name": null,
"noise_dataset_name": null
```

Stored directly in metadata; never inferred automatically.

### Clean path template

```json
"clean_path_template": null
```

Maps path components to known provenance fields. For example, given `wav/hindi/speaker001/file.wav`, the template:

```text
{clean_language}/{clean_speaker_id}/{file}
```

infers `clean_language = hindi` and `clean_speaker_id = speaker001`. Only placeholders matching known provenance fields are accepted — the generator won't guess arbitrary metadata.

### Sidecar metadata CSV

```json
"clean_metadata_csv": "clean_metadata.csv",
"noise_metadata_csv": "noise_metadata.csv"
```

Each CSV must contain a `path` column, relative to `--root`. Supported metadata columns:

- Clean: `clean_dataset`, `clean_speaker_id`, `clean_language`, `clean_accent`
- Noise: `noise_dataset`, `noise_category`

```csv
path,clean_dataset,clean_speaker_id,clean_language,clean_accent
wav/hindi/speaker01/a.wav,CommonVoice,spk01,hindi,Indian
wav/english/speaker02/b.wav,CommonVoice,spk02,english,Indian
```

Sidecar values take precedence over inferred or configured values whenever a non-empty value is supplied.

### Maximum attempts

```json
"max_attempts_per_sample": 20
```

A sample attempt can fail due to silent clean speech, silent noise, invalid audio, insufficient noise length, invalid category selection, clipping, or zero measured power. Each retry uses a new deterministic attempt seed; after 20 failures the sample generation fails outright. Every failed attempt is logged to `metadata/failed_attempts.jsonl`.

### Performance

```json
"probe_sources": "lazy",
"standardized_cache": true,
"cache_min_seconds": 20.0,
"cache_size": 16,
"cache_dir": null
```

| Key | Meaning |
|---|---|
| `probe_sources` | `"lazy"` (fast startup, validate on first use) or `"upfront"` (validate everything first). |
| `standardized_cache` | When enabled, long non-48 kHz sources are resampled once and cached to `.npy`. |
| `cache_min_seconds` | Only sources at least this long are disk-cached (default 20s). |
| `cache_size` | Number of shorter non-48 kHz sources kept in each worker's in-memory LRU cache. |
| `cache_dir` | When `null`, the CLI uses `<output>/.cache/std48k`. |

## Output structure

```text
output/
├── train/
│   ├── clean/
│   │   ├── 000001.wav
│   │   └── ...
│   └── noisy/
│       ├── 000001.wav
│       └── ...
├── validation/
│   ├── clean/
│   └── noisy/
├── test/
│   ├── clean/
│   └── noisy/
├── metadata/
│   ├── dataset_metadata.jsonl
│   ├── dataset_summary.json
│   ├── run_config.json
│   ├── source_inventory.json
│   └── failed_attempts.jsonl
└── .cache/
    └── std48k/
```

Only splits present in the configuration are created.

- **Clean output** (e.g. `train/clean/000001.wav`) is the clean reference matching `train/noisy/000001.wav` — both 48 kHz mono, same duration.
- **Noisy output** is conceptually `noisy = output_gain × (clean + composite_noise_scale × composite_noise)`, where `composite_noise` is built from one or more selected noise sources.

## Metadata

### `metadata/dataset_metadata.jsonl`

One JSON record per successfully generated sample:

```json
{
    "sample_id": "000001",
    "split": "train",
    "clean_source": "wav/speaker/file.wav",
    "clean_output": "train/clean/000001.wav",
    "noisy_output": "train/noisy/000001.wav",
    "target_snr_db": 10.0,
    "measured_snr_db": 9.9998,
    "snr_error_db": -0.0002,
    "composite_noise_scale": 0.123,
    "output_gain": 0.91,
    "duration_seconds": 8.2,
    "sample_rate": 48000,
    "channels": 1,
    "output_subtype": "PCM_16",
    "random_seed": 123456789
}
```

The real records contain substantially more provenance, including a per-noise breakdown of: `source`, `noise_dataset`, `noise_category`, `relative_gain_db`, `normalization_rms`, `start_sample`, `end_sample`, `start_time_seconds`, `looped`, `chunks`, `source_original_sample_rate`, `source_original_channels`, and `source_standardized_num_samples`. This is what makes each generated noise reconstructable.

### `metadata/dataset_summary.json`

Dataset-level statistics, including:

- Totals: `total_samples`, `samples_per_split`, `total_duration_hours`, `sample_rate`, `channels`, `output_subtype`
- Source usage: `number_of_clean_source_files(_used)`, `number_of_noise_source_files(_used)`, `number_of_unique_clean_speakers`, `number_of_noise_categories(_used)`
- SNR stats: `snr_min/max/mean`, `target_snr_min/max/mean`, `snr_error_mean_db`, `snr_error_max_abs_db`, `snr_distribution`
- Distributions: `noise_count_distribution`, `clean_source_usage`, `noise_source_usage`, `noise_category_usage`
- Robustness: `samples_with_clipping_prevention_gain`, `noise_usages_with_looping`, `unreadable_source_files`, `integrity`

This makes the dataset auditable without parsing the full JSONL by hand.

### `metadata/run_config.json`

Records the effective configuration and reproducibility fingerprint: `generator_version`, `config`, `config_hash`, `source_inventory_hash`, `root`, `number_of_clean_source_files`, `number_of_noise_source_files`.

`config_hash` deliberately excludes settings that don't affect generated audio/metadata semantics — `splits`, `summary_snr_bin_db`, `standardized_cache`, `cache_min_seconds`, `cache_dir`, `cache_size` — so purely operational settings can change without invalidating a resume.

### `metadata/source_inventory.json`

Records every discovered source, with separate clean/noise inventories and an explicit list of unreadable sources:

```json
{
    "clean": [
        { "path": "wav/a.wav", "size_bytes": 1234567 }
    ],
    "noise": [
        { "path": "noise/gunshot/a.wav", "size_bytes": 234567 }
    ],
    "unreadable": { "clean": [], "noise": [] }
}
```

### `metadata/failed_attempts.jsonl`

Failed attempts are never silently discarded:

```json
{
    "sample_id": "000042",
    "attempt": 3,
    "random_seed": 123456,
    "error_type": "SampleRejected",
    "error": "noise segment is (near) silent"
}
```

Useful for diagnosing dataset-generation problems at scale.

## Reconstruction model

Implemented in `reconstruct.py`. For each noise:

```text
noise segment → divide by normalization RMS → apply relative gain
             → place at destination position → sum into composite noise
```

Then:

```text
mixture      = clean + composite_noise_scale × composite_noise
clean_output = output_gain × clean
noisy_output = output_gain × mixture
```

## Reproducibility

Randomness is deterministic at the sample-attempt level: each seed is derived from `base_seed`, `sample_id`, and `attempt` via NumPy's `SeedSequence`. So `sample 000001 attempt 0` and `sample 000001 attempt 1` get different, but fully deterministic, seeds — and the actual seed used is recorded in metadata.

Random decisions covered by this scheme include: clean source selection, clean crop position, number of noises, noise selection, noise category selection, noise segment position, noise looping position, relative noise gains, and target SNR. Generation is therefore not driven by a single continuously advancing global RNG.

## Audio pipeline

### Source standardisation

All generated samples are 48,000 Hz mono. Sources already at 48 kHz only undergo channel conversion; others are resampled with `scipy.signal.resample_poly`. Multi-channel audio is downmixed via the channel mean. Files containing `NaN` or `Inf` values are rejected.

### Noise RMS normalisation

Each selected noise segment is RMS-normalised, then scaled by its relative gain:

```text
normalised_noise = noise / noise_rms
noise_i = normalised_noise × 10^(relative_gain_db / 20)
```

The first noise is the 0 dB reference. All resulting noise signals are summed to form the composite noise.

### SNR scaling

Let `Pclean = mean(clean_active²)` and `Pnoise = mean(composite_noise_active²)`, with target SNR `SNRtarget`. The scaling factor is:

```text
scale = sqrt(Pclean / (Pnoise × 10^(SNRtarget / 10)))
noisy = clean + scale × composite_noise
```

The requested SNR is therefore imposed *after* combining the individual noise recordings. The generator writes the WAV files, then re-measures SNR on the decoded audio, storing both `target_snr_db` and `measured_snr_db` alongside `snr_error_db`.

### Clipping prevention

The mixture peak is checked before writing:

```text
if peak <= peak_limit:
    output_gain = 1
elif clipping_policy == "scale":
    output_gain = peak_limit / peak
```

The same `output_gain` is applied to both `clean` and `noisy`, which preserves their relative SNR.

## Crash recovery & resume

### Crash recovery

Workers write to a temporary directory first:

```text
output/.tmp/000001_clean.wav
output/.tmp/000001_noisy.wav
```

The committer then, in order:

1. Moves the clean WAV into its final location.
2. Moves the noisy WAV into its final location.
3. Appends the metadata record to `dataset_metadata.jsonl`.

This ordering is intentional — audio is committed before metadata. On startup, `recover_state()` repairs a torn final JSONL line, reads existing metadata, determines which audio files are expected, removes orphan audio files, clears temporary files, and only then continues generation.

### Resume behaviour

Resuming is not simply "continue counting files." The generator compares `generator_version`, `config_hash`, and `source_inventory_hash` against the previous run, and rejects the resume if source files or semantically relevant configuration changed — otherwise sample IDs could end up pointing at different source material than they did originally.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Successful generation / verification |
| `1` | Generation or integrity failure |
| `2` | Configuration / setup error |
| `130` | Interrupted with Ctrl-C |

A non-zero exit code should be treated as a signal that the dataset needs inspection.

## Typical complete workflow

```bash
# Generate
python -m noisygen \
    --root /data/noisygen \
    --output /data/noisygen/output \
    --config config.json \
    --workers 8

# Inspect the default configuration
python -m noisygen --print-default-config

# Rebuild the summary without regenerating
python -m noisygen --summary-only --output /data/noisygen/output

# Resume an interrupted run
python -m noisygen \
    --resume \
    --root /data/noisygen \
    --output /data/noisygen/output \
    --config config.json

# Verify reconstruction of a specific sample
python -m noisygen \
    --root /data/noisygen \
    --output /data/noisygen/output \
    --reconstruct 000123
```

Resulting output:

```text
output/
├── train/
│   ├── clean/
│   └── noisy/
└── metadata/
    ├── dataset_metadata.jsonl
    ├── dataset_summary.json
    ├── run_config.json
    ├── source_inventory.json
    └── failed_attempts.jsonl
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `constants.py` | Version, sample rate, channels, metadata field names. |
| `errors.py` | `ConfigError` and `SampleRejected`. |
| `util.py` | Logging and periodic progress ticker. |
| `config.py` | Default configuration, validation, distributions, and configuration hashing. |
| `sources.py` | Source discovery, probing, metadata, sidecars, noise categorisation. |
| `audio.py` | Standardisation, resampling, caching, slicing, activity detection, PCM encode/decode. |
| `selection.py` | Deterministic random selection and noise chunk planning. |
| `sample.py` | Construction of one complete clean/noisy sample and its metadata record. |
| `worker.py` | Process-pool worker initialisation and retry handling. |
| `storage.py` | JSONL storage, crash recovery, verification, and committing. |
| `runner.py` | Multi-process orchestration and progress reporting. |
| `summary.py` | Dataset-level statistics. |
| `reconstruct.py` | Reconstruction and stored-audio verification. |
| `cli.py` | Command-line interface and overall orchestration. |
| `__main__.py` | Enables `python -m noisygen`. |
| `generate_dataset.py` | Standalone executable entry point. |

## High-level data transformation

```text
clean source
    ├── standardise → 48 kHz mono
    └── crop
         ↓
     clean signal

noise source 1 ─┐
noise source 2 ─┤→ normalise individual RMS → apply relative gains → composite noise
noise source 3 ─┘
                                                                          ↓
                                                          clean signal + target-SNR-scaled composite noise
                                                                          ↓
                                                                 clipping prevention
                                                                          ↓
                                                        clean WAV + noisy WAV → decoded verification
                                                                          ↓
                                                                metadata + provenance
```

## Key design property

The generated audio is never treated as an opaque artifact. For every sample, the metadata records enough to answer:

- Which clean file was used, and which part of it?
- Which noise files were used, and which parts of them?
- Where was each noise placed, and was it looped?
- What was each noise's original RMS, and what relative gain was applied?
- What target SNR was selected, and what SNR was actually measured?
- What common output gain was applied?
- Which random seed, and which generator version, produced the sample?

That provenance is what makes the dataset auditable, and what allows individual samples to be independently reconstructed.