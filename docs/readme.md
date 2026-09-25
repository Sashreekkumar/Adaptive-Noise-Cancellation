# noisygen

`noisygen` is a reproducible, provenance-preserving synthetic noisy-speech dataset generator.

It takes a collection of clean speech recordings and a collection of noise recordings, automatically selects clean/noise sources, constructs composite noise, scales that noise to a requested SNR, prevents clipping, writes clean/noisy WAV pairs, and records enough metadata to reproduce and audit every generated sample.

The generator is designed around four principles:

1. **Reproducibility** — every sample and retry has a deterministic random seed.
2. **Provenance preservation** — every generated sample records its exact source files, source ranges, gains, SNR, crop, and configuration.
3. **Fault tolerance** — failed attempts are recorded rather than silently discarded, and interrupted runs can be resumed.
4. **Integrity** — audio is committed before metadata is appended, allowing the output directory to be recovered after crashes.

---

# 1. Installation

The package requires Python and the following major dependencies:

```text
numpy
scipy
soundfile
```

Install the project according to your project's packaging setup, then run:

```bash
python -m noisygen --help
```

There is also a standalone entry point:

```bash
python generate_dataset.py --help
```

Both invoke the same CLI implementation.

---

# 2. Input Directory Structure

The default input structure is:

```text
project/
│
├── wav/
│   ├── speech1.wav
│   ├── speech2.wav
│   ├── hindi/
│   │   ├── speaker001/
│   │   │   ├── 001.wav
│   │   │   └── 002.wav
│   │   └── speaker002/
│   │       └── 001.wav
│   └── ...
│
├── noise/
│   ├── gunshot/
│   │   ├── noise001.wav
│   │   └── noise002.wav
│   ├── vehicle/
│   │   ├── vehicle001.wav
│   │   └── vehicle002.wav
│   └── ...
│
└── generate_dataset.py
```

The directories do not have to be this exact shape internally.

Both `wav/` and `noise/` are scanned recursively.

The default extensions are:

```json
[".wav", ".flac"]
```

Other extensions can be added through the configuration.

---

# 3. Basic CLI Usage

Generate 1,000 training samples:

```bash
python -m noisygen --num-samples 1000
```

Equivalent standalone command:

```bash
python generate_dataset.py --num-samples 1000
```

Generate using a custom root:

```bash
python -m noisygen --root /path/to/project --num-samples 1000
```

Specify the output directory:

```bash
python -m noisygen \
    --root /path/to/project \
    --output /path/to/dataset \
    --num-samples 1000
```

Specify a configuration file:

```bash
python -m noisygen \
    --config config.json \
    --num-samples 10000
```

Override the seed:

```bash
python -m noisygen \
    --seed 12345 \
    --num-samples 10000
```

Use a specific number of worker processes:

```bash
python -m noisygen \
    --workers 8 \
    --num-samples 10000
```

---

# 4. Complete CLI Reference

## `--root`

```bash
--root PATH
```

Root directory containing the clean and noise directories.

Default:

```text
.
```

The generator expects:

```text
<root>/wav/
<root>/noise/
```

Example:

```bash
python -m noisygen --root /data/noisy_dataset
```

---

## `--output`

```bash
--output PATH
```

Directory where the generated dataset will be stored.

Default:

```text
<root>/output
```

Example:

```bash
python -m noisygen \
    --root /data/noisy_dataset \
    --output /data/generated_dataset
```

---

## `--config`

```bash
--config PATH
```

Loads a JSON configuration file.

The configuration is merged into `DEFAULT_CONFIG`.

Nested dictionaries are recursively merged.

Example:

```bash
python -m noisygen --config config.json
```

CLI overrides take precedence over configuration-file values for options that have explicit CLI arguments.

Currently:

```text
--num-samples
--seed
--probe-all
```

can override the corresponding configuration behavior.

---

## `--num-samples`

```bash
--num-samples N
```

Shortcut for:

```json
{
    "splits": {
        "train": N
    }
}
```

Example:

```bash
python -m noisygen --num-samples 50000
```

generates:

```text
train/clean/000001.wav
train/noisy/000001.wav
...
train/clean/050000.wav
train/noisy/050000.wav
```

The sample IDs are global and sequential.

---

## `--seed`

```bash
--seed N
```

Overrides:

```json
"base_seed": 12345
```

The base seed controls deterministic random selection.

Example:

```bash
python -m noisygen --seed 98765 --num-samples 10000
```

The same source inventory, configuration, generator version, and seed produce the same random decisions.

---

## `--workers`

```bash
--workers N
```

Number of worker processes.

Default:

```text
min(cpu_count, 8)
```

For example:

```bash
python -m noisygen --workers 8 --num-samples 50000
```

The main process acts as the coordinator and metadata committer while worker processes generate samples.

The generator keeps up to approximately:

```text
workers × 3
```

tasks pending.

Therefore, with:

```text
workers = 8
```

the scheduler can maintain up to:

```text
24
```

pending tasks.

---

## `--probe-all`

```bash
--probe-all
```

Validates all discovered audio sources before generation starts.

Without this option, source information is probed lazily when a file is first used.

Use:

```bash
python -m noisygen --probe-all
```

when you want source validation before generation begins.

The trade-off is a slower startup because every source must be opened/probed.

---

## `--cache-dir`

```bash
--cache-dir PATH
```

Specifies where standardised audio cache files are stored.

Default:

```text
<output>/.cache/std48k
```

The cache contains standardised 48 kHz mono NumPy arrays for sufficiently long non-48-kHz source files.

Example:

```bash
python -m noisygen \
    --cache-dir /fast_ssd/noisygen_cache
```

---

## `--resume`

```bash
--resume
```

Continue a previously interrupted generation run.

Example:

```bash
python -m noisygen \
    --resume \
    --output /data/generated_dataset
```

Before resuming, the generator checks:

```text
generator version
configuration hash
source inventory hash
```

If these differ from the previous run, generation is rejected because continuing could violate reproducibility.

---

## `--overwrite`

```bash
--overwrite
```

Deletes previously generated dataset content before starting again.

Example:

```bash
python -m noisygen \
    --overwrite \
    --num-samples 50000
```

This should be used when you intentionally want a fresh generation.

---

## `--summary-only`

```bash
--summary-only
```

Does not generate new samples.

Instead it verifies the existing output and rebuilds:

```text
metadata/dataset_summary.json
```

Example:

```bash
python -m noisygen \
    --summary-only \
    --output /data/generated_dataset
```

---

## `--reconstruct`

```bash
--reconstruct SAMPLE_ID
```

Reconstructs a generated sample from its metadata and compares the reconstruction with the stored WAV files.

Example:

```bash
python -m noisygen \
    --root /data/source \
    --output /data/generated_dataset \
    --reconstruct 000123
```

The reconstruction uses:

```text
clean source
clean start/end
noise source
noise chunks
noise RMS
relative noise gain
noise placement
composite noise scale
output gain
output subtype
```

It then reports the maximum difference between the reconstructed and stored encoded samples.

A successful reconstruction should normally report differences no greater than 1 integer LSB.

---

## `--print-default-config`

```bash
--print-default-config
```

Prints the complete built-in configuration as JSON.

Example:

```bash
python -m noisygen --print-default-config
```

This is useful as a starting point for creating your own `config.json`.

---

# 5. Configuration Reference

The default configuration is defined in `config.py`.

## Dataset splits

```json
"splits": {
    "train": 50000
}
```

Defines the number of samples in each split.

For example:

```json
"splits": {
    "train": 80000,
    "validation": 10000,
    "test": 10000
}
```

The IDs remain globally sequential:

```text
train:
000001 ...
080000

validation:
080001 ...
090000

test:
090001 ...
100000
```

---

# 6. Input Directories

```json
"clean_dir": "wav",
"noise_dir": "noise"
```

These are interpreted relative to `--root`.

For example:

```text
root/
├── speech/
└── noises/
```

can be configured as:

```json
{
    "clean_dir": "speech",
    "noise_dir": "noises"
}
```

---

# 7. Audio Extensions

```json
"audio_extensions": [
    ".wav",
    ".flac"
]
```

The source scanner recursively searches for these extensions.

Extension matching is case-insensitive.

---

# 8. Output Audio Format

```json
"output_subtype": "PCM_16"
```

Supported values:

```text
PCM_16
PCM_24
FLOAT
```

All generated files are:

```text
48,000 Hz
mono
WAV
```

The subtype controls the stored sample representation.

---

# 9. Number of Noises Per Sample

```json
"num_noises": [1, 3]
```

This means each generated sample receives a random number of noise recordings between 1 and 3 inclusive.

A fixed number can also be used:

```json
"num_noises": 2
```

Then every sample contains exactly two noise recordings.

---

# 10. Required Noise Categories

```json
"required_noise_categories": []
```

Example:

```json
"required_noise_categories": [
    "gunshot"
]
```

Every generated sample must then contain one recording from the `gunshot` category.

The configuration requires at least one additional non-required noise:

```text
num_noises >= number_of_required_categories + 1
```

For example:

```json
"required_noise_categories": ["gunshot", "vehicle"],
"num_noises": [3, 5]
```

means every sample contains:

```text
1 gunshot
+
1 vehicle
+
1–3 other noises
```

The additional noises are selected only from non-required categories.

---

# 11. Noise Category Mode

```json
"category_mode": "prefer_different"
```

Supported values:

```text
any
prefer_different
require_different
```

### `any`

Noise files are selected from the available files without requiring category diversity.

### `prefer_different`

The generator attempts to use different categories first.

If there are not enough categories, it fills the remaining selections using additional files.

### `require_different`

Every selected noise must come from a different category.

If there are not enough categories, the attempt is rejected.

---

# 12. Noise Category Level

```json
"noise_category_level": "parent"
```

Supported values:

```text
parent
top
```

Suppose the noise structure is:

```text
noise/
└── military/
    └── gunfire/
        └── recording.wav
```

With:

```json
"noise_category_level": "parent"
```

the category is:

```text
gunfire
```

With:

```json
"noise_category_level": "top"
```

the category is:

```text
military
```

---

# 13. Short Noise Policy

```json
"short_noise_policy": "loop"
```

Supported values:

```text
loop
reject
```

If a noise recording is shorter than the required sample duration:

### `loop`

The recording is repeated until enough samples are available.

### `reject`

The generation attempt is rejected.

---

# 14. Noise Placement

```json
"noise_placement": {
    "mode": "full",
    "max_start_fraction": 0.5
}
```

Supported placement modes:

```text
full
random_start
```

With:

```text
full
```

the noise begins at sample 0.

With:

```text
random_start
```

the noise begins at a randomly selected point within:

```text
0 → max_start_fraction × sample_length
```

For example:

```json
{
    "mode": "random_start",
    "max_start_fraction": 0.5
}
```

allows the noise to begin anywhere from the beginning to halfway through the generated sample.

---

# 15. Relative Noise Gain

```json
"relative_gain_db": {
    "type": "uniform",
    "min": -10.0,
    "max": 0.0
}
```

The first selected noise is always the 0 dB reference.

Every subsequent noise receives a relative gain.

Supported distribution types are:

```text
fixed
choice
uniform
```

Fixed:

```json
{
    "type": "fixed",
    "value": -5
}
```

Choice:

```json
{
    "type": "choice",
    "values": [-10, -5, 0]
}
```

Uniform:

```json
{
    "type": "uniform",
    "min": -10,
    "max": 0
}
```

---

# 16. Minimum Noise RMS

```json
"min_noise_rms": 1e-6
```

Noise segments with an RMS below this value are considered effectively silent and the attempt is rejected.

This prevents silent recordings from becoming meaningless noise sources.

---

# 17. Target SNR

```json
"snr_db": {
    "type": "uniform",
    "min": 5.0,
    "max": 20.0
}
```

Supported distributions:

```text
fixed
choice
uniform
```

Fixed:

```json
{
    "type": "fixed",
    "value": 10
}
```

Choice:

```json
{
    "type": "choice",
    "values": [0, 5, 10, 15, 20]
}
```

Uniform:

```json
{
    "type": "uniform",
    "min": 5,
    "max": 20
}
```

The selected target value is stored in:

```text
target_snr_db
```

The actual SNR measured after encoding and decoding is stored in:

```text
measured_snr_db
```

---

# 18. SNR Measurement Method

```json
"snr_method": "active_rms"
```

Supported:

```text
active_rms
full_rms
```

### `active_rms`

SNR is calculated using detected active speech regions.

### `full_rms`

SNR is calculated over the entire sample.

For speech datasets, `active_rms` prevents long silent sections from dominating the SNR measurement.

---

# 19. Activity Detection

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

The detector calculates short-time frame energy.

The frame is considered active when it is sufficiently close to the 95th-percentile frame level and above the absolute floor.

Adjacent active regions separated by less than `min_gap_ms` are merged.

Regions shorter than `min_region_ms` are discarded.

---

# 20. Clean Speech Cropping

```json
"clean_crop": {
    "min_duration_seconds": 1.0,
    "max_duration_seconds": null
}
```

Every clean recording must contain at least:

```text
1 second
```

If:

```text
max_duration_seconds = null
```

the full recording is used.

For example:

```json
"clean_crop": {
    "min_duration_seconds": 1.0,
    "max_duration_seconds": 10.0
}
```

means recordings longer than 10 seconds are randomly cropped to 10 seconds.

The crop location is deterministic because it is generated from the sample's random seed.

---

# 21. Clipping Policy

```json
"peak_limit": 0.99,
"clipping_policy": "scale"
```

Supported policies:

```text
scale
reject
```

If the generated noisy mixture exceeds `peak_limit`:

### `scale`

A single gain is applied to both clean and noisy outputs.

This preserves the SNR relationship.

### `reject`

The attempt is rejected.

---

# 22. Provenance Configuration

```json
"clean_dataset_name": null,
"noise_dataset_name": null
```

These values are stored in the metadata.

They are never automatically guessed.

---

# 23. Clean Path Template

```json
"clean_path_template": null
```

This can infer explicit metadata from the directory structure.

For example, suppose:

```text
wav/
├── hindi/
│   └── speaker001/
│       └── file.wav
```

A template can be used to map path components to metadata fields.

The implementation only accepts placeholders corresponding to known provenance fields.

For example:

```text
{clean_language}/{clean_speaker_id}/{file}
```

can infer:

```text
clean_language = hindi
clean_speaker_id = speaker001
```

The generator deliberately avoids guessing arbitrary metadata.

---

# 24. Sidecar Metadata CSV

Clean metadata:

```json
"clean_metadata_csv": "clean_metadata.csv"
```

Noise metadata:

```json
"noise_metadata_csv": "noise_metadata.csv"
```

The CSV must contain a:

```text
path
```

column.

The path is relative to the root.

For clean speech, supported metadata includes:

```text
clean_dataset
clean_speaker_id
clean_language
clean_accent
```

For noise:

```text
noise_dataset
noise_category
```

Example:

```csv
path,clean_dataset,clean_speaker_id,clean_language,clean_accent
wav/hindi/speaker01/a.wav,CommonVoice,spk01,hindi,Indian
wav/english/speaker02/b.wav,CommonVoice,spk02,english,Indian
```

Sidecar information takes precedence over inferred/configured values when a non-empty value is supplied.

---

# 25. Maximum Attempts

```json
"max_attempts_per_sample": 20
```

A sample can fail for reasons such as:

```text
silent clean speech
silent noise
invalid audio
insufficient noise length
invalid category selection
clipping
zero measured power
```

The generator retries using a different deterministic attempt seed.

After 20 failed attempts, the sample generation fails.

Every failed attempt is written to:

```text
metadata/failed_attempts.jsonl
```

---

# 26. Performance Configuration

## Source probing

```json
"probe_sources": "lazy"
```

Options:

```text
lazy
upfront
```

`lazy` is faster at startup for large datasets.

`upfront` validates all files before generation.

---

## Standardised cache

```json
"standardized_cache": true
```

When enabled, sufficiently long non-48-kHz sources are resampled once and stored as `.npy`.

---

## Cache threshold

```json
"cache_min_seconds": 20.0
```

Only sources at least 20 seconds long are disk-cached.

---

## Cache size

```json
"cache_size": 16
```

Number of shorter non-48-kHz sources retained in the worker's in-memory LRU cache.

---

## Cache directory

```json
"cache_dir": null
```

When null, the CLI uses:

```text
<output>/.cache/std48k
```

---

# 27. Generated Output Structure

A normal generated dataset looks like:

```text
output/
│
├── train/
│   ├── clean/
│   │   ├── 000001.wav
│   │   ├── 000002.wav
│   │   └── ...
│   │
│   └── noisy/
│       ├── 000001.wav
│       ├── 000002.wav
│       └── ...
│
├── validation/
│   ├── clean/
│   └── noisy/
│
├── test/
│   ├── clean/
│   └── noisy/
│
├── metadata/
│   ├── dataset_metadata.jsonl
│   ├── dataset_summary.json
│   ├── run_config.json
│   ├── source_inventory.json
│   └── failed_attempts.jsonl
│
└── .cache/
    └── std48k/
```

Only splits specified in the configuration are created.

---

# 28. Clean Output

Example:

```text
train/clean/000001.wav
```

This is the clean reference corresponding to:

```text
train/noisy/000001.wav
```

Both have:

```text
48 kHz
mono
```

and the same duration.

---

# 29. Noisy Output

Example:

```text
train/noisy/000001.wav
```

Conceptually:

```text
noisy = output_gain × (
    clean +
    composite_noise_scale × composite_noise
)
```

The composite noise itself is constructed from one or more selected noise sources.

---

# 30. Metadata

The primary metadata file is:

```text
metadata/dataset_metadata.jsonl
```

There is exactly one JSON record per successfully generated sample.

Example structure:

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

The actual records contain substantially more provenance.

---

# 31. Noise Provenance

Each noise entry records information such as:

```text
source
noise_dataset
noise_category
relative_gain_db
normalization_rms
start_sample
end_sample
start_time_seconds
looped
chunks
source_original_sample_rate
source_original_channels
source_standardized_num_samples
```

This makes the generated noise reconstructable.

---

# 32. Reconstruction Model

The metadata describes enough information to reconstruct the generated sample.

For each noise:

```text
noise segment
    ↓
divide by normalization RMS
    ↓
apply relative gain
    ↓
place at destination position
    ↓
sum into composite noise
```

Then:

```text
mixture =
    clean +
    composite_noise_scale × composite_noise
```

Finally:

```text
clean_output = output_gain × clean

noisy_output =
    output_gain × mixture
```

This is implemented by `reconstruct.py`.

---

# 33. `dataset_summary.json`

The summary contains dataset-level statistics.

Examples include:

```text
total_samples
samples_per_split
total_duration_hours
sample_rate
channels
output_subtype

number_of_clean_source_files
number_of_clean_source_files_used

number_of_noise_source_files
number_of_noise_source_files_used

number_of_unique_clean_speakers

number_of_noise_categories
number_of_noise_categories_used

snr_min
snr_max
snr_mean

target_snr_min
target_snr_max
target_snr_mean

snr_error_mean_db
snr_error_max_abs_db

snr_distribution
noise_count_distribution

clean_source_usage
noise_source_usage
noise_category_usage

samples_with_clipping_prevention_gain
noise_usages_with_looping

unreadable_source_files
integrity
```

This makes the generated dataset auditable without parsing the entire JSONL manually.

---

# 34. `run_config.json`

This file records the effective configuration and reproducibility information.

It contains:

```text
generator_version
config
config_hash
source_inventory_hash
root
number_of_clean_source_files
number_of_noise_source_files
```

The configuration hash excludes settings that do not affect generated audio/metadata semantics, such as:

```text
splits
summary_snr_bin_db
standardized_cache
cache_min_seconds
cache_dir
cache_size
```

This allows some operational settings to change without invalidating a resume.

---

# 35. `source_inventory.json`

This file records every discovered source.

It contains separate clean and noise inventories.

For every source, information includes:

```text
path
size_bytes
metadata
```

Unreadable sources are also explicitly recorded.

Example:

```json
{
    "clean": [
        {
            "path": "wav/a.wav",
            "size_bytes": 1234567
        }
    ],
    "noise": [
        {
            "path": "noise/gunshot/a.wav",
            "size_bytes": 234567
        }
    ],
    "unreadable": {
        "clean": [],
        "noise": []
    }
}
```

---

# 36. `failed_attempts.jsonl`

Failed generation attempts are never silently discarded.

Each record contains:

```text
sample_id
attempt
random_seed
error_type
error
```

For example:

```json
{
    "sample_id": "000042",
    "attempt": 3,
    "random_seed": 123456,
    "error_type": "SampleRejected",
    "error": "noise segment is (near) silent"
}
```

This is especially useful for diagnosing dataset-generation problems.

---

# 37. Reproducibility

Randomness is deterministic at the sample-attempt level.

The seed is derived from:

```text
base_seed
sample_id
attempt
```

using NumPy's `SeedSequence`.

Therefore:

```text
sample 000001 attempt 0
```

and:

```text
sample 000001 attempt 1
```

have different deterministic seeds.

The generator records the actual seed in metadata.

Random decisions include:

```text
clean source selection
clean crop position
number of noises
noise selection
noise category selection
noise segment position
noise looping position
relative noise gains
target SNR
```

This means generation is not simply controlled by one continuously advancing global RNG.

---

# 38. Source Standardisation

All generated samples use:

```text
48,000 Hz
mono
```

If the input source is already 48 kHz, only the required channel conversion is performed.

If it has another sample rate, it is resampled using:

```text
scipy.signal.resample_poly
```

Multi-channel audio is downmixed using the channel mean.

The generator checks for non-finite values and rejects files containing:

```text
NaN
Inf
```

---

# 39. Noise RMS Normalisation

Before SNR scaling, every selected noise segment is normalised by its RMS:

```text
normalised_noise = noise / noise_rms
```

For additional noises, a relative gain is then applied:

```text
noise_i =
    normalised_noise × 10^(relative_gain_db / 20)
```

The first noise is the 0 dB reference.

All resulting noise signals are added together to form the composite noise.

---

# 40. SNR Scaling

Let:

```text
Pclean = mean(clean_active²)

Pnoise = mean(composite_noise_active²)
```

and let the requested SNR be:

```text
SNRtarget
```

The scaling factor is:

```text
scale =
sqrt(
    Pclean /
    (Pnoise × 10^(SNRtarget / 10))
)
```

The noisy signal is then:

```text
noisy = clean + scale × composite_noise
```

This means the requested SNR is imposed after the individual noise recordings have been combined.

The generator subsequently writes the WAV files and measures SNR again on the decoded files.

Therefore metadata contains both:

```text
target_snr_db
```

and:

```text
measured_snr_db
```

along with:

```text
snr_error_db
```

---

# 41. Clipping Prevention

The mixture peak is checked before writing.

If:

```text
peak <= peak_limit
```

then:

```text
output_gain = 1
```

Otherwise, when:

```text
clipping_policy = scale
```

the gain becomes:

```text
output_gain =
    peak_limit / peak
```

The same gain is applied to both clean and noisy signals:

```text
clean_output = clean × output_gain

noisy_output = noisy × output_gain
```

Applying the same gain to both signals preserves their relative SNR.

---

# 42. Crash Recovery

The generator uses a temporary directory:

```text
output/.tmp/
```

A worker first writes:

```text
.tmp/000001_clean.wav
.tmp/000001_noisy.wav
```

The committer then:

```text
1. moves clean WAV into its final location
2. moves noisy WAV into its final location
3. appends metadata JSON to dataset_metadata.jsonl
```

This ordering is intentional.

On startup, `recover_state()`:

```text
repairs a torn final JSONL line
        ↓
reads existing metadata
        ↓
determines which audio files are expected
        ↓
removes orphan audio files
        ↓
clears temporary files
        ↓
continues generation
```

This provides protection against interrupted runs.

---

# 43. Resume Behaviour

A resume is not simply:

```text
continue counting files
```

The generator verifies that the existing dataset was produced under compatible conditions.

It compares:

```text
generator_version
config_hash
source_inventory_hash
```

If source files or semantic configuration changed, resume is rejected.

This is important because otherwise sample IDs could point to different source material than they did during the original generation.

---

# 44. Typical Complete Workflow

A typical generation workflow is:

```bash
python -m noisygen \
    --root /data/noisygen \
    --output /data/noisygen/output \
    --config config.json \
    --workers 8
```

After completion:

```text
output/
├── train/
│   ├── clean/
│   └── noisy/
│
└── metadata/
    ├── dataset_metadata.jsonl
    ├── dataset_summary.json
    ├── run_config.json
    ├── source_inventory.json
    └── failed_attempts.jsonl
```

To inspect the default configuration:

```bash
python -m noisygen --print-default-config
```

To rebuild the summary:

```bash
python -m noisygen \
    --summary-only \
    --output /data/noisygen/output
```

To resume an interrupted run:

```bash
python -m noisygen \
    --resume \
    --root /data/noisygen \
    --output /data/noisygen/output \
    --config config.json
```

To verify reconstruction of sample `000123`:

```bash
python -m noisygen \
    --root /data/noisygen \
    --output /data/noisygen/output \
    --reconstruct 000123
```

---

# 45. Exit Codes

The CLI uses the following general behavior:

```text
0    successful generation / verification
1    generation or integrity failure
2    configuration / setup error
130  interrupted with Ctrl-C
```

A non-zero exit code should be treated as a signal that the dataset requires inspection.

---

# 46. Module Responsibilities

```text
constants.py
    Version, sample rate, channels, metadata field names.

errors.py
    ConfigError and SampleRejected.

util.py
    Logging and periodic progress ticker.

config.py
    Default configuration, validation, distributions and
    configuration hashing.

sources.py
    Source discovery, probing, metadata, sidecars and
    noise categorisation.

audio.py
    Audio standardisation, resampling, caching, slicing,
    activity detection and PCM encoding/decoding.

selection.py
    All deterministic random selection and noise chunk planning.

sample.py
    Construction of one complete clean/noisy sample and
    its metadata record.

worker.py
    Process-pool worker initialisation and retry handling.

storage.py
    JSONL storage, crash recovery, verification and committing.

runner.py
    Multi-process orchestration and progress reporting.

summary.py
    Dataset-level statistics.

reconstruct.py
    Reconstruction and stored-audio verification.

cli.py
    Command-line interface and overall orchestration.

__main__.py
    Enables `python -m noisygen`.

generate_dataset.py
    Standalone executable entry point.
```

---

# 47. High-Level Data Transformation

The generator can be viewed mathematically as:

```text
clean source
    │
    ├── standardise → 48 kHz mono
    │
    └── crop
         │
         ▼
      clean signal
         │
         │
noise source 1 ──┐
noise source 2 ──┤
noise source 3 ──┤
                 ▼
        normalise individual RMS
                 │
                 ▼
        apply relative gains
                 │
                 ▼
        composite noise
                 │
                 ▼
       target-SNR scaling
                 │
                 ▼
      clean + scaled noise
                 │
                 ▼
         clipping prevention
                 │
                 ▼
       clean WAV + noisy WAV
                 │
                 ▼
        decoded verification
                 │
                 ▼
       metadata + provenance
```

---

# 48. Key Design Property

The most important architectural property is that the generated audio is not treated as an opaque artifact.

For each sample, the metadata records the information needed to answer:

```text
Which clean file was used?

Which part of that clean file was used?

Which noise files were used?

Which parts of those noise files were used?

Where was each noise placed?

Was it looped?

What was its original RMS?

What relative gain was applied?

What target SNR was selected?

What SNR was actually measured?

What common output gain was applied?

Which random seed generated the sample?

Which generator version produced it?
```

That provenance makes the dataset auditable and allows individual samples to be reconstructed independently.
