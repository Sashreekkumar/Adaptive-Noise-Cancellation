# noisygen — modular noisy-speech dataset generator (v1.2.0)

Same behaviour and CLI as the original single script, split into a package.

```
python generate_dataset.py --root /path/with/wav_and_noise --num-samples 1000 --workers 4
python -m noisygen --root ... --resume | --summary-only | --reconstruct 000001 | --print-default-config
```
Requires: numpy, scipy, soundfile.

| Module | Responsibility |
|---|---|
| `constants.py` | version, `SR`, provenance field names |
| `errors.py` | `ConfigError`, `SampleRejected` |
| `util.py` | `log`, `Ticker` |
| `config.py` | `DEFAULT_CONFIG`, validation, distributions, config hash |
| `sources.py` | `Source`, scanning, lazy probing, sidecar CSVs, noise grouping |
| `audio.py` | 48 kHz mono standardisation, disk cache, slicing, activity detection, PCM encode/decode |
| `selection.py` | seeds, noise selection, chunk planning (all randomness) |
| `sample.py` | `make_sample()` – one clean/noisy pair + metadata record |
| `worker.py` | process-pool `init_worker` / `run_sample` |
| `storage.py` | JSONL I/O, crash recovery, verification, `Committer` |
| `runner.py` | multi-process generation loop |
| `summary.py` | `dataset_summary.json` |
| `reconstruct.py` | `--reconstruct` check |
| `cli.py` | argparse + `main()` |

Import graph is acyclic: constants/errors/util → config/sources → audio/selection → sample → worker → runner → cli.

## New: mp3 output

Set `"output_format": "mp3"` (default is `"wav"`). Tune size/quality with `"mp3_compression_level"`
(0.0 = highest quality/largest files, up to but excluding 1.0 = smaller/lower quality; it's VBR, so the
actual kbps depends on the content). Requires `soundfile`/`libsndfile` >= 1.1 built with mp3 support
(check with `python -c "import soundfile as sf; print('MP3' in sf.available_formats())"`).

Internally, each sample is still generated and SNR/clipping-verified as lossless PCM first (unchanged from
before), then compressed to mp3 as a final step once that's passed. `measured_snr_db` reflects that
lossless stage; `final_peak_amplitude` reflects the actual stored mp3 file. `--reconstruct` still works for
mp3 output — encoding is deterministic, so it re-runs the same encode and compares files byte-for-byte —
but it needs the original clean/noise source files to still be on disk (see the deletion feature below).

## New: delete_clean_source_after_use

Set `"delete_clean_source_after_use": true` to delete each file under `clean_dir/` once every sample in
the current run that needs it has been generated, to avoid keeping the full clean speech corpus and the
generated clean+noisy copies all on disk at once. A few things to know:

- It only deletes **clean speech** sources, never noise sources.
- If the same clean file is used by several samples, it's kept until the last of them is done, not deleted
  after the first use.
- It plays correctly with `--resume`: each run only tracks the sources its own remaining work still needs.
- `--reconstruct` can't rebuild a sample whose clean source has since been deleted — that's an unavoidable
  trade-off of not keeping the source around.
- Very occasionally, a failed sample's retry may land on a clean source that has already been deleted
  (because everything that source was originally needed for is done); that read just fails and is retried
  like any other rejected attempt, it doesn't crash the run.

## New: impulsive noise events (gunfire, explosions, ...)

By default every noise pick runs as one continuous segment through the sample (looped if the source is
shorter than needed). For inherently short, bursty sources - gunfire, artillery, explosions - that's the
wrong shape: list their categories in `"impulsive_noise_categories"`, and a pick from one of those
categories instead scatters several short excerpts of it at random (possibly overlapping) times across the
sample, each with a short fade in/out so splicing it in doesn't click:

```json
{
  "impulsive_noise_categories": ["gunfire", "artillery", "explosion"],
  "impulsive_events": {
    "count": [1, 4],
    "event_duration_seconds": [0.15, 1.5],
    "fade_ms": 5.0
  }
}
```

`noise_placement` and `short_noise_policy` only affect the remaining, non-impulsive categories - an
impulsive pick ignores them entirely (a source shorter than the requested event duration just gives a
shorter, clamped event rather than looping or being rejected). Each noise's metadata record carries
`"placement": "continuous"` or `"placement": "impulsive"`, and impulsive records list their `events`
(source/destination sample ranges) instead of `chunks`/`start_sample`. `--reconstruct` and everything else
(mp3 output, delete_clean_source_after_use, resume) work the same either way.
