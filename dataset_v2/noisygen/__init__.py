"""
noisygen - reproducible, provenance-preserving synthetic noisy-speech dataset generator.

Layout expected under --root (default "."):

    wav/    clean speech, any depth            (recursively scanned)
    noise/  noise recordings, any depth        (recursively scanned; categories are auto-discovered)

Layout produced under --output (default <root>/output):

    <split>/clean/000001.wav        clean reference (48 kHz, mono)
    <split>/noisy/000001.wav        clean + scaled composite noise
    metadata/dataset_metadata.jsonl exactly one JSON record per generated sample
    metadata/dataset_summary.json   statistics computed from the JSONL
    metadata/run_config.json        effective config, hashes, source inventory hash
    metadata/source_inventory.json  every discovered source with inferred attributes
    metadata/failed_attempts.jsonl  every rejected/failed attempt (never silent)

Composite noise reconstruction (what the metadata makes possible):
    Continuous placement (the default):
        seg_i   = concat(source_i[c.source_start_sample:c.source_end_sample] for c in chunks_i)
        comp   += place(seg_i / normalization_rms_i * 10^(relative_gain_db_i/20), start_sample_i)
    Impulsive placement (noise categories listed in impulsive_noise_categories - gunfire, explosions, ...):
        for each event e in events_i:
            comp[e.destination_start_sample:e.destination_end_sample] +=
                fade(source_i[e.source_start_sample:e.source_end_sample], fade_samples_i)
                / normalization_rms_i * 10^(relative_gain_db_i/20)
    noisy   = output_gain * (clean + composite_noise_scale * comp)
    clean_output = output_gain * clean

Module map
    constants.py    version, sample rate, provenance field names
    errors.py       ConfigError, SampleRejected
    util.py         log(), Ticker
    config.py       DEFAULT_CONFIG, validation, distributions, config hash
    sources.py      Source dataclass, discovery, probing, sidecar CSVs, noise grouping
    audio.py        standardisation (48 kHz mono), disk cache, slicing, activity detection, PCM encode/decode
    selection.py    seeds, noise selection, chunk planning (all random choices)
    sample.py       make_sample(): builds one clean/noisy pair + its metadata record
    worker.py       process-pool worker entry points
    storage.py      JSONL I/O, crash recovery, verification, Committer
    runner.py       multi-process generation loop
    summary.py      dataset_summary.json
    reconstruct.py  --reconstruct check
    cli.py          argument parsing and main()
"""
from .constants import __version__

__all__ = ["__version__"]
