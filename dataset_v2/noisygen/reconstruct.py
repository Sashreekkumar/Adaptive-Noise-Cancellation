"""--reconstruct: rebuild a sample from its metadata alone and compare with the stored audio."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import READ_DTYPE, apply_fade, encode, get_slice, write_mp3
from .sources import source_from_path
from .storage import read_records


def _rebuild_clean_and_noisy(r: dict, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the exact float clean/noisy signals (post output_gain, pre encoding) from metadata alone."""
    clean = get_slice(source_from_path(root, r["clean_source"]), r["clean_start_sample"], r["clean_end_sample"]).astype(np.float64)
    comp = np.zeros(len(clean))
    for nz in r["noises"]:
        src = source_from_path(root, nz["source"])
        gain = 10 ** (nz["relative_gain_db"] / 20)
        if nz.get("placement") == "impulsive":
            for ev in nz["events"]:
                seg = get_slice(src, ev["source_start_sample"], ev["source_end_sample"]).astype(np.float64)
                seg = apply_fade(seg, nz["fade_samples"])
                comp[ev["destination_start_sample"]:ev["destination_end_sample"]] += seg / nz["normalization_rms"] * gain
        else:
            seg = np.concatenate([get_slice(src, c["source_start_sample"], c["source_end_sample"])
                                  for c in nz["chunks"]]).astype(np.float64)
            comp[nz["start_sample"]:] += seg / nz["normalization_rms"] * gain
    g = r["output_gain"]
    return clean * g, (clean + comp * r["composite_noise_scale"]) * g


def reconstruct_check(out: Path, root: Path, sample_id: str) -> int:
    recs = [r for r in read_records(out / "metadata" / "dataset_metadata.jsonl") if r["sample_id"] == sample_id]
    if not recs:
        print(f"no record for sample {sample_id}")
        return 1
    r = recs[0]
    sub = r["output_subtype"]
    # branch on the actual stored extension (not a possibly-missing metadata field), so older wav-only
    # metadata from before this feature existed still reconstructs exactly as it always did
    container = Path(r["clean_output"]).suffix.lstrip(".").lower()
    try:
        rebuilt_clean, rebuilt_noisy = _rebuild_clean_and_noisy(r, root)
    except FileNotFoundError as e:
        print(f"sample {sample_id}: cannot reconstruct - a source file is missing ({e}). "
              f"If delete_clean_source_after_use was on, its clean source has since been deleted.")
        return 1

    if container != "mp3":
        rc = encode(rebuilt_clean, sub)
        rn = encode(rebuilt_noisy, sub)
        ac = sf.read(str(out / r["clean_output"]), dtype=READ_DTYPE[sub])[0]
        an = sf.read(str(out / r["noisy_output"]), dtype=READ_DTYPE[sub])[0]
        dc = int(np.max(np.abs(rc.astype(np.int64) - ac)))
        dn = int(np.max(np.abs(rn.astype(np.int64) - an)))
        print(f"sample {sample_id}: max |reconstructed - stored| clean={dc}, noisy={dn} (integer LSBs)")
        return 0 if dc <= 1 and dn <= 1 else 1

    # mp3: libsndfile/LAME encoding is deterministic (same input + compression_level -> identical bytes),
    # so this re-runs the exact same encode used at generation time and compares files byte-for-byte,
    # rather than trying to make an exactness claim about lossy audio itself.
    level = r.get("mp3_compression_level")
    if level is None:
        print(f"sample {sample_id}: mp3 output but no mp3_compression_level recorded in metadata; cannot "
              f"reproduce the exact encode.")
        return 1
    stored_c, stored_n = (out / r["clean_output"]).read_bytes(), (out / r["noisy_output"]).read_bytes()
    with tempfile.TemporaryDirectory() as td:
        tc, tn = Path(td) / "c.mp3", Path(td) / "n.mp3"
        write_mp3(tc, encode(rebuilt_clean, "PCM_16"), level)
        write_mp3(tn, encode(rebuilt_noisy, "PCM_16"), level)
        match_c, match_n = tc.read_bytes() == stored_c, tn.read_bytes() == stored_n
    print(f"sample {sample_id}: re-encoded mp3 byte-identical to stored file? clean={match_c}, noisy={match_n}")
    return 0 if match_c and match_n else 1
