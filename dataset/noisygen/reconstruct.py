"""--reconstruct: rebuild a sample from its metadata alone and compare with the stored audio."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import READ_DTYPE, encode, get_slice
from .sources import source_from_path
from .storage import read_records


def reconstruct_check(out: Path, root: Path, sample_id: str) -> int:
    recs = [r for r in read_records(out / "metadata" / "dataset_metadata.jsonl") if r["sample_id"] == sample_id]
    if not recs:
        print(f"no record for sample {sample_id}")
        return 1
    r = recs[0]
    sub = r["output_subtype"]
    clean = get_slice(source_from_path(root, r["clean_source"]), r["clean_start_sample"], r["clean_end_sample"]).astype(np.float64)
    comp = np.zeros(len(clean))
    for nz in r["noises"]:
        src = source_from_path(root, nz["source"])
        seg = np.concatenate([get_slice(src, c["source_start_sample"], c["source_end_sample"])
                              for c in nz["chunks"]]).astype(np.float64)
        comp[nz["start_sample"]:] += seg / nz["normalization_rms"] * 10 ** (nz["relative_gain_db"] / 20)
    g = r["output_gain"]
    rc = encode(clean * g, sub)
    rn = encode((clean + comp * r["composite_noise_scale"]) * g, sub)
    ac = sf.read(str(out / r["clean_output"]), dtype=READ_DTYPE[sub])[0]
    an = sf.read(str(out / r["noisy_output"]), dtype=READ_DTYPE[sub])[0]
    dc, dn = int(np.max(np.abs(rc.astype(np.int64) - ac))), int(np.max(np.abs(rn.astype(np.int64) - an)))
    print(f"sample {sample_id}: max |reconstructed - stored| clean={dc}, noisy={dn} (integer LSBs)")
    return 0 if dc <= 1 and dn <= 1 else 1
