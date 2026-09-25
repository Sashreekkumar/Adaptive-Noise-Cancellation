"""All random choices: per-sample seeds, noise selection, source-chunk planning."""
from __future__ import annotations

import numpy as np

from .errors import SampleRejected


def derive_seed(base_seed: int, k: int, attempt: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(k), int(attempt)]).generate_state(1, dtype=np.uint32)[0])


def _pick_from(groups: list[list[int]], n: int, mode: str, rng: np.random.Generator) -> list[int]:
    """Pick n distinct files from category groups according to category_mode."""
    files = sorted(i for g in groups for i in g)
    n = min(n, len(files))
    ng = len(groups)
    if n <= 0:
        return []
    if mode == "any":
        return [int(files[int(i)]) for i in rng.choice(len(files), size=n, replace=False)]
    if ng >= n:  # enough categories: one file from each of n distinct categories
        gi = rng.choice(ng, size=n, replace=False)
        return [int(groups[int(g)][int(rng.integers(len(groups[int(g)])))]) for g in gi]
    if mode == "require_different":
        raise SampleRejected("require_different but not enough categories")
    chosen = [int(g[int(rng.integers(len(g)))]) for g in groups]  # prefer_different: every category once...
    taken = set(chosen)
    rest = [i for i in files if i not in taken]  # ...then fill from the remaining files
    chosen += [int(rest[int(i)]) for i in rng.choice(len(rest), size=n - ng, replace=False)]
    return [chosen[int(i)] for i in rng.permutation(len(chosen))]


def select_noises(keys: list, groups: list[list[int]], n: int, mode: str, required: list[str],
                  rng: np.random.Generator) -> list[int]:
    if not required:
        return _pick_from(groups, n, mode, rng)
    req_idx = [i for i, k in enumerate(keys) if k in required]
    picks = [int(groups[i][int(rng.integers(len(groups[i])))]) for i in req_idx]  # one per required category
    others = [g for i, g in enumerate(groups) if i not in set(req_idx)]  # extras never come from required categories
    picks += _pick_from(others, n - len(picks), mode, rng)
    return [picks[int(i)] for i in rng.permutation(len(picks))]  # random order -> random 0 dB reference noise


def plan_noise_chunks(L: int, need: int, policy: str, rng: np.random.Generator):
    """Decide which source ranges cover `need` samples. Returns (chunks[(src_start, src_end, dest_offset)], looped)."""
    if L >= need:
        st = int(rng.integers(0, L - need + 1))
        return [(st, st + need, 0)], False
    if policy == "reject":
        raise SampleRejected("noise shorter than required and short_noise_policy=reject")
    cur = int(rng.integers(0, L))
    chunks, filled = [], 0
    while filled < need:
        take = min(L - cur, need - filled)
        chunks.append((cur, cur + take, filled))
        filled += take
        cur = 0
    return chunks, True
