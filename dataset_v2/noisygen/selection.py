"""All random choices: per-sample seeds, noise selection, source-chunk planning."""
from __future__ import annotations

import numpy as np

from .errors import SampleRejected


def derive_seed(base_seed: int, k: int, attempt: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(k), int(attempt)]).generate_state(1, dtype=np.uint32)[0])


def predict_clean_source(base_seed: int, k: int, cleans: list) -> str:
    """The clean source `make_sample` will draw on attempt 0 for sample k, computed without touching any
    audio. This mirrors the first random draw in make_sample() exactly (same seed derivation, same
    `cleans` list/order, same rng call), so it is only valid when called with the identical `cleans` list
    that generation itself will use. Used to plan `delete_clean_source_after_use` deletions ahead of time;
    a retry that ends up using a different source is handled separately and does not rely on this being
    exact for every sample, only for the common (attempt 0 succeeds) case."""
    rng = np.random.default_rng(derive_seed(base_seed, k, 0))
    return cleans[int(rng.integers(len(cleans)))].rel


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


def plan_impulsive_events(N: int, Ln: int, count_range: tuple[int, int], dur_range_samples: tuple[int, int],
                          rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """Scatter short events of this noise source across the sample. Returns (source_start, source_end,
    destination_start) per event, in samples. Events may overlap each other in the destination timeline -
    real impulsive noises like rapid gunfire often do - and each event's length is clamped to whatever is
    actually available (the source's own length, and the sample's length)."""
    lo_c, hi_c = count_range
    n = lo_c if lo_c == hi_c else int(rng.integers(lo_c, hi_c + 1))
    lo_d, hi_d = dur_range_samples
    events = []
    for _ in range(n):
        dur = lo_d if lo_d == hi_d else int(rng.integers(lo_d, hi_d + 1))
        dur = max(1, min(dur, Ln, N))
        src_start = int(rng.integers(0, Ln - dur + 1)) if Ln > dur else 0
        dest_start = int(rng.integers(0, N - dur + 1)) if N > dur else 0
        events.append((src_start, src_start + dur, dest_start))
    return events


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
