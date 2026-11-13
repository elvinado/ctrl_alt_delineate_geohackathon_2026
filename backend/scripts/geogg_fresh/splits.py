"""Whole-well split planning: balanced outer folds and nested inner folds.

Why whole wells?  Neighbouring 4-ms samples of one well are almost copies of each
other (same wavelet, same tie error, same geology).  If some samples of a well are in
training and others in testing, the score measures memory, not prediction.  So every
fold holds complete wells (Roberts et al. 2017, *Ecography*).

Why exhaustive?  With 12 development wells and 4 folds of 3 there are only 15,400 ways
to group them.  We score every grouping by how evenly the scarce labels are spread
(sand bins, hydrocarbon-potential bins, ...) and keep the best one - deterministic, no
random seed to argue about.  The same is done for the 3 inner folds of each outer
training set (280 groupings of 9 wells).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def group_sizes(n: int, k: int) -> list[int]:
    """Sizes of k groups as equal as possible (larger groups first)."""
    base, extra = divmod(n, k)
    return [base + 1] * extra + [base] * (k - extra)


def n_partitions(n: int, sizes: list[int]) -> int:
    """Number of distinct ways to split n items into unlabelled groups of the given sizes."""
    total, left = 1, n
    for s in sizes:
        total *= comb(left, s)
        left -= s
    from collections import Counter
    from math import factorial

    for _, m in Counter(sizes).items():
        total //= factorial(m)
    return total


def partitions(items: list, sizes: list[int]):
    """Yield every unordered partition of ``items`` into groups of the given sizes exactly once.

    The first remaining item always opens the next group; only the SIZE of that group is
    chosen (each distinct remaining size once).  The group containing a given item is
    therefore built exactly once, so no partition repeats and none is missed.
    """
    def rec(rest, sizes_left, acc):
        if not sizes_left:
            yield list(acc)
            return
        first, others = rest[0], rest[1:]
        for size in sorted(set(sizes_left), reverse=True):
            left = list(sizes_left)
            left.remove(size)
            for combo in combinations(others, size - 1):
                remaining = [x for x in others if x not in combo]
                yield from rec(remaining, left, acc + [(first, *combo)])

    yield from rec(list(items), list(sizes), [])


def score_partition(groups: list[tuple], counts: pd.DataFrame, weights: dict[str, float]) -> float:
    """Sum over targets of squared relative deviations of fold totals from the fold mean.

    A fold with zero examples of a target that exists elsewhere costs an extra 10 points,
    because its test metrics for that target would be undefined.
    """
    cols = [c for c in weights if c in counts.columns]
    M = counts[cols].to_numpy(float)
    pos = {w: i for i, w in enumerate(counts.index)}
    w = np.array([weights[c] for c in cols])
    return _score(groups, M, pos, w)


def _score(groups, M, pos, w) -> float:
    tot = np.stack([M[[pos[x] for x in g]].sum(0) for g in groups])        # [k, targets]
    mean = tot.sum(0) / len(groups)
    ok = mean > 0
    if not ok.any():
        return 0.0
    rel = (tot[:, ok] - mean[ok]) / mean[ok]
    return float((w[ok] * (rel ** 2).sum(0)).sum() + (w[ok] * 10.0 * (tot[:, ok] == 0).sum(0)).sum())


def best_partition(wells: list[str], k: int, counts: pd.DataFrame, weights: dict[str, float],
                   max_enumerate: int = 300_000) -> tuple[list[tuple], float, int]:
    """Best-balanced grouping of ``wells`` into ``k`` folds (exhaustive when feasible)."""
    wells = sorted(wells)
    k = max(1, min(k, len(wells)))
    sizes = group_sizes(len(wells), k)
    n = n_partitions(len(wells), sizes)
    if n > max_enumerate:
        raise ValueError(f"{n} partitions: too many to enumerate - reduce wells or raise max_enumerate")
    cols = [c for c in weights if c in counts.columns]
    M = counts[cols].to_numpy(float)
    pos = {w: i for i, w in enumerate(counts.index)}
    wv = np.array([weights[c] for c in cols])
    best, best_key = None, None
    for p in partitions(wells, sizes):
        s = round(_score(p, M, pos, wv), 12)
        if best_key is None or s < best_key[0] or (s == best_key[0] and sorted(p) < best_key[1]):
            best, best_key = p, (s, sorted(p))
    best = sorted([tuple(sorted(g)) for g in best])
    return best, float(best_key[0]), n


def nested_plan(dev_wells: list[str], counts: pd.DataFrame, weights: dict[str, float], n_outer: int,
                n_inner: int) -> dict:
    """Outer test folds + inner validation folds for each outer training set."""
    outer, outer_score, n_outer_parts = best_partition(dev_wells, n_outer, counts, weights)
    plan = {"outer": outer, "outer_score": outer_score, "n_outer_partitions_scored": n_outer_parts, "inner": {}}
    for f, test in enumerate(outer):
        train = [w for w in dev_wells if w not in test]
        k_in = min(n_inner, len(train))
        if k_in < 2:
            plan["inner"][f] = {"folds": [], "score": np.nan, "note": "fewer than 2 training wells: no inner folds"}
            continue
        inner, s, n_parts = best_partition(train, k_in, counts, weights)
        plan["inner"][f] = {"folds": inner, "score": s, "n_partitions_scored": n_parts}
    return plan
