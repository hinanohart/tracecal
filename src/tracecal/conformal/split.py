"""Embodiment-grouped calibration/holdout splitting.

The conformal coverage guarantee assumes calibration and holdout items are exchangeable. On
robot data this is violated when episodes from the *same embodiment* (sharing kinematics,
control stack, operator) appear on both sides of the split: calibration effectively "sees" the
holdout, and measured coverage is optimistic relative to a genuinely new robot.

``grouped_split`` keeps an entire embodiment wholly in calibration or wholly in holdout. When
embodiment ids are missing it degrades to a naive row-level split and records an explicit
caveat — it never silently pretends the leakage-safe guarantee holds. Transcribed from the
foldgauge cluster-grouped splitter (cluster_id → embodiment_id).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitResult:
    """Indices of a calibration/holdout split plus provenance about how it was made."""

    cal_idx: np.ndarray
    holdout_idx: np.ndarray
    grouped: bool
    n_groups: int
    caveat: str | None


def naive_split(
    n: int, *, holdout_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Row-level random split (no exchangeability protection)."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1).")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_hold = max(1, int(round(holdout_fraction * n)))
    n_hold = min(n_hold, n - 1)
    return np.sort(idx[n_hold:]), np.sort(idx[:n_hold])


def grouped_split(
    group_ids: list[str | None], *, holdout_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole embodiment groups into calibration/holdout. Raises if any id is missing."""
    if any(g is None for g in group_ids):
        raise ValueError("grouped_split requires every item to have a group id.")
    n = len(group_ids)
    arr = np.asarray(group_ids, dtype=object)
    uniq = np.array(sorted({g for g in group_ids if g is not None}), dtype=object)
    if len(uniq) < 2:
        raise ValueError("grouped_split needs at least 2 groups.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    target = holdout_fraction * n
    holdout_groups: set[object] = set()
    running = 0
    for j in order:
        g = uniq[j]
        if running >= target:
            break
        holdout_groups.add(g)
        running += int(np.sum(arr == g))
    if not holdout_groups:
        holdout_groups.add(uniq[order[0]])
    if len(holdout_groups) == len(uniq):
        holdout_groups.discard(uniq[order[-1]])
    hold_mask = np.array([g in holdout_groups for g in group_ids])
    holdout_idx = np.nonzero(hold_mask)[0]
    cal_idx = np.nonzero(~hold_mask)[0]
    return cal_idx, holdout_idx


def make_split(
    group_ids: list[str | None],
    *,
    group_by_embodiment: bool,
    holdout_fraction: float = 0.3,
    seed: int = 0,
) -> SplitResult:
    """Dispatch to a grouped or naive split and report which was used.

    Falls back to a naive split (with a caveat) when grouping is requested but impossible:
    missing embodiment ids, or fewer than two embodiments.
    """
    n = len(group_ids)
    if n < 2:
        raise ValueError(
            f"need at least 2 episodes to form non-empty calibration and holdout sets; got {n}."
        )
    n_groups = len({g for g in group_ids if g is not None})
    can_group = group_by_embodiment and all(g is not None for g in group_ids) and n_groups >= 2

    if can_group:
        cal_idx, holdout_idx = grouped_split(
            group_ids, holdout_fraction=holdout_fraction, seed=seed
        )
        return SplitResult(cal_idx, holdout_idx, grouped=True, n_groups=n_groups, caveat=None)

    if group_by_embodiment:
        if any(g is None for g in group_ids):
            caveat = (
                "embodiment-grouped split requested but some episodes lack an embodiment id; "
                "fell back to a naive row-level split. Coverage is a reference value, not a "
                "leakage-safe guarantee."
            )
        else:
            caveat = (
                "embodiment-grouped split requested but fewer than 2 embodiments are present; "
                "fell back to a naive row-level split. Coverage is a reference value, not a "
                "leakage-safe guarantee."
            )
    else:
        caveat = (
            "naive (non-embodiment-grouped) split: coverage assumes row-level exchangeability "
            "and is not protected against per-embodiment leakage."
        )
    cal_idx, holdout_idx = naive_split(n, holdout_fraction=holdout_fraction, seed=seed)
    return SplitResult(cal_idx, holdout_idx, grouped=False, n_groups=n_groups, caveat=caveat)
