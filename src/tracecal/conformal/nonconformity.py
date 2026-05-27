"""Nonconformity primitives for split-conformal binary validity prediction.

Transcribed from the foldgauge split-conformal core (numpy-only). A single scalar validity
score ``s`` per episode is oriented so that "higher = more like the valid (positive) class".
The nonconformity of assigning class ``c`` to an item is::

    A(s, 1) = -s_oriented      A(s, 0) = +s_oriented

These primitives are shared by the pooled and Mondrian calibrators in
:mod:`tracecal.conformal.calibrate`. The self-supervised :func:`reference_mode_flags` is a
separate, explicitly-heuristic path used when no validity labels exist.
"""

from __future__ import annotations

import math

import numpy as np

from tracecal.schema import ScoreOrientation


def orient(scores: np.ndarray, orientation: ScoreOrientation) -> np.ndarray:
    """Flip scores so that larger always means "more like the valid (positive) class"."""
    return scores if orientation == "higher_better" else -scores


def require_finite(values: np.ndarray, *, what: str) -> np.ndarray:
    """Return ``values`` as float, raising if any element is NaN/inf.

    A non-finite score would sort to the array end and silently corrupt the rank count in
    :func:`numpy.searchsorted`, producing a wrong conformal p-value. Refuse it rather than
    fabricate a number.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} must all be finite (no NaN/inf).")
    return arr


def validate_binary(labels: np.ndarray) -> np.ndarray:
    """Coerce labels to float and require they are exactly {0, 1}.

    A non-binary label (e.g. ``1.7``) would be truncated by ``int()`` elsewhere and silently
    miscount coverage; reject it at the boundary.
    """
    arr = np.asarray(labels, dtype=float)
    if arr.size:
        uniq = set(np.unique(arr).tolist())
        if not uniq <= {0.0, 1.0}:
            raise ValueError(f"binary labels required in {{0, 1}}; got {sorted(uniq)}.")
    return arr


def conformal_quantile(sorted_alphas: np.ndarray, alpha: float) -> float:
    """The ``ceil((n+1)(1-alpha))``-th smallest calibration nonconformity.

    Returns ``+inf`` when that rank exceeds ``n`` (the calibration set is too small to certify
    coverage at this ``alpha``), which makes the positive class always included — the
    conservative, honest behaviour.
    """
    n = len(sorted_alphas)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(sorted_alphas[k - 1])  # k is 1-indexed


def calibration_nonconformity(oriented_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nonconformity of each calibration item *under its true label*: y=1 → -s, y=0 → +s."""
    return np.where(labels == 1.0, -oriented_scores, oriented_scores)


def reference_mode_flags(
    scores: np.ndarray,
    groups: list[str] | None,
    *,
    alpha: float = 0.1,
    min_group: int = 20,
) -> np.ndarray:
    """Self-supervised abstention heuristic for the *no-label* (reference) mode.

    Flags episodes whose validity score sits in the lower ``alpha`` tail of their embodiment
    group's score distribution (or the pooled distribution when a group is too small). This is
    a descriptive outlier flag, **not** a distribution-free coverage guarantee — callers must
    keep ``coverage=None`` when using it.
    """
    s = require_finite(scores, what="reference-mode scores")
    n = len(s)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")

    def _flag(apply_idx: np.ndarray, pop_idx: np.ndarray) -> None:
        # Threshold is the lower-``alpha`` quantile of the *population* ``pop_idx``; it is
        # applied to ``apply_idx``. For large groups population == group; for undersized groups
        # the population is the pooled set (a 3-point per-group quantile would be noise).
        if apply_idx.size == 0 or pop_idx.size == 0:
            return
        thr = float(np.quantile(s[pop_idx], alpha))
        flags[apply_idx] = s[apply_idx] <= thr

    if groups is None:
        _flag(np.arange(n), np.arange(n))
        return flags

    if len(groups) != n:
        raise ValueError("groups and scores length mismatch.")
    arr = np.asarray(groups, dtype=object)
    all_idx = np.arange(n)
    small_mask = np.ones(n, dtype=bool)
    for g in sorted(set(groups)):
        idx = np.nonzero(arr == g)[0]
        if idx.size >= min_group:
            _flag(idx, idx)  # per-group quantile
            small_mask[idx] = False
    # episodes in undersized groups: flag against the pooled distribution
    if small_mask.any():
        _flag(all_idx[small_mask], all_idx)
    return flags
