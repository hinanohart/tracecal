"""Measure holdout coverage with a bootstrap confidence interval.

Coverage is always computed from a holdout set of episodes whose true validity labels are
known; there is no path that produces an assumed number. The resulting :class:`CoverageReport`
carries ``source="measured"`` by type (see :mod:`tracecal.schema`). Transcribed from foldgauge,
with per-group breakdown keyed by embodiment.
"""

from __future__ import annotations

import numpy as np

from tracecal.conformal.nonconformity import validate_binary
from tracecal.schema import CoverageReport, GroupCoverage


def coverage_indicators(
    prediction_sets: list[frozenset[int]], true_labels: np.ndarray
) -> np.ndarray:
    """Boolean array: was each episode's true validity label in its prediction set?"""
    y = validate_binary(true_labels)
    if len(prediction_sets) != len(y):
        raise ValueError("prediction_sets and true_labels length mismatch.")
    return np.array([int(y[i]) in prediction_sets[i] for i in range(len(y))], dtype=bool)


def bootstrap_ci(
    covered: np.ndarray, *, ci_level: float = 0.95, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a boolean coverage indicator."""
    covered = np.asarray(covered, dtype=float)
    n = len(covered)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = covered[idx].mean(axis=1)
    lo = (1.0 - ci_level) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(means, lo)), float(np.quantile(means, hi))


def build_coverage_report(
    prediction_sets: list[frozenset[int]],
    true_labels: np.ndarray,
    group_ids: list[str | None] | None,
    *,
    target_coverage: float,
    min_group_size: int = 20,
    ci_level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    exchangeability_caveat: str | None = None,
) -> CoverageReport:
    """Assemble a measured :class:`CoverageReport` with per-embodiment breakdown."""
    covered = coverage_indicators(prediction_sets, true_labels)
    empirical = float(covered.mean())
    ci_low, ci_high = bootstrap_ci(covered, ci_level=ci_level, n_boot=n_boot, seed=seed)

    per_group: dict[str, GroupCoverage] = {}
    if group_ids is not None:
        groups = np.array([g if g is not None else "<none>" for g in group_ids], dtype=object)
        for g in sorted(set(groups.tolist())):
            mask = groups == g
            n_g = int(mask.sum())
            if n_g < min_group_size:
                continue
            cov_g = float(covered[mask].mean())
            per_group[str(g)] = GroupCoverage(
                n=n_g, empirical_coverage=cov_g, nominal_violated=cov_g < target_coverage
            )

    return CoverageReport(
        target_coverage=target_coverage,
        empirical_coverage=empirical,
        n_holdout=len(covered),
        ci_level=ci_level,
        ci_low=ci_low,
        ci_high=ci_high,
        per_group=per_group,
        nominal_violated=empirical < target_coverage,
        exchangeability_caveat=exchangeability_caveat,
    )
