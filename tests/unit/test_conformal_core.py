"""S2: split-conformal core — golden values, adversarial guards, determinism, coverage."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.conformal.calibrate import SplitConformalBinary, fit_split_conformal
from tracecal.conformal.coverage import bootstrap_ci, coverage_indicators
from tracecal.conformal.nonconformity import conformal_quantile
from tracecal.conformal.split import make_split

# A hand-computable calibration set (verified by the (n+1) inductive p-value rule by hand):
#   scores [0.1,0.4,0.6,0.9] labels [0,0,1,1], higher_better.
#   nonconformity under true label = [+0.1,+0.4,-0.6,-0.9]; sorted=[-0.9,-0.6,0.1,0.4].
#   test s=0.5 -> p0=(#{a>=+0.5}+1)/5=1/5=0.2 ; p1=(#{a>=-0.5}+1)/5=3/5=0.6.
_CAL_S = np.array([0.1, 0.4, 0.6, 0.9])
_CAL_Y = np.array([0.0, 0.0, 1.0, 1.0])


def _fit() -> SplitConformalBinary:
    return fit_split_conformal(_CAL_S, _CAL_Y, score_key="validity", alpha=0.1)


def test_golden_pvalues() -> None:
    m = _fit()
    p = m.predict_p(np.array([0.5]))
    assert p.shape == (1, 2)
    np.testing.assert_allclose(p[0], [0.2, 0.6], rtol=0, atol=1e-12)


def test_golden_prediction_sets() -> None:
    m = _fit()
    # alpha=0.1: both classes' p > 0.1 -> uninformative {0,1}
    assert m.predict_set(np.array([0.5]), alpha=0.1) == [frozenset({0, 1})]
    # alpha=0.3: only class 1 survives (p1=0.6 > 0.3, p0=0.2 !> 0.3)
    assert m.predict_set(np.array([0.5]), alpha=0.3) == [frozenset({1})]


def test_positive_pvalue_monotone_nondecreasing() -> None:
    m = _fit()
    grid = np.linspace(-2.0, 2.0, 200)
    p1 = m.positive_pvalue(grid)
    assert np.all(np.diff(p1) >= -1e-12), "positive p-value must be non-decreasing in score"


def test_determinism_100x() -> None:
    ref = _fit().predict_p(np.array([0.2, 0.5, 0.8]))
    for _ in range(100):
        got = fit_split_conformal(_CAL_S, _CAL_Y, score_key="v", alpha=0.1).predict_p(
            np.array([0.2, 0.5, 0.8])
        )
        np.testing.assert_array_equal(got, ref)


def test_orientation_flip_symmetry() -> None:
    hi = fit_split_conformal(_CAL_S, _CAL_Y, score_key="v", alpha=0.1, orientation="higher_better")
    lo = fit_split_conformal(-_CAL_S, _CAL_Y, score_key="v", alpha=0.1, orientation="lower_better")
    np.testing.assert_allclose(hi.predict_p(np.array([0.5])), lo.predict_p(np.array([-0.5])))


# --- adversarial guards (each must fail loudly, never fabricate) ---
def test_guard_rejects_nan_calibration_score() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_split_conformal(np.array([0.1, np.nan]), np.array([0.0, 1.0]), score_key="v", alpha=0.1)


def test_guard_rejects_inf_test_score() -> None:
    m = _fit()
    with pytest.raises(ValueError, match="finite"):
        m.predict_p(np.array([np.inf]))


def test_guard_rejects_nonbinary_label() -> None:
    with pytest.raises(ValueError, match="binary"):
        fit_split_conformal(np.array([0.1, 0.4]), np.array([0.0, 1.7]), score_key="v", alpha=0.1)


def test_guard_rejects_empty_and_bad_alpha() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_split_conformal(np.array([]), np.array([]), score_key="v", alpha=0.1)
    with pytest.raises(ValueError, match="alpha"):
        fit_split_conformal(_CAL_S, _CAL_Y, score_key="v", alpha=1.5)


def test_guard_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_split_conformal(
            np.array([0.1, 0.2, 0.3]), np.array([0.0, 1.0]), score_key="v", alpha=0.1
        )


def test_conformal_quantile_returns_inf_when_too_small() -> None:
    # n=2, alpha=0.01 -> ceil(3*0.99)=3 > 2 -> inf (cannot certify; conservative)
    assert conformal_quantile(np.array([0.1, 0.2]), 0.01) == np.inf
    assert np.isfinite(conformal_quantile(np.array([0.1, 0.2]), 0.5))


def test_coverage_indicator_rejects_nonbinary_holdout() -> None:
    with pytest.raises(ValueError, match="binary"):
        coverage_indicators([frozenset({1})], np.array([1.7]))


def test_bootstrap_ci_empty_is_nan_not_silent_pass() -> None:
    lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_marginal_coverage_holds_on_exchangeable_data() -> None:
    """Over many exchangeable draws the mean holdout coverage must meet the target."""
    alpha = 0.1
    target = 1.0 - alpha
    covs = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = 400
        y = rng.integers(0, 2, size=n).astype(float)
        # informative but noisy score: valid episodes score higher on average
        s = y * 1.0 + rng.normal(0, 1.0, size=n)
        sp = make_split([f"g{i % 4}" for i in range(n)], group_by_embodiment=True, seed=seed)
        m = fit_split_conformal(s[sp.cal_idx], y[sp.cal_idx], score_key="v", alpha=alpha)
        sets = m.predict_set(s[sp.holdout_idx], alpha=alpha)
        cov = coverage_indicators(sets, y[sp.holdout_idx]).mean()
        covs.append(cov)
    assert float(np.mean(covs)) >= target - 0.02
