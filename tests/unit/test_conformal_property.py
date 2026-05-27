"""S2: property-based invariants of the conformal core (hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tracecal.conformal.calibrate import fit_split_conformal
from tracecal.conformal.coverage import coverage_indicators

_floats = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def _cal_set(draw):
    n = draw(st.integers(min_value=2, max_value=60))
    scores = draw(st.lists(_floats, min_size=n, max_size=n))
    labels = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    return np.array(scores), np.array(labels)


@settings(max_examples=150, deadline=None)
@given(_cal_set(), st.lists(_floats, min_size=1, max_size=30))
def test_pvalues_in_unit_interval_and_shape(cal, test_scores) -> None:
    s, y = cal
    m = fit_split_conformal(s, y, score_key="v", alpha=0.1)
    p = m.predict_p(np.array(test_scores))
    assert p.shape == (len(test_scores), 2)
    # inductive p-values are in (0, 1]
    assert np.all(p > 0.0) and np.all(p <= 1.0 + 1e-12)


@settings(max_examples=150, deadline=None)
@given(_cal_set())
def test_positive_pvalue_monotone(cal) -> None:
    s, y = cal
    m = fit_split_conformal(s, y, score_key="v", alpha=0.1)
    grid = np.linspace(-1100, 1100, 50)
    p1 = m.positive_pvalue(grid)
    assert np.all(np.diff(p1) >= -1e-12)


@settings(max_examples=80, deadline=None)
@given(st.integers(min_value=2, max_value=40), st.sampled_from([0.0, 1.0]))
def test_single_class_calibration_does_not_crash(n, cls) -> None:
    # Degenerate: every calibration label identical. Must not crash; p-values stay valid.
    s = np.linspace(0, 1, n)
    y = np.full(n, cls)
    m = fit_split_conformal(s, y, score_key="v", alpha=0.1)
    p = m.predict_p(np.array([0.5]))
    assert p.shape == (1, 2)
    assert np.all((p > 0.0) & (p <= 1.0 + 1e-12))


@settings(max_examples=60, deadline=None)
@given(_cal_set())
def test_fit_is_deterministic(cal) -> None:
    s, y = cal
    a = fit_split_conformal(s, y, score_key="v", alpha=0.1).predict_p(np.array([0.0, 0.5]))
    b = fit_split_conformal(s, y, score_key="v", alpha=0.1).predict_p(np.array([0.0, 0.5]))
    np.testing.assert_array_equal(a, b)


def test_empirical_marginal_coverage_holds_in_house() -> None:
    """The central guarantee, self-verified without the optional crepes cross-check.

    Under exchangeability, split-conformal prediction sets cover the true label with
    probability >= 1 - alpha marginally. We pool the holdout coverage over many independent
    exchangeable draws and assert the empirical rate is at least 1 - alpha (within a small
    finite-sample slack). This is the value-level proof of the core claim that otherwise lives
    only behind the `crosscheck` extra (tests/unit/test_crosscheck.py).
    """
    alpha = 0.1
    covered: list[bool] = []
    for seed in range(120):
        rng = np.random.default_rng(seed)
        n = 400
        y = rng.integers(0, 2, size=n).astype(float)
        scores = y + rng.normal(0.0, 1.0, size=n)  # exchangeable, informative score
        n_cal = n // 2
        m = fit_split_conformal(scores[:n_cal], y[:n_cal], score_key="v", alpha=alpha)
        sets = m.predict_set(scores[n_cal:], alpha=alpha)
        covered.extend(bool(c) for c in coverage_indicators(sets, y[n_cal:]))
    empirical = float(np.mean(covered))
    # ~24k pooled holdout points: the 1-alpha guarantee holds with a small slack for noise.
    assert empirical >= (1.0 - alpha) - 0.02, f"empirical coverage {empirical:.4f} < target"
    assert empirical <= 1.0
