"""S2: property-based invariants of the conformal core (hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tracecal.conformal.calibrate import fit_split_conformal

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
