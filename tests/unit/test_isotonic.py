"""S2: in-house PAVA isotonic + ECE diagnostic (cross-checked against sklearn)."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.conformal.calibrate import expected_calibration_error, isotonic_fit, pava


def test_pava_is_nondecreasing_and_idempotent() -> None:
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    f = pava(y)
    assert np.all(np.diff(f) >= -1e-12)
    np.testing.assert_allclose(pava(f), f)  # already monotone -> unchanged


def test_pava_known_blocks() -> None:
    # [1,0] violates -> merge to 0.5 ; then 0.5 < 1 ok. Expect [0.5,0.5,1.0].
    np.testing.assert_allclose(pava(np.array([1.0, 0.0, 1.0])), [0.5, 0.5, 1.0])


def test_pava_matches_sklearn() -> None:
    skl = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, size=200) + np.arange(200) * 1e-3)  # strictly increasing
    y = (rng.uniform(size=200) < (x - x.min()) / (x.max() - x.min())).astype(float)
    ours = pava(y)  # x already ascending
    ref = skl.IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y).predict(x)
    np.testing.assert_allclose(ours, ref, atol=1e-9)


def test_isotonic_calibrator_clamped_and_monotone() -> None:
    rng = np.random.default_rng(2)
    s = rng.normal(size=300)
    y = (s + rng.normal(0, 0.3, size=300) > 0).astype(float)
    cal = isotonic_fit(s, y)
    grid = np.linspace(-3, 3, 100)
    p = cal.predict(grid)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) >= -1e-12)


def test_ece_zero_for_perfectly_calibrated() -> None:
    # probs exactly equal empirical frequency in each bin -> ECE 0
    probs = np.array([0.0] * 50 + [1.0] * 50)
    labels = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-12)


def test_ece_detects_miscalibration() -> None:
    # model says 0.9 everywhere but only 50% are valid -> ECE ~ 0.4
    probs = np.full(100, 0.9)
    labels = np.array([0.0, 1.0] * 50)
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.4, abs=1e-9)


def test_ece_rejects_nonfinite_and_nonbinary() -> None:
    with pytest.raises(ValueError, match="finite"):
        expected_calibration_error(np.array([np.nan, 0.5]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error(np.array([0.5, 0.5]), np.array([0.0, 2.0]))
