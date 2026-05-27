"""S2: native Mondrian (group-conditional) conformal calibration."""

from __future__ import annotations

import numpy as np

from tracecal.conformal.calibrate import fit_mondrian_conformal, fit_split_conformal


def _data(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    bins = np.array([f"emb{i % 3}" for i in range(n)])
    y = rng.integers(0, 2, size=n).astype(float)
    # per-embodiment score offset so a pooled threshold would be wrong for some bins
    offset = {"emb0": 0.0, "emb1": 3.0, "emb2": -3.0}
    s = y + np.array([offset[b] for b in bins]) + rng.normal(0, 0.5, size=n)
    return s, y, bins


def test_mondrian_fits_one_calibrator_per_large_bin() -> None:
    s, y, bins = _data()
    m = fit_mondrian_conformal(s, y, bins, score_key="v", alpha=0.1, min_bin_size=20)
    assert set(m.calibrators) == {"emb0", "emb1", "emb2"}
    # routing returns the bin-specific calibrator, not the fallback
    routed = m._route(np.array(["emb1", "emb0"]))
    assert routed[0] is m.calibrators["emb1"]
    assert routed[1] is m.calibrators["emb0"]


def test_mondrian_unseen_bin_falls_back() -> None:
    s, y, bins = _data()
    m = fit_mondrian_conformal(s, y, bins, score_key="v", alpha=0.1)
    routed = m._route(np.array(["never_seen"]))
    assert routed[0] is m.fallback


def test_mondrian_undersized_bin_uses_fallback_not_noise() -> None:
    # one big bin + a 3-sample bin: the tiny bin must NOT get its own calibrator
    s = np.concatenate([np.linspace(0, 1, 50), np.array([5.0, 5.1, 5.2])])
    y = np.concatenate([np.array([0.0, 1.0] * 25), np.array([1.0, 1.0, 1.0])])
    bins = np.array(["big"] * 50 + ["tiny"] * 3)
    m = fit_mondrian_conformal(s, y, bins, score_key="v", alpha=0.1, min_bin_size=20)
    assert "big" in m.calibrators
    assert "tiny" not in m.calibrators
    assert m._route(np.array(["tiny"]))[0] is m.fallback


def test_mondrian_predict_shapes_and_alignment() -> None:
    s, y, bins = _data()
    m = fit_mondrian_conformal(s, y, bins, score_key="v", alpha=0.1)
    p = m.predict_p(s[:5], bins[:5])
    assert p.shape == (5, 2)
    # per-item routing must equal calling the routed calibrator directly
    for i in range(5):
        cal = m.calibrators.get(bins[i], m.fallback)
        np.testing.assert_allclose(p[i], cal.predict_p(s[i : i + 1])[0])


def test_mondrian_pooled_equivalence_when_single_bin() -> None:
    rng = np.random.default_rng(1)
    n = 100
    s = rng.normal(size=n)
    y = (s > 0).astype(float)
    bins = np.array(["only"] * n)
    m = fit_mondrian_conformal(s, y, bins, score_key="v", alpha=0.1, min_bin_size=20)
    pooled = fit_split_conformal(s, y, score_key="v", alpha=0.1)
    np.testing.assert_allclose(m.predict_p(s, bins), pooled.predict_p(s))  # one big bin == pooled
