"""S2: cross-check the in-house split-conformal p-values against crepes (independent reference).

crepes is never imported at runtime; this test (crosscheck extra) is the only place it appears.
The in-house ``predict_p`` must match ``crepes.ConformalClassifier.predict_p(smoothing=False)``
to machine precision — the same (n+1) inductive rule, implemented independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.conformal.calibrate import fit_split_conformal
from tracecal.conformal.nonconformity import calibration_nonconformity, orient

pytestmark = pytest.mark.crosscheck


def test_pvalues_match_crepes() -> None:
    crepes = pytest.importorskip("crepes")
    rng = np.random.default_rng(0)
    n_cal, n_test = 300, 120
    s_cal = rng.normal(size=n_cal)
    y_cal = (s_cal + rng.normal(0, 0.5, size=n_cal) > 0).astype(float)
    s_test = rng.normal(size=n_test)

    ours = fit_split_conformal(s_cal, y_cal, score_key="v", alpha=0.1)
    p_ours = ours.predict_p(s_test)

    # Build the SAME nonconformity inputs crepes expects: 1D calibration alphas (under true
    # label) and an (n_test, 2) test-alpha matrix [class0=+s, class1=-s].
    oriented_cal = orient(s_cal, "higher_better")
    cal_alphas = calibration_nonconformity(oriented_cal, y_cal)
    oriented_test = orient(s_test, "higher_better")
    test_alphas = np.stack([oriented_test, -oriented_test], axis=1)

    cc = crepes.ConformalClassifier()
    cc.fit(cal_alphas)
    p_crepes = cc.predict_p(test_alphas, smoothing=False)

    np.testing.assert_allclose(p_ours, p_crepes, atol=1e-12, rtol=0)


def test_prediction_set_coverage_matches_crepes_marginal() -> None:
    crepes = pytest.importorskip("crepes")
    rng = np.random.default_rng(3)
    n = 500
    s = rng.normal(size=n)
    y = (s + rng.normal(0, 0.4, size=n) > 0).astype(float)
    cut = 350
    ours = fit_split_conformal(s[:cut], y[:cut], score_key="v", alpha=0.1)
    sets_ours = ours.predict_set(s[cut:], alpha=0.1)

    cal_alphas = calibration_nonconformity(orient(s[:cut], "higher_better"), y[:cut])
    o_test = orient(s[cut:], "higher_better")
    test_alphas = np.stack([o_test, -o_test], axis=1)
    cc = crepes.ConformalClassifier()
    cc.fit(cal_alphas)
    p_crepes = cc.predict_p(test_alphas, smoothing=False)
    sets_crepes = [
        frozenset(int(c) for c in (0, 1) if p_crepes[i, c] > 0.1) for i in range(len(o_test))
    ]
    assert sets_ours == sets_crepes
