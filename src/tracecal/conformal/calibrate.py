"""Split (inductive) conformal calibration of binary episode validity.

The correctness-critical core. Given a calibration set of (validity-score, binary-label)
pairs it produces, for any test score, conformal p-values per class and a prediction set with
a finite-sample, distribution-free coverage guarantee under exchangeability. The pooled
calibrator (:class:`SplitConformalBinary`) is transcribed from foldgauge; these p-values match
``crepes.ConformalClassifier.predict_p(..., smoothing=False)`` exactly (cross-checked in
``tests/unit/test_crosscheck.py``). crepes is an independent reference, never imported at runtime.

:class:`MondrianConformal` implements group-conditional (per embodiment/task) calibration
natively in numpy — the piece foldgauge deferred — so coverage can be conditioned on
embodiment without leaking a global threshold across very different robots.

:func:`pava` / :func:`isotonic_fit` / :func:`expected_calibration_error` provide an isotonic
reliability diagnostic (cross-checked against ``sklearn.isotonic`` in tests). They are *not*
the coverage guarantee — they describe how well the raw score matches empirical validity.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from tracecal.conformal.nonconformity import (
    calibration_nonconformity,
    conformal_quantile,
    orient,
    require_finite,
    validate_binary,
)
from tracecal.schema import ScoreOrientation


@dataclass
class SplitConformalBinary:
    """A fitted standard (pooled) inductive conformal binary classifier."""

    score_key: str
    alpha: float
    orientation: ScoreOrientation
    cal_alphas_sorted: np.ndarray  # ascending nonconformity of calibration true-labels
    n_cal: int
    q_hat: float  # diagnostic only; prediction uses the full p-value rule

    def predict_p(self, scores: np.ndarray) -> np.ndarray:
        """Return p-values of shape ``(n, 2)`` as columns ``[p_class0, p_class1]``."""
        oriented = orient(require_finite(scores, what="test scores"), self.orientation)
        test_alphas = np.stack([oriented, -oriented], axis=1)
        counts = self.n_cal - np.searchsorted(self.cal_alphas_sorted, test_alphas, side="left")
        return (counts + 1.0) / (self.n_cal + 1.0)

    def predict_set(self, scores: np.ndarray, alpha: float | None = None) -> list[frozenset[int]]:
        """Prediction set ``{c : p_c > alpha}`` per item."""
        a = self.alpha if alpha is None else alpha
        p = self.predict_p(scores)
        return [frozenset(int(c) for c in (0, 1) if p[i, c] > a) for i in range(len(p))]

    def positive_pvalue(self, scores: np.ndarray) -> np.ndarray:
        """Conformal p-value for the valid (class 1) label; used for triage ranking."""
        return self.predict_p(scores)[:, 1]


def fit_split_conformal(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    score_key: str,
    alpha: float,
    orientation: ScoreOrientation = "higher_better",
) -> SplitConformalBinary:
    """Fit a standard pooled split-conformal binary classifier."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    scores = require_finite(scores, what="calibration scores")
    labels = validate_binary(labels)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if len(scores) == 0:
        raise ValueError("empty calibration set.")
    oriented = orient(scores, orientation)
    cal_alphas = calibration_nonconformity(oriented, labels)
    cal_alphas_sorted = np.sort(cal_alphas)
    q_hat = conformal_quantile(cal_alphas_sorted, alpha)
    return SplitConformalBinary(
        score_key=score_key,
        alpha=alpha,
        orientation=orientation,
        cal_alphas_sorted=cal_alphas_sorted,
        n_cal=len(scores),
        q_hat=q_hat,
    )


@dataclass
class MondrianConformal:
    """Group-conditional (Mondrian) conformal calibrator.

    Holds one :class:`SplitConformalBinary` per bin (embodiment/task) that had enough
    calibration items, plus a pooled fallback for unseen or undersized bins. Each test item is
    routed to its bin's calibrator (or the fallback), giving per-group conditional coverage.
    """

    by: str
    alpha: float
    calibrators: dict[Hashable, SplitConformalBinary]
    fallback: SplitConformalBinary

    def _route(self, bins: np.ndarray) -> list[SplitConformalBinary]:
        return [self.calibrators.get(b, self.fallback) for b in bins.tolist()]

    def predict_p(self, scores: np.ndarray, bins: np.ndarray) -> np.ndarray:
        scores = require_finite(scores, what="test scores")
        bins = np.asarray(bins)
        if len(scores) != len(bins):
            raise ValueError("scores and bins must have the same length.")
        out = np.empty((len(scores), 2), dtype=float)
        for i, cal in enumerate(self._route(bins)):
            out[i] = cal.predict_p(scores[i : i + 1])[0]
        return out

    def predict_set(
        self, scores: np.ndarray, bins: np.ndarray, alpha: float | None = None
    ) -> list[frozenset[int]]:
        a = self.alpha if alpha is None else alpha
        p = self.predict_p(scores, bins)
        return [frozenset(int(c) for c in (0, 1) if p[i, c] > a) for i in range(len(p))]

    def positive_pvalue(self, scores: np.ndarray, bins: np.ndarray) -> np.ndarray:
        return self.predict_p(scores, bins)[:, 1]


def fit_mondrian_conformal(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: np.ndarray,
    *,
    score_key: str,
    alpha: float,
    orientation: ScoreOrientation = "higher_better",
    min_bin_size: int = 20,
    by: str = "embodiment",
) -> MondrianConformal:
    """Fit one pooled calibrator per bin with >= ``min_bin_size`` items, plus a pooled fallback.

    Undersized bins are intentionally *not* given their own calibrator (a per-bin threshold from
    a handful of points is noise); their test items fall back to the pooled calibrator. This is
    the native Mondrian body foldgauge left as ``NotImplementedError``.
    """
    scores = require_finite(scores, what="calibration scores")
    labels = validate_binary(labels)
    bins = np.asarray(bins)
    if not (len(scores) == len(labels) == len(bins)):
        raise ValueError("scores, labels and bins must have the same length.")
    fallback = fit_split_conformal(
        scores, labels, score_key=score_key, alpha=alpha, orientation=orientation
    )
    calibrators: dict[Hashable, SplitConformalBinary] = {}
    for b in np.unique(bins).tolist():
        mask = bins == b
        if int(mask.sum()) >= min_bin_size:
            calibrators[b] = fit_split_conformal(
                scores[mask],
                labels[mask],
                score_key=score_key,
                alpha=alpha,
                orientation=orientation,
            )
    return MondrianConformal(by=by, alpha=alpha, calibrators=calibrators, fallback=fallback)


# --------------------------------------------------------------------------- #
# Isotonic reliability diagnostic (PAVA) — NOT the coverage guarantee.
# --------------------------------------------------------------------------- #
def pava(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pool-adjacent-violators isotonic (non-decreasing) regression of ``y``.

    Returns the fitted non-decreasing sequence with the same length as ``y``. ``y`` is assumed
    already ordered by the covariate (ascending score). Matches ``sklearn.isotonic`` on the
    fitted values (cross-checked in tests).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    # Stack of (value, weight, count) blocks; merge while the monotone constraint is violated.
    vals: list[float] = []
    wts: list[float] = []
    cnts: list[int] = []
    for i in range(n):
        vals.append(y[i])
        wts.append(w[i])
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            merged_w = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / merged_w)
            wts.append(merged_w)
            cnts.append(c1 + c2)
    out = np.empty(n, dtype=float)
    pos = 0
    for v, c in zip(vals, cnts, strict=True):
        out[pos : pos + c] = v
        pos += c
    return out


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone (isotonic) map from validity score to calibrated P(valid)."""

    x_sorted: np.ndarray  # ascending unique-ish scores
    y_fitted: np.ndarray  # non-decreasing fitted probabilities

    def predict(self, scores: np.ndarray) -> np.ndarray:
        s = require_finite(scores, what="scores")
        if len(self.x_sorted) == 0:
            return np.full(len(s), np.nan)
        # piecewise-constant / linear interpolation, clamped to [0, 1]
        p = np.interp(
            s, self.x_sorted, self.y_fitted, left=self.y_fitted[0], right=self.y_fitted[-1]
        )
        return np.asarray(np.clip(p, 0.0, 1.0), dtype=float)


def isotonic_fit(
    scores: np.ndarray, labels: np.ndarray, *, orientation: ScoreOrientation = "higher_better"
) -> IsotonicCalibrator:
    """Fit an isotonic (PAVA) calibrator mapping oriented score → P(valid)."""
    s = orient(require_finite(scores, what="scores"), orientation)
    y = validate_binary(labels)
    if len(s) != len(y):
        raise ValueError("scores and labels must have the same length.")
    if len(s) == 0:
        raise ValueError("empty calibration set.")
    order = np.argsort(s, kind="stable")
    y_fit = pava(y[order])
    return IsotonicCalibrator(x_sorted=s[order], y_fitted=y_fit)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    """Equal-width binned ECE: weighted mean |mean(prob) - mean(label)| over occupied bins."""
    p = require_finite(probs, what="probabilities")
    y = validate_binary(labels)
    if len(p) != len(y):
        raise ValueError("probs and labels must have the same length.")
    if len(p) == 0:
        raise ValueError("empty input.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    # ECE is only meaningful for probabilities; a public diagnostic must reject out-of-range
    # input rather than silently binning, say, 1.5 into the top bin and reporting a number.
    if float(p.min()) < 0.0 or float(p.max()) > 1.0:
        raise ValueError("probabilities must lie in [0, 1].")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # clip so prob==1.0 lands in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return ece
