"""S2: measured coverage report + bootstrap CI + per-embodiment breakdown."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.conformal.coverage import bootstrap_ci, build_coverage_report, coverage_indicators


def test_coverage_indicators_basic() -> None:
    sets = [frozenset({1}), frozenset({0}), frozenset({0, 1}), frozenset()]
    y = np.array([1.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(coverage_indicators(sets, y), [True, False, True, False])


def test_build_report_is_measured_and_flags_violation() -> None:
    sets = [frozenset({1})] * 9 + [frozenset({0})]  # 9/10 cover label 1
    y = np.ones(10)
    rep = build_coverage_report(sets, y, None, target_coverage=0.95)
    assert rep.source == "measured"
    assert rep.empirical_coverage == pytest.approx(0.9)
    assert rep.nominal_violated is True
    assert rep.ci_low <= rep.empirical_coverage <= rep.ci_high


def test_per_group_breakdown_respects_min_size() -> None:
    n = 60
    sets = [frozenset({1})] * n
    y = np.ones(n)
    groups = [f"emb{i % 2}" for i in range(n)]  # 2 groups of 30 each
    rep = build_coverage_report(sets, y, groups, target_coverage=0.9, min_group_size=20)
    assert set(rep.per_group) == {"emb0", "emb1"}
    # a group below min_group_size is omitted (not silently merged)
    rep2 = build_coverage_report(sets, y, groups, target_coverage=0.9, min_group_size=40)
    assert rep2.per_group == {}


def test_bootstrap_ci_deterministic_and_ordered() -> None:
    covered = np.array([1.0] * 80 + [0.0] * 20)
    lo1, hi1 = bootstrap_ci(covered, seed=0)
    lo2, hi2 = bootstrap_ci(covered, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1
    assert 0.0 <= lo1 <= 1.0 and 0.0 <= hi1 <= 1.0


def test_exchangeability_caveat_propagates() -> None:
    sets = [frozenset({1})] * 10
    y = np.ones(10)
    rep = build_coverage_report(
        sets, y, None, target_coverage=0.9, exchangeability_caveat="naive split"
    )
    assert rep.exchangeability_caveat == "naive split"
