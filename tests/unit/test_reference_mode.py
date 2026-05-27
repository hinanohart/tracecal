"""S2: self-supervised reference-mode abstention (no labels) — heuristic, not a guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.conformal.nonconformity import reference_mode_flags


def test_flags_lower_tail_pooled() -> None:
    scores = np.arange(100, dtype=float)  # 0..99
    flags = reference_mode_flags(scores, None, alpha=0.1)
    # bottom ~10% flagged
    assert flags[:10].all()
    assert not flags[20:].any()


def test_per_group_quantiles_used_when_large_enough() -> None:
    # two groups with very different score scales; flagging must be per-group
    g0 = np.linspace(0, 1, 50)
    g1 = np.linspace(100, 200, 50)
    scores = np.concatenate([g0, g1])
    groups = ["a"] * 50 + ["b"] * 50
    flags = reference_mode_flags(scores, groups, alpha=0.1, min_group=20)
    # the low end of group b (≈100) must NOT be flagged just for being below group a's high end
    assert flags[50:55].any()  # bottom of group b flagged within its own group
    assert not flags[60:].any()


def test_small_group_uses_pooled() -> None:
    scores = np.concatenate([np.linspace(0, 1, 50), np.array([0.001, 0.002, 0.003])])
    groups = ["big"] * 50 + ["tiny"] * 3
    flags = reference_mode_flags(scores, groups, alpha=0.1, min_group=20)
    # tiny-group episodes are very low vs pooled -> flagged via pooled distribution
    assert flags[50:].all()


def test_empty_and_bad_alpha() -> None:
    assert reference_mode_flags(np.array([]), None).tolist() == []
    with pytest.raises(ValueError, match="alpha"):
        reference_mode_flags(np.array([1.0, 2.0]), None, alpha=2.0)


def test_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        reference_mode_flags(np.array([1.0, np.inf]), None)
