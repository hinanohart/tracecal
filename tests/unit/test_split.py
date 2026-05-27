"""S2: embodiment-grouped split keeps the coverage guarantee leakage-honest."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.conformal.split import grouped_split, make_split, naive_split


def test_grouped_split_keeps_embodiments_whole() -> None:
    groups = [f"emb{i % 5}" for i in range(100)]
    cal, hold = grouped_split(groups, holdout_fraction=0.3, seed=0)
    cal_g = {groups[i] for i in cal}
    hold_g = {groups[i] for i in hold}
    assert cal_g.isdisjoint(hold_g), "an embodiment must not appear on both sides"
    assert len(cal) + len(hold) == 100
    assert len(cal) > 0 and len(hold) > 0


def test_grouped_split_requires_ids_and_two_groups() -> None:
    with pytest.raises(ValueError, match="group id"):
        grouped_split(["a", None, "b"])
    with pytest.raises(ValueError, match="2 groups"):
        grouped_split(["a", "a", "a"])


def test_make_split_grouped_path_no_caveat() -> None:
    groups: list[str | None] = [f"emb{i % 4}" for i in range(80)]
    r = make_split(groups, group_by_embodiment=True, seed=1)
    assert r.grouped is True
    assert r.caveat is None
    assert r.n_groups == 4


def test_make_split_degrades_with_caveat_on_missing_ids() -> None:
    groups: list[str | None] = ["emb0", None, "emb1", "emb1"]
    r = make_split(groups, group_by_embodiment=True, seed=1)
    assert r.grouped is False
    assert r.caveat is not None and "reference value" in r.caveat


def test_make_split_naive_when_not_requested() -> None:
    groups: list[str | None] = [f"emb{i % 4}" for i in range(40)]
    r = make_split(groups, group_by_embodiment=False, seed=1)
    assert r.grouped is False
    assert r.caveat is not None and "row-level exchangeability" in r.caveat


def test_make_split_too_few_episodes_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        make_split(["emb0"], group_by_embodiment=True)


def test_naive_split_both_sides_nonempty_and_deterministic() -> None:
    a1 = naive_split(10, seed=3)
    a2 = naive_split(10, seed=3)
    np.testing.assert_array_equal(a1[0], a2[0])
    np.testing.assert_array_equal(a1[1], a2[1])
    assert len(a1[0]) > 0 and len(a1[1]) > 0
