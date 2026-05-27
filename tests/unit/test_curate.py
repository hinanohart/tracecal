"""S3: thin curate() over a DatasetReport."""

from __future__ import annotations

import pytest

from tracecal.curate import curate
from tracecal.schema import DatasetReport, EpisodeVerdict


def _report() -> DatasetReport:
    vs = (
        EpisodeVerdict(
            episode_id="a", verdict="accept", Q=0.9, hard_valid=True, abstain=False, degraded=False
        ),
        EpisodeVerdict(
            episode_id="h", verdict="hold", Q=0.4, hard_valid=True, abstain=True, degraded=False
        ),
        EpisodeVerdict(
            episode_id="r", verdict="reject", Q=0.0, hard_valid=False, abstain=False, degraded=False
        ),
    )
    return DatasetReport(
        dataset="d", n_episodes=3, n_accept=1, n_hold=1, n_reject=1, n_degraded=0, verdicts=vs
    )


def test_curate_partitions_and_weights() -> None:
    r = curate(_report())
    assert r.kept_ids == ("a",)
    assert r.held_ids == ("h",)
    assert r.rejected_ids == ("r",)
    assert r.weights == {"a": 1.0, "h": 0.0, "r": 0.0}
    assert r.n_kept == 1


def test_hold_weight_downweights_not_drops() -> None:
    r = curate(_report(), hold_weight=0.25)
    assert r.weights["h"] == 0.25
    assert r.weights["r"] == 0.0  # rejected always 0 regardless of hold_weight


def test_hold_weight_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="hold_weight"):
        curate(_report(), hold_weight=2.0)
