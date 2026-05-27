"""S5: the pytest plugin's coverage assertion, weight artifact, and fixture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tracecal.io.lerobot_v3 import from_arrays
from tracecal.pytest_plugin import assert_coverage_holds, write_weights
from tracecal.schema import CoverageReport, DatasetReport, EpisodeVerdict


def _report(coverage: CoverageReport | None) -> DatasetReport:
    v = EpisodeVerdict(
        episode_id="a", verdict="accept", Q=0.9, hard_valid=True, abstain=False, degraded=False
    )
    r = EpisodeVerdict(
        episode_id="r", verdict="reject", Q=0.0, hard_valid=False, abstain=False, degraded=False
    )
    return DatasetReport(
        dataset="d",
        n_episodes=2,
        n_accept=1,
        n_hold=0,
        n_reject=1,
        n_degraded=0,
        coverage=coverage,
        verdicts=(v, r),
    )


def test_assert_coverage_holds_noop_in_reference_mode() -> None:
    assert_coverage_holds(_report(None))  # must not raise: no coverage claim to check


def test_assert_coverage_holds_passes_when_met() -> None:
    cov = CoverageReport(
        target_coverage=0.9,
        empirical_coverage=0.93,
        n_holdout=50,
        ci_low=0.88,
        ci_high=0.97,
        nominal_violated=False,
    )
    assert_coverage_holds(_report(cov))


def test_assert_coverage_holds_fails_on_breach() -> None:
    cov = CoverageReport(
        target_coverage=0.9,
        empirical_coverage=0.6,
        n_holdout=50,
        ci_low=0.5,
        ci_high=0.7,
        nominal_violated=True,
    )
    with pytest.raises(Exception, match="below target"):
        assert_coverage_holds(_report(cov))


def test_write_weights_artifact(tmp_path) -> None:
    path = tmp_path / "w.json"
    weights = write_weights(_report(None), str(path))
    assert weights == {"a": 1.0, "r": 0.0}
    assert json.loads(path.read_text()) == {"a": 1.0, "r": 0.0}


def test_tracecal_audit_fixture(tracecal_audit) -> None:
    # the plugin fixture audits a dataset; so101 degrades -> coverage None -> assertion is a no-op
    t = np.linspace(0, 1, 15)
    clean = np.stack([0.5 * np.sin(t), 0.3 * np.cos(t)], axis=1)
    ds = from_arrays(
        robot_type="so101",
        fps=10.0,
        joint_names=("j1", "j2"),
        episodes=[("e0", clean), ("e1", clean)],
        dataset="toy",
    )
    report = tracecal_audit(ds)
    assert report.n_degraded == 2
    tracecal_audit.assert_coverage_holds(report)  # no-op (reference-mode)
