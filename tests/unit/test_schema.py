"""S1: the IR contract enforces correctness rules at the type level."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tracecal.schema import (
    CoverageReport,
    DatasetReport,
    EmbodimentSpec,
    EpisodeRecord,
    EpisodeVerdict,
    JointLimits,
    PhysicsCheckResult,
)


def test_joint_limits_order_enforced() -> None:
    JointLimits(name="j1", lower=-1.0, upper=1.0, velocity=2.0)
    with pytest.raises(ValidationError):
        JointLimits(name="j1", lower=1.0, upper=-1.0)  # lower !< upper
    with pytest.raises(ValidationError):
        JointLimits(name="j1", lower=-1.0, upper=1.0, velocity=0.0)  # non-positive velocity


def test_joint_limits_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        JointLimits(name="j1", lower=-1.0, upper=1.0, bogus=3)  # type: ignore[call-arg]


def test_embodiment_resolved_requires_joints() -> None:
    j = JointLimits(name="j1", lower=-1.0, upper=1.0)
    spec = EmbodimentSpec(robot_type="panda", resolved=True, source="rd:panda", dof=1, joints=(j,))
    assert spec.degraded is False
    with pytest.raises(ValidationError):
        EmbodimentSpec(robot_type="panda", resolved=True, source="rd:panda", dof=0, joints=())


def test_embodiment_degraded_forbids_joints_and_is_degraded() -> None:
    deg = EmbodimentSpec(robot_type="so101", resolved=False, source="degraded:no-urdf", dof=0)
    assert deg.degraded is True
    assert deg.joints == ()
    j = JointLimits(name="j1", lower=-1.0, upper=1.0)
    with pytest.raises(ValidationError):
        EmbodimentSpec(robot_type="so101", resolved=False, source="x", dof=1, joints=(j,))


def test_embodiment_dof_must_match_joint_count() -> None:
    j = JointLimits(name="j1", lower=-1.0, upper=1.0)
    with pytest.raises(ValidationError):
        EmbodimentSpec(robot_type="panda", resolved=True, source="rd", dof=2, joints=(j,))


def test_coverage_source_is_measured_only() -> None:
    rep = CoverageReport(
        target_coverage=0.9,
        empirical_coverage=0.91,
        n_holdout=50,
        ci_low=0.85,
        ci_high=0.96,
        nominal_violated=False,
    )
    assert rep.source == "measured"
    # No path can set source to anything else.
    with pytest.raises(ValidationError):
        CoverageReport(
            target_coverage=0.9,
            empirical_coverage=0.5,
            n_holdout=50,
            ci_low=0.4,
            ci_high=0.6,
            nominal_violated=True,
            source="assumed",  # type: ignore[arg-type]
        )


def test_physics_result_degrade_implies_hard_valid_none() -> None:
    PhysicsCheckResult(
        episode_id="e0",
        degraded=True,
        hard_valid=None,
        n_steps=10,
        n_steps_invalid=0,
        pass_rate=1.0,
    )
    with pytest.raises(ValidationError):
        PhysicsCheckResult(
            episode_id="e0",
            degraded=True,
            hard_valid=True,
            n_steps=10,
            n_steps_invalid=0,
            pass_rate=1.0,
        )
    with pytest.raises(ValidationError):
        PhysicsCheckResult(
            episode_id="e0",
            degraded=False,
            hard_valid=None,
            n_steps=10,
            n_steps_invalid=0,
            pass_rate=1.0,
        )


def test_episode_record_requires_positive_fps_and_steps() -> None:
    EpisodeRecord(id="e0", embodiment_id="panda", fps=30.0, n_steps=100, joint_names=("j1", "j2"))
    with pytest.raises(ValidationError):
        EpisodeRecord(id="e0", embodiment_id="panda", fps=0.0, n_steps=100, joint_names=("j1",))
    with pytest.raises(ValidationError):
        EpisodeRecord(id="e0", embodiment_id="panda", fps=30.0, n_steps=0, joint_names=("j1",))


def test_dataset_report_counts_must_sum() -> None:
    v = EpisodeVerdict(
        episode_id="e0", verdict="accept", Q=0.8, hard_valid=True, abstain=False, degraded=False
    )
    DatasetReport(
        dataset="d", n_episodes=1, n_accept=1, n_hold=0, n_reject=0, n_degraded=0, verdicts=(v,)
    )
    with pytest.raises(ValidationError):
        DatasetReport(dataset="d", n_episodes=3, n_accept=1, n_hold=0, n_reject=0, n_degraded=0)


def test_episode_verdict_q_bounds() -> None:
    with pytest.raises(ValidationError):
        EpisodeVerdict(
            episode_id="e0", verdict="accept", Q=1.5, hard_valid=True, abstain=False, degraded=False
        )
