"""S3: per-step kinematic gate — valid passes, violations caught, fail-closed on NaN/dim."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.physics.kinematics import check_episode
from tracecal.schema import EmbodimentSpec, JointLimits


def test_clean_trajectory_is_hard_valid(synth_2dof_spec, clean_trajectory) -> None:
    r = check_episode(clean_trajectory, spec=synth_2dof_spec, fps=10.0, episode_id="e0")
    assert r.hard_valid is True
    assert r.degraded is False
    assert r.n_steps_invalid == 0
    assert r.pass_rate == 1.0
    assert r.checks == {"finite": True, "joint_limit": True, "velocity": True, "dim": True}


def test_position_limit_violation_caught(synth_2dof_spec) -> None:
    pos = np.zeros((10, 2))
    pos[5, 0] = 2.0  # exceeds j1 upper (1.5708)
    r = check_episode(pos, spec=synth_2dof_spec, fps=10.0, episode_id="e1")
    assert r.hard_valid is False
    assert r.n_steps_invalid >= 1
    assert r.checks["joint_limit"] is False


def test_velocity_limit_violation_caught(synth_2dof_spec) -> None:
    # a 1.0 rad jump in one frame at fps=100 -> 100 rad/s, far above j1's 2.0 rad/s limit
    pos = np.zeros((4, 2))
    pos[2, 0] = 1.0
    r = check_episode(pos, spec=synth_2dof_spec, fps=100.0, episode_id="e2")
    assert r.hard_valid is False
    assert r.checks["velocity"] is False


def test_nan_is_failclosed_not_silent_pass(synth_2dof_spec) -> None:
    pos = np.zeros((8, 2))
    pos[3, 1] = np.nan  # a non-finite sample must count as invalid, never pass
    r = check_episode(pos, spec=synth_2dof_spec, fps=10.0, episode_id="e3")
    assert r.hard_valid is False
    assert r.checks["finite"] is False
    assert r.n_steps_invalid >= 1


def test_inf_is_failclosed(synth_2dof_spec) -> None:
    pos = np.zeros((8, 2))
    pos[0, 0] = np.inf
    r = check_episode(pos, spec=synth_2dof_spec, fps=10.0, episode_id="e4")
    assert r.hard_valid is False


def test_dim_mismatch_raises(synth_2dof_spec) -> None:
    with pytest.raises(ValueError, match="joint columns"):
        check_episode(np.zeros((10, 3)), spec=synth_2dof_spec, fps=10.0, episode_id="e5")


def test_degraded_spec_rejected(synth_2dof_spec) -> None:
    deg = EmbodimentSpec(robot_type="so101", resolved=False, source="degraded:x", dof=0)
    with pytest.raises(ValueError, match="degraded"):
        check_episode(np.zeros((5, 2)), spec=deg, fps=10.0, episode_id="e6")


def test_joint_without_velocity_limit_skips_velocity_check() -> None:
    spec = EmbodimentSpec(
        robot_type="novel",
        resolved=True,
        source="x",
        dof=1,
        joints=(JointLimits(name="j", lower=-10.0, upper=10.0, velocity=None),),
    )
    pos = np.array([[0.0], [5.0], [-5.0]])  # huge jumps, but no velocity limit declared
    r = check_episode(pos, spec=spec, fps=100.0, episode_id="e7")
    assert r.checks["velocity"] is True  # not penalised when the URDF omits a velocity limit
    assert r.hard_valid is True


def test_pass_rate_fraction(synth_2dof_spec) -> None:
    # Slow ramp past j1's upper limit (1.5708): per-step delta is tiny so velocity stays in
    # range; only the position check fails, on exactly the last 3 of 10 steps -> pass_rate 0.7.
    pos = np.zeros((10, 2))
    pos[:, 0] = np.linspace(1.5, 1.6, 10)
    r = check_episode(pos, spec=synth_2dof_spec, fps=10.0, episode_id="e8")
    assert r.checks["velocity"] is True
    assert r.checks["joint_limit"] is False
    assert r.pass_rate == pytest.approx(0.7)
    assert r.n_steps_invalid == 3
