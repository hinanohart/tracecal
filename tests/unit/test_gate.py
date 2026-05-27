"""S3: multiplicative physics gate (Q = hard_valid * quality), fail-closed on non-bool."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tracecal.physics.gate import (
    RoboticsConstraintBackend,
    degraded_result,
    gated_score,
)
from tracecal.schema import PhysicsCheckResult


def _valid(eid: str = "e", pass_rate: float = 1.0) -> PhysicsCheckResult:
    return PhysicsCheckResult(
        episode_id=eid,
        degraded=False,
        hard_valid=True,
        n_steps=10,
        n_steps_invalid=0,
        pass_rate=pass_rate,
    )


def _invalid(eid: str = "e") -> PhysicsCheckResult:
    return PhysicsCheckResult(
        episode_id=eid,
        degraded=False,
        hard_valid=False,
        n_steps=10,
        n_steps_invalid=3,
        pass_rate=0.7,
    )


def test_valid_keeps_quality() -> None:
    g = gated_score(0.8, _valid())
    assert g.Q == pytest.approx(0.8)
    assert g.gated_out is False
    assert g.degraded is False


def test_invalid_forces_zero() -> None:
    g = gated_score(0.99, _invalid())
    assert g.Q == 0.0
    assert g.gated_out is True


def test_degraded_passthrough_ungated() -> None:
    g = gated_score(0.6, degraded_result("e", "no-urdf-known", n_steps=10))
    assert g.Q == pytest.approx(0.6)
    assert g.gated_out is False
    assert g.degraded is True


def test_quality_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="quality"):
        gated_score(1.5, _valid())


def test_failclosed_on_nonbool_hard_valid() -> None:
    # A numeric truthy hard_valid (0.4) must NOT pass the gate as True (foldconsensus lesson).
    fake = SimpleNamespace(episode_id="e", degraded=False, hard_valid=0.4, pass_rate=0.4)
    with pytest.raises(ValueError, match="genuine bool"):
        gated_score(0.9, fake)  # type: ignore[arg-type]


def test_degraded_with_nonnull_hard_valid_rejected() -> None:
    fake = SimpleNamespace(episode_id="e", degraded=True, hard_valid=True, pass_rate=0.0)
    with pytest.raises(ValueError, match="hard_valid=None"):
        gated_score(0.5, fake)  # type: ignore[arg-type]


def test_backend_protocol_and_name() -> None:
    from tracecal.physics.gate import ConstraintBackend

    backend = RoboticsConstraintBackend()
    assert backend.name == "robotics"
    assert isinstance(backend, ConstraintBackend)  # runtime_checkable protocol
