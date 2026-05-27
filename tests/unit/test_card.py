"""S4: the accept/hold/reject decision rule."""

from __future__ import annotations

import pytest

from tracecal.physics.gate import degraded_result, gated_score
from tracecal.report.card import decide_verdict
from tracecal.schema import PhysicsCheckResult


def _phys(eid, hard_valid, pass_rate=1.0, checks=None):
    return PhysicsCheckResult(
        episode_id=eid,
        degraded=False,
        hard_valid=hard_valid,
        n_steps=10,
        n_steps_invalid=0 if hard_valid else 3,
        pass_rate=pass_rate,
        checks=checks or {"finite": True, "joint_limit": hard_valid, "velocity": True, "dim": True},
    )


def test_reject_on_hard_violation() -> None:
    p = _phys("e", hard_valid=False, pass_rate=0.7, checks={"joint_limit": False, "velocity": True})
    v = decide_verdict(p, gated_score(0.7, p), abstain=False)
    assert v.verdict == "reject"
    assert v.Q == 0.0
    assert v.hard_valid is False
    assert "joint_limit" in v.reasons[0]


def test_accept_when_valid_and_confident() -> None:
    p = _phys("e", hard_valid=True)
    v = decide_verdict(p, gated_score(0.95, p), abstain=False)
    assert v.verdict == "accept"
    assert v.Q == pytest.approx(0.95)
    assert v.reasons == ()


def test_hold_when_valid_but_abstain() -> None:
    p = _phys("e", hard_valid=True)
    v = decide_verdict(p, gated_score(0.8, p), abstain=True, abstain_reason="non-singleton {0, 1}")
    assert v.verdict == "hold"
    assert v.hard_valid is True
    assert v.abstain is True
    assert "non-singleton" in v.reasons[0]


def test_hold_on_degrade_takes_precedence_over_abstain() -> None:
    p = degraded_result("e", "no-urdf-known", n_steps=10)
    v = decide_verdict(p, gated_score(0.5, p), abstain=False)
    assert v.verdict == "hold"
    assert v.hard_valid is None
    assert v.degraded is True
    assert "physics-skipped" in v.reasons[0]


def test_episode_id_mismatch_raises() -> None:
    p = _phys("a", hard_valid=True)
    g = gated_score(0.9, _phys("b", hard_valid=True))
    with pytest.raises(ValueError, match="different episodes"):
        decide_verdict(p, g, abstain=False)
