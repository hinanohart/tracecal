"""Multiplicative physics gate — the v0.1.0a1 CLAIM.

A kinematically invalid episode is *invalid*, not merely low-quality: a hard-check failure
forces the final score ``Q`` to exactly 0, regardless of any continuous quality signal::

    Q = hard_valid(episode) * quality(episode)        # hard_valid ∈ {0, 1}

The gate is deliberately separate from how ``quality`` is produced (physics pass-rate in
reference-mode, or the calibrated P(valid) when labels exist), so the value-add never depends on
an untuned quality weighting. A degraded episode (no resolvable URDF) is *not* gated — its
physics is unknown, so it is passed through ungated and flagged for a ``hold`` verdict upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from tracecal.physics.kinematics import check_episode
from tracecal.schema import EmbodimentSpec, PhysicsCheckResult


@dataclass(frozen=True)
class GatedScore:
    """Final gated score for one episode."""

    episode_id: str
    Q: float
    gated_out: bool  # True iff a hard physics failure forced Q to 0
    degraded: bool  # True iff physics was skipped (ungated pass-through)
    components: dict[str, float] = field(default_factory=dict)


def degraded_result(episode_id: str, reason: str, n_steps: int) -> PhysicsCheckResult:
    """Build a degraded :class:`PhysicsCheckResult` (physics skipped).

    ``pass_rate`` is 0.0 and **meaningless** when ``degraded`` is True; consumers must branch on
    ``degraded`` / ``hard_valid is None`` rather than read it.
    """
    return PhysicsCheckResult(
        episode_id=episode_id,
        degraded=True,
        hard_valid=None,
        n_steps=max(0, n_steps),
        n_steps_invalid=0,
        pass_rate=0.0,
        checks={},
        degrade_reason=reason,
    )


def gated_score(quality: float, physics: PhysicsCheckResult) -> GatedScore:
    """Apply the multiplicative hard gate to a continuous ``quality`` in [0, 1].

    Fail-closed on a non-boolean ``hard_valid``: a numeric truthy value (e.g. 0.4) must never be
    allowed to pass the gate as if it were ``True`` (the foldconsensus lesson). For a degraded
    episode (``hard_valid is None``) no gate is applied and ``Q = quality`` with ``degraded=True``.
    """
    if not (0.0 <= quality <= 1.0):
        raise ValueError(f"quality must be in [0, 1]; got {quality}.")

    if physics.degraded:
        if physics.hard_valid is not None:
            raise ValueError("degraded physics must carry hard_valid=None.")
        return GatedScore(
            episode_id=physics.episode_id,
            Q=float(quality),
            gated_out=False,
            degraded=True,
            components={"quality": float(quality), "hard_valid": float("nan")},
        )

    if not isinstance(physics.hard_valid, bool):
        kind = type(physics.hard_valid).__name__
        raise ValueError(f"hard_valid must be a genuine bool for the gate; got {kind}.")
    valid = physics.hard_valid
    Q = float(quality) if valid else 0.0
    return GatedScore(
        episode_id=physics.episode_id,
        Q=Q,
        gated_out=not valid,
        degraded=False,
        components={
            "quality": float(quality),
            "pass_rate": float(physics.pass_rate),
            "hard_valid": float(valid),
        },
    )


@runtime_checkable
class ConstraintBackend(Protocol):
    """The v0.1 extension seam: a backend turns an episode trajectory into a check result.

    Exactly one backend ships in v0.1.0a1 (:class:`RoboticsConstraintBackend`). The protocol is
    real so a v0.2 backend (e.g. self-collision, dynamics) can register via the
    ``tracecal.backends`` entry point without touching the core.
    """

    name: str

    def evaluate(
        self, positions: np.ndarray, *, spec: EmbodimentSpec, fps: float, episode_id: str
    ) -> PhysicsCheckResult: ...


@dataclass(frozen=True)
class RoboticsConstraintBackend:
    """URDF joint position/velocity hard-gate backend (the only v0.1.0a1 backend)."""

    name: str = "robotics"

    def evaluate(
        self, positions: np.ndarray, *, spec: EmbodimentSpec, fps: float, episode_id: str
    ) -> PhysicsCheckResult:
        return check_episode(positions, spec=spec, fps=fps, episode_id=episode_id)
