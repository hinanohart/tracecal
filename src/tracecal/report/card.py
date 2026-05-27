"""The pure accept / hold / reject decision rule.

Combines the multiplicative physics gate with conformal (or reference-mode) abstention:

* **reject** ⟺ a hard kinematic violation forced ``Q = 0`` (physics ran, episode invalid).
* **hold**   ⟺ physics was skipped (degrade) OR the episode is hard-valid but tracecal cannot
  confidently call it (non-singleton conformal set, or a reference-mode outlier flag).
* **accept** ⟺ hard-valid and confidently valid.

This module is deliberately free of numpy/IO so the rule is trivially testable and auditable.
"""

from __future__ import annotations

from tracecal.physics.gate import GatedScore
from tracecal.schema import EpisodeVerdict, PhysicsCheckResult


def decide_verdict(
    physics: PhysicsCheckResult,
    gated: GatedScore,
    *,
    abstain: bool,
    abstain_reason: str | None = None,
) -> EpisodeVerdict:
    """Return the :class:`EpisodeVerdict` for one episode.

    ``abstain`` is supplied by the caller from the conformal prediction set (non-singleton) or
    the reference-mode outlier flag. It only matters for hard-valid episodes — a hard physics
    rejection and a degrade both take precedence over (and ignore) ``abstain``.
    """
    if physics.episode_id != gated.episode_id:
        raise ValueError("physics and gated results refer to different episodes.")

    if physics.degraded:
        reason = physics.degrade_reason or "no resolvable URDF"
        return EpisodeVerdict(
            episode_id=physics.episode_id,
            verdict="hold",
            Q=gated.Q,
            hard_valid=None,
            abstain=True,
            degraded=True,
            reasons=(f"physics-skipped: {reason}",),
        )

    if gated.gated_out:  # hard_valid is False -> Q forced to 0
        viol = [k for k, ok in physics.checks.items() if not ok]
        return EpisodeVerdict(
            episode_id=physics.episode_id,
            verdict="reject",
            Q=gated.Q,
            hard_valid=False,
            abstain=False,
            degraded=False,
            reasons=(f"hard kinematic violation: {', '.join(viol) or 'unknown'}",),
        )

    if abstain:
        return EpisodeVerdict(
            episode_id=physics.episode_id,
            verdict="hold",
            Q=gated.Q,
            hard_valid=True,
            abstain=True,
            degraded=False,
            reasons=(abstain_reason or "not confident at the requested risk level",),
        )

    return EpisodeVerdict(
        episode_id=physics.episode_id,
        verdict="accept",
        Q=gated.Q,
        hard_valid=True,
        abstain=False,
        degraded=False,
        reasons=(),
    )
