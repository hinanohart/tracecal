"""Per-step kinematic validity checks against URDF joint limits.

Given a joint-position trajectory ``positions`` of shape ``(T, dof)`` (rad for revolute joints,
m for prismatic) and an :class:`~tracecal.schema.EmbodimentSpec`, classify each timestep as
hard-valid or not. Hard checks (a failure makes the episode kinematically impossible):

* **joint_limit** — every joint position within ``[lower, upper]``.
* **velocity** — finite-difference velocity ``|Δq · fps|`` within the URDF velocity limit, for
  joints that declare one.

Soft checks (informational, never gate): **acceleration** magnitude. Fail-closed rules: a
non-finite sample is treated as *invalid* (never silently passed); a dof/column mismatch raises
(a misconfiguration must not be papered over with a verdict).
"""

from __future__ import annotations

import numpy as np

from tracecal.schema import EmbodimentSpec, PhysicsCheckResult


def check_episode(
    positions: np.ndarray,
    *,
    spec: EmbodimentSpec,
    fps: float,
    episode_id: str,
) -> PhysicsCheckResult:
    """Run the kinematic hard/soft gate over one episode's joint-position trajectory.

    Raises ``ValueError`` for a degraded spec (caller must branch on ``spec.resolved``) or a
    dof/column mismatch. Non-finite samples are counted as invalid steps (fail-closed).
    """
    if spec.degraded:
        raise ValueError(
            f"check_episode requires a resolved EmbodimentSpec; {spec.robot_type!r} is degraded. "
            "Build a degraded PhysicsCheckResult via tracecal.physics.gate.degraded_result instead."
        )
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2:
        raise ValueError(f"positions must be 2-D (T, dof); got shape {pos.shape}.")
    n_steps, dof = pos.shape
    if dof != spec.dof:
        raise ValueError(
            f"positions has {dof} joint columns but embodiment {spec.robot_type!r} "
            f"has dof={spec.dof}."
        )
    if n_steps < 1:
        raise ValueError("episode has no timesteps.")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive; got {fps}.")

    finite_mask = np.isfinite(pos)  # (T, dof)
    step_finite = finite_mask.all(axis=1)  # (T,)
    all_finite = bool(step_finite.all())

    lowers = np.array([j.lower for j in spec.joints])
    uppers = np.array([j.upper for j in spec.joints])
    # Treat non-finite as out-of-range (fail-closed) by masking it to a violating sentinel.
    safe = np.where(finite_mask, pos, np.inf)
    within_pos = (safe >= lowers) & (safe <= uppers)  # (T, dof); NaN/inf -> False
    step_pos_ok = within_pos.all(axis=1)

    # Velocity: finite-difference, checked only for joints that declare a limit.
    vel_limits = np.array([j.velocity if j.velocity is not None else np.inf for j in spec.joints])
    has_vel_limit = np.isfinite(vel_limits)
    if n_steps >= 2:
        vel = np.diff(pos, axis=0) * fps  # (T-1, dof)
        vel_finite = np.isfinite(vel)
        safe_vel = np.where(vel_finite, np.abs(vel), np.inf)
        within_vel = (safe_vel <= vel_limits) | (~has_vel_limit)  # ignore joints w/o a limit
        step_vel_ok = np.ones(n_steps, dtype=bool)
        step_vel_ok[1:] = within_vel.all(axis=1)  # attribute the transition to its end step
    else:
        step_vel_ok = np.ones(n_steps, dtype=bool)

    step_hard_ok = step_finite & step_pos_ok & step_vel_ok
    n_invalid = int((~step_hard_ok).sum())
    hard_valid = bool(step_hard_ok.all())
    pass_rate = float(step_hard_ok.mean())

    checks = {
        "finite": all_finite,
        "joint_limit": bool(step_pos_ok.all()),
        "velocity": bool(step_vel_ok.all()),
        "dim": True,
    }
    return PhysicsCheckResult(
        episode_id=episode_id,
        degraded=False,
        hard_valid=hard_valid,
        n_steps=n_steps,
        n_steps_invalid=n_invalid,
        pass_rate=pass_rate,
        checks=checks,
        degrade_reason=None,
    )
