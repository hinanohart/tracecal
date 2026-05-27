"""Typed intermediate representation (IR) — the contract across the pipeline.

These pydantic models are the contract between the loaders (:mod:`tracecal.io`), the
physics gate (:mod:`tracecal.physics`), the conformal core (:mod:`tracecal.conformal`)
and the report layer (:mod:`tracecal.report`). Heavy per-step trajectory arrays are
*not* carried in the IR — they flow as raw ``numpy`` arrays straight into the physics
functions; the IR carries metadata and per-episode *results*. Three choices encode
correctness rules at the type level:

* ``EmbodimentSpec.resolved`` makes "we could not find a URDF for this arm" a first-class,
  visible state (``resolved=False`` ⇒ degrade-mode, physics-skipped) rather than a silent
  pass.
* ``CoverageReport.source`` is ``Literal["measured"]``. There is deliberately no constructor
  path that produces an assumed/synthetic coverage number — coverage must come from a holdout
  measurement against real labels. This is the type-level enforcement of "never fabricate
  coverage".
* ``EpisodeVerdict.hard_valid`` is ``bool | None``; ``None`` means *physics was skipped*
  (degrade-mode), which is distinct from ``False`` (physics ran and the episode is invalid).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ScoreOrientation = Literal["higher_better", "lower_better"]
"""Direction of a validity score relative to the *valid* (positive) label.

``higher_better`` means a larger score makes "valid" more likely. The conformal core flips
``lower_better`` scores internally so nonconformity is always "smaller = less conforming to valid".
"""

Verdict = Literal["accept", "hold", "reject"]
"""Per-episode triage outcome. ``reject`` ⟺ a hard physics failure forced ``Q = 0``;
``hold`` ⟺ abstention (non-confident) or physics was skipped (degrade); ``accept`` ⟺
hard-valid and confidently called."""


class JointLimits(BaseModel):
    """Hard kinematic limits for one actuated joint, extracted from a URDF."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    lower: float = Field(..., description="Lower position limit (rad or m).")
    upper: float = Field(..., description="Upper position limit (rad or m).")
    velocity: float | None = Field(
        default=None, description="Max |velocity| (rad/s or m/s); None if the URDF omits it."
    )
    effort: float | None = Field(default=None, description="Max effort; informational only.")

    @model_validator(mode="after")
    def _check_order(self) -> JointLimits:
        if not self.lower < self.upper:
            raise ValueError(
                f"joint {self.name!r}: lower ({self.lower}) must be < upper ({self.upper})."
            )
        if self.velocity is not None and self.velocity <= 0.0:
            raise ValueError(
                f"joint {self.name!r}: velocity limit must be positive, got {self.velocity}."
            )
        return self


class EmbodimentSpec(BaseModel):
    """Resolved (or degraded) physical description of a robot embodiment.

    ``resolved=False`` means no URDF could be mapped for ``robot_type`` (e.g. SO-101/Koch/LeKiwi):
    the physics gate is *skipped* for episodes of this embodiment and they are reported in
    degrade-mode. ``joints`` is then empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    robot_type: str = Field(
        ..., description="Embodiment key as it appears in the dataset (e.g. 'panda')."
    )
    resolved: bool = Field(
        ..., description="True iff a URDF was resolved and joint limits extracted."
    )
    source: str = Field(
        ...,
        description="Provenance, e.g. 'robot_descriptions:panda_description' or 'degraded:no-urdf'",
    )
    dof: int = Field(..., ge=0, description="Number of limited joints (0 when degraded).")
    joints: tuple[JointLimits, ...] = Field(
        default=(), description="Per-joint limits; empty when degraded."
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> EmbodimentSpec:
        if self.resolved and (self.dof == 0 or len(self.joints) == 0):
            raise ValueError(f"embodiment {self.robot_type!r}: resolved=True requires >=1 joint.")
        if not self.resolved and len(self.joints) != 0:
            raise ValueError(
                f"embodiment {self.robot_type!r}: degraded (resolved=False) must carry no joints."
            )
        if self.resolved and self.dof != len(self.joints):
            raise ValueError(
                f"embodiment {self.robot_type!r}: dof {self.dof} != len(joints) {len(self.joints)}."
            )
        return self

    @property
    def degraded(self) -> bool:
        return not self.resolved


class EpisodeRecord(BaseModel):
    """Metadata for one LeRobot episode (the trajectory arrays flow separately as numpy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    embodiment_id: str = Field(
        ..., description="Key into the run's embodiment specs (often == robot_type)."
    )
    fps: float = Field(
        ..., gt=0.0, description="Sampling rate (Hz); used to derive velocity from positions."
    )
    n_steps: int = Field(..., ge=1)
    joint_names: tuple[str, ...] = Field(
        ..., description="Names of the joint columns in the action/state arrays."
    )
    task: str | None = None
    dataset: str | None = None
    success: float | None = Field(
        default=None,
        description="Optional binary success/validity label in {0,1} if the dataset has one.",
    )


class PhysicsCheckResult(BaseModel):
    """Outcome of the URDF kinematic gate for one episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    degraded: bool = Field(..., description="True iff physics was skipped (no resolvable URDF).")
    hard_valid: bool | None = Field(
        ..., description="All hard checks pass; None iff degraded (physics not run)."
    )
    n_steps: int = Field(..., ge=0)
    n_steps_invalid: int = Field(..., ge=0, description="Steps with >=1 hard-check violation.")
    pass_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of steps that are hard-valid."
    )
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-check overall pass/fail (joint_limit, velocity, finite, dim).",
    )
    degrade_reason: str | None = None

    @model_validator(mode="after")
    def _check_degrade(self) -> PhysicsCheckResult:
        if self.degraded and self.hard_valid is not None:
            raise ValueError(
                f"episode {self.episode_id!r}: degraded result must have hard_valid=None."
            )
        if not self.degraded and self.hard_valid is None:
            raise ValueError(
                f"episode {self.episode_id!r}: non-degraded result must have a bool hard_valid."
            )
        return self


class GroupCoverage(BaseModel):
    """Empirical coverage for a single group (embodiment or task)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    empirical_coverage: float
    nominal_violated: bool


class CoverageReport(BaseModel):
    """Holdout-measured coverage of a conformal calibration run.

    ``source`` is fixed to ``"measured"`` by type. There is no constructor path producing an
    assumed/synthetic coverage number. A run with no validity labels does not build a
    ``CoverageReport`` at all (it stays ``None`` on the dataset report = reference-mode).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_coverage: float = Field(..., description="Nominal target = 1 - alpha.")
    empirical_coverage: float = Field(..., description="Fraction covered on the holdout set.")
    n_holdout: int
    ci_level: float = 0.95
    ci_low: float
    ci_high: float
    source: Literal["measured"] = "measured"
    per_group: dict[str, GroupCoverage] = Field(default_factory=dict)
    nominal_violated: bool
    exchangeability_caveat: str | None = Field(
        default=None,
        description="Set when a naive (non-embodiment-grouped) split was used; coverage is a "
        "reference value, not a leakage-safe guarantee.",
    )


class EpisodeVerdict(BaseModel):
    """Per-episode triage decision combining the physics gate and conformal abstention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    verdict: Verdict
    Q: float = Field(..., ge=0.0, le=1.0, description="Gated score; exactly 0 iff hard-rejected.")
    hard_valid: bool | None = Field(..., description="None iff physics was skipped (degrade).")
    abstain: bool
    degraded: bool
    reasons: tuple[str, ...] = ()


class DatasetReport(BaseModel):
    """Top-level result of auditing one dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str | None
    n_episodes: int = Field(..., ge=0)
    n_accept: int = Field(..., ge=0)
    n_hold: int = Field(..., ge=0)
    n_reject: int = Field(..., ge=0)
    n_degraded: int = Field(..., ge=0)
    coverage: CoverageReport | None = Field(
        default=None, description="None ⇒ reference-mode (no validity labels supplied)."
    )
    embodiments: dict[str, EmbodimentSpec] = Field(default_factory=dict)
    verdicts: tuple[EpisodeVerdict, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_counts(self) -> DatasetReport:
        s = self.n_accept + self.n_hold + self.n_reject
        if s != self.n_episodes:
            raise ValueError(f"verdict counts {s} (a+h+r) != n_episodes {self.n_episodes}.")
        return self


__all__ = [
    "ScoreOrientation",
    "Verdict",
    "JointLimits",
    "EmbodimentSpec",
    "EpisodeRecord",
    "PhysicsCheckResult",
    "GroupCoverage",
    "CoverageReport",
    "EpisodeVerdict",
    "DatasetReport",
]
