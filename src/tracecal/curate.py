"""Thin curation sugar over a :class:`~tracecal.schema.DatasetReport`.

``curate`` does not re-implement any logic: it reads the per-episode verdicts produced by the
audit engine and turns them into the artifacts a training pipeline wants — a kept-id list, an
abstain mask, and per-episode sample weights. This is the small A2-derived convenience layer;
the value is entirely in the engine (conformal + physics gate) it wraps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tracecal.schema import DatasetReport


@dataclass(frozen=True)
class CurateResult:
    """Curation artifacts derived from a dataset's verdicts."""

    kept_ids: tuple[str, ...]  # verdict == accept
    held_ids: tuple[str, ...]  # verdict == hold (abstained or physics-skipped)
    rejected_ids: tuple[str, ...]  # verdict == reject (hard physics failure, Q=0)
    weights: dict[str, float] = field(default_factory=dict)  # accept→1.0, else 0.0

    @property
    def n_kept(self) -> int:
        return len(self.kept_ids)


def curate(report: DatasetReport, *, hold_weight: float = 0.0) -> CurateResult:
    """Turn a :class:`DatasetReport` into kept/held/rejected ids and sample weights.

    ``hold_weight`` (default 0.0) is the weight given to abstained/held episodes; set it to a
    small value to down-weight rather than drop uncertain episodes. Rejected (hard-invalid)
    episodes always get weight 0.0 — that is the whole point of the gate.
    """
    if not (0.0 <= hold_weight <= 1.0):
        raise ValueError(f"hold_weight must be in [0, 1]; got {hold_weight}.")
    kept: list[str] = []
    held: list[str] = []
    rejected: list[str] = []
    weights: dict[str, float] = {}
    for v in report.verdicts:
        if v.verdict == "accept":
            kept.append(v.episode_id)
            weights[v.episode_id] = 1.0
        elif v.verdict == "hold":
            held.append(v.episode_id)
            weights[v.episode_id] = hold_weight
        else:  # reject
            rejected.append(v.episode_id)
            weights[v.episode_id] = 0.0
    return CurateResult(
        kept_ids=tuple(kept),
        held_ids=tuple(held),
        rejected_ids=tuple(rejected),
        weights=weights,
    )
