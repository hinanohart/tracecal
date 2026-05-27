"""Public API: audit a LeRobot dataset and return a calibrated :class:`DatasetReport`.

``evaluate_dataset`` ties the pipeline together: load (local dir / Hub repo id / in-memory) →
resolve each embodiment's URDF (degrade if none) → run the kinematic gate → calibrate validity
(supervised split-conformal when binary labels exist, else reference-mode with no coverage
claim) → decide accept/hold/reject per episode. It is GPU-free and torch-free.
"""

from __future__ import annotations

import os
import platform
from datetime import date

import numpy as np

from tracecal import __version__
from tracecal.conformal.calibrate import (
    MondrianConformal,
    SplitConformalBinary,
    fit_mondrian_conformal,
    fit_split_conformal,
)
from tracecal.conformal.coverage import build_coverage_report
from tracecal.conformal.nonconformity import reference_mode_flags
from tracecal.conformal.split import make_split
from tracecal.io.lerobot_v3 import LeRobotV3Dataset, load_hub, load_local
from tracecal.physics.gate import degraded_result, gated_score
from tracecal.physics.kinematics import check_episode
from tracecal.physics.resolver import resolve_embodiment
from tracecal.report.card import decide_verdict
from tracecal.schema import CoverageReport, DatasetReport, EmbodimentSpec, EpisodeVerdict

MIN_LABELED_TOTAL = 8  # below this, calibration is too small to claim coverage → reference-mode


def _load(source: object, *, state_key: str, max_episodes: int | None) -> LeRobotV3Dataset:
    if isinstance(source, LeRobotV3Dataset):
        return source
    s = str(source)
    if os.path.exists(s):
        return load_local(s, state_key=state_key, max_episodes=max_episodes)
    return load_hub(s, state_key=state_key, max_episodes=max_episodes)


def _resolve_specs(
    ds: LeRobotV3Dataset, *, physics: str, urdf_overrides: dict[str, str]
) -> dict[str, EmbodimentSpec]:
    specs: dict[str, EmbodimentSpec] = {}
    for ep in ds.episodes:
        rt = ep.record.embodiment_id
        if rt in specs:
            continue
        if physics == "off":
            specs[rt] = EmbodimentSpec(
                robot_type=rt, resolved=False, source="degraded:physics-off", dof=0
            )
        else:
            specs[rt] = resolve_embodiment(
                rt, ep.record.joint_names, urdf_path=urdf_overrides.get(rt)
            )
    return specs


def evaluate_dataset(
    source: str | os.PathLike[str] | LeRobotV3Dataset,
    *,
    confidence: float = 0.9,
    mondrian_by: str | None = "embodiment",
    physics: str = "auto",
    urdf_overrides: dict[str, str] | None = None,
    max_episodes: int | None = None,
    state_key: str = "observation.state",
    seed: int = 0,
    holdout_fraction: float = 0.3,
) -> DatasetReport:
    """Audit ``source`` and return a :class:`DatasetReport`.

    Args:
        source: a local v3 dataset dir, a Hugging Face dataset repo id, or a pre-loaded
            :class:`~tracecal.io.lerobot_v3.LeRobotV3Dataset`.
        confidence: target coverage ``1 - alpha`` for the conformal prediction sets.
        mondrian_by: ``"embodiment"`` for group-conditional calibration, or ``None`` for pooled.
        physics: ``"auto"`` (resolve URDFs, degrade if none), ``"off"`` (skip physics entirely).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
    if mondrian_by not in (None, "embodiment"):
        raise ValueError("mondrian_by must be 'embodiment' or None (task-conditional is v0.2).")
    if physics not in ("auto", "off"):
        raise ValueError("physics must be 'auto' or 'off'.")
    alpha = 1.0 - confidence
    urdf_overrides = urdf_overrides or {}

    ds = _load(source, state_key=state_key, max_episodes=max_episodes)
    specs = _resolve_specs(ds, physics=physics, urdf_overrides=urdf_overrides)
    n = len(ds.episodes)

    # --- physics gate per episode ---
    phys = []
    for ep in ds.episodes:
        spec = specs[ep.record.embodiment_id]
        if spec.resolved:
            try:
                phys.append(
                    check_episode(
                        ep.positions, spec=spec, fps=ep.record.fps, episode_id=ep.record.id
                    )
                )
                continue
            except ValueError as exc:  # malformed trajectory -> degrade this episode, don't crash
                reason = f"physics-error:{type(exc).__name__}"
        else:
            reason = spec.source.removeprefix("degraded:")
        phys.append(degraded_result(ep.record.id, reason, ep.record.n_steps))

    scores = np.array([p.pass_rate for p in phys], dtype=float)
    embs = [ep.record.embodiment_id for ep in ds.episodes]
    labels = [ep.record.success for ep in ds.episodes]
    resolved_idx = [i for i, p in enumerate(phys) if not p.degraded]

    abstain = [False] * n
    abstain_reason: list[str | None] = [None] * n
    coverage: CoverageReport | None = None
    warnings: list[str] = []

    labeled_idx = [i for i in resolved_idx if labels[i] is not None]
    lab_y = np.array([labels[i] for i in labeled_idx], dtype=float) if labeled_idx else np.array([])
    supervised = len(labeled_idx) >= MIN_LABELED_TOTAL and set(lab_y.tolist()) == {0.0, 1.0}

    if supervised:
        coverage, sup_warnings = _supervised_calibration(
            scores=scores,
            embs=embs,
            labeled_idx=labeled_idx,
            lab_y=lab_y,
            resolved_idx=resolved_idx,
            confidence=confidence,
            mondrian_by=mondrian_by,
            holdout_fraction=holdout_fraction,
            seed=seed,
            abstain=abstain,
            abstain_reason=abstain_reason,
        )
        warnings.extend(sup_warnings)
    else:
        if resolved_idx:
            flags = reference_mode_flags(
                scores[resolved_idx], [embs[i] for i in resolved_idx], alpha=alpha
            )
            for k, i in enumerate(resolved_idx):
                if bool(flags[k]):
                    abstain[i] = True
                    abstain_reason[i] = (
                        "reference-mode outlier (low validity score for its embodiment)"
                    )
        warnings.append(
            "reference-mode: no binary validity labels available; coverage is not claimed "
            "(coverage=None). Abstention is a self-supervised heuristic, not a guarantee."
        )

    # --- per-episode verdicts ---
    verdicts: list[EpisodeVerdict] = []
    for i in range(n):
        p = phys[i]
        quality = p.pass_rate if not p.degraded else 0.0
        g = gated_score(quality, p)
        verdicts.append(decide_verdict(p, g, abstain=abstain[i], abstain_reason=abstain_reason[i]))

    n_accept = sum(1 for v in verdicts if v.verdict == "accept")
    n_hold = sum(1 for v in verdicts if v.verdict == "hold")
    n_reject = sum(1 for v in verdicts if v.verdict == "reject")
    n_degraded = sum(1 for p in phys if p.degraded)

    return DatasetReport(
        dataset=ds.episodes[0].record.dataset if ds.episodes else None,
        n_episodes=n,
        n_accept=n_accept,
        n_hold=n_hold,
        n_reject=n_reject,
        n_degraded=n_degraded,
        coverage=coverage,
        embodiments=specs,
        verdicts=tuple(verdicts),
        warnings=tuple(warnings),
        provenance={
            "tool": "tracecal",
            "version": __version__,
            "python": platform.python_version(),
            "os": platform.system(),
            "date": date.today().isoformat(),
            "source": ds.source,
            "mode": "supervised" if supervised else "reference",
            "confidence": f"{confidence}",
            "seed": str(seed),
        },
    )


def _supervised_calibration(
    *,
    scores: np.ndarray,
    embs: list[str],
    labeled_idx: list[int],
    lab_y: np.ndarray,
    resolved_idx: list[int],
    confidence: float,
    mondrian_by: str | None,
    holdout_fraction: float,
    seed: int,
    abstain: list[bool],
    abstain_reason: list[str | None],
) -> tuple[CoverageReport, list[str]]:
    """Fit split-conformal on labelled episodes, measure holdout coverage, set abstain flags."""
    warnings: list[str] = []
    lab_scores = scores[labeled_idx]
    lab_embs = [embs[i] for i in labeled_idx]
    split = make_split(
        lab_embs,
        group_by_embodiment=(mondrian_by == "embodiment"),
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    if split.caveat:
        warnings.append(split.caveat)
    cal_s, cal_y = lab_scores[split.cal_idx], lab_y[split.cal_idx]
    cal_bins = np.array([lab_embs[i] for i in split.cal_idx], dtype=object)
    alpha = 1.0 - confidence

    model: MondrianConformal | SplitConformalBinary
    if mondrian_by == "embodiment":
        model = fit_mondrian_conformal(cal_s, cal_y, cal_bins, score_key="validity", alpha=alpha)
    else:
        model = fit_split_conformal(cal_s, cal_y, score_key="validity", alpha=alpha)

    def _sets(idx: list[int]) -> list[frozenset[int]]:
        s = scores[idx]
        if isinstance(model, MondrianConformal):
            return model.predict_set(s, np.array([embs[i] for i in idx], dtype=object), alpha=alpha)
        return model.predict_set(s, alpha=alpha)

    # holdout coverage (measured, against real labels)
    hold_idx_global = [labeled_idx[i] for i in split.holdout_idx]
    hold_sets = _sets(hold_idx_global)
    coverage = build_coverage_report(
        hold_sets,
        lab_y[split.holdout_idx],
        [embs[i] for i in hold_idx_global],
        target_coverage=confidence,
        exchangeability_caveat=split.caveat,
    )
    significant_violation = (not np.isfinite(coverage.ci_high)) or coverage.ci_high < confidence
    if significant_violation:
        warnings.append(
            f"holdout coverage {coverage.empirical_coverage:.3f} is significantly below the target "
            f"{confidence:.3f} (95% CI upper {coverage.ci_high:.3f}); all episodes set to hold."
        )

    # abstain on non-singleton-valid prediction sets for every resolved episode
    res_sets = _sets(resolved_idx)
    for k, i in enumerate(resolved_idx):
        if significant_violation or res_sets[k] != frozenset({1}):
            abstain[i] = True
            abstain_reason[i] = (
                "blanket abstain (coverage breach)"
                if significant_violation
                else f"non-singleton conformal set {set(res_sets[k])}"
            )
    return coverage, warnings
