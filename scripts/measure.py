#!/usr/bin/env python3
"""S6 measurement: produce results/*.json with honest, reproducible metrics.

Two tiers (the README numbers are generated from these files; never hand-written):

* ``results/gate_v0.1.0a1.json`` — the v0.1.0a1 CLAIM. A REAL industrial-arm URDF (Franka Panda
  via robot_descriptions) plus an *adversarial synthetic* joint trajectory: a kinematically
  impossible episode is forced to ``Q = 0`` (gated_out), a clean one is not. mode="synthetic"
  (the trajectory is synthetic — a real robot will not violate its own limits) with a disclaimer,
  and ``data_provenance`` citing the real URDF source.
* ``results/calibration_v0.1.0a1.json`` — split-conformal coverage converges to its target and the
  isotonic ECE on synthetic exchangeable data. mode="synthetic", algorithm-validation only.

Optionally, set ``TRACECAL_REAL_DATASET=<hf repo id>`` to also run a real public LeRobot dataset
(``results/real_v0.1.0a1.json``, mode="live"); on any failure that tier is honestly recorded as
deferred to v0.1.1 rather than faked.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import date
from pathlib import Path

import numpy as np

from tracecal import __version__
from tracecal.conformal.calibrate import (
    expected_calibration_error,
    fit_split_conformal,
    isotonic_fit,
)
from tracecal.conformal.coverage import coverage_indicators
from tracecal.conformal.split import make_split
from tracecal.physics.gate import gated_score
from tracecal.physics.kinematics import check_episode
from tracecal.physics.resolver import resolve_embodiment

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 0
GATE_DISCLAIMER = (
    "Adversarial SYNTHETIC joint trajectory evaluated against a REAL Franka Panda URDF. The "
    "trajectory is synthetic because a physical robot does not violate its own joint limits; "
    "this demonstrates the gate logic (algorithm validation only), not a measured failure rate "
    "of any real dataset."
)
CAL_DISCLAIMER = (
    "Synthetic exchangeable (score, label) data; algorithm validation only. Real-dataset "
    "conformal coverage requires real binary validity labels (reference-mode otherwise)."
)


def _provenance(mode: str, extra: dict[str, str]) -> dict[str, str]:
    p = {
        "tool": "tracecal",
        "version": __version__,
        "python": platform.python_version(),
        "os": platform.system(),
        "date": date.today().isoformat(),
        "seed": str(SEED),
        "mode": mode,
    }
    p.update(extra)
    return p


def measure_gate() -> dict:
    """Demonstrate the multiplicative hard gate on a real Panda URDF + synthetic trajectories."""
    spec = resolve_embodiment("panda")  # fetches the real URDF via robot_descriptions
    if not spec.resolved:
        raise RuntimeError(f"could not resolve a real Panda URDF for the gate demo: {spec.source}")
    dof = spec.dof
    lowers = np.array([j.lower for j in spec.joints])
    uppers = np.array([j.upper for j in spec.joints])
    mids = (lowers + uppers) / 2.0
    span = (uppers - lowers) / 2.0
    rng = np.random.default_rng(SEED)
    fps = 10.0
    T = 40

    def clean_traj() -> np.ndarray:
        t = np.linspace(0, 1, T)
        # gentle within-limit oscillation at 30% of each joint's half-range
        return mids[None, :] + 0.3 * span[None, :] * np.sin(2 * np.pi * t)[:, None]

    n_clean = n_bad = 10
    n_bad_gated = 0
    example = None
    for _ in range(n_clean):
        pc = check_episode(clean_traj(), spec=spec, fps=fps, episode_id="clean")
        assert pc.hard_valid is True  # a within-limit trajectory must pass
    for k in range(n_bad):
        bad = clean_traj()
        j = rng.integers(0, dof)
        bad[T // 2, j] = uppers[j] + (0.5 + rng.random())  # push one joint clearly past its limit
        pb = check_episode(bad, spec=spec, fps=fps, episode_id=f"bad{k}")
        gb = gated_score(pb.pass_rate, pb)
        if gb.gated_out and pb.hard_valid is False and gb.Q == 0.0:
            n_bad_gated += 1
            if example is None:
                example = {"gated_out": gb.gated_out, "hard_valid": pb.hard_valid, "Q": gb.Q}
    caught_rate = n_bad_gated / n_bad

    assert example is not None
    example["physics_caught_rate"] = caught_rate
    return {
        "mode": "synthetic",
        "disclaimer": GATE_DISCLAIMER,
        "data_provenance": spec.source,
        "embodiment": {"robot_type": spec.robot_type, "dof": dof, "source": spec.source},
        "gate_demonstration": example,
        "summary": {
            "n_clean": n_clean,
            "n_violating": n_bad,
            "n_gated_out_on_violation": n_bad_gated,
            "physics_caught_rate": caught_rate,
        },
        "provenance": _provenance("synthetic", {"urdf": spec.source}),
    }


def measure_calibration() -> dict:
    """Split-conformal coverage vs target and isotonic ECE on synthetic exchangeable data."""
    alpha = 0.1
    target = 1.0 - alpha
    n_trials = 80
    covs = []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        n = 400
        y = rng.integers(0, 2, size=n).astype(float)
        s = y + rng.normal(0, 1.0, size=n)
        sp = make_split([f"emb{i % 4}" for i in range(n)], group_by_embodiment=True, seed=seed)
        m = fit_split_conformal(s[sp.cal_idx], y[sp.cal_idx], score_key="v", alpha=alpha)
        sets = m.predict_set(s[sp.holdout_idx], alpha=alpha)
        covs.append(float(coverage_indicators(sets, y[sp.holdout_idx]).mean()))
    empirical = float(np.mean(covs))

    # ECE of the isotonic-calibrated score on a fresh synthetic draw
    rng = np.random.default_rng(SEED)
    n = 2000
    y = rng.integers(0, 2, size=n).astype(float)
    s = y + rng.normal(0, 1.0, size=n)
    cal = isotonic_fit(s[: n // 2], y[: n // 2])
    probs = cal.predict(s[n // 2 :])
    ece = expected_calibration_error(probs, y[n // 2 :], n_bins=10)

    return {
        "mode": "synthetic",
        "disclaimer": CAL_DISCLAIMER,
        "target_coverage": target,
        "empirical_coverage": empirical,
        "n_trials": n_trials,
        "ece": ece,
        "provenance": _provenance("synthetic", {}),
    }


def measure_real(repo_id: str) -> dict:
    """Best-effort real public dataset run (mode=live). Caller wraps failures into a deferral."""
    from tracecal.api import evaluate_dataset

    report = evaluate_dataset(repo_id, max_episodes=50)
    return {
        "mode": "live",
        "data_provenance": repo_id,
        "summary": {
            "n_episodes": report.n_episodes,
            "n_accept": report.n_accept,
            "n_hold": report.n_hold,
            "n_reject": report.n_reject,
            "n_degraded": report.n_degraded,
            "coverage": None if report.coverage is None else report.coverage.empirical_coverage,
            "reference_mode": report.coverage is None,
        },
        "embodiments": {k: s.source for k, s in report.embodiments.items()},
        "provenance": _provenance(
            "live", {"source": report.source if hasattr(report, "source") else repo_id}
        ),
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    gate = measure_gate()
    (RESULTS / "gate_v0.1.0a1.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    g_sum = gate["summary"]
    print(
        f"gate: caught {g_sum['n_gated_out_on_violation']}/{g_sum['n_violating']} "
        f"violations (Q=0); URDF={gate['data_provenance']}"
    )

    cal = measure_calibration()
    (RESULTS / "calibration_v0.1.0a1.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(
        f"calibration: empirical coverage {cal['empirical_coverage']:.3f} "
        f"(target {cal['target_coverage']:.3f}), ECE {cal['ece']:.3f}"
    )

    repo = os.environ.get("TRACECAL_REAL_DATASET")
    if repo:
        try:
            real = measure_real(repo)
            (RESULTS / "real_v0.1.0a1.json").write_text(
                json.dumps(real, indent=2), encoding="utf-8"
            )
            print(f"real: {repo} -> {real['summary']}")
        except Exception as exc:  # noqa: BLE001 - honest deferral, never a fake number
            deferred = {
                "mode": "real",
                "status": "deferred-to-v0.1.1",
                "data_provenance": repo,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "provenance": _provenance("real", {"source": repo}),
            }
            (RESULTS / "real_v0.1.0a1.json").write_text(
                json.dumps(deferred, indent=2), encoding="utf-8"
            )
            print(f"real: {repo} FAILED -> recorded as deferred-to-v0.1.1 ({type(exc).__name__})")
    else:
        print("real: skipped (set TRACECAL_REAL_DATASET=<hf repo id> to run a live dataset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
