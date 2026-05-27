"""S4 (physics extra): full evaluate_dataset pipeline over a fixture URDF (offline)."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal import evaluate_dataset
from tracecal.io.lerobot_v3 import from_arrays

pytestmark = pytest.mark.physics


def _clean(n: int = 25) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.stack([0.5 * np.sin(t), 0.3 * np.cos(t)], axis=1)


def _violating(n: int = 25) -> np.ndarray:
    pos = _clean(n)
    pos[:, 0] = np.linspace(1.4, 1.7, n)  # ramp past j1 limit (1.5708)
    return pos


def test_end_to_end_gate_rejects_invalid(urdf_dir) -> None:
    ds = from_arrays(
        robot_type="synth_2dof",
        fps=10.0,
        joint_names=("j1", "j2"),
        episodes=[("clean0", _clean()), ("clean1", _clean()), ("bad0", _violating())],
        dataset="synthetic",
    )
    report = evaluate_dataset(ds, urdf_overrides={"synth_2dof": str(urdf_dir / "synth_2dof.urdf")})
    assert report.embodiments["synth_2dof"].resolved is True
    assert report.n_reject == 1
    assert report.n_degraded == 0
    by_id = {v.episode_id: v for v in report.verdicts}
    assert by_id["bad0"].verdict == "reject"
    assert by_id["bad0"].Q == 0.0
    # no labels -> reference-mode, no coverage claim
    assert report.coverage is None
    assert report.provenance["mode"] == "reference"


def test_supervised_run_produces_measured_coverage(urdf_dir) -> None:
    # 12 valid episodes (within limits) with explicit binary labels across 2 embodiment groups
    rng = np.random.default_rng(0)
    eps = []
    for i in range(12):
        t = np.linspace(0, 1, 20)
        amp = 0.4 + 0.1 * rng.random()
        pos = np.stack([amp * np.sin(t), 0.2 * np.cos(t)], axis=1)
        label = float(i % 2)  # both classes present
        eps.append((f"e{i}", pos, label))
    ds = from_arrays(
        robot_type="synth_2dof", fps=10.0, joint_names=("j1", "j2"), episodes=eps, dataset="sup"
    )
    report = evaluate_dataset(
        ds, urdf_overrides={"synth_2dof": str(urdf_dir / "synth_2dof.urdf")}, mondrian_by=None
    )
    assert report.provenance["mode"] == "supervised"
    assert report.coverage is not None
    assert report.coverage.source == "measured"
    assert report.coverage.n_holdout >= 1
