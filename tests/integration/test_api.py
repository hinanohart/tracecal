"""S4 integration: evaluate_dataset plumbing in reference-mode and physics-off (no yourdfpy)."""

from __future__ import annotations

import numpy as np

from tracecal import evaluate_dataset
from tracecal.io.lerobot_v3 import from_arrays
from tracecal.schema import DatasetReport


def _ds(robot_type: str = "so101"):
    t = np.linspace(0, 1, 20)
    clean = np.stack([0.5 * np.sin(t), 0.3 * np.cos(t)], axis=1)
    return from_arrays(
        robot_type=robot_type,
        fps=10.0,
        joint_names=("j1", "j2"),
        episodes=[("e0", clean), ("e1", clean), ("e2", clean)],
        dataset="toy",
    )


def test_degraded_embodiment_all_hold_reference_mode() -> None:
    # so101 has no resolvable URDF -> every episode degrades to hold, coverage stays None.
    report = evaluate_dataset(_ds("so101"))
    assert isinstance(report, DatasetReport)
    assert report.n_degraded == 3
    assert report.n_hold == 3
    assert report.n_accept == 0 and report.n_reject == 0
    assert report.coverage is None  # reference-mode (also no labels)
    assert report.provenance["mode"] == "reference"
    assert any("reference-mode" in w for w in report.warnings)


def test_physics_off_skips_gate() -> None:
    report = evaluate_dataset(_ds("panda"), physics="off")
    assert report.n_degraded == 3  # physics forced off -> all degraded/hold
    assert report.embodiments["panda"].resolved is False
    assert "physics-off" in report.embodiments["panda"].source


def test_confidence_bounds_validated() -> None:
    import pytest

    with pytest.raises(ValueError, match="confidence"):
        evaluate_dataset(_ds(), confidence=1.5)


def test_counts_sum_to_n_episodes() -> None:
    report = evaluate_dataset(_ds("so101"))
    assert report.n_accept + report.n_hold + report.n_reject == report.n_episodes
