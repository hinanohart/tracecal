"""S3: in-memory dataset construction (the offline io path, no pyarrow)."""

from __future__ import annotations

import numpy as np
import pytest

from tracecal.io.lerobot_v3 import from_arrays


def test_from_arrays_builds_records() -> None:
    ds = from_arrays(
        robot_type="panda",
        fps=30.0,
        joint_names=("j1", "j2"),
        episodes=[("e0", np.zeros((50, 2)), 1.0), ("e1", np.ones((40, 2)), None)],
        dataset="toy",
    )
    assert len(ds) == 2
    assert ds.joint_names == ("j1", "j2")
    e0 = ds.episodes[0]
    assert e0.record.id == "e0"
    assert e0.record.n_steps == 50
    assert e0.record.embodiment_id == "panda"
    assert e0.record.success == 1.0
    assert e0.record.dataset == "toy"
    assert ds.episodes[1].record.success is None


def test_from_arrays_dof_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="positions must be"):
        from_arrays(
            robot_type="panda",
            fps=30.0,
            joint_names=("j1", "j2", "j3"),
            episodes=[("e0", np.zeros((10, 2)))],
        )


def test_from_arrays_two_tuple_defaults_success_none() -> None:
    ds = from_arrays(
        robot_type="iiwa",
        fps=20.0,
        joint_names=("a",),
        episodes=[("e0", np.zeros((5, 1)))],
    )
    assert ds.episodes[0].record.success is None
