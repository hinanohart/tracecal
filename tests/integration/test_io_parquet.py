"""S3 (hub extra): round-trip a minimal LeRobot-v3 layout through load_local via pyarrow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.hub

from tracecal.io.lerobot_v3 import load_local  # noqa: E402


def _write_v3(root: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 20,
        "robot_type": "panda",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["j1", "j2"]}
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    # two episodes (0,1), 3 frames each; episode 1 has a terminal success label
    ep_idx, fr_idx, states, success = [], [], [], []
    for e in (0, 1):
        for f in range(3):
            ep_idx.append(e)
            fr_idx.append(f)
            states.append([0.1 * f, -0.1 * f])
            success.append(1 if (e == 1 and f == 2) else 0)
    table = pa.table(
        {
            "episode_index": ep_idx,
            "frame_index": fr_idx,
            "observation.state": states,
            "next.success": success,
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def test_load_local_round_trip(tmp_path) -> None:
    _write_v3(tmp_path)
    ds = load_local(tmp_path)
    assert ds.robot_type == "panda"
    assert ds.fps == 20.0
    assert ds.joint_names == ("j1", "j2")
    assert len(ds) == 2
    e0, e1 = ds.episodes
    assert e0.record.n_steps == 3
    assert e0.positions.shape == (3, 2)
    # frames must be ordered; row f=1 is [0.1, -0.1]
    np.testing.assert_allclose(e0.positions[1], [0.1, -0.1])
    # episode 1 has terminal success -> label 1.0; episode 0 -> 0.0
    assert e1.record.success == 1.0
    assert e0.record.success == 0.0
