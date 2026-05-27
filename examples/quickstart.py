#!/usr/bin/env python3
"""tracecal quickstart — audit a synthetic LeRobot-style dataset, no network, no GPU.

Builds three episodes for a 2-DOF arm (two clean, one driving a joint past its hard limit),
resolves the URDF, runs the kinematic gate + reference-mode abstention, and prints the verdicts.
With the ``[physics]`` extra installed the bad episode is *rejected* (Q=0); without it the arm
degrades to a physics-skipped ``hold`` (degrade-first-class) — either way nothing is silently
"validated".

Run:  python examples/quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from tracecal import evaluate_dataset
from tracecal.io.lerobot_v3 import from_arrays
from tracecal.report.emit import to_html

# A minimal, hand-authored 2-DOF arm (not a manufacturer model).
_URDF = """<?xml version="1.0"?>
<robot name="demo_arm">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/><axis xyz="0 0 1"/>
    <limit lower="-3.1416" upper="3.1416" velocity="3.0" effort="10"/>
  </joint>
</robot>
"""


def main() -> None:
    t = np.linspace(0, 1, 30)
    clean = np.stack([0.5 * np.sin(t), 0.3 * np.cos(t)], axis=1)
    bad = clean.copy()
    bad[:, 0] = np.linspace(1.4, 1.7, 30)  # ramps past j1's 1.5708 limit

    dataset = from_arrays(
        robot_type="demo_arm",
        fps=10.0,
        joint_names=("j1", "j2"),
        episodes=[("clean_0", clean), ("clean_1", clean), ("over_limit", bad)],
        dataset="quickstart-synthetic",
    )

    with tempfile.TemporaryDirectory() as tmp:
        urdf_path = Path(tmp) / "demo_arm.urdf"
        urdf_path.write_text(_URDF, encoding="utf-8")
        report = evaluate_dataset(dataset, urdf_overrides={"demo_arm": str(urdf_path)})

        print(f"dataset: {report.dataset}  ({report.n_episodes} episodes)")
        print(f"mode: {report.provenance['mode']}  | coverage: {report.coverage}")
        print(
            f"accept={report.n_accept} hold={report.n_hold} reject={report.n_reject} "
            f"degraded={report.n_degraded}"
        )
        for v in report.verdicts:
            print(f"  {v.episode_id:12s} -> {v.verdict:6s} Q={v.Q:.3f}  {'; '.join(v.reasons)}")

        out = Path(tmp) / "report.html"  # self-contained HTML card
        out.write_text(to_html(report), encoding="utf-8")
        print(f"\n(HTML card would be written to {out.name}; {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
