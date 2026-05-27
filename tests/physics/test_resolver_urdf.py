"""S3 (physics extra): URDF joint-limit extraction via yourdfpy over offline synthetic fixtures.

No manufacturer URDF is downloaded; these parse the hand-authored fixtures in tests/fixtures/urdf.
"""

from __future__ import annotations

import pytest

from tracecal.physics.resolver import resolve_embodiment

pytestmark = pytest.mark.physics


def test_resolve_from_urdf_path_2dof(urdf_dir) -> None:
    spec = resolve_embodiment(
        "synth_2dof", joint_names=("j1", "j2"), urdf_path=str(urdf_dir / "synth_2dof.urdf")
    )
    assert spec.resolved is True
    assert spec.dof == 2
    assert spec.source.startswith("urdf:")
    j1, j2 = spec.joints
    assert (j1.name, j1.lower, j1.upper, j1.velocity) == ("j1", -1.5708, 1.5708, 2.0)
    assert (j2.lower, j2.upper, j2.velocity) == (-3.1416, 3.1416, 3.0)


def test_resolve_ignores_continuous_joint_and_supports_prismatic(urdf_dir) -> None:
    spec = resolve_embodiment("synth_mixed", urdf_path=str(urdf_dir / "synth_3dof_mixed.urdf"))
    assert spec.resolved is True
    names = {j.name for j in spec.joints}
    assert names == {"rev1", "prism1", "rev2"}  # j_free (continuous, no limit) excluded
    assert spec.dof == 3


def test_joint_name_mismatch_degrades(urdf_dir) -> None:
    spec = resolve_embodiment(
        "synth_2dof", joint_names=("nope1", "nope2"), urdf_path=str(urdf_dir / "synth_2dof.urdf")
    )
    assert spec.resolved is False
    assert "joint-name-mismatch" in spec.source


def test_unparseable_urdf_degrades(tmp_path) -> None:
    bad = tmp_path / "bad.urdf"
    bad.write_text("<robot><not valid urdf", encoding="utf-8")
    spec = resolve_embodiment("x", urdf_path=str(bad))
    assert spec.resolved is False
    assert "urdf-parse-error" in spec.source or "no-limited-joints" in spec.source
