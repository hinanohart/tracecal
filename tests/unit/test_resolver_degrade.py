"""S3: resolver degrade-first-class paths that need no URDF parsing (no yourdfpy/network)."""

from __future__ import annotations

from tracecal.physics.resolver import resolve_embodiment


def test_unmapped_robot_type_degrades() -> None:
    spec = resolve_embodiment("totally_unknown_robot_xyz")
    assert spec.resolved is False
    assert spec.degraded is True
    assert "unmapped-robot-type" in spec.source
    assert spec.joints == ()
    assert spec.dof == 0


def test_known_no_urdf_arm_degrades_with_precise_reason() -> None:
    spec = resolve_embodiment("so101")
    assert spec.resolved is False
    assert "no-urdf-known" in spec.source


def test_degraded_spec_is_well_formed() -> None:
    # A degraded spec must satisfy the schema invariants (resolved=False -> no joints).
    spec = resolve_embodiment("koch")
    assert spec.dof == 0 and spec.joints == ()
