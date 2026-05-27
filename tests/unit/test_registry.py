"""S3: robot_type -> robot_descriptions registry + import allowlist (no network)."""

from __future__ import annotations

from tracecal.physics.registry import (
    ALLOWED_MODULES,
    is_known_no_urdf,
    normalize_robot_type,
    resolve_module,
)


def test_normalize_collapses_separators() -> None:
    assert normalize_robot_type("SO-101") == "so101"
    assert normalize_robot_type("so_101") == "so101"
    assert normalize_robot_type(" Panda ") == "panda"


def test_resolve_known_industrial_arms() -> None:
    assert resolve_module("panda") == "panda_description"
    assert resolve_module("franka_panda") == "panda_description"
    assert resolve_module("iiwa14") == "iiwa14_description"
    assert resolve_module("kuka_iiwa") == "iiwa14_description"


def test_resolve_unmapped_is_none() -> None:
    assert resolve_module("totally_unknown_robot_xyz") is None


def test_known_no_urdf_flagged() -> None:
    assert is_known_no_urdf("so101") is True
    assert is_known_no_urdf("koch") is True
    assert is_known_no_urdf("aloha") is True
    assert is_known_no_urdf("panda") is False


def test_allowlist_matches_dispatch_keys() -> None:
    # Guard against registry/dispatch drift: every mapped module must be importable via the
    # resolver's static dispatch (and vice versa).
    from tracecal.physics.resolver import _MODULE_IMPORTERS

    assert set(_MODULE_IMPORTERS) == set(ALLOWED_MODULES)
