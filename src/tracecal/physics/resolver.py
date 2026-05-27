"""Resolve a ``robot_type`` (+ optional URDF path) into a usable :class:`EmbodimentSpec`.

Resolution order:

1. an explicit ``urdf_path`` (user-supplied or a vendored fixture), else
2. a ``robot_descriptions`` module looked up via :mod:`tracecal.physics.registry`.

A spec is *resolved* only when joint limits could be extracted for the requested
``joint_names`` (or all limited joints, if names are not given). Anything else — no mapping,
no plain ``URDF_PATH``, missing parser, or a state-column/URDF name mismatch — returns a
**degraded** spec with an explicit reason. The physics gate is then skipped for that embodiment;
it is never silently "validated".
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

from tracecal.physics.registry import is_known_no_urdf, resolve_module
from tracecal.schema import EmbodimentSpec, JointLimits

# Static dispatch with *literal* module strings: the only robot_descriptions submodules this
# package can import. Each thunk is lazy (the import triggers robot_descriptions' on-demand URDF
# fetch), and the keys are exactly the registry's module names. This is the import whitelist —
# a value not present here can never reach importlib.import_module.
_MODULE_IMPORTERS: dict[str, Callable[[], ModuleType]] = {
    "panda_description": lambda: importlib.import_module("robot_descriptions.panda_description"),
    "iiwa14_description": lambda: importlib.import_module("robot_descriptions.iiwa14_description"),
    "ur10e_description": lambda: importlib.import_module("robot_descriptions.ur10e_description"),
    "ur10_description": lambda: importlib.import_module("robot_descriptions.ur10_description"),
    "ur5e_description": lambda: importlib.import_module("robot_descriptions.ur5e_description"),
    "ur5_description": lambda: importlib.import_module("robot_descriptions.ur5_description"),
    "ur3e_description": lambda: importlib.import_module("robot_descriptions.ur3e_description"),
    "ur3_description": lambda: importlib.import_module("robot_descriptions.ur3_description"),
    "xarm7_description": lambda: importlib.import_module("robot_descriptions.xarm7_description"),
    "xarm6_description": lambda: importlib.import_module("robot_descriptions.xarm6_description"),
}


def _degraded(robot_type: str, reason: str) -> EmbodimentSpec:
    return EmbodimentSpec(
        robot_type=robot_type, resolved=False, source=f"degraded:{reason}", dof=0, joints=()
    )


def _load_urdf_joint_limits(urdf_path: str) -> dict[str, JointLimits]:
    """Parse a URDF and return {joint_name: JointLimits} for revolute/prismatic limited joints."""
    import yourdfpy  # lazy: only needed when a URDF is actually resolved

    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=False)
    out: dict[str, JointLimits] = {}
    for joint in urdf.robot.joints:
        if joint.type not in ("revolute", "prismatic"):
            continue
        limit = joint.limit
        if limit is None or limit.lower is None or limit.upper is None:
            continue
        if not limit.lower < limit.upper:
            continue  # a fixed/degenerate joint range is not a usable limit
        vel = float(limit.velocity) if getattr(limit, "velocity", None) else None
        out[joint.name] = JointLimits(
            name=joint.name,
            lower=float(limit.lower),
            upper=float(limit.upper),
            velocity=vel if (vel is not None and vel > 0.0) else None,
            effort=float(limit.effort) if getattr(limit, "effort", None) else None,
        )
    return out


def _urdf_path_for_module(module: str) -> str | None:
    """Return ``robot_descriptions.<module>.URDF_PATH`` if present, else None (degrade).

    ``module`` is dispatched through :data:`_MODULE_IMPORTERS`, whose keys are literal module
    names; an unknown name is refused before any import, so no arbitrary module can be loaded.
    """
    importer = _MODULE_IMPORTERS.get(module)
    if importer is None:
        return None
    try:
        mod = importer()
    except Exception:
        return None
    return getattr(mod, "URDF_PATH", None)


def _align(
    limits: dict[str, JointLimits], joint_names: tuple[str, ...] | None
) -> tuple[JointLimits, ...] | None:
    """Order limits to match ``joint_names`` (best-effort name matching), or all if names absent.

    Returns None when a requested name cannot be matched to a URDF joint (→ degrade).
    """
    if joint_names is None:
        return tuple(limits[k] for k in sorted(limits))
    by_norm = {k.lower(): v for k, v in limits.items()}
    ordered: list[JointLimits] = []
    for name in joint_names:
        cand = limits.get(name) or by_norm.get(name.lower())
        if cand is None:
            # suffix/substring fallback (e.g. state column 'joint1' vs URDF 'panda_joint1')
            matches = [
                v
                for k, v in limits.items()
                if name.lower() in k.lower() or k.lower() in name.lower()
            ]
            if len(matches) == 1:
                cand = matches[0]
        if cand is None:
            return None
        ordered.append(cand)
    return tuple(ordered)


def resolve_embodiment(
    robot_type: str,
    joint_names: tuple[str, ...] | None = None,
    *,
    urdf_path: str | None = None,
) -> EmbodimentSpec:
    """Resolve ``robot_type`` to an :class:`EmbodimentSpec`, degrading honestly on any failure."""
    path = urdf_path
    source: str
    if path is not None:
        source = f"urdf:{path}"
    else:
        module = resolve_module(robot_type)
        if module is None:
            reason = "no-urdf-known" if is_known_no_urdf(robot_type) else "unmapped-robot-type"
            return _degraded(robot_type, reason)
        path = _urdf_path_for_module(module)
        if path is None:
            return _degraded(robot_type, f"no-plain-urdf:{module}")
        source = f"robot_descriptions:{module}"

    try:
        limits = _load_urdf_joint_limits(path)
    except Exception as exc:  # noqa: BLE001 — any parse failure must degrade, not crash a run
        return _degraded(robot_type, f"urdf-parse-error:{type(exc).__name__}")
    if not limits:
        return _degraded(robot_type, "no-limited-joints")

    aligned = _align(limits, joint_names)
    if aligned is None:
        return _degraded(robot_type, "joint-name-mismatch")

    return EmbodimentSpec(
        robot_type=robot_type, resolved=True, source=source, dof=len(aligned), joints=aligned
    )
