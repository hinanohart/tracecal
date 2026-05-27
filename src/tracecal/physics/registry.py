"""Map a LeRobot ``robot_type`` string to a ``robot_descriptions`` module.

Only industrial arms with a directly-loadable URDF are confidently supported by v0.1.0a1
(Franka Panda, KUKA iiwa — confirmed to expose ``URDF_PATH`` with joint limits). UR / xArm
entries are present but their ``robot_descriptions`` modules may ship MJCF/xacro only; when a
plain ``URDF_PATH`` is unavailable the resolver degrades rather than guessing.
"""

from __future__ import annotations

# Normalised substring -> robot_descriptions module name. Order matters: more specific first.
_REGISTRY: tuple[tuple[str, str], ...] = (
    ("panda", "panda_description"),
    ("franka", "panda_description"),
    ("iiwa14", "iiwa14_description"),
    ("iiwa", "iiwa14_description"),
    ("kuka", "iiwa14_description"),
    ("ur10e", "ur10e_description"),
    ("ur10", "ur10_description"),
    ("ur5e", "ur5e_description"),
    ("ur5", "ur5_description"),
    ("ur3e", "ur3e_description"),
    ("ur3", "ur3_description"),
    ("xarm7", "xarm7_description"),
    ("xarm6", "xarm6_description"),
    ("xarm", "xarm7_description"),
)

# Embodiments known to lack a resolvable URDF in robot_descriptions (degrade-first-class).
# Listed only for clearer degrade messages; absence here is NOT treated as resolvable.
_KNOWN_NO_URDF: frozenset[str] = frozenset(
    {"so100", "so101", "so-100", "so-101", "koch", "lekiwi", "aloha", "moss", "stretch", "widowx"}
)


# The complete set of module names this package will ever import from robot_descriptions.
# Used as an import allowlist so a dynamic import can never load an arbitrary module.
ALLOWED_MODULES: frozenset[str] = frozenset(module for _, module in _REGISTRY)


def normalize_robot_type(robot_type: str) -> str:
    """Lowercase and strip separators so 'SO-101' / 'so_101' / 'so101' compare equal."""
    return robot_type.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def resolve_module(robot_type: str) -> str | None:
    """Return the ``robot_descriptions`` module for ``robot_type``, or None if unmapped."""
    norm = normalize_robot_type(robot_type)
    for key, module in _REGISTRY:
        if key.replace("_", "").replace("-", "") in norm:
            return module
    return None


def is_known_no_urdf(robot_type: str) -> bool:
    """True for embodiments we know have no resolvable URDF (gives a precise degrade reason)."""
    norm = normalize_robot_type(robot_type)
    return any(k.replace("-", "") in norm for k in _KNOWN_NO_URDF)
