"""URDF joint-limit physics gate (the moat, pillar 2).

For industrial arms whose URDF resolves (Franka Panda, KUKA iiwa, and any arm the user points
at a plain URDF), an episode that drives a joint past its hard position/velocity limit is
*kinematically impossible*. :mod:`tracecal.physics.kinematics` turns a joint-position trajectory
into per-step hard/soft checks; :mod:`tracecal.physics.gate` combines them multiplicatively so
one hard violation forces ``Q = 0``. Embodiments with no resolvable URDF are returned as a
degraded :class:`~tracecal.schema.EmbodimentSpec` (physics-skipped, never silently validated).

``yourdfpy`` and ``robot_descriptions`` are imported lazily, only when a URDF is actually
resolved, so importing :mod:`tracecal.physics` stays dependency-light.
"""

from __future__ import annotations
