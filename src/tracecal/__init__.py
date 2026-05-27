"""tracecal — conformal-calibrated, URDF physics-gated validity auditing for LeRobot datasets.

The public surface is intentionally small at import time and **torch-free**: importing
``tracecal`` (or anything under :mod:`tracecal.conformal`) never imports ``lerobot``,
``torch`` or a deep-learning framework. The Hub loader (:mod:`tracecal.io`) and the URDF
physics gate (:mod:`tracecal.physics`) lazy-import their optional dependencies
(``huggingface_hub``/``pyarrow`` and ``yourdfpy``/``robot_descriptions``) only when called.

v0.1.0a1 CLAIM = the multiplicative URDF joint-limit physics gate on **industrial arms**
(Franka Panda / Universal Robots / KUKA iiwa / UFACTORY xArm): a kinematically impossible
episode is forced to ``Q = 0`` regardless of any quality signal. Conformal coverage is a
*reference-mode* diagnostic by default (it is a validated guarantee only when real binary
validity labels are supplied); cross-embodiment normalization is deferred to v0.2. Embodiments
without a resolvable URDF (e.g. SO-101 / Koch / LeKiwi) are handled in degrade-first-class
mode (physics-skipped, ``coverage=None``, verdict ``hold``) — never silently "validated".
"""

__version__ = "0.1.0a1"

__all__ = ["__version__"]
