"""Distribution-free conformal calibration of episode validity (the moat, pillar 1).

Numpy-only, correctness-critical, transcribed from the foldgauge split-conformal core and
adapted to robot episodes: calibration/holdout splits are grouped by *embodiment* (the
robotics analogue of foldgauge's homology clusters) to keep the coverage guarantee honest,
and group-conditional (Mondrian) calibration is implemented natively per embodiment/task.

The guarantee is real only when binary validity labels are supplied. With no labels the
caller stays in reference-mode (``coverage=None``) and uses
:func:`tracecal.conformal.nonconformity.reference_mode_flags` for self-supervised abstention,
which is explicitly *not* a coverage guarantee.
"""

from __future__ import annotations
