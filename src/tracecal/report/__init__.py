"""Per-episode verdicts and serialisation (pillar 3 of the report card UX).

:mod:`tracecal.report.card` is the pure decision rule combining the physics gate and conformal
abstention into accept/hold/reject; :mod:`tracecal.report.emit` serialises a
:class:`~tracecal.schema.DatasetReport` to JSON / CSV / a self-contained HTML card.
"""

from __future__ import annotations
