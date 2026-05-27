"""Best-effort detection of an explicit per-episode binary *validity/success* label.

LeRobot v3 datasets usually carry **no** success label, so the common outcome here is ``None``
→ the audit runs in reference-mode (no coverage claim). We detect only an *explicit* success
column. We deliberately do NOT derive success from a reward signal: a reward→success rule is
both heuristic and circular with what tracecal is trying to certify (success-probability
coverage is out of scope for v0.1.0a1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Explicit success columns, in priority order.
SUCCESS_KEYS: tuple[str, ...] = (
    "next.success",
    "success",
    "is_success",
    "task_success",
    "episode_success",
)


def _to_binary(value: Any) -> float | None:
    """Coerce a single success value to {0.0, 1.0}, or None if it is not clearly binary."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if f in (0.0, 1.0):
            return f
        return None
    return None


def detect_episode_label(columns: dict[str, np.ndarray]) -> tuple[float | None, str | None]:
    """Return ``(label, source)`` for one episode from its per-step success column.

    An episode is labelled valid (1.0) iff its success column is *terminally* true (the last
    step), which is the LeRobot convention for ``next.success``. Returns ``(None, None)`` when no
    explicit, clearly-binary success column is present.
    """
    for key in SUCCESS_KEYS:
        if key not in columns:
            continue
        arr = np.asarray(columns[key]).ravel()
        if arr.size == 0:
            continue
        terminal = _to_binary(arr[-1].item() if hasattr(arr[-1], "item") else arr[-1])
        if terminal is not None:
            return terminal, key
        # fall back to "any step true" only if every value is clearly binary
        bins = [_to_binary(v.item() if hasattr(v, "item") else v) for v in arr]
        if all(b is not None for b in bins):
            return (1.0 if any(b == 1.0 for b in bins) else 0.0), key
    return None, None
