"""S3: explicit success-label detection (reward is intentionally ignored)."""

from __future__ import annotations

import numpy as np

from tracecal.io.labels import detect_episode_label


def test_terminal_success_true() -> None:
    label, src = detect_episode_label({"next.success": np.array([0, 0, 0, 1])})
    assert label == 1.0
    assert src == "next.success"


def test_terminal_success_false() -> None:
    label, src = detect_episode_label({"success": np.array([0, 0, 0, 0])})
    assert label == 0.0
    assert src == "success"


def test_bool_success_column() -> None:
    label, _ = detect_episode_label({"is_success": np.array([False, False, True])})
    assert label == 1.0


def test_no_success_column_returns_none() -> None:
    label, src = detect_episode_label({"observation.state": np.zeros((5, 3))})
    assert label is None
    assert src is None


def test_reward_is_not_treated_as_success() -> None:
    # A reward column must NOT be coerced into a success label (circular / out of scope).
    label, src = detect_episode_label({"next.reward": np.array([0.1, 0.5, 2.3])})
    assert label is None
    assert src is None


def test_nonbinary_success_values_rejected() -> None:
    label, _ = detect_episode_label({"success": np.array([0.0, 0.3, 0.7])})
    assert label is None
