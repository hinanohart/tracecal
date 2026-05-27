"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tracecal.schema import EmbodimentSpec, JointLimits

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def urdf_dir() -> Path:
    return FIXTURES / "urdf"


@pytest.fixture
def synth_2dof_spec() -> EmbodimentSpec:
    """Hand-built spec matching tests/fixtures/urdf/synth_2dof.urdf (no yourdfpy needed)."""
    return EmbodimentSpec(
        robot_type="synth_2dof",
        resolved=True,
        source="fixture:synth_2dof",
        dof=2,
        joints=(
            JointLimits(name="j1", lower=-1.5708, upper=1.5708, velocity=2.0),
            JointLimits(name="j2", lower=-3.1416, upper=3.1416, velocity=3.0),
        ),
    )


@pytest.fixture
def clean_trajectory() -> np.ndarray:
    """A valid (within-limit, smooth) 2-DOF trajectory at fps=10."""
    t = np.linspace(0, 1, 30)
    return np.stack([0.5 * np.sin(t), 0.3 * np.cos(t)], axis=1)
