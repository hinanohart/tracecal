"""pytest plugin entry point (registered via the ``pytest11`` entry point).

The real fixtures/assertions that fail a test run when a dataset's calibrated coverage
breaches its target are added in S5 (they need the public API from S4). At import time this
module must stay dependency-light and side-effect-free so that simply having ``tracecal``
installed never perturbs an unrelated test suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register tracecal's markers so ``-W error::pytest.PytestUnknownMarkWarning`` stays clean."""
    config.addinivalue_line(
        "markers",
        "tracecal_audit: mark a test that audits a LeRobot dataset's calibrated validity coverage.",
    )
