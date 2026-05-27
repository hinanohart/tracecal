"""pytest plugin: fail a test run when a LeRobot dataset's calibrated coverage breaches target.

Registered via the ``pytest11`` entry point, so installing tracecal makes the ``tracecal_audit``
fixture and the marker available automatically. Import stays light: ``numpy``/the pipeline are
imported lazily inside the fixture, never at plugin-load time, so merely having tracecal installed
never perturbs an unrelated suite.

Typical use::

    def test_my_dataset_is_clean(tracecal_audit):
        report = tracecal_audit("path/or/repo_id", confidence=0.9)
        tracecal_audit.assert_coverage_holds(report)        # no-op in reference-mode
        tracecal_audit.write_weights(report, "weights.json")  # curated sample weights artifact
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

    from tracecal.schema import DatasetReport


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "tracecal_audit: mark a test that audits a LeRobot dataset's calibrated validity coverage.",
    )


def assert_coverage_holds(report: DatasetReport, *, confidence: float | None = None) -> None:
    """Fail the test if a *measured* coverage is significantly below target.

    In reference-mode (``coverage is None``) there is no coverage claim, so this is a no-op —
    the test must not pretend a guarantee that was never made. Raises ``AssertionError`` (the
    natural failure for an ``assert_*`` helper, catchable with ``pytest.raises``) on a breach.
    """
    cov = report.coverage
    if cov is None:
        return
    target = confidence if confidence is not None else cov.target_coverage
    if (cov.ci_high != cov.ci_high) or cov.ci_high < target:  # NaN or below target
        raise AssertionError(
            f"tracecal: holdout coverage {cov.empirical_coverage:.3f} (95% CI upper "
            f"{cov.ci_high:.3f}) is below target {target:.3f}"
        )


def write_weights(report: DatasetReport, path: str) -> dict[str, float]:
    """Write curated per-episode sample weights to ``path`` (JSON) and return them."""
    from tracecal.curate import curate

    weights = curate(report).weights
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(weights, fh, indent=2, sort_keys=True)
    return weights


class _Auditor:
    """Callable returned by the ``tracecal_audit`` fixture, with helper methods attached."""

    def __call__(self, source: Any, **kwargs: Any) -> DatasetReport:
        from tracecal.api import evaluate_dataset

        return evaluate_dataset(source, **kwargs)

    assert_coverage_holds = staticmethod(assert_coverage_holds)
    write_weights = staticmethod(write_weights)


def _make_auditor() -> _Auditor:
    return _Auditor()


# Define the fixture only when pytest is importable (i.e. under a test run). This keeps a plain
# ``import tracecal.pytest_plugin`` from hard-failing if pytest is absent.
try:
    import pytest as _pytest

    @_pytest.fixture
    def tracecal_audit() -> _Auditor:
        """Audit a LeRobot dataset from a test; see module docstring for usage."""
        return _make_auditor()

except ImportError:  # pragma: no cover - pytest is always present when the plugin is loaded
    pass
