"""S4: JSON/CSV/HTML serialisation surfaces verdicts and honesty caveats."""

from __future__ import annotations

import json

from tracecal.report.emit import to_csv, to_html, to_json
from tracecal.schema import (
    CoverageReport,
    DatasetReport,
    EmbodimentSpec,
    EpisodeVerdict,
    JointLimits,
)


def _report(coverage: CoverageReport | None) -> DatasetReport:
    vs = (
        EpisodeVerdict(
            episode_id="a", verdict="accept", Q=0.9, hard_valid=True, abstain=False, degraded=False
        ),
        EpisodeVerdict(
            episode_id="d",
            verdict="hold",
            Q=0.0,
            hard_valid=None,
            abstain=True,
            degraded=True,
            reasons=("physics-skipped: no-urdf-known",),
        ),
        EpisodeVerdict(
            episode_id="r",
            verdict="reject",
            Q=0.0,
            hard_valid=False,
            abstain=False,
            degraded=False,
            reasons=("hard kinematic violation: joint_limit",),
        ),
    )
    specs = {
        "panda": EmbodimentSpec(
            robot_type="panda",
            resolved=True,
            source="rd:panda",
            dof=1,
            joints=(JointLimits(name="j", lower=-1.0, upper=1.0),),
        ),
        "so101": EmbodimentSpec(
            robot_type="so101", resolved=False, source="degraded:no-urdf-known", dof=0
        ),
    }
    return DatasetReport(
        dataset="toy",
        n_episodes=3,
        n_accept=1,
        n_hold=1,
        n_reject=1,
        n_degraded=1,
        coverage=coverage,
        embodiments=specs,
        verdicts=vs,
        warnings=("reference-mode: no binary validity labels available;",),
        provenance={"tool": "tracecal", "version": "0.1.0a1"},
    )


def _coverage() -> CoverageReport:
    return CoverageReport(
        target_coverage=0.9,
        empirical_coverage=0.92,
        n_holdout=40,
        ci_low=0.85,
        ci_high=0.97,
        nominal_violated=False,
    )


def test_json_roundtrips_and_includes_verdicts() -> None:
    out = json.loads(to_json(_report(_coverage())))
    assert out["n_episodes"] == 3
    assert out["coverage"]["source"] == "measured"
    assert {v["episode_id"] for v in out["verdicts"]} == {"a", "d", "r"}


def test_csv_has_header_and_rows() -> None:
    rows = to_csv(_report(None)).strip().splitlines()
    assert rows[0].startswith("episode_id,verdict,Q,hard_valid")
    assert len(rows) == 4  # header + 3 episodes
    assert ",reject," in to_csv(_report(None))


def test_html_is_self_contained() -> None:
    h = to_html(_report(_coverage()))
    assert h.startswith("<!doctype html>")
    # no external resources -> self-contained
    assert "http://" not in h and "https://" not in h
    assert "<script" not in h.lower()
    assert "accept 1" in h and "reject 1" in h


def test_html_reference_mode_caveat_when_no_coverage() -> None:
    h = to_html(_report(None))
    assert "Reference-mode" in h
    assert "coverage = None" in h


def test_html_escapes_content() -> None:
    rep = _report(None)
    h = to_html(rep)
    assert "<script>" not in h  # any injected markup would be escaped
