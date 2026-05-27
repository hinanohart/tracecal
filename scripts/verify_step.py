#!/usr/bin/env python3
"""Per-step verification harness for the tracecal autonomous build.

Usage: ``python scripts/verify_step.py S2``

Every ``check_S*`` asserts *real* artifacts and behaviour for that phase. A check
must never be vacuous (no empty loop, no always-True) — that is audited by the S8
critic gate. Exits 0 only when the phase's concrete acceptance conditions hold;
non-zero otherwise. When a phase's artifacts do not exist yet the relevant
``_require_*`` fails, which is the intended fail-safe (a future step can never be
marked done before it is built).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


class StepFailure(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"FAIL: {msg}")


def _ok(msg: str) -> None:
    print(f"  ok: {msg}")


def _require_file(rel: str, *, nonempty: bool = True) -> Path:
    p = ROOT / rel
    if not p.is_file():
        raise StepFailure(f"missing file: {rel}")
    if nonempty and p.stat().st_size == 0:
        raise StepFailure(f"empty file: {rel}")
    _ok(f"file: {rel}")
    return p


# Every module this harness imports is first-party and hard-coded here. No external
# or user input can ever reach a dynamic import (the build tool is self-contained).
_ALLOWED_MODULES = frozenset(
    {
        "tracecal",
        "tracecal.schema",
        "tracecal.cli",
        "tracecal.api",
        "tracecal.curate",
        "tracecal.pytest_plugin",
        "tracecal.io.lerobot_v3",
        "tracecal.io.labels",
        "tracecal.physics.kinematics",
        "tracecal.physics.resolver",
        "tracecal.physics.registry",
        "tracecal.physics.gate",
        "tracecal.conformal.nonconformity",
        "tracecal.conformal.split",
        "tracecal.conformal.calibrate",
        "tracecal.conformal.coverage",
        "tracecal.report.card",
        "tracecal.report.emit",
    }
)


def _require_import(module: str) -> None:
    if module not in _ALLOWED_MODULES:
        raise StepFailure(f"module not in first-party allowlist: {module!r}")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StepFailure(f"import {module!r}: {result.stderr.strip()}")
    _ok(f"import: {module}")


def _run_pytest(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pytest", "-q", *args]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise StepFailure(f"pytest {' '.join(args)} -> rc={result.returncode}")
    _ok(f"pytest {' '.join(args)}")


# --------------------------------------------------------------------------- #
# Per-phase checks
# --------------------------------------------------------------------------- #
def check_S0_5() -> None:
    for rel in (
        "pyproject.toml",
        "LICENSE",
        "NOTICE",
        "README.md",
        ".gitignore",
        "src/tracecal/__init__.py",
        "scripts/verify_step.py",
        "scripts/check_marketing.sh",
        ".github/workflows/ci.yml",
    ):
        _require_file(rel)
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise StepFailure("LICENSE is not the full Apache-2.0 text")
    _require_import("tracecal")
    _ok("S0.5 scaffold verified")


def check_S1() -> None:
    _require_import("tracecal.schema")
    _run_pytest(["tests/unit/test_schema.py"])
    _ok("S1 IR/schema verified")


def check_S2() -> None:
    for module in (
        "tracecal.conformal.nonconformity",
        "tracecal.conformal.split",
        "tracecal.conformal.calibrate",
        "tracecal.conformal.coverage",
    ):
        _require_import(module)
    _run_pytest(["tests/unit/"])
    _ok("S2 conformal core verified")


def check_S3() -> None:
    for module in (
        "tracecal.physics.kinematics",
        "tracecal.physics.registry",
        "tracecal.physics.resolver",
        "tracecal.physics.gate",
        "tracecal.io.lerobot_v3",
        "tracecal.io.labels",
        "tracecal.curate",
    ):
        _require_import(module)
    _run_pytest(["tests/unit/", "tests/integration/"])
    _ok("S3 physics + io + curate verified")


def check_S4() -> None:
    for module in ("tracecal.report.card", "tracecal.report.emit", "tracecal.api", "tracecal.cli"):
        _require_import(module)
    result = subprocess.run(
        [sys.executable, "-m", "tracecal.cli", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StepFailure(f"`tracecal --help` rc={result.returncode}: {result.stderr}")
    _run_pytest(["tests/"])
    _ok("S4 report + CLI + public API verified")


def check_S5() -> None:
    _require_import("tracecal.pytest_plugin")
    _require_file("examples/quickstart.py")
    _run_pytest(["tests/"])
    _ok("S5 pytest plugin + demo verified")


def validate_s6_artifacts(named_datas: list[tuple[str, dict]]) -> None:
    """Pure S6 validation over (name, parsed-JSON) pairs — testable without results/.

    Enforces: every artifact declares an honest mode; synthetic carries a disclaimer;
    real cites provenance; and the physics-gate CLAIM is actually demonstrated (a
    high-quality but kinematically-invalid episode zeroed to Q=0) in at least one file.
    """
    if not named_datas:
        raise StepFailure("no measured results JSON in results/")
    gate_demonstrated = False
    for name, data in named_datas:
        mode = data.get("dataset", {}).get("mode") or data.get("mode")
        if mode not in {"live", "synthetic", "real"}:
            raise StepFailure(f"{name}: mode must be live|real|synthetic, got {mode!r}")
        if mode == "synthetic" and "disclaimer" not in json.dumps(data).lower():
            raise StepFailure(f"{name}: synthetic results require an explicit disclaimer")
        if mode in {"real", "live"} and not (
            data.get("data_provenance") or data.get("dataset", {}).get("source")
        ):
            raise StepFailure(f"{name}: real/live-mode results require data_provenance")
        demo = data.get("gate_demonstration")
        if demo is not None:
            q = demo.get("Q")
            fired = (
                demo.get("gated_out") is True
                and demo.get("hard_valid") is False
                and isinstance(q, (int, float))
                and not isinstance(q, bool)
                and float(q) == 0.0
            )
            if not fired:
                raise StepFailure(
                    f"{name}: gate_demonstration must show gated_out=true, hard_valid=false, Q=0"
                )
            if int(data.get("summary", {}).get("n_gated_out_on_violation", 0)) < 1:
                raise StepFailure(f"{name}: summary must record >=1 gated-out episode")
            gate_demonstrated = True
            _ok(f"{name}: physics gate fired (Q=0 on a kinematically-invalid episode)")
    if not gate_demonstrated:
        raise StepFailure(
            "no gate_demonstration found: the v0.1.0a1 CLAIM (joint-limit gate forcing Q=0 on "
            "a kinematically-invalid episode) must be recorded in a results/*.json artifact"
        )
    _ok("S6 measured metrics present, mode-honest, and physics-gate CLAIM demonstrated")


def check_S6() -> None:
    paths = sorted((ROOT / "results").glob("*.json"))
    named_datas = [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    validate_s6_artifacts(named_datas)


# This harness covers the phases that have local, machine-checkable artifacts (S0.5–S6).
# Later phases are gated outside this file by design: S7 (honest-marketing) by
# scripts/check_marketing.sh, S8 by the multi-agent critic gate, and S9–S11 by live GitHub
# state (CI conclusion, branch protection, release, clean-clone repro) — none of which is a
# local assertion this script can make.
_CHECKS = {
    "S0.5": check_S0_5,
    "S0_5": check_S0_5,
    "S1": check_S1,
    "S2": check_S2,
    "S3": check_S3,
    "S4": check_S4,
    "S5": check_S5,
    "S6": check_S6,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in _CHECKS:
        print(f"usage: verify_step.py <step>  (known: {', '.join(sorted(_CHECKS))})")
        return 2
    step = argv[1]
    print(f"verify_step {step}:")
    _CHECKS[step]()
    print(f"PASS: {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
