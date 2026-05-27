"""S7: the honest-marketing gate is not vacuous — it passes clean and fails on tampering."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="check_marketing.sh is a bash script (POSIX shells only)"
)


def _layout(dst: Path) -> None:
    (dst / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "README.md", dst / "README.md")
    shutil.copy(ROOT / "scripts" / "check_marketing.sh", dst / "scripts" / "check_marketing.sh")
    shutil.copytree(ROOT / "results", dst / "results")


def _run(dst: Path) -> int:
    return subprocess.run(
        ["bash", str(dst / "scripts" / "check_marketing.sh")], capture_output=True, text=True
    ).returncode


def test_clean_repo_passes(tmp_path) -> None:
    if not (ROOT / "results" / "gate_v0.1.0a1.json").exists():
        pytest.skip("results/*.json not present (run scripts/measure.py first)")
    _layout(tmp_path)
    assert _run(tmp_path) == 0


def test_missing_disclaimer_fails(tmp_path) -> None:
    if not (ROOT / "results" / "gate_v0.1.0a1.json").exists():
        pytest.skip("results/*.json not present")
    _layout(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("algorithm validation only", "great results"),
        encoding="utf-8",
    )
    assert _run(tmp_path) != 0


def test_untraceable_number_fails(tmp_path) -> None:
    if not (ROOT / "results" / "gate_v0.1.0a1.json").exists():
        pytest.skip("results/*.json not present")
    _layout(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("**1.00**", "**0.42**"),
        encoding="utf-8",
    )
    assert _run(tmp_path) != 0


def test_overclaim_phrase_fails(tmp_path) -> None:
    if not (ROOT / "results" / "gate_v0.1.0a1.json").exists():
        pytest.skip("results/*.json not present")
    _layout(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n\nThis is the world's first fully automatic tool.\n",
        encoding="utf-8",
    )
    assert _run(tmp_path) != 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
