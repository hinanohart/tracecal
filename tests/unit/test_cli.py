"""S4: CLI commands that run without network/GPU/yourdfpy."""

from __future__ import annotations

import pytest

from tracecal.cli import build_parser, main


def test_selftest_passes(capsys) -> None:
    assert main(["selftest"]) == 0
    out = capsys.readouterr().out
    assert "selftest OK" in out


def test_list_embodiments(capsys) -> None:
    assert main(["list-embodiments"]) == 0
    out = capsys.readouterr().out
    assert "panda_description" in out
    assert "iiwa14_description" in out
    assert "degrade-first-class" in out


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_version_flag() -> None:
    with pytest.raises(SystemExit):
        main(["--version"])


def test_run_subparser_defaults() -> None:
    args = build_parser().parse_args(["run", "some/source"])
    assert args.confidence == 0.9
    assert args.physics == "auto"
    assert args.mondrian_by == "embodiment"
    assert args.format == "json"
