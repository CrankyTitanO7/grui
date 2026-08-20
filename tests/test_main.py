"""Top-level ``grui`` dispatcher tests (app/main.py): --help + subcommand forwarding."""

from __future__ import annotations

import pytest


def _run(argv: list[str], capsys) -> int:
    import sys

    from app.main import main

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", argv)
        return main()


@pytest.mark.parametrize("flag", ["--help", "-h", "help"])
def test_help_lists_all_subcommands(flag, capsys):
    code = _run(["grui", flag], capsys)
    assert code == 0
    out = capsys.readouterr().out
    for name in ("dataset", "perception", "annotation", "train", "agent", "locate"):
        assert name in out


def test_help_descriptions_present(capsys):
    _run(["grui", "--help"], capsys)
    out = capsys.readouterr().out
    assert "dataset health" in out  # from dataset.cli parser description
    assert "annotation" in out


def test_subcommand_help_forwards_to_module_parser(capsys):
    with pytest.raises(SystemExit) as exc:
        _run(["grui", "annotation", "--help"], capsys)
    assert exc.value.code == 0
    assert "grui annotation" in capsys.readouterr().out


def test_dataset_help_forwards(capsys):
    with pytest.raises(SystemExit) as exc:
        _run(["grui", "dataset", "--help"], capsys)
    assert exc.value.code == 0
    assert "grui dataset" in capsys.readouterr().out


def test_dispatcher_registers_annotation_and_locate():
    from app.main import _SUBCOMMANDS

    assert "annotation" in _SUBCOMMANDS
    assert "locate" in _SUBCOMMANDS