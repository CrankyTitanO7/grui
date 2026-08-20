"""Application entry point for GRUI (Grand Unified Imitation)."""

from __future__ import annotations

import importlib
import logging
import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

# name -> (module, build_parser, runner(argv) -> int)
_SUBCOMMANDS = {
    "dataset": ("dataset.cli", "build_parser", "run"),
    "perception": ("perception.cli", "build_parser", "run"),
    "annotation": ("annotation.cli", "build_parser", "run"),
    "train": ("ml.train", "build_parser", "main"),
    "agent": ("ml.inject", "build_parser", "run_agent"),
    "locate": ("ml.locate", "build_parser", "main"),
}


def _print_cli_help() -> int:
    """List the available ``grui`` subcommands with their one-line purpose."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="grui",
        description="GRUI (Grand Unified Imitation): record demonstrations, then "
        "run perception, annotation, dataset-quality and training tools.",
        add_help=False,
    )
    parser.add_argument(
        "-h", "--help", action="help",
        help="show this help message and exit",
    )
    sub = parser.add_subparsers(title="commands", metavar="COMMAND ...")
    for name, (module, builder, _runner) in sorted(_SUBCOMMANDS.items()):
        built = getattr(importlib.import_module(module), builder)()
        sub.add_parser(name, help=(built.description or "no description"))
    parser.print_help()
    return 0


def _dispatch(argv: list[str]) -> int:
    module_name, _builder, runner_name = _SUBCOMMANDS[argv[1]]
    runner = getattr(importlib.import_module(module_name), runner_name)
    return runner(argv[2:])


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help", "help"):
        return _print_cli_help()
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCOMMANDS:
        return _dispatch(sys.argv)

    app = QApplication(sys.argv)
    app.setApplicationName("GRUI")
    window = MainWindow()
    window.show()
    app.aboutToQuit.connect(window.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
