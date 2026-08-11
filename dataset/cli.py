"""Command-line interface for dataset generation (``grui dataset ...``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset.build import DatasetConfig, build_dataset
from storage.recording import load_recording


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grui dataset",
        description="Convert raw recordings into observation->action datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build a dataset from one raw recording")
    build.add_argument("recording_dir", help="raw recording directory")
    build.add_argument(
        "--out", metavar="DIR", default=None,
        help="output directory (default: <parent>/<recording>_dataset)",
    )
    build.add_argument(
        "--obs-duration", type=float, default=3.0,
        help="observation window in seconds (default: 3.0)",
    )
    build.add_argument(
        "--fps", type=float, default=15.0,
        help="observation sampling rate (default: 15.0)",
    )
    build.add_argument(
        "--stride", type=float, default=1.0,
        help="seconds between samples (default: 1.0)",
    )
    build.add_argument(
        "--horizon", type=float, default=0.2,
        help="prediction horizon in seconds, stored for trainers (default: 0.2)",
    )
    return parser


def _cmd_build(args: argparse.Namespace) -> int:
    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else recording.directory.parent / f"{recording.directory.name}_dataset"
    config = DatasetConfig(
        observation_duration=args.obs_duration,
        fps=args.fps,
        stride=args.stride,
        prediction_horizon=args.horizon,
    )
    try:
        build_dataset(recording, config, out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"built dataset: {out}")
    return 0


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        return _cmd_build(args)
    return 2
