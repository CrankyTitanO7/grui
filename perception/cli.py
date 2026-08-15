"""Command-line interface for perception (``grui perception ...``).

Analyzes existing recordings with an optional perception provider::

    grui perception providers
    grui perception analyze <recording> --provider locate_anything --prompt "boss" --fps 2

Analysis is a derived operation: it never touches the raw recording,
``events.jsonl`` or the video. Results land in
``<recording>/perception/{manifest.json,results.jsonl}``.
"""

from __future__ import annotations

import argparse
import json
import sys

from perception import get, list_providers, provider_info
from perception.base import with_options
from perception.runner import analyze_recording
from storage.recording import load_recording


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grui perception",
        description="Analyze existing recordings with optional perception providers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    providers = sub.add_parser("providers", help="list registered perception providers")
    providers.add_argument("--json", action="store_true", dest="as_json",
                           help="print provider info as JSON")

    analyze = sub.add_parser(
        "analyze",
        help="run a perception provider over an existing recording (derived results only)",
    )
    analyze.add_argument("recording_dir", help="raw recording directory")
    analyze.add_argument("--provider", default="locate_anything",
                         help="perception provider name (default: locate_anything)")
    analyze.add_argument("--prompt", action="append", metavar="TEXT",
                         help="natural-language prompt to localize (repeatable)")
    analyze.add_argument("--fps", type=float, default=None,
                         help="sample ~F frames per second, e.g. --fps 2 on a 30 FPS recording")
    analyze.add_argument("--every", type=int, default=None, metavar="N",
                         help="process every Nth frame (alternative to --fps)")
    analyze.add_argument("--device", default=None,
                         help="override the provider's default device, e.g. cuda:1 or cpu")
    analyze.add_argument("--model", default=None, metavar="PATH",
                         help="model weights file (yolo provider only, e.g. yolov8n.pt)")
    analyze.add_argument("--conf", type=float, default=None,
                         help="confidence threshold (yolo provider only, default: 0.25)")
    analyze.add_argument("--allow-download", action="store_true",
                         help="let ultralytics download the weights if missing (yolo only); "
                              "off by default: GRUI never downloads weights silently")
    analyze.add_argument("--quantize", default=None, choices=("none", "8bit", "4bit"),
                         help="bitsandbytes weight quantization to cut VRAM: "
                              "8bit ~4 GB, 4bit ~2 GB (needs transformers+bitsandbytes)")
    analyze.add_argument("--max-pixels", type=int, default=None, metavar="N",
                         help="downscale frames to at most N pixels before inference "
                              "to shrink the vision encoder's VRAM (e.g. 786432 ≈ "
                              "1024x768); boxes are rescaled to the original frame "
                              "size in the results")
    analyze.add_argument("--force", action="store_true",
                         help="re-run even if matching cached results exist")
    return parser


def _cmd_providers(args: argparse.Namespace) -> int:
    infos = [provider_info(p) for p in list_providers()]
    if args.as_json:
        print(json.dumps([info.to_dict() for info in infos], indent=2))
        return 0
    if not infos:
        print("No perception providers registered.")
        return 0
    print("Available perception providers:")
    for info in infos:
        print(f"\n{info.name}")
        print(f"    version:   {info.version}")
        print(f"    available: {'yes' if info.available else 'no'}")
        if info.model:
            print(f"    model:     {info.model}")
        if info.description:
            print(f"    about:     {info.description}")
        if not info.available and info.install_hint:
            print(f"    install:   {info.install_hint}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        provider = with_options(
            get(args.provider),
            device=args.device,
            model=args.model,
            conf=args.conf,
            allow_download=args.allow_download,
            quantize=args.quantize,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    prompts = list(args.prompt or [])
    if not prompts:
        print("error: give at least one --prompt TEXT", file=sys.stderr)
        return 2
    if args.fps is not None and args.every is not None:
        print("error: give either --fps or --every, not both", file=sys.stderr)
        return 2
    if not provider.is_available():
        info = provider_info(provider)
        hint = info.install_hint or "install the provider's optional dependencies."
        print(
            f"error: perception provider {args.provider!r} is unavailable.\n"
            f"    {hint}",
            file=sys.stderr,
        )
        return 1
    try:
        analyze_recording(
            recording,
            provider,
            prompts,
            every=args.every,
            fps=args.fps,
            max_pixels=args.max_pixels,
            force=args.force,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "providers":
        return _cmd_providers(args)
    if args.command == "analyze":
        return _cmd_analyze(args)
    return 2
