"""Command-line interface for annotations (``grui annotation ...``).

Works on the derived ``<recording>/annotations/`` layer only — raw
recordings and perception results are never modified::

    grui annotation import <recording>             # model proposals -> annotations
    grui annotation list <recording> [--status ...] [--frame N]
    grui annotation stats <recording>
    grui annotation export <recording> --out boxes.json
    grui annotation import-human <recording> <file.jsonl>   # human-created records
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from annotation.store import AnnotationStore, load_annotations
from annotation.types import Annotation, AnnotationStatus
from perception.runner import CachedAnalysis
from perception.types import BoundingBox
from storage.recording import load_recording

_STATUS_CHOICES = [s.value for s in AnnotationStatus]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grui annotation",
        description="Inspect and manage human-verified annotations (derived data).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="import perception detections as annotations")
    imp.add_argument("recording_dir", help="raw recording directory")

    lst = sub.add_parser("list", help="list annotations")
    lst.add_argument("recording_dir")
    lst.add_argument("--status", action="append", choices=_STATUS_CHOICES, default=None,
                     help="only annotations with this status (repeatable)")
    lst.add_argument("--source", action="append", default=None,
                     help="only annotations from this source (model|human|imported|derived)")
    lst.add_argument("--label", action="append", default=None, help="only this label (repeatable)")
    lst.add_argument("--frame", type=int, default=None, metavar="N", help="only frame N")

    stats = sub.add_parser("stats", help="annotation summary for a recording")
    stats.add_argument("recording_dir")

    export = sub.add_parser("export", help="export annotations as JSON")
    export.add_argument("recording_dir")
    export.add_argument("--out", required=True, metavar="PATH", help="output .json path")
    export.add_argument("--status", action="append", choices=_STATUS_CHOICES, default=None)

    human = sub.add_parser("import-human", help="import human-created annotations from JSONL")
    human.add_argument("recording_dir")
    human.add_argument("file", metavar="FILE.jsonl",
                       help="JSONL lines: {label, bbox:{x1,y1,x2,y2}, frame_index, t, confidence?}")
    return parser


def _load_store(recording_dir: str) -> tuple[Path, AnnotationStore]:
    recording = load_recording(recording_dir)  # validates it is a recording
    store = load_annotations(recording.directory)
    return recording.directory, store


def _cmd_import(args: argparse.Namespace) -> int:
    directory, store = _load_store(args.recording_dir)
    cached = CachedAnalysis(directory / "perception")
    if not cached.exists:
        print("error: no perception results — run `grui perception analyze` first", file=sys.stderr)
        return 1
    recording = load_recording(directory)
    imported = store.import_perception(
        cached.read_results(),
        frame_size=(recording.width, recording.height),
    )
    store.save()
    print(f"imported {imported} model detections as annotations → {store.annotations_path}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    _, store = _load_store(args.recording_dir)
    statuses = {AnnotationStatus(s) for s in (args.status or [])}
    annotations = store.filter(
        statuses=statuses or None,
        sources=set(args.source) if args.source else None,
        labels=set(args.label) if args.label else None,
    )
    if args.frame is not None:
        annotations = [a for a in annotations if a.frame_index == args.frame]
    if not annotations:
        print("no matching annotations")
        return 0
    for a in sorted(annotations, key=lambda a: (a.frame_index, a.bbox.x1)):
        flag = {"verified": "✓", "rejected": "✕", "predicted": "?", "reviewed": "~", "corrected": "✎"}.get(
            a.status.value, "?"
        )
        mark = a.prediction.label if a.prediction and a.prediction.label != a.label else ""
        provenance = f"  model: {mark!r}" if mark else ""
        print(
            f"[{a.status.value:9s}] {flag} frame {a.frame_index:>6d}  t={a.t:8.3f}  "
            f"{a.label!r} {a.bbox.to_dict()}{provenance}"
        )
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    _, store = _load_store(args.recording_dir)
    by_status = {}
    for a in store:
        by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
    by_source = {}
    for a in store:
        by_source[a.source] = by_source.get(a.source, 0) + 1
    print(f"annotations:        {len(store)}")
    print(f"verified:           {store.verified_count}")
    print(f"rejected:           {store.rejected_count}")
    print(f"by status:          {json.dumps(by_status, sort_keys=True)}")
    print(f"by source:          {json.dumps(by_source, sort_keys=True)}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    _, store = _load_store(args.recording_dir)
    statuses = {AnnotationStatus(s) for s in (args.status or [])}
    annotations = store.filter(statuses=statuses or None)
    Path(args.out).write_text(
        json.dumps([a.to_dict() for a in annotations], indent=2), encoding="utf-8"
    )
    print(f"exported {len(annotations)} annotations → {args.out}")
    return 0


def _cmd_import_human(args: argparse.Namespace) -> int:
    directory, store = _load_store(args.recording_dir)
    recording = load_recording(directory)
    count = 0
    with Path(args.file).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bbox = row["bbox"]
            store.create(
                str(row["label"]),
                BoundingBox(
                    float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
                ),
                int(row["frame_index"]),
                float(row.get("t", recording.frame_time(int(row["frame_index"])))),
                source=str(row.get("source") or "human"),
                status=AnnotationStatus.from_value(row.get("status") or "reviewed"),
                confidence=row.get("confidence"),
                notes=str(row.get("notes") or ""),
            )
            count += 1
    store.save()
    print(f"imported {count} human annotations → {store.annotations_path}")
    return 0


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "import": _cmd_import,
        "list": _cmd_list,
        "stats": _cmd_stats,
        "export": _cmd_export,
        "import-human": _cmd_import_human,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, KeyError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1