"""Command-line interface for dataset generation (``grui dataset ...``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset.build import DatasetConfig, build_dataset
from storage.recording import list_recordings, load_recording


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grui dataset",
        description="Convert raw recordings into observation->action datasets and inspect dataset health.",
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

    health = sub.add_parser("health", help="dataset statistics over a recordings root")
    health.add_argument("root", help="recordings root directory")

    actions = sub.add_parser("actions", help="action distribution for one recording")
    actions.add_argument("recording_dir", help="raw recording directory")

    report = sub.add_parser("report", help="dataset-quality report for one recording")
    report.add_argument("recording_dir", help="raw recording directory")
    report.add_argument("--check-duplicates", action="store_true",
                        help="also scan sampled video frames for near-duplicates (reads the video)")

    episodes = sub.add_parser("episodes", help="episode segmentation (derived metadata)")
    episodes_sub = episodes.add_subparsers(dest="ep_sub", required=True)
    suggest = episodes_sub.add_parser("suggest", help="suggest episode boundaries")
    suggest.add_argument("recording_dir")
    suggest.add_argument("--min-inactivity", type=float, default=5.0,
                         help="inactivity gap (s) that suggests a boundary (default: 5.0)")
    suggest.add_argument("--no-markers", action="store_true", help="ignore episode: markers")
    suggest.add_argument("--visual", action="store_true",
                         help="also detect scene changes in the video (reads frames)")
    suggest.add_argument("--max-episode-s", type=float, default=None,
                         help="split episodes longer than N seconds")
    suggest.add_argument("--save", action="store_true",
                         help="write the suggestions to <recording>/episodes.jsonl")
    ep_list = episodes_sub.add_parser("list", help="list stored episodes")
    ep_list.add_argument("recording_dir")
    ep_set = episodes_sub.add_parser("set", help="replace episodes manually: START END [START END ...]")
    ep_set.add_argument("recording_dir")
    ep_set.add_argument("ranges", nargs="+", metavar="TIME",
                        help="pairs of start/end times in seconds")

    version = sub.add_parser("version", help="dataset versioning (immutable derived metadata)")
    version_sub = version.add_subparsers(dest="version_sub", required=True)
    vcreate = version_sub.add_parser("create", help="create a dataset version")
    vcreate.add_argument("root", help="dataset root (holds versions.json)")
    vcreate.add_argument("--name", default=None, help="version name (default: next vN)")
    vcreate.add_argument("--recording", action="append", default=None, metavar="DIR",
                         help="source recording dir (repeatable; default: all under root/..)")
    vcreate.add_argument("--exclude", action="append", default=None, metavar="NAME",
                         help="recording name to exclude (repeatable)")
    vcreate.add_argument("--status", action="append", default=None,
                         choices=("predicted", "reviewed", "verified", "corrected", "rejected"),
                         help="annotation statuses included (repeatable)")
    vcreate.add_argument("--no-perception", action="store_true",
                         help="do not include perception predictions in the version")
    vcreate.add_argument("--parent", default=None, help="parent version name")
    vcreate.add_argument("--notes", default="", help="free-form notes")
    vlist = version_sub.add_parser("list", help="list versions")
    vlist.add_argument("root")
    vdiff = version_sub.add_parser("diff", help="diff two versions")
    vdiff.add_argument("root")
    vdiff.add_argument("old_name")
    vdiff.add_argument("new_name")
    vstats = version_sub.add_parser("stats", help="statistics of a version")
    vstats.add_argument("root")
    vstats.add_argument("name")

    split = sub.add_parser("split", help="safe train/validation/test split by demonstration")
    split.add_argument("recordings_root", help="root containing the recordings")
    split.add_argument("--train", type=float, default=0.7)
    split.add_argument("--validation", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--out", metavar="PATH", default=None,
                       help="write split.json (else print the assignment)")
    split.add_argument("--only", action="append", default=None, metavar="NAME",
                       help="only split these recordings (repeatable)")

    review = sub.add_parser("review", help="review queue (active learning)")
    review_sub = review.add_subparsers(dest="review_sub", required=True)
    rbuild = review_sub.add_parser("build", help="(re)build the queue for a recording")
    rbuild.add_argument("recording_dir")
    rbuild.add_argument("--strategy", action="append", default=None,
                        choices=("uncertainty", "rare_action", "novelty",
                                 "annotation_uncertainty", "transition", "coverage"),
                        help="ranking strategy (repeatable; default: all)")
    rbuild.add_argument("--limit", type=int, default=200)
    rbuild.add_argument("--recording-root", default=None, metavar="ROOT",
                        help="recordings root used by the coverage strategy "
                             "(situations under-covered across demos)")
    rlist = review_sub.add_parser("list", help="list pending queue items")
    rlist.add_argument("recording_dir")
    rlist.add_argument("--json", action="store_true", help="output as JSON")
    rdecide = review_sub.add_parser("decide", help="accept/reject/skip a frame")
    rdecide.add_argument("recording_dir")
    rdecide.add_argument("frame", type=int)
    rdecide.add_argument("verdict", choices=("accept", "reject", "skip"))

    coverage = sub.add_parser(
        "coverage",
        help="coverage analysis: which situations are represented, and how well",
    )
    coverage.add_argument(
        "root",
        help="recordings root directory, or a single recording directory",
    )
    coverage.add_argument(
        "--source", default="auto", choices=("auto", "annotations", "perception"),
        help="label layer to analyse (default: annotations if present, else perception)",
    )
    coverage.add_argument(
        "--min-demos", type=int, default=2,
        help="flag situations present in fewer than N demonstrations (default: 2)",
    )
    coverage.add_argument(
        "--max-situations", type=int, default=15,
        help="most common situations to list (default: 15)",
    )
    coverage.add_argument(
        "--only", action="append", default=None, metavar="NAME",
        help="only report a recording whose folder name or session_id matches (repeatable)",
    )
    coverage.add_argument("--json", action="store_true", help="output raw counts as JSON")

    filterdemos = sub.add_parser(
        "filter-demos",
        help="find demonstrations containing given action chords (section 19: rare actions)",
    )
    filterdemos.add_argument("root", help="recordings root directory, or a single recording directory")
    filterdemos.add_argument("--contains", action="append", default=None, metavar="CHORD",
                             help='action chord to look for, e.g. "A + SPACE" or KeyQ (repeatable, OR-ed)')
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


def _cmd_actions(args: argparse.Namespace) -> int:
    from dataset.health import action_distribution, render_action_distribution

    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_action_distribution(action_distribution(recording)))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from dataset.health import create_quality_issues, render_quality_report

    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    issues = create_quality_issues(recording, check_duplicates=args.check_duplicates)
    print(render_quality_report(recording, issues))
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    from dataset.health import analyze_recordings, render_dataset_statistics

    try:
        stats, per_demo = analyze_recordings(args.root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_dataset_statistics(stats))
    if not per_demo:
        print(
            f"no recordings found under {args.root} "
            "(is it a recordings root?)",
            file=sys.stderr,
        )
    elif len([r for r in per_demo if r.duration < 5]) > max(1, len(per_demo) // 2):
        print("\n! most demonstrations are under 5s - the dataset may be too sparse.")
    return 0


def _cmd_episodes_suggest(args: argparse.Namespace) -> int:
    from dataset.episodes import suggest_episodes, write_episodes

    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    episodes = suggest_episodes(
        recording,
        min_inactivity=args.min_inactivity,
        use_markers=not args.no_markers,
        use_visual=args.visual,
        max_episode_s=args.max_episode_s,
    )
    for i, episode in enumerate(episodes, 1):
        print(f"Episode {i}  {episode.start:7.2f}s -> {episode.end:7.2f}s  ({episode.reason})")
    if args.save:
        path = write_episodes(recording.directory, episodes)
        print(f"saved → {path}")
    return 0


def _cmd_episodes_list(args: argparse.Namespace) -> int:
    from dataset.episodes import read_episodes

    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    episodes = read_episodes(recording.directory)
    if not episodes:
        print("no episodes stored — run `grui dataset episodes suggest --save`")
        return 0
    for i, episode in enumerate(episodes, 1):
        print(f"Episode {i}  {episode.start:7.2f}s -> {episode.end:7.2f}s  ({episode.reason})")
    return 0


def _cmd_episodes_set(args: argparse.Namespace) -> int:
    from dataset.episodes import Episode, write_episodes

    values = [float(v) for v in args.ranges]
    if len(values) % 2 != 0:
        print("error: give pairs of START END times", file=sys.stderr)
        return 2
    episodes = [
        Episode(start, end, reason="manual")
        for start, end in zip(values[::2], values[1::2])
        if end > start
    ]
    episodes.sort(key=lambda e: e.start)
    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for episode in episodes:
        if recording.duration and episode.end > recording.duration + 1e-6:
            print(
                f"error: episode end {episode.end:.2f}s beyond recording duration "
                f"{recording.duration:.2f}s",
                file=sys.stderr,
            )
            return 1
    path = write_episodes(recording.directory, episodes)
    print(f"wrote {len(episodes)} episodes → {path}")
    return 0


def _cmd_version_handler(args: argparse.Namespace) -> int:
    from dataset.versioning import (
        VersionStore,
        create_version,
        diff_versions,
        version_statistics,
    )
    from dataset.health import render_dataset_statistics

    if args.version_sub == "create":
        try:
            version = create_version(
                args.root,
                sources=args.recording,
                excluded=args.exclude,
                annotation_statuses=args.status,
                include_perception=not args.no_perception,
                parent=args.parent,
                notes=args.notes,
                name=args.name,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"created {version.name} ({len(version.sources)} sources) in {args.root}")
        return 0
    if args.version_sub == "list":
        store = VersionStore(args.root)
        versions = store.load()
        if not versions:
            print("no versions")
            return 0
        for version in versions:
            parent = f" (parent: {version.parent})" if version.parent else ""
            print(f"{version.name}: {len(version.sources)} sources, "
                  f"statuses={','.join(version.annotation_statuses)}{parent}")
        return 0
    if args.version_sub == "diff":
        try:
            old = VersionStore(args.root).get(args.old_name)
            new = VersionStore(args.root).get(args.new_name)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            diff = diff_versions(old, new, Path(args.root).parent)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(diff.render())
        return 0
    if args.version_sub == "stats":
        store = VersionStore(args.root)
        try:
            version = store.get(args.name)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        stats = version_statistics(version, Path(args.root).parent)
        print(render_dataset_statistics(stats))
        return 0
    return 2


def _cmd_split(args: argparse.Namespace) -> int:
    from dataset.versioning import save_split, split_demonstrations

    try:
        split = split_demonstrations(
            args.recordings_root,
            train=args.train,
            validation=args.validation,
            test=args.test,
            seed=args.seed,
            only=args.only,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        names = split.train + split.validation + split.test
        path = save_split(args.out, names, split)
        print(f"split written → {path}")
    else:
        print(f"train ({len(split.train)}):")
        for name in split.train:
            print(f"  {name}")
        print(f"validation ({len(split.validation)}):")
        for name in split.validation:
            print(f"  {name}")
        print(f"test ({len(split.test)}):")
        for name in split.test:
            print(f"  {name}")
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    from dataset.coverage import analyze, render_report, report_to_dict

    if (Path(args.root) / "metadata.json").exists():
        try:
            recordings = [load_recording(args.root)]
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            recordings = [load_recording(p) for p in list_recordings(args.root)]
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.only:
        wanted = set(args.only)
        recordings = [
            r for r in recordings
            if r.directory.name in wanted or r.session_id in wanted
        ]
    try:
        report = analyze(recordings, source=args.source)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
        return 0
    print(
        render_report(
            report,
            min_demos=args.min_demos,
            max_situations=args.max_situations,
            recordings=recordings,
        )
    )
    return 0


def _cmd_filter_demos(args: argparse.Namespace) -> int:
    from dataset.health import filter_demos, parse_chord

    contains = [parse_chord(text) for text in (args.contains or [])]
    if not contains:
        print("error: give at least one --contains CHORD", file=sys.stderr)
        return 2
    described = ", ".join(" + ".join(codes) for codes in contains)
    try:
        matches = filter_demos(args.root, contains)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not matches:
        print(f"no demonstrations under {args.root} contain {described}")
        return 0
    print(f"{len(matches)} demonstration(s) under {args.root} contain {described}:")
    for name, total, count in matches:
        print(f"  {name}: {count} matching frame(s) of {total}")
    return 0


def _cmd_review_handler(args: argparse.Namespace) -> int:
    from dataset.review import ReviewQueue

    try:
        recording = load_recording(args.recording_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    queue = ReviewQueue(recording)
    if args.review_sub == "build":
        items = queue.refresh(
            strategies=args.strategy,
            limit=args.limit,
            recording_root=args.recording_root,
        )
        print(f"review queue: {len(items)} candidates "
              f"({len(queue.pending())} pending) → {queue.path}")
        return 0
    if args.review_sub == "list":
        pending = queue.pending()
        if args.json:
            import json as _json

            print(_json.dumps([item.to_dict() for item in pending], indent=2))
            return 0
        if not pending:
            print("no pending review items")
            return 0
        for i, item in enumerate(pending, 1):
            print(f"{i}. frame {item.frame_index:>6d}  t={item.t:8.3f}  "
                  f"[{item.kind}] {item.reason}  priority={item.priority:.0f}")
        return 0
    if args.review_sub == "decide":
        verdict = args.verdict
        handler = {"accept": queue.accept, "reject": queue.reject, "skip": queue.skip}[verdict]
        handler(args.frame)
        print(f"{verdict}d frame {args.frame}")
        return 0
    return 2


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "health":
        return _cmd_health(args)
    if args.command == "actions":
        return _cmd_actions(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "episodes":
        handlers = {
            "suggest": _cmd_episodes_suggest,
            "list": _cmd_episodes_list,
            "set": _cmd_episodes_set,
        }
        return handlers[args.ep_sub](args)
    if args.command == "version":
        return _cmd_version_handler(args)
    if args.command == "split":
        return _cmd_split(args)
    if args.command == "review":
        return _cmd_review_handler(args)
    if args.command == "coverage":
        return _cmd_coverage(args)
    if args.command == "filter-demos":
        return _cmd_filter_demos(args)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(run())

