"""Dataset coverage analysis: which situations are represented, and how well.

A "situation" is the set of labels present on a frame, derived from the
annotation layer when it exists (human truth), otherwise from perception
results (model guesses); whatever the user has configured. The report
shows:

* situation coverage - exact label sets co-occurring on a frame
* label presence - share of analysed frames each label appears on
* per-demonstration matrix - which demos capture which situations
* under-covered situations - present in too few demos (highlighted)

Pure analysis over finished recordings; nothing is modified. The report
can be dumped as JSON for machine consumption.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annotation.store import load_annotations
from annotation.types import AnnotationStatus
from storage.recording import RecordingData

SOURCES = ("auto", "annotations", "perception")


def _situation_name(labels: tuple[str, ...]) -> str:
    return " + ".join(labels)


def _has_annotations(recording: RecordingData) -> bool:
    return (recording.directory / "annotations" / "annotations.jsonl").exists()


def _has_perception(recording: RecordingData) -> bool:
    from perception.runner import CachedAnalysis

    return CachedAnalysis(recording.directory / "perception").exists


def frame_situations(
    recording: RecordingData, source: str = "auto"
) -> tuple[str, dict[int, frozenset[str]]]:
    """Label sets per frame from the chosen layer.

    Returns ``(effective_source, frame_index -> labels)`` for every frame
    with at least one label. ``auto`` prefers annotations (human truth),
    falls back to perception results, and is silent (empty) when neither
    layer exists. An explicitly requested but missing source raises
    ``ValueError``.
    """
    if source == "auto":
        if _has_annotations(recording):
            source = "annotations"
        elif _has_perception(recording):
            source = "perception"
        else:
            return "auto", {}
    if source not in SOURCES:
        raise ValueError(f"unknown coverage source: {source!r}")

    if source == "annotations":
        if not _has_annotations(recording):
            raise ValueError(f"no annotations layer: {recording.directory}")
        by_frame: dict[int, set[str]] = {}
        for annotation in load_annotations(recording.directory):
            if annotation.status == AnnotationStatus.REJECTED:
                continue  # a human said it is wrong - not coverage
            by_frame.setdefault(annotation.frame_index, set()).add(annotation.label)
        return source, {fi: frozenset(labels) for fi, labels in by_frame.items()}

    from perception.runner import CachedAnalysis

    if not _has_perception(recording):
        raise ValueError(f"no perception results: {recording.directory}")
    by_frame = {}
    for result in CachedAnalysis(recording.directory / "perception").read_results():
        labels = frozenset(d.label for d in result.detections)
        if labels:
            by_frame[result.frame_index] = labels
    return source, by_frame


# ----------------------------------------------------------------- data model


@dataclass(frozen=True)
class DemoCoverage:
    """Situation coverage of a single demonstration."""

    demo: str
    analysed_frames: int  # frames with at least one label
    labels: Counter[str]  # label -> analysed frames
    situations: Counter[tuple[str, ...]]  # label set -> analysed frames

    def situation_names(self) -> frozenset[str]:
        return frozenset(_situation_name(s) for s in self.situations)


@dataclass(frozen=True)
class CoverageReport:
    """Aggregate coverage over many demonstrations."""

    demos: list[DemoCoverage]
    source: str  # "annotations" | "perception"

    @property
    def total_analysed_frames(self) -> int:
        return sum(d.analysed_frames for d in self.demos)

    @property
    def labels(self) -> Counter[str]:
        labels: Counter[str] = Counter()
        for demo in self.demos:
            labels.update(demo.labels)
        return labels

    @property
    def situations(self) -> Counter[tuple[str, ...]]:
        situations: Counter[tuple[str, ...]] = Counter()
        for demo in self.demos:
            situations.update(demo.situations)
        return situations

    def under_covered(self, min_demos: int = 2) -> list[tuple[str, int, int]]:
        """(situation, demos_present, total_demos) below the demo floor."""
        total = len(self.demos)
        if total == 0:
            return []
        present: Counter[str] = Counter()
        for demo in self.demos:
            present.update(demo.situation_names())
        return sorted(
            (name, count, total)
            for name, count in present.items()
            if count < min_demos
        )


def analyze(
    recordings: list[RecordingData], source: str = "auto"
) -> CoverageReport:
    """Coverage report across demonstrations (any source per recording)."""
    demos: list[DemoCoverage] = []
    used = "auto"
    for recording in recordings:
        source_used, by_frame = frame_situations(recording, source)
        if used == "auto" and source_used != "auto":
            used = source_used
        labels: Counter[str] = Counter()
        situations: Counter[tuple[str, ...]] = Counter()
        for frame_labels in by_frame.values():
            for label in frame_labels:
                labels[label] += 1
            situations[tuple(sorted(frame_labels))] += 1
        demos.append(
            DemoCoverage(
                demo=recording.directory.name,
                analysed_frames=len(by_frame),
                labels=labels,
                situations=situations,
            )
        )
    return CoverageReport(demos=demos, source=used)


# ------------------------------------------------------------------ rendering


def _bar(fraction: float, width: int = 30) -> str:
    return "#" * max(1, int(fraction * width))


def render_report(
    report: CoverageReport,
    *,
    min_demos: int = 2,
    max_situations: int = 15,
) -> str:
    """Human-readable ASCII coverage report (cp1252-safe glyphs only)."""
    lines = [
        f"Coverage Report ({len(report.demos)} demonstration"
        f"{'' if len(report.demos) == 1 else 's'}) - source: {report.source}",
        "=============================================",
    ]
    if not report.demos:
        lines.append("(no demonstrations)")
        return "\n".join(lines)

    if not report.total_analysed_frames:
        lines.append("\nNo frames carry labels - run perception and import/verify annotations first.")
        return "\n".join(lines)

    total = report.total_analysed_frames
    lines.append(f"\nAnalysed frames with label info: {total}")

    situations = report.situations.most_common(max_situations)
    if situations:
        lines.append(
            f"\nSituation coverage - label sets co-occurring on a frame "
            f"({report.source} layer):"
        )
        for labels, count in situations:
            name = _situation_name(labels)
            lines.append(f"  {count:>6d} ({count / total * 100:5.1f}%): {name}")

    labels = report.labels.most_common()
    if labels:
        lines.append("\nLabel presence (share of analysed frames):")
        for label, count in labels:
            lines.append(f"  {label:<16} {_bar(count / total)} {count / total * 100:5.1f}%")

    if len(report.demos) > 1:
        lines.append("\nPer-demonstration coverage:")
        names = [demo.demo for demo in report.demos]
        columns = max(len(n) for n in names) if names else 0
        lines.append(
            f"  {'situation':<20} " + "  ".join(f"{n:>{columns}}" for n in names)
        )
        for labels, _count in situations:
            cells = [
                f"{'[x]' if _situation_name(labels) in demo.situation_names() else '[ ]':>{columns}}"
                for demo in report.demos
            ]
            lines.append(f"  {_situation_name(labels):<20} " + "  ".join(cells))

    if min_demos > 1:
        under = report.under_covered(min_demos)
        if under:
            lines.append("")
            for name, count, demos in under:
                lines.append(
                    f"! under-covered: {name} - only {count} of {demos} demonstrations"
                )
        else:
            lines.append(f"\nAll situations appear in at least {min_demos} demonstrations.")

    return "\n".join(lines)


def report_to_dict(report: CoverageReport) -> dict[str, Any]:
    def demo_dict(demo: DemoCoverage) -> dict[str, Any]:
        return {
            "demo": demo.demo,
            "analysed_frames": demo.analysed_frames,
            "labels": dict(demo.labels),
            "situations": {
                _situation_name(tuple(sorted(labels))): count
                for labels, count in demo.situations.items()
            },
        }

    return {
        "source": report.source,
        "demos": [demo_dict(d) for d in report.demos],
        "aggregate": {
            "analysed_frames": report.total_analysed_frames,
            "labels": dict(report.labels),
            "situations": {
                _situation_name(tuple(sorted(labels))): count
                for labels, count in report.situations.items()
            },
        },
    }