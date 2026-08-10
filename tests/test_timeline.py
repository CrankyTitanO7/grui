"""Timeline edit operations, undo/redo and event remapping tests."""

import numpy as np
import pytest

from editor.timeline import EditSession, Timeline, remap_events

FRAMES = np.round(np.arange(0.0, 10.05, 0.1), 10)  # 101 frames, t = 0.0 .. 10.0


def make_session() -> EditSession:
    return EditSession(10.0, FRAMES)


def ranges(session: EditSession) -> list[list[float]]:
    return session.timeline.snapshot()


def test_initial_full_timeline():
    session = make_session()
    assert ranges(session) == [[0.0, 10.0]]
    assert session.timeline.duration == 10.0


def test_cut_middle():
    session = make_session()
    clipboard = session.cut(2.0, 5.0)
    assert ranges(session) == [[0.0, 2.0], [5.0, 10.0]]
    assert session.timeline.duration == 7.0
    assert clipboard.snapshot() == [[2.0, 5.0]]
    assert clipboard.duration == 3.0


def test_cut_at_boundaries():
    session = make_session()
    clipboard = session.cut(0.0, 5.0)
    assert ranges(session) == [[5.0, 10.0]]
    assert clipboard.snapshot() == [[0.0, 5.0]]
    session2 = make_session()
    session2.cut(5.0, 10.0)
    assert ranges(session2) == [[0.0, 5.0]]


def test_cut_crossing_clip_boundary_after_paste():
    session = make_session()
    session.cut(2.0, 5.0)  # clips: (0,2)@0, (5,10)@2
    clipboard = session.copy(1.0, 3.0)  # edited 1-3 spans both clips
    assert clipboard.snapshot() == [[1.0, 2.0], [5.0, 6.0]]
    session.paste(7.0, clipboard)
    assert ranges(session) == [[0.0, 2.0], [5.0, 10.0], [1.0, 2.0], [5.0, 6.0]]
    assert session.timeline.duration == 9.0
    # now cut across the second clip boundary
    session.cut(1.5, 7.5)
    assert ranges(session) == [[0.0, 1.5], [1.5, 2.0], [5.0, 6.0]]
    assert session.timeline.duration == 3.0


def test_copy_does_not_modify():
    session = make_session()
    clipboard = session.copy(2.0, 5.0)
    assert ranges(session) == [[0.0, 10.0]]
    assert clipboard.snapshot() == [[2.0, 5.0]]


def test_paste_shifts_and_duplicates():
    session = make_session()
    session.cut(2.0, 5.0)
    clipboard = session.copy(1.0, 3.0)
    session.paste(7.0, clipboard)
    assert ranges(session) == [[0.0, 2.0], [5.0, 10.0], [1.0, 2.0], [5.0, 6.0]]
    assert session.timeline.duration == 9.0
    starts = [c.start for c in session.timeline.clips]
    assert starts == [0.0, 2.0, 7.0, 8.0]


def test_paste_empty_is_noop():
    session = make_session()
    empty = Timeline(10.0, FRAMES)
    session.paste(5.0, empty)
    assert ranges(session) == [[0.0, 10.0]]


def test_delete_region():
    session = make_session()
    session.delete(0.0, 1.0)
    assert ranges(session) == [[1.0, 10.0]]
    assert session.timeline.duration == 9.0


def test_trim_to_selection():
    session = make_session()
    session.trim(0.5, 2.5)
    assert ranges(session) == [[0.5, 2.5]]
    assert session.timeline.duration == 2.0


def test_trim_empty_selection_clears():
    session = make_session()
    session.trim(2.0, 2.0)
    assert session.timeline.clips == []


def test_undo_redo():
    session = make_session()
    session.cut(2.0, 5.0)
    assert session.timeline.duration == 7.0
    assert session.undo()
    assert session.timeline.duration == 10.0
    assert session.undo() is False  # nothing before
    assert session.redo()
    assert session.timeline.duration == 7.0
    assert session.redo() is False


def test_undo_after_paste_restores():
    session = make_session()
    session.cut(2.0, 5.0)
    clipboard = session.copy(1.0, 3.0)
    session.paste(7.0, clipboard)
    assert session.timeline.duration == 9.0
    session.undo()
    assert ranges(session) == [[0.0, 2.0], [5.0, 10.0]]
    session.undo()
    assert ranges(session) == [[0.0, 10.0]]


def test_reset():
    session = make_session()
    session.cut(2.0, 5.0)
    session.reset()
    assert ranges(session) == [[0.0, 10.0]]


def test_snap_to_frame():
    session = make_session()
    assert session.timeline.snap(2.03) == 2.0
    assert session.timeline.snap(2.07) == 2.1
    assert session.timeline.snap(-5.0) == 0.0
    assert session.timeline.snap(50.0) == 10.0


def test_remap_events_through_cut():
    session = make_session()
    session.cut(2.0, 5.0)
    events = [
        {"t": 1.5, "device": "keyboard", "event": "down", "code": "KeyW"},
        {"t": 3.5, "device": "keyboard", "event": "down", "code": "KeyA"},  # cut region
        {"t": 6.0, "device": "mouse", "event": "move", "x": 1, "y": 2},
    ]
    out = remap_events(events, session.timeline)
    assert [(e["t"], e["device"], e["event"]) for e in out] == [
        (1.5, "keyboard", "down"),
        (3.0, "mouse", "move"),
    ]


def test_remap_duplicates_events_for_paste():
    session = make_session()
    session.cut(2.0, 5.0)
    clipboard = session.copy(1.0, 3.0)
    session.paste(7.0, clipboard)
    events = [{"t": 1.2, "device": "keyboard", "event": "down", "code": "KeyW"}]
    out = remap_events(events, session.timeline)
    assert [(e["t"], e["code"]) for e in out] == [
        (1.2, "KeyW"),
        (pytest.approx(7.2, abs=1e-9), "KeyW"),
    ]


def test_remap_events_at_clip_boundary_dropped():
    session = make_session()
    session.cut(5.0, 6.0)  # kept (0,5)@0 and (6,10)@5
    out = remap_events([{"t": 5.0, "device": "keyboard", "event": "down", "code": "KeyW"}], session.timeline)
    assert out == []  # t=5.0 is the exclusive end of kept clip (0,5)
    out = remap_events([{"t": 4.99, "device": "keyboard", "event": "down", "code": "KeyW"}], session.timeline)
    assert len(out) == 1
    assert out[0]["t"] == 4.99


def test_history_logged():
    session = make_session()
    session.cut(2.0, 5.0)
    session.paste(7.0, session.copy(1.0, 3.0))
    ops = [h["op"] for h in session.history]
    assert ops == ["cut", "paste"]
