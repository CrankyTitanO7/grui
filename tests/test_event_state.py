"""KeyStateTimeline query tests (held keys/buttons, mouse position)."""

from player.event_state import KeyStateTimeline


def make_timeline() -> KeyStateTimeline:
    return KeyStateTimeline(
        events=[
            {"t": 0.0, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 0.5, "device": "keyboard", "event": "down", "code": "KeyA"},
            {"t": 0.7, "device": "keyboard", "event": "up", "code": "KeyW"},
            {"t": 1.0, "device": "keyboard", "event": "up", "code": "KeyA"},
            {"t": 0.4, "device": "mouse", "event": "button_down", "button": "left"},
            {"t": 0.8, "device": "mouse", "event": "button_up", "button": "left"},
            {"t": 0.6, "device": "mouse", "event": "move", "x": 100, "y": 200},
        ]
    )


def test_active_keys_before_any_event():
    tl = make_timeline()
    assert tl.active_keys_at(-1.0) == set()


def test_active_keys_at_down_event():
    tl = make_timeline()
    assert tl.active_keys_at(0.0) == {"KeyW"}


def test_active_keys_during_hold():
    tl = make_timeline()
    assert tl.active_keys_at(0.2) == {"KeyW"}
    assert tl.active_keys_at(0.6) == {"KeyW", "KeyA"}


def test_active_keys_after_release():
    tl = make_timeline()
    assert tl.active_keys_at(0.8) == {"KeyA"}
    assert tl.active_keys_at(1.5) == set()


def test_active_buttons():
    tl = make_timeline()
    assert tl.active_buttons_at(0.3) == set()
    assert tl.active_buttons_at(0.5) == {"left"}
    assert tl.active_buttons_at(1.5) == set()


def test_mouse_position():
    tl = make_timeline()
    # before any move: unknown
    assert tl.mouse_at(0.0) is None
    assert tl.mouse_at(0.6) == (100, 200)
    # position persists after the move
    assert tl.mouse_at(3.0) == (100, 200)


def test_unsorted_input_is_sorted():
    tl = KeyStateTimeline(
        events=[
            {"t": 0.5, "device": "keyboard", "event": "up", "code": "KeyW"},
            {"t": 0.0, "device": "keyboard", "event": "down", "code": "KeyW"},
        ]
    )
    assert tl.active_keys_at(0.2) == {"KeyW"}
    assert tl.active_keys_at(0.7) == set()


def test_used_codes_property():
    tl = make_timeline()
    assert sorted(tl.used_codes) == ["KeyA", "KeyW"]


def test_irrelevant_devices_ignored():
    tl = KeyStateTimeline(
        events=[
            {"t": 0.0, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 0.5, "device": "keyboard", "event": "move", "x": 5, "y": 6},
            {"t": 1.0, "device": "mouse", "event": "move", "x": 5, "y": 6},
        ]
    )
    assert tl.active_keys_at(0.8) == {"KeyW"}
    assert tl.mouse_at(1.5) == (5, 6)


def test_events_without_t_dropped():
    tl = KeyStateTimeline(events=[{"device": "keyboard", "event": "down", "code": "KeyW"}])
    assert tl.active_keys_at(5.0) == set()
    assert tl.used_codes == set()
