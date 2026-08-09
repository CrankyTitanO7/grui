"""Tests for key/button serialization and screen helper functions."""

import numpy as np
import pytest

from recorder.keyboard import serialize_key
from recorder.mouse import button_name
from recorder.screen import _bgra_to_bgr, _resolve_region


def test_serialize_char_key():
    from pynput.keyboard import KeyCode

    code, char = serialize_key(KeyCode(char="w"))
    assert code == "KeyW"
    assert char == "w"


def test_serialize_special_key():
    from pynput.keyboard import Key

    code, char = serialize_key(Key.space)
    assert code == "Key.space"
    assert char is None


def test_serialize_unknown_vk():
    from pynput.keyboard import KeyCode

    code, char = serialize_key(KeyCode(vk=1234))
    assert code == "Key.vk_1234"
    assert char is None


def test_button_names():
    from pynput.mouse import Button

    assert button_name(Button.left) == "left"
    assert button_name(Button.right) == "right"
    assert button_name(Button.middle) == "middle"


def test_resolve_region_all_monitors():
    monitors = [
        {"left": -1920, "top": 0, "width": 3840, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},
    ]
    assert _resolve_region(monitors, -1)["width"] == 3840
    assert _resolve_region(monitors, 0) is monitors[1]
    assert _resolve_region(monitors, 1) is monitors[2]


def test_resolve_region_out_of_range():
    monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]
    with pytest.raises(ValueError):
        _resolve_region(monitors, 1)


def test_bgra_to_bgr_converts_and_detaches():
    bgra = np.array(
        [[[10, 20, 30, 255], [1, 2, 3, 255]], [[4, 5, 6, 0], [7, 8, 9, 0]]],
        dtype=np.uint8,
    )

    bgra_bytes = bgra.tobytes()

    class FakeShot:
        bgra = bgra_bytes
        width = 2
        height = 2

    out = _bgra_to_bgr(FakeShot())
    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)
    assert out.tolist() == [[[10, 20, 30], [1, 2, 3]], [[4, 5, 6], [7, 8, 9]]]
    assert not np.shares_memory(out, bgra)
