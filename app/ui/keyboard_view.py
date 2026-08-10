"""Live full-keyboard/mouse state view for the player.

The keyboard and mouse live in separate titled areas (boxes): the keyboard
shows the complete layout (function row, numbers, QWERTY, modifiers, nav
cluster, arrows, numpad) with the mouse buttons stacked to the right of the
keys; the mouse box shows a mini screen with the pointer position. Keycaps
shrink/expand with the window so the whole keyboard fits at once.
"""

from __future__ import annotations

import logging
from typing import Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_ACTIVE_STYLE = (
    "background: #e74c3c; color: white; border: 1px solid #e74c3c; border-radius: 4px;"
)
_INACTIVE_STYLE = (
    "background: #222222; color: #cccccc; border: 1px solid #666666; border-radius: 4px;"
)

_BUTTON_LABELS = {"left": "LMB", "right": "RMB", "middle": "MMB"}

# Special key codes -> short display labels. Char-coded keys (KeyQ, Key1,
# Key[, ...) display their character and need no entry here.
_SPECIAL_LABELS: Mapping[str, str] = {
    "Key.esc": "Esc",
    "Key.print_screen": "PrtSc",
    "Key.scroll_lock": "ScrLk",
    "Key.pause": "Pause",
    "Key.backspace": "Bksp",
    "Key.tab": "Tab",
    "Key.caps_lock": "Caps",
    "Key.enter": "Enter",
    "Key.shift": "Shift",
    "Key.ctrl": "Ctrl",
    "Key.cmd": "Win",
    "Key.alt": "Alt",
    "Key.alt_gr": "AltGr",
    "Key.menu": "Menu",
    "Key.space": "Space",
    "Key.insert": "Ins",
    "Key.delete": "Del",
    "Key.home": "Home",
    "Key.end": "End",
    "Key.page_up": "PgUp",
    "Key.page_down": "PgDn",
    "Key.up": "↑",
    "Key.down": "↓",
    "Key.left": "←",
    "Key.right": "→",
    "Key.num_lock": "NumLk",
    "Key.cmd_l": "Win", "Key.cmd_r": "Win",
    "Key.shift_l": "Shift", "Key.shift_r": "Shift",
    "Key.ctrl_l": "Ctrl", "Key.ctrl_r": "Ctrl",
    "Key.alt_l": "Alt", "Key.alt_r": "Alt",
}

# Alias variants that should light up the same keycap (left/right modifiers).
_ALIASES: Mapping[str, str] = {
    "Key.shift_l": "Key.shift", "Key.shift_r": "Key.shift",
    "Key.ctrl_l": "Key.ctrl", "Key.ctrl_r": "Key.ctrl",
    "Key.alt_l": "Key.alt", "Key.alt_r": "Key.alt",
    "Key.cmd_l": "Key.cmd", "Key.cmd_r": "Key.cmd",
    "Key.alt_gr": "Key.alt",
}

# Full ANSI keyboard, laid out as (row, col, colspan, rowspan, code).
# Numpad digits/operators share pynput char codes with the main block, so
# both caps light up when those keys are pressed (a pynput limitation).
_GRID_PLACEMENT: list[tuple[int, int, int, int, str]] = [
    # function row
    (0, 0, 1, 1, "Key.esc"),
    (0, 1, 1, 1, "Key.f1"), (0, 2, 1, 1, "Key.f2"), (0, 3, 1, 1, "Key.f3"),
    (0, 4, 1, 1, "Key.f4"), (0, 5, 1, 1, "Key.f5"), (0, 6, 1, 1, "Key.f6"),
    (0, 7, 1, 1, "Key.f7"), (0, 8, 1, 1, "Key.f8"), (0, 9, 1, 1, "Key.f9"),
    (0, 10, 1, 1, "Key.f10"), (0, 11, 1, 1, "Key.f11"), (0, 12, 1, 1, "Key.f12"),
    (0, 13, 1, 1, "Key.print_screen"), (0, 14, 1, 1, "Key.scroll_lock"),
    (0, 15, 1, 1, "Key.pause"),
    # number row
    (1, 0, 1, 1, "Key`"), (1, 1, 1, 1, "Key1"), (1, 2, 1, 1, "Key2"),
    (1, 3, 1, 1, "Key3"), (1, 4, 1, 1, "Key4"), (1, 5, 1, 1, "Key5"),
    (1, 6, 1, 1, "Key6"), (1, 7, 1, 1, "Key7"), (1, 8, 1, 1, "Key8"),
    (1, 9, 1, 1, "Key9"), (1, 10, 1, 1, "Key0"),
    (1, 11, 1, 1, "Key-"), (1, 12, 1, 1, "Key="), (1, 13, 3, 1, "Key.backspace"),
    # qwerty row
    (2, 0, 2, 1, "Key.tab"),
    (2, 2, 1, 1, "KeyQ"), (2, 3, 1, 1, "KeyW"), (2, 4, 1, 1, "KeyE"),
    (2, 5, 1, 1, "KeyR"), (2, 6, 1, 1, "KeyT"), (2, 7, 1, 1, "KeyY"),
    (2, 8, 1, 1, "KeyU"), (2, 9, 1, 1, "KeyI"), (2, 10, 1, 1, "KeyO"),
    (2, 11, 1, 1, "KeyP"), (2, 12, 1, 1, "Key["), (2, 13, 1, 1, "Key]"),
    (2, 14, 2, 1, "Key\\"),
    # home row
    (3, 0, 2, 1, "Key.caps_lock"),
    (3, 2, 1, 1, "KeyA"), (3, 3, 1, 1, "KeyS"), (3, 4, 1, 1, "KeyD"),
    (3, 5, 1, 1, "KeyF"), (3, 6, 1, 1, "KeyG"), (3, 7, 1, 1, "KeyH"),
    (3, 8, 1, 1, "KeyJ"), (3, 9, 1, 1, "KeyK"), (3, 10, 1, 1, "KeyL"),
    (3, 11, 1, 1, "Key;"), (3, 12, 1, 1, "Key'"), (3, 13, 3, 1, "Key.enter"),
    # shift row
    (4, 0, 2, 1, "Key.shift"),
    (4, 2, 1, 1, "KeyZ"), (4, 3, 1, 1, "KeyX"), (4, 4, 1, 1, "KeyC"),
    (4, 5, 1, 1, "KeyV"), (4, 6, 1, 1, "KeyB"), (4, 7, 1, 1, "KeyN"),
    (4, 8, 1, 1, "KeyM"), (4, 9, 1, 1, "Key,"), (4, 10, 1, 1, "Key."),
    (4, 11, 1, 1, "Key/"), (4, 12, 4, 1, "Key.shift"),
    # bottom row
    (5, 0, 2, 1, "Key.ctrl"), (5, 2, 1, 1, "Key.cmd"), (5, 3, 1, 1, "Key.alt"),
    (5, 4, 6, 1, "Key.space"), (5, 10, 1, 1, "Key.alt_gr"), (5, 11, 1, 1, "Key.menu"),
    (5, 12, 4, 1, "Key.ctrl"),
    # nav cluster
    (1, 16, 1, 1, "Key.insert"), (1, 17, 1, 1, "Key.home"), (1, 18, 1, 1, "Key.page_up"),
    (2, 16, 1, 1, "Key.delete"), (2, 17, 1, 1, "Key.end"), (2, 18, 1, 1, "Key.page_down"),
    (4, 17, 1, 1, "Key.up"),
    (5, 16, 1, 1, "Key.left"), (5, 17, 1, 1, "Key.down"), (5, 18, 1, 1, "Key.right"),
    # numpad
    (1, 20, 1, 1, "Key.num_lock"), (1, 21, 1, 1, "Key/"), (1, 22, 1, 1, "Key*"),
    (1, 23, 1, 1, "Key-"),
    (2, 20, 1, 1, "Key7"), (2, 21, 1, 1, "Key8"), (2, 22, 1, 1, "Key9"),
    (2, 23, 1, 2, "Key+"),
    (3, 20, 1, 1, "Key4"), (3, 21, 1, 1, "Key5"), (3, 22, 1, 1, "Key6"),
    (4, 20, 1, 1, "Key1"), (4, 21, 1, 1, "Key2"), (4, 22, 1, 1, "Key3"),
    (4, 23, 1, 2, "Key.enter"),
    (5, 20, 2, 1, "Key0"), (5, 22, 1, 1, "Key."),
    # mouse buttons (stacked to the right of all keyboard keys)
    (1, 25, 1, 1, "button:left"), (2, 25, 1, 1, "button:right"),
    (3, 25, 1, 1, "button:middle"),
]


def _cap_label(code: str) -> str:
    if code in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[code]
    if code.startswith("button:"):
        name = code.split(":", 1)[1]
        return _BUTTON_LABELS.get(name, name)
    if code.startswith("Key."):
        return code[len("Key."):].replace("_", " ").title()
    if code.startswith("Key") and len(code) > 3:
        return code[3:]
    return code


def _canonical(code: str) -> str:
    return _ALIASES.get(code, code)


class KeyCap(QLabel):
    """A single keycap that lights up when its key is held."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.font()
        font.setPointSize(8)
        self.setFont(font)
        self.setMinimumHeight(22)
        self.setMinimumWidth(24)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(_ACTIVE_STYLE if active else _INACTIVE_STYLE)


class MouseSurface(QWidget):
    """Small screen mock showing the pointer position during playback."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(56)
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._screen_w = 1920.0
        self._screen_h = 1080.0
        self._pos: tuple[float, float] | None = None

    def set_screen_size(self, width: int, height: int) -> None:
        if width > 0 and height > 0:
            self._screen_w = float(width)
            self._screen_h = float(height)

    def set_pos(self, pos: tuple[int, int] | None) -> None:
        self._pos = (float(pos[0]), float(pos[1])) if pos is not None else None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setPen(Qt.GlobalColor.darkGray)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._pos is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "mouse: —")
            return
        x = self._pos[0] / self._screen_w * self.width()
        y = self._pos[1] / self._screen_h * self.height()
        painter.setBrush(Qt.GlobalColor.red)
        painter.setPen(Qt.GlobalColor.red)
        painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
        painter.setPen(Qt.GlobalColor.gray)
        painter.drawText(
            self.rect().adjusted(0, 0, 0, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            f"x={int(self._pos[0])}  y={int(self._pos[1])}",
        )


class KeyboardView(QWidget):
    """Full keyboard + mouse state showing what is pressed right now."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._caps: dict[str, list[KeyCap]] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._keyboard_group = QGroupBox("Keyboard")
        kb_layout = QVBoxLayout(self._keyboard_group)
        kb_layout.setContentsMargins(8, 4, 8, 6)
        kb_layout.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        self._grid = QGridLayout(host)
        self._grid.setSpacing(4)
        self._grid.setContentsMargins(0, 0, 0, 0)
        for row, col, colspan, rowspan, code in _GRID_PLACEMENT:
            cap = KeyCap(_cap_label(code))
            self._caps.setdefault(code, []).append(cap)
            self._grid.addWidget(cap, row, col, rowspan, colspan)
        scroll.setWidget(host)
        kb_layout.addWidget(scroll)
        kb_layout.addStretch(1)
        root.addWidget(self._keyboard_group)

        self._mouse_group = QGroupBox("Mouse")
        mouse_layout = QVBoxLayout(self._mouse_group)
        mouse_layout.setContentsMargins(8, 4, 8, 6)
        mouse_layout.setSpacing(4)
        self._mouse_surface = MouseSurface()
        mouse_layout.addWidget(self._mouse_surface)
        root.addWidget(self._mouse_group)

    # ------------------------------------------------------------ config

    def configure(self, used_codes: set[str], screen_size: tuple[int, int] | None = None) -> None:
        if screen_size is not None:
            self._mouse_surface.set_screen_size(*screen_size)

    # ------------------------------------------------------------ state

    def set_state(
        self, keys: set[str], buttons: set[str], mouse_pos: tuple[int, int] | None
    ) -> None:
        active = {_canonical(code) for code in keys}
        for code, caps in self._caps.items():
            if code.startswith("button:"):
                lit = code.split(":", 1)[1] in buttons
            else:
                lit = code in active
            for cap in caps:
                cap.set_active(lit)
        self._mouse_surface.set_pos(mouse_pos)
