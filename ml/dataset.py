"""PyTorch dataset over a built dataset directory.

Loads the artifacts produced by ``grui dataset build`` (``samples.jsonl``,
``frames.jsonl``, ``vocab.json``, PNG frames) and serves tensor pairs: an
observation window ``[T, 3, H, W]`` (float in [0, 1]) and the action at the
sample time — one-hot held-key / held-button vectors plus pointer velocity
``dx``/``dy`` and a ``mouse_valid`` mask (the mouse may be unknown for early
samples). All action dimensions are indexed by the fixed vocabulary, so a
checkpoint trained on one set of recordings can be evaluated on another.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImitationDataset(Dataset):
    """One sample per line of ``samples.jsonl`` in a built dataset dir."""

    def __init__(
        self,
        dataset_dir: Path | str,
        *,
        keys: list[str] | None = None,
        buttons: list[str] | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        root = Path(dataset_dir)
        self.root = root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self._samples = [
            json.loads(line)
            for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self._frame_paths: dict[int, Path] = {}
        for line in (root / "frames.jsonl").read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            self._frame_paths[int(entry["frame_index"])] = root / entry["path"]
        built_in = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        self.keys = list(keys) if keys is not None else list(built_in["keys"])
        self.buttons = list(buttons) if buttons is not None else list(built_in["buttons"])
        self._key_index = {code: i for i, code in enumerate(self.keys)}
        self._button_index = {code: i for i, code in enumerate(self.buttons)}
        self._target = tuple(int(v) for v in target_size) if target_size else None
        self._frame_cache: dict[int, np.ndarray] = {}
        self.config = manifest["config"]

    # ------------------------------------------------------------------ API

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def n_buttons(self) -> int:
        return len(self.buttons)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._samples[index]
        observation = torch.stack(
            [self._load_frame(idx) for idx in sample["observation"]]
        )  # [T, 3, H, W] float in [0, 1]
        action = sample["action"]
        keys = torch.zeros(self.n_keys, dtype=torch.float32)
        for code in action["keys"]:
            if code in self._key_index:
                keys[self._key_index[code]] = 1.0
        buttons = torch.zeros(self.n_buttons, dtype=torch.float32)
        for code in action["buttons"]:
            if code in self._button_index:
                buttons[self._button_index[code]] = 1.0
        mouse = action["mouse"]
        if mouse is None:
            dx = dy = torch.tensor(0.0)
            valid = torch.tensor(0.0)
        else:
            dx = torch.tensor(float(mouse["dx"]))
            dy = torch.tensor(float(mouse["dy"]))
            valid = torch.tensor(1.0)
        return {
            "observation": observation,
            "keys": keys,
            "buttons": buttons,
            "dx": dx,
            "dy": dy,
            "mouse_valid": valid,
            "t": torch.tensor(float(sample["t"])),
        }

    # ------------------------------------------------------------- internal

    def _load_frame(self, frame_index: int) -> torch.Tensor:
        if frame_index not in self._frame_cache:
            frame = cv2.imread(str(self._frame_paths[frame_index]))  # BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self._target is not None:
                frame = cv2.resize(frame, self._target)
            if len(self._frame_cache) > 512:
                self._frame_cache.clear()
            self._frame_cache[frame_index] = frame
        frame = self._frame_cache[frame_index]
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor
