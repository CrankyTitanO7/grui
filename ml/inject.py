"""Run a trained policy on a dataset and inject its actions (``grui agent``).

Walks the samples of a built dataset in order, feeds each observation window
to a checkpoint, and either prints the predicted actions (dry run, the
default) or injects them into the OS via pynput — pressing/releasing keys
and mouse buttons and moving the pointer by ``dx``/``dy``. Injection is
deliberately opt-in: ``--inject`` is required to touch the real desktop.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from ml.dataset import ImitationDataset
from ml.policy import load_checkpoint

logger = logging.getLogger(__name__)


def _code_to_key(code: str):
    """Reverse of :func:`recorder.keyboard.serialize_key`."""
    from pynput.keyboard import Key, KeyCode

    if code.startswith("Key."):
        name = code[len("Key."):]
        if name.startswith("vk_"):
            return KeyCode.from_vk(int(name[len("vk_"):]))
        return getattr(Key, name)
    if code.startswith("Key") and len(code) > 3:
        return KeyCode.from_char(code[3])
    return KeyCode.from_char(code)


def _button_for(name: str):
    from pynput.mouse import Button

    return getattr(Button, name, Button.left)


class Agent:
    """A trained policy plus the vocabulary needed to decode its outputs."""

    def __init__(self, checkpoint: Path | str, device: str = "auto") -> None:
        resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.policy, self.data = load_checkpoint(checkpoint, resolved)
        self.keys: list[str] = list(self.data["vocab"]["keys"])
        self.buttons: list[str] = list(self.data["vocab"]["buttons"])

    def act(self, observation: torch.Tensor, threshold: float = 0.5) -> dict:
        """Observation ``[T, 3, H, W]`` -> predicted action dict."""
        with torch.no_grad():
            out = self.policy(observation.unsqueeze(0).to(next(self.policy.parameters()).device))
        probs = torch.sigmoid(out.keys_logits[0])
        held_keys = [code for code, prob in zip(self.keys, probs) if float(prob) >= threshold]
        probs = torch.sigmoid(out.buttons_logits[0])
        held_buttons = [code for code, prob in zip(self.buttons, probs) if float(prob) >= threshold]
        return {
            "keys": held_keys,
            "buttons": held_buttons,
            "dx": float(out.dx[0]),
            "dy": float(out.dy[0]),
        }


class InputInjector:
    """Holds the currently pressed keys/buttons; releases them on finish."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller as KeyboardController
        from pynput.mouse import Controller as MouseController

        self._keyboard = KeyboardController()
        self._mouse = MouseController()
        self._held_keys: set = set()
        self._held_buttons: set = set()

    def apply(self, action: dict) -> None:
        wanted_keys = {_code_to_key(code) for code in action["keys"]}
        for key in self._held_keys - wanted_keys:
            self._keyboard.release(key)
        for key in wanted_keys - self._held_keys:
            self._keyboard.press(key)
        self._held_keys = wanted_keys

        wanted_buttons = {_button_for(name) for name in action["buttons"]}
        for button in self._held_buttons - wanted_buttons:
            self._mouse.release(button)
        for button in wanted_buttons - self._held_buttons:
            self._mouse.press(button)
        self._held_buttons = wanted_buttons

        self._mouse.move(int(round(action["dx"])), int(round(action["dy"])))

    def finish(self) -> None:
        for key in self._held_keys:
            try:
                self._keyboard.release(key)
            except Exception:  # noqa: BLE001
                logger.exception("failed to release key")
        for button in self._held_buttons:
            try:
                self._mouse.release(button)
            except Exception:  # noqa: BLE001
                logger.exception("failed to release button")
        self._held_keys.clear()
        self._held_buttons.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grui agent",
        description="Run a trained policy over a dataset and (optionally) inject its actions.",
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH", help="trained .pt checkpoint")
    parser.add_argument("--dataset", required=True, metavar="DIR", help="built dataset directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="key/button activation threshold")
    parser.add_argument("--resize", metavar="WxH", default=None,
                        help="resize frames to the size used during training, e.g. 160x120")
    parser.add_argument("--device", default="auto", help="cpu, cuda or auto")
    parser.add_argument("--inject", action="store_true",
                        help="actually press keys / move the mouse (default: dry run)")
    parser.add_argument("--max-samples", type=int, default=0, help="stop after N samples (0 = all)")
    return parser


def _parse_resize(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    width, height = (int(part) for part in value.lower().split("x"))
    return width, height


def run_agent(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        agent = Agent(args.checkpoint, device=args.device)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: could not load checkpoint: {exc}", file=sys.stderr)
        return 1
    dataset = ImitationDataset(args.dataset, keys=agent.keys, buttons=agent.buttons,
                               target_size=_parse_resize(args.resize))
    injector = InputInjector() if args.inject else None
    print(f"policy: {args.checkpoint}  dataset: {len(dataset)} samples  "
          f"mode: {'inject' if injector else 'dry run'}")
    try:
        for index in range(len(dataset)):
            if args.max_samples and index >= args.max_samples:
                break
            sample = dataset[index]
            action = agent.act(sample["observation"], threshold=args.threshold)
            t = float(sample["t"])
            mouse = f"dx={action['dx']:+.0f} dy={action['dy']:+.0f}" if action["dx"] or action["dy"] else "idle"
            print(f"t={t:.2f}  keys={','.join(action['keys']) or '-'}  "
                  f"buttons={','.join(action['buttons']) or '-'}  {mouse}")
            if injector is not None:
                injector.apply(action)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        if injector is not None:
            injector.finish()
    return 0
