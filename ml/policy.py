"""Policy network: observation window -> key/button/pointer commands.

A per-frame CNN encodes every frame of the observation window, a GRU
summarizes the sequence, and small heads emit per-key / per-button logits
(multi-label, sigmoid+BCE during training) plus pointer velocity ``dx``/``dy``
(regression). The encoder ends in adaptive pooling, so the network is
size-agnostic to frame resolution and one checkpoint works across recordings
of any screen size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

_CHECKPOINT_FORMAT = 1


@dataclass
class PolicyOutput:
    keys_logits: torch.Tensor  # [B, n_keys]
    buttons_logits: torch.Tensor  # [B, n_buttons]
    dx: torch.Tensor  # [B]
    dy: torch.Tensor  # [B]


class ImitationPolicy(nn.Module):
    """Behavior-cloning policy: observation window -> action distribution."""

    def __init__(self, n_keys: int, n_buttons: int, hidden: int = 128) -> None:
        super().__init__()
        self.n_keys = n_keys
        self.n_buttons = n_buttons
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(64, hidden, batch_first=True)
        self.key_head = nn.Linear(hidden, n_keys)
        self.button_head = nn.Linear(hidden, n_buttons)
        self.dx_head = nn.Linear(hidden, 1)
        self.dy_head = nn.Linear(hidden, 1)

    def forward(self, observation: torch.Tensor) -> PolicyOutput:
        """``observation``: [B, T, 3, H, W] floats in [0, 1]."""
        batch, frames, channels, height, width = observation.shape
        features = self.encoder(
            observation.reshape(batch * frames, channels, height, width)
        ).reshape(batch, frames, -1)
        _, hidden = self.temporal(features)
        context = hidden[-1]  # last timestep: [B, hidden]
        return PolicyOutput(
            keys_logits=self.key_head(context),
            buttons_logits=self.button_head(context),
            dx=self.dx_head(context).squeeze(-1),
            dy=self.dy_head(context).squeeze(-1),
        )


def save_checkpoint(
    policy: ImitationPolicy,
    path: Path | str,
    keys: list[str],
    buttons: list[str],
    config: dict,
) -> None:
    """Persist weights, vocabulary and training config in one file."""
    torch.save(
        {
            "format_version": _CHECKPOINT_FORMAT,
            "vocab": {"keys": keys, "buttons": buttons},
            "architecture": {"hidden": policy.hidden},
            "config": config,
            "state_dict": policy.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path | str, device: str | torch.device = "cpu"
) -> tuple[ImitationPolicy, dict]:
    """Load a checkpoint into an eval-mode policy. Returns (policy, metadata)."""
    data = torch.load(path, map_location=device)
    vocab = data["vocab"]
    hidden = int(data.get("architecture", {}).get("hidden", 128))
    policy = ImitationPolicy(len(vocab["keys"]), len(vocab["buttons"]), hidden=hidden)
    policy.load_state_dict(data["state_dict"])
    policy.to(device)
    policy.eval()
    return policy, data
