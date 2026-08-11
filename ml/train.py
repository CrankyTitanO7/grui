"""Behavior-cloning training over built datasets (``grui train``).

Trains an :class:`~ml.policy.ImitationPolicy` to reproduce the demonstrated
actions: held keys and buttons are multi-label targets (BCE with logits),
pointer velocity ``dx``/``dy`` is a regression target (MSE, masked to
samples where the mouse position is known). Accepts one or more dataset
directories; vocabularies are the union across them, so demonstrations from
different recordings share one model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from ml.dataset import ImitationDataset
from ml.monitor import MetricsLogger, ProgressBar
from ml.policy import ImitationPolicy, save_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grui train",
        description="Train a behavior-cloning policy on built datasets.",
    )
    parser.add_argument("--dataset", action="append", required=True, metavar="DIR",
                        help="built dataset directory (repeatable)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128, help="GRU width")
    parser.add_argument("--resize", metavar="WxH", default=None,
                        help="resize frames to a fixed size, e.g. 160x120 (needed when mixing recordings of different resolutions)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="cpu, cuda or auto")
    parser.add_argument("--out", required=True, metavar="PATH", help="checkpoint path (.pt)")
    parser.add_argument("--metrics", metavar="PATH", default=None,
                        help="metrics log path (default: <out>.metrics.jsonl)")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable the live progress bar (auto-disabled when piped)")
    return parser


def _parse_resize(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"invalid --resize {value!r} (expected WxH)") from exc
    return width, height


def _vocab_union(dataset_dirs: list[str]) -> tuple[list[str], list[str]]:
    keys: set[str] = set()
    buttons: set[str] = set()
    for directory in dataset_dirs:
        vocab = json.loads((Path(directory) / "vocab.json").read_text(encoding="utf-8"))
        keys.update(vocab["keys"])
        buttons.update(vocab["buttons"])
    return sorted(keys), sorted(buttons)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1 (got {args.epochs})")
    torch.manual_seed(args.seed)
    device = _device(args.device)
    keys, buttons = _vocab_union(args.dataset)
    target_size = _parse_resize(args.resize)
    datasets = [
        ImitationDataset(directory, keys=keys, buttons=buttons, target_size=target_size)
        for directory in args.dataset
    ]
    if not any(len(ds) for ds in datasets):
        raise ValueError("no samples in the given datasets")
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    policy = ImitationPolicy(len(keys), len(buttons), hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    key_loss = nn.BCEWithLogitsLoss()
    button_loss = nn.BCEWithLogitsLoss()
    mouse_loss = nn.MSELoss()

    metrics_path = args.metrics or str(Path(args.out).with_suffix(".metrics.jsonl"))
    metrics = MetricsLogger(metrics_path)
    total_samples = sum(len(ds) for ds in datasets)
    print(f"device: {device}  keys: {len(keys)}  buttons: {len(buttons)}  "
          f"samples: {total_samples}  metrics: {metrics_path}")
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        policy.train()
        bar = ProgressBar(len(loader), prefix=f"epoch {epoch}/{args.epochs}",
                          enabled=False if args.no_progress else None)
        total = 0.0
        batches = 0
        key_hits = key_total = button_hits = button_total = 0
        dx_abs = dy_abs = 0.0
        mouse_count = 0
        for batch in loader:
            observation = batch["observation"].to(device)
            out = policy(observation)
            loss = key_loss(out.keys_logits, batch["keys"].to(device))
            loss = loss + button_loss(out.buttons_logits, batch["buttons"].to(device))
            mouse = batch["mouse_valid"].to(device) == 1
            if bool(mouse.any()):
                dx = batch["dx"].to(device)[mouse]
                dy = batch["dy"].to(device)[mouse]
                loss = loss + mouse_loss(out.dx[mouse], dx)
                loss = loss + mouse_loss(out.dy[mouse], dy)
                mouse_count += int(mouse.sum())
                dx_abs += float(out.dx[mouse].abs().sum().detach())
                dy_abs += float(out.dy[mouse].abs().sum().detach())
            keys_bool = batch["keys"].to(device) == 1
            key_hits += int(((torch.sigmoid(out.keys_logits) >= 0.5) == keys_bool).sum().detach())
            key_total += keys_bool.numel()
            buttons_bool = batch["buttons"].to(device) == 1
            button_hits += int(((torch.sigmoid(out.buttons_logits) >= 0.5) == buttons_bool).sum().detach())
            button_total += buttons_bool.numel()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            bar.update(loss=float(loss.detach()))
        bar.close()
        epoch_loss = total / max(1, batches)
        key_acc = key_hits / max(1, key_total)
        button_acc = button_hits / max(1, button_total)
        dx_mae = dx_abs / max(1, mouse_count)
        dy_mae = dy_abs / max(1, mouse_count)
        print(f"epoch {epoch}/{args.epochs}: loss={epoch_loss:.4f} "
              f"keys_acc={key_acc:.2f} buttons_acc={button_acc:.2f} "
              f"dx_mae={dx_mae:.2f} dy_mae={dy_mae:.2f}")
        metrics.write(
            {
                "event": "epoch",
                "epoch": epoch,
                "loss": epoch_loss,
                "key_acc": key_acc,
                "button_acc": button_acc,
                "dx_mae": dx_mae,
                "dy_mae": dy_mae,
                "samples": total_samples,
                "elapsed_s": round(time.monotonic() - started, 2),
            }
        )
    metrics.write(
        {
            "event": "summary",
            "epochs": args.epochs,
            "final_loss": epoch_loss,
            "samples": total_samples,
            "total_s": round(time.monotonic() - started, 2),
            "checkpoint": str(args.out),
        }
    )

    save_checkpoint(
        policy.cpu(),
        args.out,
        keys,
        buttons,
        {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "lr": args.lr,
            "seed": args.seed,
            "datasets": list(args.dataset),
            "resize": args.resize,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"saved checkpoint: {args.out}")
    return 0
