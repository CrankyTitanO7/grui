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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

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
    parser.add_argument("--val-fraction", type=float, default=0.2, metavar="FRAC",
                        help="fraction of each dataset held out for validation (default: 0.2; 0 disables)")
    parser.add_argument("--val-dataset", action="append", metavar="DIR",
                        help="dataset dirs used exclusively for validation (repeatable; overrides --val-fraction)")
    parser.add_argument("--early-stop", type=int, default=0, metavar="N",
                        help="stop after N consecutive epochs without validation-loss improvement (default: 0 = never)")
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


def _split(
    dataset: ImitationDataset, fraction: float, rng: torch.Generator
) -> tuple[Subset, Subset]:
    """Deterministic (seed-based) train/validation split of one dataset."""
    indices = torch.randperm(len(dataset), generator=rng).tolist()
    n_val = int(round(len(dataset) * fraction))
    return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


def _evaluate(
    policy: ImitationPolicy,
    loader: DataLoader,
    device: torch.device,
    key_loss: nn.Module,
    button_loss: nn.Module,
    mouse_loss: nn.Module,
) -> dict[str, float]:
    """Validation pass in eval mode. Loss is batch-size-weighted."""
    policy.eval()
    total = 0.0
    samples = 0
    key_hits = key_total = button_hits = button_total = 0
    dx_abs = dy_abs = 0.0
    mouse_count = 0
    with torch.no_grad():
        for batch in loader:
            size = batch["observation"].shape[0]
            out = policy(batch["observation"].to(device))
            if batch["keys"].numel():
                keys_t = batch["keys"].to(device)
                total += float(key_loss(out.keys_logits, keys_t)) * size
                keys_bool = keys_t == 1
                key_hits += int(((torch.sigmoid(out.keys_logits) >= 0.5) == keys_bool).sum())
                key_total += int(keys_bool.numel())
            if batch["buttons"].numel():
                buttons_t = batch["buttons"].to(device)
                total += float(button_loss(out.buttons_logits, buttons_t)) * size
                buttons_bool = buttons_t == 1
                button_hits += int(((torch.sigmoid(out.buttons_logits) >= 0.5) == buttons_bool).sum())
                button_total += int(buttons_bool.numel())
            mouse = batch["mouse_valid"].to(device) == 1
            if bool(mouse.any()):
                dx = batch["dx"].to(device)[mouse]
                dy = batch["dy"].to(device)[mouse]
                total += float(mouse_loss(out.dx[mouse], dx) + mouse_loss(out.dy[mouse], dy)) * size
                mouse_count += int(mouse.sum())
                dx_abs += float(out.dx[mouse].abs().sum())
                dy_abs += float(out.dy[mouse].abs().sum())
            samples += size
    return {
        "loss": total / max(1, samples),
        "keys_acc": key_hits / max(1, key_total),
        "buttons_acc": button_hits / max(1, button_total),
        "dx_mae": dx_abs / max(1, mouse_count),
        "dy_mae": dy_abs / max(1, mouse_count),
        "samples": samples,
    }


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
    if not 0 <= args.val_fraction < 1:
        raise ValueError(f"--val-fraction must be in [0, 1) (got {args.val_fraction})")
    if args.val_dataset and args.val_fraction:
        raise ValueError("use either --val-dataset or --val-fraction, not both")
    if args.early_stop < 0:
        raise ValueError(f"--early-stop must be >= 0 (got {args.early_stop})")
    torch.manual_seed(args.seed)
    device = _device(args.device)
    keys, buttons = _vocab_union(args.dataset)
    target_size = _parse_resize(args.resize)
    rng = torch.Generator().manual_seed(args.seed)

    train_sets: list[Dataset] = []
    val_sets: list[Dataset] = []
    for directory in args.dataset:
        dataset = ImitationDataset(directory, keys=keys, buttons=buttons, target_size=target_size)
        if args.val_dataset:
            train_sets.append(dataset)
        elif args.val_fraction > 0:
            train_part, val_part = _split(dataset, args.val_fraction, rng)
            train_sets.append(train_part)
            val_sets.append(val_part)
        else:
            train_sets.append(dataset)
    for directory in args.val_dataset or []:
        val_sets.append(
            ImitationDataset(directory, keys=keys, buttons=buttons, target_size=target_size)
        )
    if not any(len(ds) for ds in train_sets):
        raise ValueError("no samples in the given datasets")
    if val_sets and not any(len(ds) for ds in val_sets):
        raise ValueError("no samples left for validation")

    loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=args.batch_size,
        shuffle=False,
    )

    policy = ImitationPolicy(len(keys), len(buttons), hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    key_loss = nn.BCEWithLogitsLoss()
    button_loss = nn.BCEWithLogitsLoss()
    mouse_loss = nn.MSELoss()

    metrics_path = args.metrics or str(Path(args.out).with_suffix(".metrics.jsonl"))
    metrics = MetricsLogger(metrics_path)
    train_samples = sum(len(ds) for ds in train_sets)
    val_samples = sum(len(ds) for ds in val_sets)
    print(f"device: {device}  keys: {len(keys)}  buttons: {len(buttons)}  "
          f"train: {train_samples}  val: {val_samples}  metrics: {metrics_path}")
    started = time.monotonic()
    best_val_loss = float("inf")
    stalled = 0
    stopped_early = False
    epoch_loss = 0.0
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
            keys_t = batch["keys"].to(device)
            buttons_t = batch["buttons"].to(device)
            dx_t = batch["dx"].to(device)
            dy_t = batch["dy"].to(device)
            mouse = batch["mouse_valid"].to(device) == 1
            for name, tensor in (
                ("observation", observation),
                ("keys", keys_t),
                ("buttons", buttons_t),
                ("dx", dx_t),
                ("dy", dy_t),
            ):
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"non-finite value in {name}: "
                        f"nan={int(torch.isnan(tensor).sum())} inf={int(torch.isinf(tensor).sum())} "
                        f"min={float(tensor.min())} max={float(tensor.max())}"
                    )
            out = policy(observation)
            terms: dict[str, torch.Tensor] = {}
            if keys_t.numel():
                terms["keys"] = key_loss(out.keys_logits, keys_t)
            if buttons_t.numel():
                terms["buttons"] = button_loss(out.buttons_logits, buttons_t)
            if bool(mouse.any()):
                dx = dx_t[mouse]
                dy = dy_t[mouse]
                terms["mouse"] = mouse_loss(out.dx[mouse], dx) + mouse_loss(out.dy[mouse], dy)
                mouse_count += int(mouse.sum())
                dx_abs += float(out.dx[mouse].abs().sum().detach())
                dy_abs += float(out.dy[mouse].abs().sum().detach())
            for name, term in terms.items():
                if not torch.isfinite(term):
                    raise ValueError(
                        f"loss term {name!r} is non-finite: "
                        f"nan={bool(torch.isnan(term))} inf={bool(torch.isinf(term))} "
                        f"sample dx range [{float(dx_t.min())}, {float(dx_t.max())}] "
                        f"dy range [{float(dy_t.min())}, {float(dy_t.max())}]"
                    )
            loss = sum(terms.values())
            keys_bool = keys_t == 1
            key_hits += int(((torch.sigmoid(out.keys_logits) >= 0.5) == keys_bool).sum().detach())
            key_total += keys_bool.numel()
            buttons_bool = buttons_t == 1
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
        line = (f"epoch {epoch}/{args.epochs}: loss={epoch_loss:.4f} "
                f"keys_acc={key_acc:.2f} buttons_acc={button_acc:.2f} "
                f"dx_mae={dx_mae:.2f} dy_mae={dy_mae:.2f}")
        record: dict = {
            "event": "epoch",
            "epoch": epoch,
            "loss": epoch_loss,
            "key_acc": key_acc,
            "button_acc": button_acc,
            "dx_mae": dx_mae,
            "dy_mae": dy_mae,
            "samples": train_samples,
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        if val_loader:
            validation = _evaluate(policy, val_loader, device, key_loss, button_loss, mouse_loss)
            line += (f"  val_loss={validation['loss']:.4f} "
                     f"val_keys_acc={validation['keys_acc']:.2f} "
                     f"val_buttons_acc={validation['buttons_acc']:.2f} "
                     f"val_dx_mae={validation['dx_mae']:.2f} val_dy_mae={validation['dy_mae']:.2f}")
            for key, value in validation.items():
                record[f"val_{key}"] = value
            if args.early_stop:
                if validation["loss"] < best_val_loss - 1e-6:
                    best_val_loss = validation["loss"]
                    stalled = 0
                else:
                    stalled += 1
                    if stalled >= args.early_stop:
                        line += f"  [early stop: val_loss not improved for {args.early_stop} epochs]"
                        print(line)
                        stopped_early = True
                        metrics.write(record)
                        break
        print(line)
        metrics.write(record)
        if stopped_early:
            break
    metrics.write(
        {
            "event": "summary",
            "epochs": args.epochs,
            "epochs_run": epoch,
            "early_stopped": stopped_early,
            "final_loss": epoch_loss,
            "train_samples": train_samples,
            "val_samples": val_samples,
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
            "epochs_run": epoch,
            "hidden": args.hidden,
            "lr": args.lr,
            "seed": args.seed,
            "datasets": list(args.dataset),
            "val_dataset": args.val_dataset,
            "val_fraction": args.val_fraction,
            "early_stop": args.early_stop,
            "resize": args.resize,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"saved checkpoint: {args.out}")
    return 0
