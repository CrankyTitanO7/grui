# Training & Agents (PyTorch)

Turns built datasets into a trained behavior-cloning agent. Requires torch:

```
pip install -e ".[ml]"        # or uv pip install -e ".[ml]"
```

## Train

```
grui train --dataset <dataset_dir> [--dataset <more...>] --out ckpt.pt \
           [--epochs 5] [--batch-size 16] [--lr 1e-3] [--hidden 128] \
           [--resize 160x120] [--seed 0] [--device auto] \
           [--val-fraction 0.2] [--early-stop 5] \
           [--metrics PATH] [--no-progress]
```

* The policy is a per-frame CNN + GRU: it watches the observation window
  `[t - duration, t]` and emits per-key and per-button logits (multi-label)
  plus pointer velocity `dx`/`dy`.
* Multiple `--dataset` directories are combined; the vocabulary is the union
  across them, so demonstrations from different recordings share one model.
  Use `--resize WxH` when the recordings have different screen resolutions.
* Loss: BCE-with-logits on keys and buttons, MSE on `dx`/`dy` (masked to
  samples where the pointer position is known).
* Validation: by default `--val-fraction 0.2` holds out 20% of each dataset
  (seeded, reproducible) for a per-epoch eval pass — `val_loss` and the
  `val_*` metrics let you spot overfitting. Use `--val-fraction 0` to train
  on everything, or `--val-dataset DIR` (repeatable) to validate on separate
  recordings. `--early-stop N` stops when `val_loss` has not improved for `N`
  consecutive epochs.
* The checkpoint (`.pt`) stores weights, the exact vocabulary, the
  architecture and the training config.

## Monitoring

While training runs you get two views:

* **Live stdout** — a progress bar per epoch (`batch 12/28 | loss=... | eta`)
  plus a per-epoch line with richer metrics: `loss`, `keys_acc`,
  `buttons_acc` (binary accuracy at threshold 0.5) and `dx_mae`/`dy_mae`.
  The bar auto-disables when stdout is piped; `--no-progress` forces that.
* **Metrics log** — one JSON record per epoch appended to
  `<out>.metrics.jsonl` (override with `--metrics PATH`), plus a final
  `summary` record. Tail it live or plot it:

  ```bash
  Get-Content ckpt.metrics.jsonl -Wait        # watch while training
  ```

  Each epoch record: `event=epoch, epoch, loss, key_acc, button_acc,
  dx_mae, dy_mae, samples, elapsed_s` plus `val_loss` and `val_*` metrics
  when validation is enabled.

## Run the agent

```
grui agent --checkpoint ckpt.pt --dataset <dataset_dir>            # dry run (default)
grui agent --checkpoint ckpt.pt --dataset <dataset_dir> --inject   # actually presses keys / moves the mouse
```

The dry run prints the predicted actions for every sample
(`keys=... buttons=... dx=... dy=...`). With `--inject`, the agent presses
and releases keys/buttons with pynput and moves the pointer by `dx`/`dy` —
all held inputs are released on exit (Ctrl+C included). Injection touches
the real desktop, so it is deliberately opt-in.

## Architecture

```
ml/
    dataset.py   ImitationDataset: dataset dir -> tensors (window, one-hot targets, dx/dy, mask)
    policy.py    ImitationPolicy (CNN + GRU) + checkpoint save/load
    monitor.py   ProgressBar + MetricsLogger (metrics.jsonl)
    train.py     grui train: behavior-cloning training loop
    inject.py    grui agent: dry-run or pynput injection of a checkpoint
```

`ImitationDataset` is usable directly from Python for custom training:

```python
from torch.utils.data import DataLoader
from ml.dataset import ImitationDataset

loader = DataLoader(ImitationDataset("path/to/dataset", target_size=(160, 120)), batch_size=16, shuffle=True)
```
