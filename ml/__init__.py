"""ML package: PyTorch dataset, policy, training and agent injection.

The raw recorder + dataset builder produce demonstrations; this package
turns them into a trained agent:

- :class:`ml.dataset.ImitationDataset` — tensor samples from a built
  dataset directory (observation windows, one-hot key/button targets,
  pointer velocity).
- :class:`ml.policy.ImitationPolicy` — CNN+GRU policy with per-key /
  per-button heads and dx/dy velocity heads.
- ``grui train`` — behavior-cloning training over one or more datasets.
- ``grui agent`` — run a checkpoint over a dataset (dry run by default)
  or inject its actions into the OS with pynput (``--inject``).

Requires ``torch`` (optional extra: ``pip install -e ".[ml]"``).
"""

from ml.dataset import ImitationDataset
from ml.policy import ImitationPolicy, load_checkpoint, save_checkpoint

__all__ = ["ImitationDataset", "ImitationPolicy", "load_checkpoint", "save_checkpoint"]
