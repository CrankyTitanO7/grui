"""Dataset generation: raw recordings -> temporally aligned samples.

The recorder deliberately produces raw demonstrations, not ML samples. This
package converts a raw recording into observation->action samples with
configurable parameters (not stored in the raw format, so the same recording
can generate many different datasets)::

    observation:
        duration: 3.0
        fps: 15

    sampling:
        stride: 1.0

    prediction:
        horizon: 0.2

Usage: ``grui dataset build <recording_dir>`` (see ``--help`` for options).
"""

from dataset.build import DatasetConfig, build_dataset

__all__ = ["DatasetConfig", "build_dataset"]
