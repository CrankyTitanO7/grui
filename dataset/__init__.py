"""Dataset generation (future).

The recorder deliberately produces raw demonstrations, not ML samples. This
package is reserved for the future dataset builder that converts a raw
recording into temporally aligned observation->action samples, e.g.::

    observation:
        duration: 3.0
        fps: 15

    sampling:
        stride: 1.0

    prediction:
        horizon: 0.2

The observation window, stride, FPS and prediction horizon are dataset-
generation parameters and are NOT stored in the recorder's raw format, so
the same recording can generate many different datasets.
"""
