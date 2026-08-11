# Dataset Builder

Converts raw recordings into temporally aligned observation->action training
samples.

```
grui dataset build <recording_dir> [--out DIR] [--obs-duration 3.0] [--fps 15] [--stride 1.0] [--horizon 0.2]
```

## Input: a raw recording directory

```
recordings/2026-08-08_22-51-03_<session-id>/
    metadata.json   # version, session_id, platform, screen config, duration
    video.mp4       # screen capture (constant frame rate)
    events.jsonl    # keyboard / mouse / lifecycle events, timestamps t
    markers.jsonl   # human annotations
    frames.jsonl    # frame_index -> capture time t (exact sync source of truth)
```

## Output layout

```
<out>/
    manifest.json   # format version, source session, config, counts
    frames.jsonl    # frame_index -> t -> frames/frame_<index>.png
    frames/         # one PNG per observation frame (shared across samples)
    samples.jsonl   # one JSON record per sample
```

Each `samples.jsonl` record:

```json
{
  "t": 3.2155,
  "observation": [17, 18, 19, 20],
  "action": {
    "keys": ["KeyW"],
    "buttons": ["left"],
    "mouse": {"x": 120, "y": 55, "dx": 20, "dy": 5}
  }
}
```

* `t` — session-clock seconds; samples ascend every `stride` seconds.
* `observation` — frame indices whose capture times cover `[t - duration, t]`
  at `fps`; frames are decoded once in a single sequential video pass, so
  overlapping windows share files on disk.
* `action` — the input state at `t`: held keys (multi-label), held mouse
  buttons (independent states), and the pointer position with `dx`/`dy`
  relative to the previous sample (`0, 0` when the position was unknown
  before).

## Pipeline

```
Raw video + timestamped events
             |
             v
      Synchronization (via frames.jsonl)
             |
             v
       Window creation (shared frames, one video pass)
             |
             v
 observation sequence -> action
 observation sequence -> action
```

Configuration is parameterized (nothing hardcoded):

```yaml
observation:
    duration: 3.0
    fps: 15

sampling:
    stride: 1.0

prediction:
    horizon: 0.2
```

The observation window, stride, FPS and prediction horizon are dataset-
generation parameters and are NOT stored in the recorder's raw format, so
the same recording can generate many different datasets. `prediction_horizon`
is recorded in the manifest for downstream trainers that offset the target.

## Contract with the recorder

* All timestamps use one monotonic timeline (`t` = seconds since session start).
* `frames.jsonl` maps video frame index -> capture time, so even dropped
  frames never desynchronize events from video.
* Keyboard events are collapsed into a multi-label held-key state; mouse
  buttons into independent states; mouse movement into `dx`/`dy` streams.
* Building is deterministic: the same recording + config produce identical
  output (no timestamps written into the dataset).
