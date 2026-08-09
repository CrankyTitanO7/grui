# Dataset Builder (planned)

This package is intentionally empty. It will convert raw recordings into
temporally aligned training samples.

## Input: a raw recording directory

```
recordings/2026-08-08_22-51-03_<session-id>/
    metadata.json   # version, session_id, platform, screen config, duration
    video.mp4       # screen capture (constant frame rate)
    events.jsonl    # keyboard / mouse / lifecycle events, timestamps t
    markers.jsonl   # human annotations
    frames.jsonl    # frame_index -> capture time t (exact sync source of truth)
```

## Planned pipeline

```
Raw video + timestamped events
             |
             v
      Synchronization (via frames.jsonl)
             |
             v
       Window creation
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

## Contract with the recorder

* All timestamps use one monotonic timeline (`t` = seconds since session start).
* `frames.jsonl` maps video frame index -> capture time, so even dropped
  frames never desynchronize events from video.
* Keyboard events can be collapsed into a multi-label held-key state;
  mouse buttons into independent states; mouse movement into `dx`/`dy`
  streams.

Not yet implemented: `imitate dataset build <recording_dir>`.
