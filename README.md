# grui — Grand Unified Imitation

Capture a user's screen and keyboard/mouse actions while they interact with
**arbitrary software** — no source code, engine, API or backend access
required. Output is a versioned, timestamped **raw demonstration** that a
separate dataset builder can later convert into temporally aligned training
examples. The recorder itself is ML-agnostic, and the built-in player can
replay, inspect and edit any recording.

https://github.com/user-attachments/assets/f6f66744-f462-42c9-a858-bea349844b9b

## full ai disclosure
**made entirely with AI**

now I had the idea, and I wanted to see if it would really work, so I asked chatgpt to think of an implementation, make a prompt, then plugged it into DeepSeek V4 Flash Free on opencode.

Then I "guided" the idea using constructive criticism. I encourage someone with real capabilities to re-build this app, fork the repo, etc. I just want this app out there. I had the idea, and now you can see it too.

All media and testing, ideas, screencaptures, etc were done by me; only the code done by ai.

## Status

Milestone 1: reliably record screen + input into a portable raw
demonstration. Milestone 2: playback and editing — load any recording,
watch the video with live key/mouse state, cut/copy/paste/trim/delete
timeline clips with undo/redo, and save as a new recording (the original
is never modified).

## Quickstart

Requires Python 3.12+.

```bash
python -m venv .venv
.\.venv\Scripts\pip install -e .        # Windows
source .venv/bin/pip install -e .       # macOS/Linux

grui                                    # or: python -m app.main
```

FFmpeg is used for encoding. `imageio-ffmpeg` (a dependency) bundles a
binary automatically; alternatively make `ffmpeg` available on `PATH`.

## Usage

1. Launch the application.
2. Select a monitor (or all monitors) and FPS.
3. Click **Start Recording** — an unobtrusive overlay appears.
4. Play / work in any software. Annotations can be added with F9
   (arbitrary labels like `boss_start`, `attack`, ...).
5. Stop with **Stop**, F8, or by closing the window. Never corrupts data —
   shutdown is graceful.

| Hotkey | Action         |
| ------ | -------------- |
| F8     | Stop recording |
| F9     | Add annotation |
| F10    | Pause/Resume   |

## Dataset builder

Convert any recording into temporally aligned observation->action samples:

```bash
grui dataset build <recording_dir> --out ./dataset
```

See `grui dataset build --help` for the observation window, FPS, stride and
prediction-horizon options; `dataset/README.md` documents the output format.

## Training & agents

Train a behavior-cloning policy on one or more datasets and run it:

```bash
pip install -e ".[ml]"                 # torch extra
grui train --dataset ./dataset --out ckpt.pt --epochs 10
grui agent --checkpoint ckpt.pt --dataset ./dataset                 # dry run
grui agent --checkpoint ckpt.pt --dataset ./dataset --inject        # presses keys / moves mouse
```

The policy is a CNN + GRU: observation window -> held keys, held mouse
buttons, and pointer velocity. See `ml/README.md` for details.

## Optional Perception

GRUI can optionally analyze **existing recordings** with a visual-perception
provider: natural-language prompts are grounded in recorded frames (e.g.
"boss", "projectile", "player") and the resulting boxes are stored as a
*derived artifact* next to the raw demonstration.

* LocateAnything is one optional provider behind a generic
  `PerceptionProvider` interface — other models can be added without
  touching the recorder, player or dataset builder.
* Perception is never required: a normal GRUI installation records, plays,
  edits, builds datasets and trains behavior-cloning policies with zero
  perception code loaded. Models are only loaded when you explicitly run an
  analysis.
* Results are written to `<recording>/perception/` (`manifest.json` +
  `results.jsonl`) using the exact frame timestamps from `frames.jsonl`.
  The raw recording, `events.jsonl` and the video are never modified.
* Everything runs locally — no data is uploaded, no telemetry.

Install the optional LocateAnything dependency stack:

```bash
uv pip install -e ".[locate]"       # uv is recommended
# or: pip install -e ".[locate]"
```

`locate-anything` expects `torch` built for your CUDA (install it from the
PyTorch index if you don't have it). It also pins `numpy==2.0.2` and
`Pillow==11.1.0`, which have no Python 3.14 / Windows wheels — this repo
relaxes them via `[tool.uv] override-dependencies`, which uv picks up
automatically. Plain pip cannot override transitive pins: on
Python 3.14/Windows use uv, or install with
`pip install --no-deps locate-anything` plus a manually resolved stack, or
use Python 3.12/3.13 where the pins have wheels.

Analyze a recording (sampling ~2 frames per second):

```bash
grui perception analyze <recording> --provider locate_anything \
    --prompt "boss" --prompt "projectile" --fps 2

grui perception providers        # list providers and availability
```

**Licensing & hardware**: `nvidia/LocateAnything-3B` is a gated Hugging Face
model (~6 GB, `huggingface-cli login`) needing a CUDA GPU with several GB of
VRAM. GRUI never bundles or silently downloads model weights; check the
model's license before commercial use.

The built-in player has an optional overlay: **Perception…** runs an
analysis, and **Show perception detections** draws the boxes on the video.
Perception is an extension — GRUI's core remains general-purpose human
demonstration recording and imitation-learning dataset generation.

## Player & editor

Click **Open Player** in the main window to browse saved recordings:

1. Pick a recording from the list (or **Browse…**). Video plays with a
   live keyboard/mouse state view that stays in exact sync via
   `frames.jsonl`. Keyboard events are drawn on the timeline as yellow
   dots (they follow cuts/pastes/undo).
2. **Drag on the timeline** to select a region (click to seek — the
   selection is drawn with clear edge borders and shows its duration;
   **Select All** / **Deselect**, `Ctrl+A`, `Esc`). **Trim** keeps only
   the selection, **Cut** removes it (also copying it to the clipboard),
   **Copy** copies without removing, **Paste** inserts the clipboard at
   the playhead, **Delete** removes without copying.
3. **Undo / Redo / Reset Edits** revert timeline changes (`Ctrl+Z`,
   `Ctrl+Y`, `Delete`, `Ctrl+X/C/V` shortcuts).
4. **Save Edits as New Recording…** exports a brand-new recording (video
   re-encoded from kept frames, events/markers/frames remapped and
   duplicated through pasted clips) — the source recording is untouched.

Edits are stored in the exported `metadata.json` under `edit_clips` (and
`edit_history`), so reopening the exported recording restores the same
timeline.

## Raw recording format (version 1)

Each session creates its own directory under `recordings/`:

```text
recordings/
└── 2026-08-08_22-51-03_<session-id>/
    ├── metadata.json
    ├── video.mp4
    ├── events.jsonl
    ├── markers.jsonl
    └── frames.jsonl
```

### metadata.json

```json
{
    "version": 1,
    "session_id": "4f3a91c2e6d1",
    "started_at": "2026-08-08T22:51:03+00:00",
    "platform": "windows",
    "screen": { "width": 1920, "height": 1080, "fps": 30, "monitor_index": 0 },
    "input": { "keyboard": true, "mouse": true },
    "duration": 183.42,
    "stats": { "frames_captured": 5503, "frames_dropped": 0, "frames_encoded": 5503, ... },
    "files": { "video": 1234567, ... }
}
```

`version` guarantees future readers can detect and handle old recordings.

### events.jsonl

One JSON object per line. **Every event carries `t` = seconds since
session start on one shared monotonic clock** (`time.perf_counter_ns()`),
so frames, inputs, annotations and lifecycle events are all directly
comparable.

```json
{"t": 4.12031, "device": "keyboard", "event": "down", "code": "KeyW", "char": "w"}
{"t": 5.10222, "device": "mouse", "event": "move", "x": 731, "y": 412, "dx": 12, "dy": -3}
{"t": 5.81222, "device": "mouse", "event": "button_down", "button": "left", "x": 731, "y": 412}
{"t": 6.01000, "device": "mouse", "event": "scroll", "dx": 0, "dy": -1, "x": 731, "y": 412}
{"t": 0.00341, "device": "session", "event": "recording_start"}
{"t": 0.00341, "device": "session", "event": "pause"}
```

* Keyboard: `code` is canonical (`KeyW`, `Key.space`, `Key.vk_<vk>`);
  `char` is included when printable. Multi-label held-key state can be
  reconstructed from down/up pairs.
* Mouse: absolute `x`/`y` plus `dx`/`dy` deltas (omitted on the first move
  after listener start) — supports both absolute and delta policies.
* Lifecycle: `recording_start`, `recording_stop`, `pause`, `resume`,
  `recording_error`.

### markers.jsonl

Human annotations, same clock:

```json
{"t": 82.193, "type": "annotation", "label": "boss_start"}
```

### frames.jsonl

Exact synchronization source of truth: maps each **actually encoded** video
frame to its capture time. If the encoder ever falls behind, frames are
dropped deliberately — the video stays at constant frame rate and
`frames.jsonl` tells you exactly what was captured when:

```json
{"frame_index": 0, "t": 0.0123}
{"frame_index": 1, "t": 0.0457}
```

Dataset builders align events to video through this file, never through
wall-clock guesses.

### Derived artifacts

Perception analysis (see [Optional Perception](#optional-perception)) writes
an extra `perception/` directory (`manifest.json` + `results.jsonl`) next to
the raw files. It is optional, derived and never required to read or edit a
recording.

## Architecture

```text
PySide6 UI (main window + overlay)
        |
        v
RecordingSession (one SessionClock, state machine, config)
        |
        +-- ScreenRecorder (thread) --+  frame queue  +-- FFmpegEncoder (thread)
        |                            |              |          |
        +-- KeyboardRecorder (thread)-+-- EventWriter---> events.jsonl
        +-- MouseRecorder (thread)   |   (thread)   |          |
        +-- markers (annotations)    |               +--> frames.jsonl
                                     +-------------------> video.mp4
```

* **One clock**: a `SessionClock` wraps `time.perf_counter_ns()`; `t=0` is
  session start. No wall-clock anywhere in the sync path.
* **Queues everywhere**: capture/input threads only enqueue; dedicated
  writer and encoder threads drain to disk. A slow encoder or disk can
  never block input capture.
* **Bounded frame queue**: when the encoder is behind, frames are dropped
  (counted in `metadata.stats.frames_dropped`) instead of stalling the
  pipeline.

## Privacy

This application records global keyboard/mouse input and screen contents —
by design, since it must observe arbitrary software. It therefore:

* only records while recording is active, with an always-visible overlay;
* keeps everything local — nothing is transmitted anywhere;
* supports pause/stop at any time (F8/F10);
* logs nothing sensitive.

Never use it to capture credentials or other people's data without consent.

## Development

```bash
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

Tests cover the clock, JSONL writers, recording directory/metadata,
session lifecycle (with fake capture components), key/button serialization
and real FFmpeg encoding of synthetic frames — plus the timeline editor
(cut/copy/paste/trim/delete, undo/redo, event remapping through pasted
clips), the export pipeline (real re-encode + remap), the playback
state timeline, and an offscreen player-window smoke test. Hardware-
dependent pieces (screen grab, pynput listeners) are injectable and
excluded from tests.

## Roadmap

- [x] Raw recorder (screen, keyboard, mouse, markers, versioned format)
- [x] Player + editor (live key view, timeline, cut/copy/paste/trim, undo/redo, save as new recording)
- [ ] Window/region capture; overlay-region exclusion
- [ ] Excluded-application list (privacy)
- [x] Dataset builder: `grui dataset build <recording>` (observation->action samples, CLI)
- [x] PyTorch dataset/agent: `grui train` (CNN+GRU behavior cloning) and `grui agent` (dry run or pynput injection)
- [x] Optional perception: `grui perception analyze <recording>` (generic provider interface + LocateAnything, derived results, player overlay)
