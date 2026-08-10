"""Editor: non-destructive timeline editing of raw recordings.

The timeline is a sequence of clips; each clip references a source range of
the original recording. Cut/copy/paste/trim/delete operate on the edited
timeline, and events/markers are remapped through clips on export. The
original recording is never modified.
"""

from editor.timeline import Clip, EditSession, Timeline, remap_events
from editor.export import export_recording

__all__ = ["Clip", "EditSession", "Timeline", "remap_events", "export_recording"]
