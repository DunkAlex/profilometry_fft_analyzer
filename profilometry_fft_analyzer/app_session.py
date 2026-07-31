"""
app_session.py
---------------
The shared state for one run of the GUI app: which config is active, which
images were processed, their result rows, and the CSV they were written to.

app.py creates exactly one AppSession and hands it to all three tabs. They
read and mutate that same object and call notify_results_changed() when
they change something, which is how a correction made in the Viewer shows
up in the Dashboard without either tab knowing about the other or anything
being re-read from disk.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import app_engine as engine


@dataclass
class AppSession:
    config: engine.BatchConfig

    # Identity of the current run: its timestamp, its overlay folder, and
    # the CSV on disk. All None until a batch has actually run.
    run_ts: Optional[str] = None
    run_dir: Optional[str] = None
    csv_path: Optional[str] = None

    # image_paths and rows are index-aligned: rows[i] describes image_paths[i].
    image_paths: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    # Per-image analysis context, keyed by full path. Kept for EVERY attempted
    # image, not just failures, so the Viewer can reopen any of them later.
    ctx_by_path: dict = field(default_factory=dict)

    # The trace-extraction settings this run used. The Viewer needs them to
    # re-run a fit for an image the Batch tab never prefilled.
    extraction_buffer: float = 120.0
    extraction_max_gap: int = 5

    # Callbacks fired by notify_results_changed; each tab appends its own.
    on_results_changed: list = field(default_factory=list)

    def notify_results_changed(self) -> None:
        """Tells every registered listener that the rows changed, so each tab
        can redraw. Iterates over a copy of the list, so a listener that
        registers or removes another one mid-notify can't corrupt the walk."""
        for cb in list(self.on_results_changed):
            cb()

    def path_for_name(self, name: str) -> Optional[str]:
        """Takes an image's bare filename and returns its full path from this
        run, or None if it wasn't part of it."""
        for p in self.image_paths:
            if os.path.basename(p) == name:
                return p
        return None

    def row_index_for_name(self, name: str) -> Optional[int]:
        """Takes an image's bare filename and returns its index into `rows`,
        or None if there's no row for it. Callers use the index rather than
        the row itself so they can replace the row in place."""
        for i, row in enumerate(self.rows):
            if row.get("Name") == name:
                return i
        return None
