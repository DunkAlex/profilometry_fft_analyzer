"""
app.py
------
Application entry point: one window, a Preferences menubar, and three tabs
(Batch, Dashboard, Viewer) sharing a single AppSession.

This is the GUI counterpart of batch_analyze.py — same app_engine.py
underneath, so results are identical either way. What it adds is in-window
batch processing with a progress bar and ETA, a results dashboard, and a
viewer for checking and correcting any processed image. Each tab's own
contract lives in batch_tab.py, dashboard_tab.py and viewer_tab.py.

Run:
    python app.py
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import app_engine as engine
from app_session import AppSession
from batch_tab import BatchTab
from dashboard_tab import DashboardTab
from preferences_gui import install_menubar
from viewer_tab import ViewerTab

WINDOW_TITLE = "Profilometry FFT Analyzer"
WINDOW_SIZE = "1200x820"


def main() -> int:
    """Builds the main window and hands control to Tk's event loop. Creates
    the shared session, wires the menubar and the three tabs to it, and
    returns 0 once the user closes the window."""
    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry(WINDOW_SIZE)
    root.minsize(1000, 700)

    # One session object; every tab reads and mutates this same instance.
    session = AppSession(config=engine.default_config())

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    batch_tab = BatchTab(notebook, session)
    dashboard_tab = DashboardTab(notebook, session)
    viewer_tab = ViewerTab(notebook, session)

    notebook.add(batch_tab, text="Batch")
    notebook.add(dashboard_tab, text="Dashboard")
    notebook.add(viewer_tab, text="Viewer")

    def on_parameters_changed() -> None:
        """Rebuild the session's config after a Preferences edit, so the next
        run picks up the new values. Only the folder-independent measurement
        settings change; the run's paths and already-collected rows stay put."""
        session.config = engine.default_config(session.config.base_dir)

    install_menubar(root, on_changed=on_parameters_changed)

    def open_in_viewer(name: str) -> None:
        """Jump the Viewer to `name` and bring that tab forward — wired to the
        Dashboard's double-click."""
        viewer_tab.show_image(name)
        notebook.select(viewer_tab)

    dashboard_tab.on_open_in_viewer = open_in_viewer

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
