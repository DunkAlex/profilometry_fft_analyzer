"""
app_ui.py
---------
Small tkinter helpers shared by app.py and its tabs (batch_tab, dashboard_tab,
viewer_tab). Kept separate from app.py itself so a tab module can import
these without importing app.py (which imports the tab modules).
"""
from __future__ import annotations
from typing import Callable, Optional


def run_tab_and_wait(notebook, tab_title: str,
                     build_fn: Callable[["object", Callable], None]):
    """
    Takes a notebook, a tab title, and a builder function; opens a temporary
    tab, waits there until the user finishes whatever it's for, then removes
    the tab and returns whatever the task reported.

    `build_fn(frame, on_done)` fills the fresh frame with its UI and arranges
    for `on_done(result)` to be called exactly once when the task ends —
    Save, Cancel and window-close all count, and what `result` means is up to
    that task.

    This is how every "pop-up" in the app actually works: the caller shows a
    banner on the main screen, this opens a second tab for the one task, and
    the tab closes itself when it's done — no floating windows.
    """
    import tkinter as tk
    from tkinter import ttk

    # Add the temporary tab and bring it to the front.
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=tab_title)
    notebook.select(frame)

    result_box: dict = {}
    done_var = tk.IntVar(master=notebook, value=0)

    def on_done(result=None):
        """Records the task's result and trips the variable being waited on."""
        result_box["result"] = result
        done_var.set(1)

    build_fn(frame, on_done)

    # Blocks here on a nested Tk event loop — the same approach
    # tkinter.simpledialog uses, and safe on the main thread.
    frame.wait_variable(done_var)

    notebook.forget(frame)
    frame.destroy()
    return result_box.get("result")


def format_hms(seconds: float) -> str:
    """Takes a duration in seconds and returns it as 'MM:SS', widening to
    'H:MM:SS' once it reaches an hour. Negatives clamp to zero."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
