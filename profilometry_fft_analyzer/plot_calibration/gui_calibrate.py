"""
gui_calibrate.py
----------------
Tkinter modal window for manual calibration when no profile matches.

Features:
  - Sequential picking: background FIRST, then plot / axis / gridline.
    After background is picked, subsequent picks exclude background pixels
    from the patch median so thin lines aren't drowned out by surroundings.
  - Scroll-to-zoom, pan, crosshair readout — provided by
    zoom_canvas.ZoomPanImageFrame (shared with the manual-fit GUI).
  - Keyboard shortcuts:
        + / =   zoom in           - / _   zoom out
        R       reset view        F       fit to window
        Arrow keys pan            U       undo last pick
        Enter   finish            Esc     cancel
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .colors import pick_color_at, infer_background, validate_profile_colors
from .profiles import build_profile, compute_histogram
from .zoom_canvas import ZoomPanImageFrame


def launch_calibration_gui(image_rgb: np.ndarray,
                           default_name: str = "profile"
                           ) -> Optional[dict]:
    """
    Standalone modal calibration window: creates its own Tk root and blocks
    until the user finishes or cancels. Returns a completed profile dict
    (from build_profile) or None if the user cancels. This is the CLI path
    (batch_analyze.py) — for embedding inside an existing app window, use
    launch_calibration_gui_embedded instead.
    """
    try:
        import tkinter as tk  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "GUI calibration needs tkinter and Pillow. "
            "Install Pillow with `pip install Pillow`."
        ) from e

    root = tk.Tk()
    result_box: dict = {}

    def on_done(result):
        result_box["result"] = result
        root.destroy()

    _CalibrationApp(root, image_rgb, default_name, on_done=on_done)
    root.mainloop()
    return result_box.get("result")


def launch_calibration_gui_embedded(parent, image_rgb: np.ndarray,
                                    on_done, default_name: str = "profile"
                                    ) -> "_CalibrationApp":
    """
    Build the calibration flow into an existing container (a ttk.Frame —
    typically a Notebook tab — or a Toplevel) instead of a dedicated Tk
    root. Non-blocking: builds the UI and returns immediately; `on_done`
    (result: dict | None) is called exactly once, when the user finishes or
    cancels. The caller owns `parent`'s lifecycle (removing/destroying the
    tab) and is never destroyed by this function.
    """
    return _CalibrationApp(parent, image_rgb, default_name, on_done=on_done)


# ---------------- Application class ----------------

class _CalibrationApp:
    """Encapsulates state for the calibration window. Builds into any
    container widget (Tk root, Toplevel, or Frame/Notebook tab) passed as
    `parent` and reports completion via `on_done(result)` instead of owning
    a mainloop itself — see launch_calibration_gui (standalone) and
    launch_calibration_gui_embedded (hosted in another window) for the two
    ways this gets driven."""

    STEPS = ["background", "plot", "axis", "gridline"]
    PROMPTS = {
        "background": "1/4  Click the background color (empty plot area).",
        "plot":       "2/4  Click the plot/trace color (the data curve).",
        "axis":       "3/4  Click the axis line color.",
        "gridline":   "4/4  Click a gridline.",
    }

    def __init__(self, parent, image_rgb: np.ndarray, default_name: str,
                on_done=None):
        import tkinter as tk

        self.tk = tk
        self.container = parent
        self.on_done = on_done
        self.image_rgb = image_rgb
        self.default_name = default_name
        self.orig_h, self.orig_w = image_rgb.shape[:2]

        # pick state
        self.step_i = 0
        self.picked: dict[str, tuple[int, int, int]] = {}

        # result / lifecycle
        self._result: Optional[dict] = None
        self._closed = False

        self._build_ui()

    # --------- Build ---------

    def _build_ui(self):
        tk = self.tk
        container = self.container

        if hasattr(container, "title"):
            container.title("Plot calibration")
        # keyboard shortcuts must reach this flow regardless of which child
        # widget has focus — bind at the toplevel that hosts `container`
        # (itself, when container already is one, e.g. Tk()/Toplevel()).
        self._bind_target = container.winfo_toplevel()

        # --- shared zoom/pan image view (toolbar + canvas) ---
        self.view = ZoomPanImageFrame(container, self.image_rgb)
        self.view.pack(side=tk.TOP)

        # --- prompt ---
        self.prompt_var = tk.StringVar()
        tk.Label(container, textvariable=self.prompt_var,
                 font=("Helvetica", 12, "bold"), pady=6).pack(side=tk.TOP, fill=tk.X)

        # --- swatch row ---
        sw_frame = tk.Frame(container)
        sw_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.swatches: dict[str, "tk.Label"] = {}
        for name in self.STEPS:
            col = tk.Frame(sw_frame); col.pack(side=tk.LEFT, padx=8)
            tk.Label(col, text=name, width=11).pack(side=tk.TOP)
            sw = tk.Label(col, width=14, height=2, relief=tk.SUNKEN,
                          bg="#dddddd", text="")
            sw.pack(side=tk.TOP)
            self.swatches[name] = sw

        # --- buttons ---
        btns = tk.Frame(container)
        btns.pack(side=tk.TOP, fill=tk.X, pady=6)
        self.undo_btn = tk.Button(btns, text="Undo (U)", command=self._undo)
        self.undo_btn.pack(side=tk.LEFT, padx=4)
        self.auto_bg_btn = tk.Button(btns, text="Auto-detect BG",
                                     command=self._auto_bg)
        self.auto_bg_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Cancel (Esc)", command=self._cancel).pack(side=tk.RIGHT, padx=4)
        tk.Button(btns, text="Finish (Enter)", command=self._finish).pack(side=tk.RIGHT, padx=4)

        # --- bindings ---
        self.view.canvas.bind("<Button-1>", self._on_left_click)
        self.view.bind_keyboard(self._bind_target)
        self._bind_target.bind("u", self._guard(lambda e: self._undo()))
        self._bind_target.bind("U", self._guard(lambda e: self._undo()))
        self._bind_target.bind("<Return>", self._guard(lambda e: self._finish()))
        self._bind_target.bind("<Escape>", self._guard(lambda e: self._cancel()))
        if hasattr(container, "protocol"):
            container.protocol("WM_DELETE_WINDOW", self._cancel)

        # deferred: wait until the canvas has a real size before first draw
        self.view.first_draw()

        self._update_prompt()

    def _guard(self, fn):
        """Wrap a root-level key binding so it's a no-op once this flow has
        finished — root-level bindings outlive this instance's widgets when
        `container` is a shared app window (a Notebook tab) rather than a
        dedicated Tk root/Toplevel that gets destroyed with them."""
        def wrapped(event=None):
            if self._closed:
                return
            return fn(event)
        return wrapped

    # --------- Event handlers ---------

    def _on_left_click(self, event):
        if self.step_i >= len(self.STEPS):
            return
        ix, iy = self.view.canvas_to_image(event.x, event.y)
        ix, iy = int(round(ix)), int(round(iy))
        if not (0 <= ix < self.orig_w and 0 <= iy < self.orig_h):
            return

        name = self.STEPS[self.step_i]

        # After background is picked, exclude it from subsequent medians
        exclude = None
        if name != "background" and "background" in self.picked:
            exclude = self.picked["background"]

        rgb = pick_color_at(self.image_rgb, ix, iy, exclude_rgb=exclude)
        self._set_swatch(name, rgb)
        self.picked[name] = rgb
        self.step_i += 1
        self._update_prompt()

    # --------- Step management ---------

    def _set_swatch(self, name: str, rgb: tuple[int, int, int], suffix: str = ""):
        hexc = "#{:02x}{:02x}{:02x}".format(*rgb)
        self.swatches[name].configure(
            bg=hexc, fg=_contrasting_fg(rgb),
            text=f"{rgb[0]},{rgb[1]},{rgb[2]}{suffix}",
        )

    def _clear_swatch(self, name: str):
        self.swatches[name].configure(bg="#dddddd", fg="black", text="")

    def _update_prompt(self):
        if self.step_i < len(self.STEPS):
            self.prompt_var.set(self.PROMPTS[self.STEPS[self.step_i]])
        else:
            self.prompt_var.set("All colors picked — press Finish (or Enter).")
        # Only enable auto-detect on the background step
        can_auto = (self.step_i == 0)
        self.auto_bg_btn.configure(state=("normal" if can_auto else "disabled"))
        # Undo enabled iff we have something to undo
        self.undo_btn.configure(state=("normal" if self.step_i > 0 else "disabled"))

    def _undo(self):
        if self.step_i == 0:
            return
        self.step_i -= 1
        name = self.STEPS[self.step_i]
        self.picked.pop(name, None)
        self._clear_swatch(name)
        self._update_prompt()

    def _auto_bg(self):
        if self.step_i != 0:
            return
        rgb = infer_background(self.image_rgb)
        self.picked["background"] = rgb
        self._set_swatch("background", rgb, suffix=" (auto)")
        self.step_i += 1
        self._update_prompt()

    def _finish(self):
        if self.step_i < len(self.STEPS):
            from tkinter import messagebox
            missing = [s for s in self.STEPS if s not in self.picked]
            messagebox.showwarning(
                "Incomplete", f"Please pick: {', '.join(missing)}",
                parent=self._bind_target,
            )
            return

        from tkinter import messagebox
        problems = validate_profile_colors(self.picked)
        if problems:
            proceed = messagebox.askyesno(
                "Colors too similar",
                "These picks look too close to tell apart:\n\n"
                + "\n".join(problems)
                + "\n\nThis usually means a click missed a thin line and "
                  "landed on background instead. Try zooming in (scroll "
                  "wheel) and redoing the affected pick(s) with Undo.\n\n"
                  "Save anyway?",
                parent=self._bind_target,
            )
            if not proceed:
                return

        from tkinter import simpledialog
        name = simpledialog.askstring(
            "Profile name", "Save this profile as:",
            initialvalue=self.default_name,
            parent=self._bind_target,
        )
        if not name:
            return
        hist = compute_histogram(self.image_rgb)
        result = build_profile(
            name=name, picked_colors=self.picked, histogram=hist,
        )
        self._finish_with(result)

    def _cancel(self):
        self._finish_with(None)

    def _finish_with(self, result: Optional[dict]) -> None:
        """Report completion exactly once. Does NOT destroy `container` —
        the caller owns that (see launch_calibration_gui /
        launch_calibration_gui_embedded)."""
        if self._closed:
            return
        self._closed = True
        self._result = result
        if self.on_done is not None:
            self.on_done(result)


# ---------------- Utility ----------------

def _contrasting_fg(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luma > 128 else "white"
