"""
manual_fit_gui.py
-----------------
Per-image manual correction window. Used for the batch pipeline's failed
images, and reused as-is by the GUI app's session viewer to re-open and
adjust any already-processed image (see viewer_tab.py) — same window, same
save contract, either way.

Workflow inside the window (all steps optional — enter as much or as little
as you want):

  1. Axis calibration — per axis, either click two points spanning a known
     distance (button arms a 2-click capture) or type a unit-per-pixel value
     directly. Auto-calibration results are prefilled when the automatic
     pipeline got that far (e.g. only the unit OCR failed).
  2. Auto trend fit — once both axes are calibrated (and the image's color
     profile is known), one click re-runs the AUTOMATIC trace extraction and
     full parameter computation with your calibration (most batch failures
     are OCR-only, so this usually succeeds). Accept the proposal, keep
     working manually to override it, or just continue if it fails.
  3. Manual features — feature type dropdown; step heights and dishing by
     two clicks on the image (drawn live as measurement bars) or typed
     directly; repeatable, with per-entry delete.
  4. Roughness — Ra/Rq/Rz text entries.

Returns (action, data): action is 'saved' | 'skipped' | 'quit'; data is the
dict consumed by manual_merge.merge_manual_row (plus pixel-space markers and
trace for the final overlay), or None when not saved.

All typed measurement values are interpreted in the batch OUTPUT_UNIT;
clicked measurements use the axis calibration and are converted to it.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

import feature_analysis as fa
from plot_calibration.zoom_canvas import ZoomPanImageFrame

UNIT_CHOICES = ["um", "nm", "mm", "pm"]

# overlay colors (tk color strings)
C_POINT = "orange"
C_HEIGHT = "#00a9f7"
C_DISHING = "red"
C_WIDTH = "magenta"
C_TRACE = "#00b7ff"
C_SMOOTH = "#ff4500"


def launch_manual_fit(image_rgb: np.ndarray,
                      image_name: str,
                      prefill: Optional[dict],
                      output_unit: str) -> tuple[str, Optional[dict]]:
    """Standalone blocking manual-fit window: creates its own Tk root and
    blocks until the user saves/skips/quits. See module docstring for the
    contract. This is the CLI path (batch_analyze.py) — for embedding
    inside an existing app window, use launch_manual_fit_embedded instead."""
    try:
        import tkinter as tk  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        print("manual fit unavailable: tkinter/Pillow missing")
        return "quit", None

    root = tk.Tk()
    result_box: dict = {}

    def on_done(result):
        """Stashes the result and closes the window, ending mainloop."""
        result_box["result"] = result
        root.destroy()

    _ManualFitApp(root, image_rgb, image_name, prefill or {}, output_unit,
                 on_done=on_done)
    root.mainloop()
    return result_box.get("result", ("skipped", None))


def launch_manual_fit_embedded(parent, image_rgb: np.ndarray, image_name: str,
                               prefill: Optional[dict], output_unit: str,
                               on_done) -> "_ManualFitApp":
    """
    Build the manual-fit flow into an existing container (a ttk.Frame —
    typically a Notebook tab — or a Toplevel) instead of a dedicated Tk
    root. Non-blocking: builds the UI and returns immediately; `on_done`
    (action, data) is called exactly once — see the module docstring for
    the (action, data) contract. The caller owns `parent`'s lifecycle
    (removing/destroying the tab) and is never destroyed by this function.
    """
    return _ManualFitApp(parent, image_rgb, image_name, prefill or {},
                         output_unit, on_done=on_done)


class _ManualFitApp:
    """Builds into any container widget (Tk root, Toplevel, or Frame/
    Notebook tab) passed as `parent` and reports completion via
    `on_done(action, data)` instead of owning a mainloop itself — see
    launch_manual_fit (standalone) and launch_manual_fit_embedded (hosted
    in another window) for the two ways this gets driven."""

    def __init__(self, parent, image_rgb, image_name, prefill, output_unit,
                on_done=None):
        """Sets up one manual-fit session over `image_rgb`. `prefill` carries
        whatever the automatic pipeline managed for this image (calibration,
        plot region, colors), which decides how much of the form starts
        filled in. Nothing is measured yet — that happens as the user clicks
        or types — and `on_done(action, data)` fires once at the end."""
        import tkinter as tk

        self.tk = tk
        self.container = parent
        self.on_done = on_done
        self.image_rgb = image_rgb
        self.image_name = image_name
        self.prefill = prefill
        self.output_unit = output_unit

        # calibration state: per axis None or
        # {'unit_per_px', 'unit', 'source': 'auto'|'user', 'conf'}
        self.cal = {"x": None, "y": None}
        for axis in ("x", "y"):
            upp = prefill.get(f"{axis}_unit_per_px")
            if upp:
                self.cal[axis] = {
                    "unit_per_px": float(upp),
                    "unit": prefill.get(f"{axis}_unit"),
                    "source": "auto",
                    "conf": prefill.get(f"{axis}_conf"),
                }

        # measurement state (values in OUTPUT_UNIT; markers image-coords or
        # None for typed entries) — lists kept index-aligned
        self.heights: list[float] = []
        self.height_markers: list[Optional[dict]] = []
        self.widths: list[float] = []
        self.width_markers: list[Optional[dict]] = []
        self.dishings: list[float] = []
        self.dishing_markers: list[Optional[dict]] = []
        self.calib_points: list[tuple[float, float]] = []

        # Which of height/width/dishing the user (or an accepted auto fit)
        # has actually touched this session — lets _on_save distinguish "the
        # user deliberately emptied this group" (report it as empty) from
        # "the user never went near this group" (leave the existing row
        # value alone). See merge_manual_row's `edited` handling.
        self.touched_groups: set[str] = set()

        # click-capture state
        self.mode: Optional[str] = None      # 'calx'|'caly'|'height'|'width'|'dishing'
        self.pending_pt: Optional[tuple[float, float]] = None

        # auto re-fit state
        self.auto: Optional[dict] = None
        self.auto_accepted = False
        self.auto_tconf: Optional[float] = None
        self.lambda_c_um: Optional[float] = None
        self.lambda_c_capped = False
        self.accepted_rough: dict[str, str] = {}
        self.manual_touched = False

        self._result: tuple[str, Optional[dict]] = ("skipped", None)
        self._closed = False
        self._build_ui()

    # ================= UI construction =================

    def _build_ui(self):
        """Builds the whole form: the image canvas on the left and, down the
        right, the four numbered steps (calibrate axes, run the auto fit,
        enter features, enter roughness) plus the Save/Skip/Quit buttons."""
        tk = self.tk
        container = self.container

        if hasattr(container, "title"):
            container.title(f"Manual fit — {self.image_name}")
        # keyboard shortcuts must reach this flow regardless of which child
        # widget has focus — bind at the toplevel that hosts `container`
        # (itself, when container already is one, e.g. Tk()/Toplevel()).
        self._bind_target = container.winfo_toplevel()

        self.view = ZoomPanImageFrame(container, self.image_rgb,
                                      max_w=820, max_h=620,
                                      on_after_redraw=self._redraw_overlay)
        self.view.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        panel = tk.Frame(container)
        panel.grid(row=0, column=1, sticky="ns", padx=6, pady=4)

        # --- status ---
        self.status_var = tk.StringVar()
        err = self.prefill.get("error")
        self.status_var.set(f"Auto analysis failed: {err}" if err
                            else "Manual entry — everything is optional.")
        tk.Label(panel, textvariable=self.status_var, wraplength=330,
                 justify="left", fg="#8b0000").pack(anchor="w", pady=(0, 4))

        # --- 1. calibration ---
        calf = tk.LabelFrame(panel, text="1. Axis calibration")
        calf.pack(fill=tk.X, pady=3)
        self.known_vars = {}
        self.unit_vars = {}
        self.direct_vars = {}
        self.cal_status_vars = {}
        for axis, default_unit in (("x", "um"), ("y", "nm")):
            row = tk.Frame(calf); row.pack(fill=tk.X, padx=4, pady=2)
            tk.Label(row, text=f"{axis.upper()}:", width=2).pack(side=tk.LEFT)
            tk.Button(row, text="Pick 2 pts", width=9,
                      command=lambda a=axis: self._arm(f"cal{a}")
                      ).pack(side=tk.LEFT, padx=2)
            kv = tk.StringVar(value="10.0"); self.known_vars[axis] = kv
            tk.Entry(row, textvariable=kv, width=7).pack(side=tk.LEFT, padx=2)
            uv = tk.StringVar(value=(self.cal[axis] or {}).get("unit")
                              or default_unit)
            self.unit_vars[axis] = uv
            tk.OptionMenu(row, uv, *UNIT_CHOICES).pack(side=tk.LEFT, padx=2)

            row2 = tk.Frame(calf); row2.pack(fill=tk.X, padx=4)
            tk.Label(row2, text="   or unit/px:").pack(side=tk.LEFT)
            dv = tk.StringVar(); self.direct_vars[axis] = dv
            tk.Entry(row2, textvariable=dv, width=9).pack(side=tk.LEFT, padx=2)
            tk.Button(row2, text="Set", width=4,
                      command=lambda a=axis: self._set_direct(a)
                      ).pack(side=tk.LEFT, padx=2)
            sv = tk.StringVar(); self.cal_status_vars[axis] = sv
            tk.Label(row2, textvariable=sv, fg="#006400").pack(side=tk.LEFT, padx=6)

        # --- 2. auto trend fit ---
        autof = tk.LabelFrame(panel, text="2. Auto trend fit (uses your calibration)")
        autof.pack(fill=tk.X, pady=3)
        arow = tk.Frame(autof); arow.pack(fill=tk.X, padx=4, pady=2)
        self.auto_btn = tk.Button(arow, text="Run auto fit", width=12,
                                  command=self._run_auto)
        self.auto_btn.pack(side=tk.LEFT, padx=2)
        self.accept_btn = tk.Button(arow, text="Accept results", width=12,
                                    state=tk.DISABLED, command=self._accept_auto)
        self.accept_btn.pack(side=tk.LEFT, padx=2)
        self.auto_status_var = tk.StringVar(value="(needs X and Y calibration)")
        tk.Label(autof, textvariable=self.auto_status_var, wraplength=330,
                 justify="left").pack(anchor="w", padx=6, pady=(0, 3))

        # --- 3. manual features ---
        featf = tk.LabelFrame(panel, text="3. Manual features")
        featf.pack(fill=tk.X, pady=3)
        trow = tk.Frame(featf); trow.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(trow, text="Type:").pack(side=tk.LEFT)
        self.ftype_var = tk.StringVar(value="")
        tk.OptionMenu(trow, self.ftype_var, "", "square", "sine", "none"
                      ).pack(side=tk.LEFT, padx=4)

        self.height_widgets = self._build_measure_block(
            featf, "Height", self.heights, self.height_markers, "height")
        self.width_widgets = self._build_measure_block(
            featf, "Width", self.widths, self.width_markers, "width")
        self.dishing_widgets = self._build_measure_block(
            featf, "Dishing", self.dishings, self.dishing_markers, "dishing")

        # --- 4. roughness ---
        rf = tk.LabelFrame(panel, text=f"4. Roughness ({self.output_unit})")
        rf.pack(fill=tk.X, pady=3)
        self.rough_vars = {}
        rrow = tk.Frame(rf); rrow.pack(fill=tk.X, padx=4, pady=2)
        for key in ("Ra", "Rq", "Rz"):
            tk.Label(rrow, text=key + ":").pack(side=tk.LEFT, padx=(4, 0))
            v = tk.StringVar(); self.rough_vars[key] = v
            tk.Entry(rrow, textvariable=v, width=8).pack(side=tk.LEFT, padx=2)

        # --- action buttons ---
        brow = tk.Frame(panel); brow.pack(fill=tk.X, pady=8)
        tk.Button(brow, text="Save & next", width=11,
                  command=self._on_save).pack(side=tk.LEFT, padx=3)
        tk.Button(brow, text="Skip image", width=10,
                  command=self._on_skip).pack(side=tk.LEFT, padx=3)
        tk.Button(brow, text="Quit review", width=10,
                  command=self._on_quit).pack(side=tk.LEFT, padx=3)

        # --- bindings ---
        self.view.canvas.bind("<Button-1>", self._on_click)
        self.view.bind_keyboard(self._bind_target)
        self._bind_target.bind("<Escape>", self._guard(lambda e: self._cancel_capture()))
        if hasattr(container, "protocol"):
            container.protocol("WM_DELETE_WINDOW", self._on_skip)

        self.view.first_draw()
        self._refresh_cal_status()
        self._refresh_buttons()

        # Viewer-initiated re-edit of an already-analyzed image: seed the
        # form with what the batch actually measured instead of opening
        # empty (an empty form makes "remove a feature" impossible and
        # "add a feature" silently overwrite the real count). `after` lets
        # this tab paint before the fit runs. Guarded on auto_btn's state
        # so a failed image with no usable plot region behaves exactly as
        # before (falls through to plain manual entry).
        if (self.prefill.get("autofit_on_open")
                and str(self.auto_btn["state"]) != tk.DISABLED):
            self._bind_target.after(50, self._guard(self._autofit_on_open))

    def _autofit_on_open(self, event=None):
        """Auto-populate the form on open by running the same auto-fit +
        accept path a user would trigger by hand (buttons 'Run auto fit' /
        'Accept results') — see the call site in _build_ui. Reusing that
        path exactly means the seeded numbers can't drift from what the
        batch/auto pipeline would compute for this image."""
        self.status_var.set("Loading the current fit…")
        self._run_auto()
        if self.auto is not None:
            self._accept_auto()
            self.status_var.set(
                "Loaded the current measurements — add/remove features below, "
                "then Save.")
        else:
            self.status_var.set(
                "Could not reload the current fit automatically — "
                + self.auto_status_var.get())

    def _guard(self, fn):
        """Wrap a root-level key binding so it's a no-op once this flow has
        finished — root-level bindings outlive this instance's widgets when
        `container` is a shared app window (a Notebook tab) rather than a
        dedicated Tk root/Toplevel that gets destroyed with them."""
        def wrapped(event=None):
            """Swallows the event once this flow has finished."""
            if self._closed:
                return
            return fn(event)
        return wrapped

    def _build_measure_block(self, parent, label, values, markers, mode):
        """One measurement group: click button, typed entry + Add, listbox
        with Delete + Clear all. Returns dict of widgets that need later
        access.

        Explicit pady above/below each block (f/f2) keeps three of these
        stacked in the same LabelFrame from crowding into each other — with
        two buttons now stacked in btncol (Del + Clear all, taller than the
        single "Del" this block used to have) there's more content per
        block than before, so the gap between one block's listbox row and
        the next block's label row needs to be explicit rather than
        incidental."""
        tk = self.tk
        f = tk.Frame(parent); f.pack(fill=tk.X, padx=4, pady=(8, 2))
        tk.Label(f, text=label + ":", width=7, anchor="w").pack(side=tk.LEFT)
        btn = tk.Button(f, text="Click 2 pts", width=9,
                        command=lambda: self._arm(mode))
        btn.pack(side=tk.LEFT, padx=2)
        ev = tk.StringVar()
        tk.Entry(f, textvariable=ev, width=7).pack(side=tk.LEFT, padx=2)
        tk.Button(f, text="Add", width=4,
                  command=lambda: self._add_typed(mode, ev)
                  ).pack(side=tk.LEFT, padx=2)

        f2 = tk.Frame(parent); f2.pack(fill=tk.X, padx=14, pady=(0, 6))
        lb = tk.Listbox(f2, height=3, width=24)
        lb.pack(side=tk.LEFT, fill=tk.Y)
        btncol = tk.Frame(f2); btncol.pack(side=tk.LEFT, padx=(6, 0), anchor="n")
        tk.Button(btncol, text="Del", width=8,
                  command=lambda: self._delete_selected(mode)
                  ).pack(side=tk.TOP, fill=tk.X)
        tk.Button(btncol, text="Clear all", width=8,
                  command=lambda: self._clear_all(mode)
                  ).pack(side=tk.TOP, fill=tk.X, pady=(3, 0))
        return {"button": btn, "entry": ev, "listbox": lb,
                "values": values, "markers": markers}

    # ================= calibration =================

    def _set_direct(self, axis):
        """Takes the typed unit-per-pixel value for one axis and records it as
        that axis's calibration. Rejects anything non-positive with a message
        rather than storing a scale that would invert the image."""
        try:
            upp = float(self.direct_vars[axis].get())
            if upp <= 0:
                raise ValueError
        except ValueError:
            self.status_var.set(f"{axis.upper()} unit/px must be a positive number.")
            return
        self.cal[axis] = {"unit_per_px": upp,
                          "unit": self.unit_vars[axis].get(),
                          "source": "user", "conf": None}
        self._refresh_cal_status()
        self._refresh_buttons()

    def _finish_axis_pick(self, axis, p1, p2):
        """Turns two clicked points into an axis calibration: measures their
        pixel separation along that axis and divides the known distance by it
        to get units per pixel. Points too close together are rejected, since
        a tiny span makes the scale wildly sensitive to a one-pixel slip."""
        span = abs(p2[0] - p1[0]) if axis == "x" else abs(p2[1] - p1[1])
        if span < 2:
            self.status_var.set("Points too close along the axis — pick again.")
            return
        try:
            known = float(self.known_vars[axis].get())
            if known <= 0:
                raise ValueError
        except ValueError:
            self.status_var.set("Known distance must be a positive number.")
            return
        self.cal[axis] = {"unit_per_px": known / span,
                          "unit": self.unit_vars[axis].get(),
                          "source": "user", "conf": None}
        self.calib_points.extend([p1, p2])
        self.status_var.set(f"{axis.upper()} calibrated: "
                            f"{known / span:.5g} {self.unit_vars[axis].get()}/px")
        self._refresh_cal_status()
        self._refresh_buttons()

    def _refresh_cal_status(self):
        for axis in ("x", "y"):
            c = self.cal[axis]
            if c is None:
                self.cal_status_vars[axis].set("not set")
            else:
                unit = c.get("unit") or "?"
                tag = "auto" if c["source"] == "auto" else "user"
                self.cal_status_vars[axis].set(
                    f"{c['unit_per_px']:.5g} {unit}/px ({tag})")
                # keep the dropdown in sync with a prefit unit
                if c.get("unit"):
                    self.unit_vars[axis].set(c["unit"])

    def _refresh_buttons(self):
        """Enables or disables the click-to-measure and auto-fit buttons based
        on what's been calibrated so far. Height and dishing are vertical
        spans so they need the Y axis; width is horizontal and needs X; the
        auto fit needs both plus a known plot region."""
        tk = self.tk
        # Height/dishing are vertical spans (need Y calibration); width is a
        # horizontal span (needs X calibration).
        can_measure_y = self.cal["y"] is not None
        for block in (self.height_widgets, self.dishing_widgets):
            block["button"].configure(
                state=(tk.NORMAL if can_measure_y else tk.DISABLED))
        self.width_widgets["button"].configure(
            state=(tk.NORMAL if self.cal["x"] is not None else tk.DISABLED))
        can_auto = (self.cal["x"] is not None and self.cal["y"] is not None
                    and self.prefill.get("plot_rgb") is not None
                    and self.prefill.get("plot_region") is not None)
        self.auto_btn.configure(state=(tk.NORMAL if can_auto else tk.DISABLED))
        if not can_auto and self.auto is None:
            reason = ("(needs X and Y calibration)"
                      if self.prefill.get("plot_rgb") is not None
                      else "(unavailable: no color profile for this image)")
            self.auto_status_var.set(reason)

    # ================= click capture =================

    def _arm(self, mode):
        """Arms the canvas for a two-click capture and tells the user which
        two points to click. Esc cancels."""
        prompts = {
            "calx": "Click two points spanning a known X distance.",
            "caly": "Click two points spanning a known Y distance.",
            "height": "Click the feature TOP, then the BASE.",
            "width": "Click the feature's LEFT edge, then its RIGHT edge.",
            "dishing": "Click the plateau SHOULDER, then the interior MINIMUM.",
        }
        self.mode = mode
        self.pending_pt = None
        self.status_var.set(prompts[mode] + "  (Esc cancels)")

    def _cancel_capture(self):
        """Aborts an armed capture and drops any first point already clicked."""
        self.mode = None
        self.pending_pt = None
        self.status_var.set("Capture cancelled.")
        self.view.redraw()

    def _on_click(self, event):
        """Handles a click on the image while a capture is armed. The first
        click is held; the second completes the pair and routes it to either
        the axis-calibration or the measurement handler. Clicks outside the
        image are ignored."""
        if self.mode is None:
            return
        ix, iy = self.view.canvas_to_image(event.x, event.y)
        if not (0 <= ix < self.view.orig_w and 0 <= iy < self.view.orig_h):
            return
        if self.pending_pt is None:
            self.pending_pt = (ix, iy)
            self.view.redraw()
            return

        p1, p2 = self.pending_pt, (ix, iy)
        mode, self.mode, self.pending_pt = self.mode, None, None

        if mode in ("calx", "caly"):
            self._finish_axis_pick(mode[-1], p1, p2)
        else:
            self._finish_measure_pick(mode, p1, p2)
        self.view.redraw()

    def _finish_measure_pick(self, mode, p1, p2):
        """Converts two clicked points into a measurement in the output unit
        and records it with a marker so it can be drawn. Width is a
        horizontal span read off the X calibration; height and dishing are
        vertical spans read off Y. A pair with no separation is rejected."""
        if mode == "width":
            # Width is a HORIZONTAL span read off the X calibration, and its
            # marker is a horizontal bar (see _draw_width_marker) — unlike
            # height/dishing, which are vertical spans off the Y calibration.
            c = self.cal["x"]
            if c is None:
                return
            px_span = abs(p2[0] - p1[0])
            if px_span < 1:
                self.status_var.set("Points have no horizontal separation — pick again.")
                return
            value = fa.convert_length(px_span * c["unit_per_px"], c["unit"],
                                      self.output_unit)
            row = (p1[1] + p2[1]) / 2.0
            marker = {"row": row,
                      "col_left": min(p1[0], p2[0]),
                      "col_right": max(p1[0], p2[0]),
                      "label": f"w={value:.4g}{self.output_unit}"}
            self._append_measurement(mode, value, marker)
            self.status_var.set(f"width = {value:.5g} {self.output_unit}")
            return

        c = self.cal["y"]
        if c is None:
            return
        px_span = abs(p2[1] - p1[1])
        if px_span < 1:
            self.status_var.set("Points have no vertical separation — pick again.")
            return
        value_axis_unit = px_span * c["unit_per_px"]
        value = fa.convert_length(value_axis_unit, c["unit"], self.output_unit)
        col = (p1[0] + p2[0]) / 2.0
        marker = {"col": int(round(col)),
                  "row_top": min(p1[1], p2[1]),
                  "row_base": max(p1[1], p2[1]),
                  "label": f"{'h' if mode == 'height' else 'd'}="
                           f"{value:.4g}{self.output_unit}"}
        self._append_measurement(mode, value, marker)
        self.status_var.set(f"{mode} = {value:.5g} {self.output_unit}")

    # ================= measurement bookkeeping =================

    def _block(self, mode):
        """Returns the widget bundle (button, entry, listbox, values, markers)
        for one measurement group."""
        return {"height": self.height_widgets,
                "width": self.width_widgets,
                "dishing": self.dishing_widgets}[mode]

    def _append_measurement(self, mode, value, marker):
        """Adds one measurement to its group: stores the value and marker,
        shows it in the listbox, and notes that the group has been touched so
        the save path knows this state is deliberate."""
        block = self._block(mode)
        block["values"].append(float(value))
        block["markers"].append(marker)
        block["listbox"].insert(self.tk.END,
                                f"{value:.5g} {self.output_unit}"
                                + ("" if marker else " (typed)"))
        self.manual_touched = True
        self.touched_groups.add(mode)

    def _add_typed(self, mode, entry_var):
        """Adds a typed-in measurement to its group and clears the entry box.
        Typed values have no marker, so nothing is drawn on the image for
        them. Non-numeric input is refused with a message."""
        try:
            value = float(entry_var.get())
        except ValueError:
            self.status_var.set("Enter a numeric value before Add.")
            return
        entry_var.set("")
        self._append_measurement(mode, value, None)

    def _delete_selected(self, mode):
        """Removes the selected entry from one group — its value, its marker,
        and its listbox row — and redraws. Does nothing if no row is
        selected."""
        block = self._block(mode)
        sel = block["listbox"].curselection()
        if not sel:
            return
        i = int(sel[0])
        block["listbox"].delete(i)
        del block["values"][i]
        del block["markers"][i]
        self.manual_touched = True
        self.touched_groups.add(mode)
        self.view.redraw()

    def _clear_all(self, mode):
        """Remove every accepted measurement for one parameter (its listbox,
        values, and markers) — a one-click reset per block, vs. Del's
        selected-row removal."""
        block = self._block(mode)
        if not block["values"] and not block["markers"]:
            return
        block["listbox"].delete(0, self.tk.END)
        block["values"].clear()
        block["markers"].clear()
        self.manual_touched = True
        self.touched_groups.add(mode)
        self.view.redraw()

    # ================= auto re-fit =================

    def _run_auto(self):
        """Re-runs the full automatic pipeline on this image using the current
        calibration, and draws the proposed trace and markers over the image
        without committing anything. The user then either accepts the result
        or keeps working by hand. Failures report why in the status line
        rather than raising."""
        pre = self.prefill
        try:
            from profilometry_fft_analyzer import extract_trend_with_info
            import measurement_overlay as mo
        except Exception as e:
            self.auto_status_var.set(f"auto fit unavailable: {e}")
            return

        inset = int(pre.get("crop_inset", 2))
        x0, y0, x1, y1 = pre["plot_region"]
        ix0, iy0 = x0 + inset, y0 + inset
        ix1, iy1 = x1 - inset, y1 - inset
        if ix1 - ix0 < 20 or iy1 - iy0 < 20:
            self.auto_status_var.set("auto fit failed: plot region degenerate")
            return
        crop = self.image_rgb[iy0:iy1, ix0:ix1]
        crop_h = crop.shape[0]

        x_cols, y_flip, info = extract_trend_with_info(
            crop, tuple(pre["plot_rgb"]),
            buffer=pre.get("buffer", 120), max_gap=pre.get("max_gap", 5),
            interp_gaps=True)
        if len(x_cols) < 16:
            self.auto_status_var.set(
                "auto fit failed: too few color-matched columns "
                "(is the plot color right for this profile?)")
            return

        cx, cy = self.cal["x"], self.cal["y"]
        x_upp_um = cx["unit_per_px"] * fa.unit_factor(cx["unit"], "um")
        y_upp_um = cy["unit_per_px"] * fa.unit_factor(cy["unit"], "um")

        y_flip = np.asarray(y_flip, dtype=float)   # increases upward
        y_um = y_flip * y_upp_um
        dx_um = x_upp_um                            # columns are 1 px apart
        rows_crop = (crop_h - 1) - y_flip

        res = fa.run_measurement_pipeline(y_um, dx_um, rows_crop=rows_crop,
                                          ext_info=info)

        cols_full = np.asarray(x_cols, dtype=float) + ix0
        rows_full = rows_crop + iy0
        baseline_um = y_um - res["y_lev"]

        def row_of_um(v_lev, i):
            """Converts a leveled height at sample `i` back to an image row,
            by re-adding the removed baseline and undoing the Y scale."""
            v_img = v_lev + baseline_um[i]
            return iy0 + (crop_h - 1) - (v_img / y_upp_um)

        def fmt(v_um):
            """Formats a micrometre value in the output unit for a label."""
            v = fa.convert_length(v_um, "um", self.output_unit)
            return f"{v:.4g}{self.output_unit}"

        h_m, d_m, w_m = mo.build_measurement_markers(
            res["feats"], cols_full, row_of_um, fmt, dx_um)

        smooth_rows = None
        if res["feats"].y_smooth is not None:
            v_img = res["feats"].y_smooth + baseline_um
            smooth_rows = iy0 + (crop_h - 1) - (v_img / y_upp_um)

        self.auto = {"res": res, "cols_full": cols_full,
                     "rows_full": rows_full, "smooth_rows": smooth_rows,
                     "h_markers": h_m, "d_markers": d_m, "w_markers": w_m}

        feats, rough, lc = res["feats"], res["rough"], res["lc"]
        cv = lambda v: fa.convert_length(v, "um", self.output_unit)
        self.auto_status_var.set(
            f"type={feats.feature_type}  n={feats.n_features}  "
            f"h_avg={cv(feats.height_avg):.4g}  "
            f"Ra={cv(rough['Ra']):.4g}  Rq={cv(rough['Rq']):.4g}  "
            f"Rz={cv(rough['Rz']):.4g}  ({self.output_unit})  "
            f"lambda_c={cv(lc['lambda_c_um']):.4g}"
            f"{' capped' if lc['capped'] else ''}  "
            f"trend_conf={res['trend_conf']:.2f}\n"
            "Review the drawn fit, then Accept — or measure manually to "
            "override.")
        self.accept_btn.configure(state=self.tk.NORMAL)
        self.view.redraw()

    def _accept_auto(self):
        """Commits the auto-fit proposal into the editable lists: replaces
        height, width and dishing wholesale, fills in feature type, roughness
        and cutoff, and marks all three groups touched. From here the user can
        delete or add entries and the edits stick."""
        if self.auto is None:
            return
        res = self.auto["res"]
        feats, rough = res["feats"], res["rough"]
        cv = lambda v: fa.convert_length(v, "um", self.output_unit)

        for block in (self.height_widgets, self.width_widgets,
                      self.dishing_widgets):
            block["listbox"].delete(0, self.tk.END)
            block["values"].clear()
            block["markers"].clear()

        self.ftype_var.set(feats.feature_type)
        for f, marker in zip(feats.features, self.auto["h_markers"]):
            self.heights.append(cv(f.height))
            self.height_markers.append(marker)
            self.height_widgets["listbox"].insert(
                self.tk.END, f"{cv(f.height):.5g} {self.output_unit} (auto)")
        width_feats = [f for f in feats.features if np.isfinite(f.width)]
        for f, marker in zip(width_feats, self.auto["w_markers"]):
            self.widths.append(cv(f.width))
            self.width_markers.append(marker)
            self.width_widgets["listbox"].insert(
                self.tk.END, f"{cv(f.width):.5g} {self.output_unit} (auto)")
        dish_feats = [f for f in feats.features if np.isfinite(f.dishing)]
        for f, marker in zip(dish_feats, self.auto["d_markers"]):
            self.dishings.append(cv(f.dishing))
            self.dishing_markers.append(marker)
            self.dishing_widgets["listbox"].insert(
                self.tk.END, f"{cv(f.dishing):.5g} {self.output_unit} (auto)")

        self.accepted_rough = {}
        for key in ("Ra", "Rq", "Rz"):
            text = f"{cv(rough[key]):.6g}"
            self.rough_vars[key].set(text)
            self.accepted_rough[key] = text

        self.lambda_c_um = res["lc"]["lambda_c_um"]
        self.lambda_c_capped = bool(res["lc"]["capped"])
        self.auto_tconf = res["trend_conf"]
        self.auto_accepted = True
        self.manual_touched = False
        # Accepting the auto fit replaces all three lists wholesale (above),
        # including down to zero entries when the fit found none of a kind
        # (e.g. a sine-type feature has no dishing) — that zero is the real
        # current state, not "untouched", so every group counts as touched
        # from here even if the user edits nothing further before Save.
        self.touched_groups.update({"height", "width", "dishing"})
        self.status_var.set("Auto results accepted — edit freely, then Save.")
        self.view.redraw()

    # ================= overlay drawing =================

    def _redraw_overlay(self):
        """Repaints every annotation on the image: the auto trace, the
        committed measurement markers, and any calibration or pending click
        points. Before the auto fit is accepted its markers are drawn as a
        preview; afterwards they live in the committed lists, so the guard
        stops them being drawn twice."""
        cv = self.view.canvas
        cv.delete("overlay")

        # auto trace (proposal or accepted)
        if self.auto is not None:
            self._draw_polyline(self.auto["cols_full"],
                                self.auto["rows_full"], C_TRACE, 1)
            if self.auto["smooth_rows"] is not None:
                self._draw_polyline(self.auto["cols_full"],
                                    self.auto["smooth_rows"], C_SMOOTH, 2)
            # Before Accept: preview the auto markers. After Accept they've
            # been moved into the committed lists below, so this guard avoids
            # double-drawing (same pattern for height/width/dishing).
            if not self.auto_accepted:
                for m in self.auto["h_markers"]:
                    self._draw_marker(m, C_HEIGHT)
                for m in self.auto["w_markers"]:
                    self._draw_width_marker(m, C_WIDTH)
                for m in self.auto["d_markers"]:
                    self._draw_marker(m, C_DISHING)

        for m in self.height_markers:
            if m:
                self._draw_marker(m, C_HEIGHT)
        for m in self.width_markers:
            if m:
                self._draw_width_marker(m, C_WIDTH)
        for m in self.dishing_markers:
            if m:
                self._draw_marker(m, C_DISHING)
        for (px, py) in self.calib_points:
            self._draw_point(px, py, C_POINT)
        if self.pending_pt is not None:
            self._draw_point(*self.pending_pt, C_POINT)

    def _draw_point(self, ix, iy, color):
        """Draws a small circle at an image coordinate — used for calibration
        points and the first half of a click pair."""
        cx, cy = self.view.image_to_canvas(ix, iy)
        self.view.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                     outline=color, width=2, tags="overlay")

    def _draw_marker(self, m, color):
        """Draws a vertical measurement bar with end caps and its label — the
        live-canvas twin of the height/dishing markers baked into the saved
        overlay."""
        cv = self.view.canvas
        x0, yt = self.view.image_to_canvas(m["col"], m["row_top"])
        _x, yb = self.view.image_to_canvas(m["col"], m["row_base"])
        cv.create_line(x0, yt, x0, yb, fill=color, width=2, tags="overlay")
        cv.create_line(x0 - 4, yt, x0 + 4, yt, fill=color, width=2, tags="overlay")
        cv.create_line(x0 - 4, yb, x0 + 4, yb, fill=color, width=2, tags="overlay")
        if m.get("label"):
            cv.create_text(x0 + 6, (yt + yb) / 1.8, text=m["label"],
                           fill=color, anchor="w", tags="overlay",
                           font=("Helvetica", 9, "bold"))

    def _draw_width_marker(self, m, color):
        """Horizontal counterpart of _draw_marker — a width bar spanning
        [col_left, col_right] at a fixed row, mirroring
        measurement_overlay._draw_horizontal_marker but on the live
        zoom/pan canvas instead of the saved overlay PNG."""
        cv = self.view.canvas
        xl, yr = self.view.image_to_canvas(m["col_left"], m["row"])
        xr, _y = self.view.image_to_canvas(m["col_right"], m["row"])
        cv.create_line(xl, yr, xr, yr, fill=color, width=2, tags="overlay")
        cv.create_line(xl, yr - 4, xl, yr + 4, fill=color, width=2, tags="overlay")
        cv.create_line(xr, yr - 4, xr, yr + 4, fill=color, width=2, tags="overlay")
        if m.get("label"):
            # Anchor the label above the LEFT end (not the center) so it
            # clears the height label, which sits at the feature center.
            cv.create_text(xl, yr - 8, text=m["label"],
                           fill=color, anchor="sw", tags="overlay",
                           font=("Helvetica", 9, "bold"))

    def _draw_polyline(self, cols, rows, color, width):
        """Draws a curve from image-space column/row arrays, thinning the
        points first so a full-width trace doesn't bog the canvas down."""
        pts = []
        step = max(1, len(cols) // 1200)   # decimate for canvas performance
        for i in range(0, len(cols), step):
            x, y = self.view.image_to_canvas(cols[i], rows[i])
            pts.extend((x, y))
        if len(pts) >= 4:
            self.view.canvas.create_line(*pts, fill=color, width=width,
                                         tags="overlay")

    # ================= save / skip / quit =================

    def _parse_rough(self):
        """Returns {'Ra': float|None, ...} or None if an entry is invalid."""
        out = {}
        for key, var in self.rough_vars.items():
            text = var.get().strip()
            if not text:
                out[key] = None
                continue
            try:
                out[key] = float(text)
            except ValueError:
                self.status_var.set(f"{key} is not a number — fix or clear it.")
                return None
        return out

    def _on_save(self):
        """Gathers the whole form into the payload manual_merge expects and
        reports it back as a save. Works out whether anything was actually
        entered by hand, which decides whether the row keeps the automatic
        trend confidence or gets the manual sentinel, and records which
        measurement groups were touched so a deliberately emptied group
        clears the row instead of being ignored. Invalid roughness entries
        stop the save."""
        rough = self._parse_rough()
        if rough is None:
            return

        rough_touched = any(
            (rough[k] is not None
             and f"{rough[k]:.6g}" != self.accepted_rough.get(k))
            for k in rough)

        data = {}
        for axis in ("x", "y"):
            c = self.cal[axis]
            if c is not None:
                data[f"{axis}_cal"] = {
                    "unit_per_px": c["unit_per_px"],
                    "unit": c.get("unit") or self.unit_vars[axis].get(),
                    "source": c["source"],
                    "conf": c.get("conf"),
                }

        data["feature_type"] = self.ftype_var.get() or None
        data["heights"] = list(self.heights)
        data["widths"] = list(self.widths)
        data["dishings"] = list(self.dishings)
        # Tells merge_manual_row which of these three lists reflect a
        # deliberate current state (so an empty one means "really zero,
        # clear the aggregates") vs. a group the user never opened (empty
        # here only because nothing was ever entered — leave the row's
        # existing value alone). See manual_merge.merge_manual_row.
        data["groups_edited"] = {
            "height": "height" in self.touched_groups,
            "width": "width" in self.touched_groups,
            "dishing": "dishing" in self.touched_groups,
        }
        data.update(rough)

        if self.auto_accepted:
            data["lambda_c"] = fa.convert_length(self.lambda_c_um, "um",
                                                 self.output_unit)
            data["lambda_c_capped"] = self.lambda_c_capped
        data["auto_accepted"] = self.auto_accepted

        anything_manual = (self.manual_touched or rough_touched
                           or (not self.auto_accepted
                               and (self.heights or self.widths or self.dishings
                                    or any(v is not None for v in rough.values())
                                    or data["feature_type"])))
        if self.auto_accepted and not anything_manual:
            data["trend_conf"] = self.auto_tconf
        elif anything_manual:
            data["trend_conf"] = -1.0
        else:
            data["trend_conf"] = None

        # committed marker lists already include accepted-auto markers and
        # reflect any deletions the user made afterward. Width is now a
        # first-class manual block like height/dishing, so it uses the same
        # committed-list pattern (no auto-only special case).
        data["markers"] = {
            "height": [m for m in self.height_markers if m],
            "width": [m for m in self.width_markers if m],
            "dishing": [m for m in self.dishing_markers if m],
        }
        if self.auto_accepted:
            data["trace"] = {
                "cols": [float(v) for v in self.auto["cols_full"]],
                "rows": [float(v) for v in self.auto["rows_full"]],
                "smooth_rows": ([float(v) for v in self.auto["smooth_rows"]]
                                if self.auto["smooth_rows"] is not None else None),
            }
        else:
            data["trace"] = None

        self._finish_with(("saved", data))

    def _on_skip(self):
        """Leaves this image untouched and moves on."""
        self._finish_with(("skipped", None))

    def _on_quit(self):
        """Ends the whole review, not just this image."""
        self._finish_with(("quit", None))

    def _finish_with(self, result: tuple[str, Optional[dict]]) -> None:
        """Report completion exactly once. Does NOT destroy `container` —
        the caller owns that (see launch_manual_fit /
        launch_manual_fit_embedded)."""
        if self._closed:
            return
        self._closed = True
        self._result = result
        if self.on_done is not None:
            self.on_done(result)
