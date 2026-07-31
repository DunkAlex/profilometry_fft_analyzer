"""
viewer_tab.py
-------------
The Viewer tab: click through this session's processed images (their review
overlays), see the row's measurements/confidence, and — using the very same
window used for failed-image correction in the batch pipeline — adjust
parameters for ANY image, not just ones that failed automatically. Saving
amends the in-memory row, the run's results CSV on disk, and re-renders that
image's overlay: manual verification/correction for low-certainty fits.

Session-only, by design: this shows session.image_paths, the images
processed since the last batch run in this app session — not an arbitrary
CSV opened from a previous session.
"""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk
from typing import Optional

import app_engine as engine
from app_ui import run_tab_and_wait
from manual_fit_gui import launch_manual_fit_embedded
from manual_merge import merge_manual_row, USER_CONFIDENCE
from plot_calibration.zoom_canvas import ZoomPanImageFrame

# Row fields worth a line in the detail panel, in display order.
DETAIL_FIELDS = [
    ("feature_type", "Feature type"),
    ("n_features", "N features"),
    ("height_avg", "Height (avg)"),
    ("width_avg", "Width (avg)"),
    ("dishing_max", "Dishing (max)"),
    ("Ra", "Ra"), ("Rq", "Rq"), ("Rz", "Rz"),
    ("lambda_c", "lambda_c"),
    ("cal_conf_x", "Cal. confidence X"),
    ("cal_conf_y", "Cal. confidence Y"),
    ("trend_conf", "Trend confidence"),
    ("error", "Error"),
]

_CONF_FIELDS = {"cal_conf_x", "cal_conf_y", "trend_conf"}


def _fmt_value(key: str, value) -> str:
    """Formats one row field for the detail panel. Blanks and NaN show as a
    dash, numbers get 4 significant digits, and a confidence carrying the
    manual-entry sentinel reads "manual entry" rather than a bare -1."""
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if key in _CONF_FIELDS and value == USER_CONFIDENCE:
            return "manual entry"
        return f"{value:.4g}"
    return str(value)


class ViewerTab(ttk.Frame):
    def __init__(self, notebook: ttk.Notebook, session):
        """Builds the viewer and subscribes it to the session so another
        tab's edit redraws it. Starts on the first processed image, or an
        empty placeholder when no batch has run yet."""
        super().__init__(notebook)
        self.notebook = notebook
        self.session = session

        self._names: list[str] = []
        self._name_to_index: dict[str, int] = {}
        self._index = 0
        self.view: Optional[ZoomPanImageFrame] = None
        # Set while _on_flag_toggle's own notify_results_changed() call is
        # in flight, so the resulting _on_results_changed re-broadcast
        # doesn't re-render this tab's image (which would reset the user's
        # zoom/pan for a click that only changed a checkbox).
        self._suppress_refresh = False

        self._build_ui()
        self.session.on_results_changed.append(self._on_results_changed)
        self._refresh_image_list(keep_current=False)

    # ================= UI construction =================

    def _build_ui(self):
        """Lays out the tab: Prev/Next nav across the top, the banner used
        while an Adjust tab is open, the zoomable image on the left, and the
        measurements panel with the flag checkbox on the right."""
        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(nav, text="◀ Prev", command=self._prev).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next ▶", command=self._next).pack(
            side=tk.LEFT, padx=(4, 0))
        self.title_var = tk.StringVar(value="No processed images yet.")
        ttk.Label(nav, textvariable=self.title_var).pack(side=tk.LEFT, padx=12)

        # banner: main-screen prompt shown while a transient "Adjust" tab is open
        self.banner_frame = ttk.Frame(self, relief=tk.RIDGE, borderwidth=1)
        self.banner_var = tk.StringVar()
        ttk.Label(self.banner_frame, textvariable=self.banner_var,
                  wraplength=520, justify="left").pack(
            side=tk.LEFT, padx=8, pady=6, fill=tk.X, expand=True)

        self.body = ttk.Frame(self)
        self.body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.image_holder = ttk.Frame(self.body)
        self.image_holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.placeholder_var = tk.StringVar(
            value="Run a batch, then select an image here or from the Dashboard.")
        self.placeholder_label = ttk.Label(self.image_holder,
                                           textvariable=self.placeholder_var)
        self.placeholder_label.pack(expand=True)

        panel = ttk.Frame(self.body, width=320)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        panel.pack_propagate(False)

        ttk.Label(panel, text="Measurements", font=("", 10, "bold")).pack(anchor="w")
        detail_frame = ttk.Frame(panel)
        detail_frame.pack(fill=tk.X, pady=(2, 10))
        self.detail_vars: dict[str, tk.StringVar] = {}
        for row_i, (key, label) in enumerate(DETAIL_FIELDS):
            ttk.Label(detail_frame, text=label + ":").grid(
                row=row_i, column=0, sticky="w", pady=1)
            v = tk.StringVar(value="-")
            self.detail_vars[key] = v
            ttk.Label(detail_frame, textvariable=v).grid(
                row=row_i, column=1, sticky="w", padx=(6, 0), pady=1)

        self.flag_var = tk.BooleanVar(value=False)
        self.flag_check = ttk.Checkbutton(
            panel, text="Flag for follow-up review", variable=self.flag_var,
            command=self._on_flag_toggle, state=tk.DISABLED)
        self.flag_check.pack(anchor="w", pady=(0, 8))

        self.adjust_btn = ttk.Button(panel, text="Adjust parameters…",
                                     command=self._on_adjust, state=tk.DISABLED)
        self.adjust_btn.pack(fill=tk.X, pady=(4, 0))

        self.saved_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.saved_var, foreground="#006400",
                  wraplength=300, justify="left").pack(anchor="w", pady=(8, 0))

    def _show_banner(self, text: str):
        """Shows the prompt strip above the image, explaining what the user
        should be doing in the transient tab that just opened."""
        self.banner_var.set(text)
        self.banner_frame.pack(fill=tk.X, padx=8, pady=(0, 6), before=self.body)

    def _hide_banner(self):
        """Takes the prompt strip back down once the transient tab closes."""
        self.banner_frame.pack_forget()

    # ================= image list =================

    def _on_results_changed(self):
        """Redraws when the session's rows change — unless this tab caused
        the change itself, in which case re-rendering would only throw away
        the user's zoom and pan for nothing."""
        if self._suppress_refresh:
            return
        self._refresh_image_list(keep_current=True)

    def _refresh_image_list(self, keep_current: bool):
        """Rebuilds the list of images from the session and redraws. With
        keep_current set, stays on the same image where possible rather than
        jumping back to the first one."""
        current = (self._names[self._index]
                  if keep_current and self._names and 0 <= self._index < len(self._names)
                  else None)
        self._names = [os.path.basename(p) for p in self.session.image_paths]
        self._name_to_index = {n: i for i, n in enumerate(self._names)}
        self._index = self._name_to_index.get(current, 0) if current else 0
        self._render_current()

    def show_image(self, name: str) -> None:
        """Jump to `name` — called by the Dashboard tab on double-click."""
        idx = self._name_to_index.get(name)
        if idx is None:
            return
        self._index = idx
        self._render_current()

    def _prev(self):
        """Steps to the previous image, wrapping around at the start."""
        if self._names:
            self._index = (self._index - 1) % len(self._names)
            self._render_current()

    def _next(self):
        """Steps to the next image, wrapping around at the end."""
        if self._names:
            self._index = (self._index + 1) % len(self._names)
            self._render_current()

    def _row_for_name(self, name: str) -> Optional[dict]:
        """Returns the results row for an image name, or None if this session
        has no row for it."""
        idx = self.session.row_index_for_name(name)
        return self.session.rows[idx] if idx is not None else None

    # ================= rendering =================

    def _render_current(self):
        """Draws whichever image is currently selected: its title, its
        measurements down the side panel, its flag state, and its overlay in
        the image area. Falls back to a placeholder when there's nothing to
        show yet."""
        self.saved_var.set("")
        if not self._names:
            self.title_var.set("No processed images yet.")
            self._clear_image("Run a batch, then select an image here or "
                              "from the Dashboard.")
            for v in self.detail_vars.values():
                v.set("-")
            self.adjust_btn.configure(state=tk.DISABLED)
            self.flag_var.set(False)
            self.flag_check.configure(state=tk.DISABLED)
            return

        name = self._names[self._index]
        row = self._row_for_name(name)
        self.title_var.set(f"{name}   ({self._index + 1} / {len(self._names)})")

        for key, _label in DETAIL_FIELDS:
            self.detail_vars[key].set(_fmt_value(key, row.get(key) if row else None))

        # Variable.set() does not invoke the Checkbutton's own `command`, so
        # syncing the displayed state here can't loop back into
        # _on_flag_toggle and re-notify/re-write the CSV on a plain Prev/Next.
        self.flag_var.set(bool(row.get("flagged")) if row else False)
        self.flag_check.configure(state=tk.NORMAL)

        self._show_image_for(name, row)
        self.adjust_btn.configure(state=tk.NORMAL)

    # ================= flag for follow-up review =================

    def _on_flag_toggle(self):
        """Persist the 'Flag for follow-up review' checkbox: update the
        in-memory row, amend the on-disk CSV if one exists (same
        update-in-place pattern as _apply_adjustment), then notify other
        tabs so the Dashboard's 'flagged' column follows along.

        _suppress_refresh brackets the notify so THIS tab's own listener
        doesn't re-render — a checkbox click is not a reason to reload the
        displayed image and reset the user's zoom/pan, unlike an actual
        measurement edit."""
        if not self._names:
            return
        name = self._names[self._index]
        idx = self.session.row_index_for_name(name)
        if idx is None:
            return

        flagged = bool(self.flag_var.get())
        self.session.rows[idx]["flagged"] = flagged

        csv_error = None
        if self.session.csv_path and os.path.isfile(self.session.csv_path):
            try:
                engine.update_results_csv_flag(self.session.csv_path, name, flagged)
            except Exception as e:
                csv_error = str(e)

        self._suppress_refresh = True
        try:
            self.session.notify_results_changed()
        finally:
            self._suppress_refresh = False

        if csv_error:
            self.saved_var.set(
                f"Flag saved in this session, but updating the CSV failed: {csv_error}")
        else:
            self.saved_var.set(
                f"{name} {'flagged' if flagged else 'unflagged'} — saved.")

    def _resolve_display_path(self, name: str, row: Optional[dict]) -> Optional[str]:
        """Picks what to actually display for an image: its review overlay if
        one was rendered and still exists, otherwise the original file. Returns
        None when neither is available."""
        overlay_rel = row.get("overlay_file") if row else None
        if isinstance(overlay_rel, str) and overlay_rel:
            full = os.path.join(self.session.config.image_fits_dir, overlay_rel)
            if os.path.isfile(full):
                return full
        return self.session.path_for_name(name)

    def _clear_image(self, message: str):
        """Tears down the image widget and shows `message` in its place —
        used when there's nothing to display or a file wouldn't load."""
        if self.view is not None:
            self.view.frame.destroy()
            self.view = None
        self.placeholder_var.set(message)
        self.placeholder_label.pack(expand=True)

    def _show_image_for(self, name: str, row: Optional[dict]):
        """Loads an image and puts it on screen in a fresh zoom/pan frame.
        A missing file or a read failure becomes a message in the image area
        rather than an exception."""
        path = self._resolve_display_path(name, row)
        if path is None:
            self._clear_image(f"No image file found for {name}.")
            return
        try:
            rgb = engine.load_rgb(path)
        except Exception as e:
            self._clear_image(f"Could not load {name}: {e}")
            return

        if self.view is not None:
            self.view.frame.destroy()
            self.view = None
        self.placeholder_label.pack_forget()
        self.view = ZoomPanImageFrame(self.image_holder, rgb)
        self.view.pack(fill=tk.BOTH, expand=True)
        self.view.first_draw()

    # ================= adjust parameters =================

    def _on_adjust(self):
        """Opens the manual-fit form for the current image in a transient tab
        and, if the user saves, applies what they entered. The form is seeded
        with this image's existing measurements (see autofit_on_open) so
        adding or removing a feature starts from what was actually measured."""
        if not self._names:
            return
        name = self._names[self._index]
        path = self.session.path_for_name(name)
        row = self._row_for_name(name)
        if path is None or row is None:
            self.saved_var.set(f"{name} is no longer available in this session.")
            return

        try:
            rgb = engine.load_rgb(path)
        except Exception as e:
            self.saved_var.set(f"Could not load {name}: {e}")
            return

        config = self.session.config
        ctx = self.session.ctx_by_path.get(path, {})
        prefill = engine.build_prefill(
            config, rgb, ctx, row.get("error"),
            self.session.extraction_buffer, self.session.extraction_max_gap)
        # Re-opening an already-processed image: seed the form with what
        # was actually measured (re-running the same auto fit) instead of
        # opening empty — otherwise there's nothing to remove a feature
        # from, and adding one silently overwrites the real count. The
        # batch's own failed-image review (manual_review_phase) does NOT
        # set this — there "Run auto fit" stays an explicit user action.
        prefill["autofit_on_open"] = True

        self._show_banner(
            f"Manual fit — {name}: complete it in the tab that just opened "
            f"(every field is optional). Saving amends the results CSV.")

        def build(frame, on_done):
            """Fills the transient tab with the manual-fit form."""
            launch_manual_fit_embedded(frame, rgb, name, prefill,
                                       config.output_unit, on_done)

        result = run_tab_and_wait(self.notebook, f"Adjust — {name}", build)
        self._hide_banner()
        if not result:
            return
        action, data = result
        if action != "saved" or not data:
            return
        self._apply_adjustment(name, path, ctx, row, data)

    def _apply_adjustment(self, name: str, path: str, ctx: dict, row: dict,
                          data: dict) -> None:
        """Takes a saved manual-fit payload and writes it everywhere it needs
        to go: merged into the in-memory row, amended into the results CSV,
        and re-rendered as this image's overlay. Other tabs are notified last
        so the Dashboard picks the change up."""
        config = self.session.config
        idx = self.session.row_index_for_name(name)
        if idx is None:
            return

        merged = merge_manual_row(row, data, config.output_unit)
        self.session.rows[idx] = merged

        if self.session.csv_path and os.path.isfile(self.session.csv_path):
            engine.update_results_csv_row(self.session.csv_path, name, data,
                                          config.output_unit)

        run_dir = self.session.run_dir or config.image_fits_dir
        overlay_error = None
        try:
            rgb = engine.load_rgb(path)
            rel = engine.render_manual_overlay(config, rgb, ctx, data, run_dir,
                                               name, merged)
            self.session.rows[idx]["overlay_file"] = rel
        except Exception as e:
            overlay_error = str(e)

        # Triggers _on_results_changed() -> _render_current(), which blanks
        # saved_var as it redraws the (now-updated) image/details — so the
        # status message must be set AFTER notifying, not before, or it
        # would be cleared before the user ever sees it.
        self.session.notify_results_changed()

        if overlay_error:
            self.saved_var.set(f"Saved, but re-rendering the overlay failed: {overlay_error}")
        else:
            self.saved_var.set(f"Saved — {name} updated in the results CSV.")
