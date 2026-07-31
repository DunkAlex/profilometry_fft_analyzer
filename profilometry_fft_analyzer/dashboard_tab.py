"""
dashboard_tab.py
-----------------
The Dashboard tab: a sortable table over the current session's results
(session.rows — the same rows written to the results CSV), with failed and
low-confidence rows highlighted so the images most likely to need a manual
look stand out instead of requiring a scroll through the whole run.

Double-clicking a row calls `self.on_open_in_viewer(name)` (wired by app.py)
to jump to that image in the Viewer tab.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import app_settings as settings
from manual_merge import USER_CONFIDENCE as MANUAL_ENTRY_SENTINEL

# Curated subset of app_engine.CSV_COLUMNS worth a column in the table —
# the full row (all CSV_COLUMNS) is still what's written to disk and what
# the Viewer shows in detail; this is just what's worth a glance here.
DISPLAY_COLUMNS = [
    "Name", "Lot", "Wafer", "Recipe ID", "Site Name", "DIEX", "DIEY",
    "flagged", "feature_type", "n_features",
    "height_avg", "width_avg", "dishing_max",
    "Ra", "Rq", "Rz",
    "cal_conf_x", "cal_conf_y", "trend_conf",
    "error",
]

# The "needs a look" bar (settings.LOW_CONFIDENCE_THRESHOLD) is user-tunable
# from Preferences. A blended [0,1] confidence under it gets the row
# highlighted. Nothing else in the codebase defines a shared "this is low"
# constant to reuse — the nearest thing, SEQ_RECONSTRUCT_MIN_OCR_CONF, uses
# the same 0.5 bar for a much narrower signal.


def _cell(value) -> str:
    """Takes any row value and returns the string the table should show.
    Floats are trimmed to 4 significant digits; None and NaN both come back
    as an empty cell rather than the word 'nan'."""
    if value is None:
        return ""
    if isinstance(value, float):
        # NaN is the only float that isn't equal to itself.
        if value != value:
            return ""
        return f"{value:.4g}"
    return str(value)


def _confidence_values(row: dict):
    """Yields the row's three confidence scores as floats, skipping any that
    are missing, unparseable, or NaN. Callers get a clean stream of real
    numbers to test against a threshold."""
    for key in ("trend_conf", "cal_conf_x", "cal_conf_y"):
        v = row.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        # Skip NaN.
        if v == v:
            yield v


def _is_manual(row: dict) -> bool:
    """True when any confidence carries the manual-entry sentinel, meaning a
    human typed that group in rather than the pipeline measuring it."""
    return any(v == MANUAL_ENTRY_SENTINEL for v in _confidence_values(row))


def _is_low_confidence(row: dict) -> bool:
    """True when any automatic confidence sits below the review threshold.
    The manual-entry sentinel (-1.0) is deliberately excluded by the `0 <= v`
    bound — a human-verified value isn't low confidence, it's the opposite."""
    return any(0 <= v < settings.LOW_CONFIDENCE_THRESHOLD
               for v in _confidence_values(row))


class DashboardTab(ttk.Frame):
    def __init__(self, notebook: ttk.Notebook, session):
        """Builds the table and subscribes it to the session, so any change
        another tab makes to the rows redraws this one. Draws once up front
        so the tab isn't blank before the first batch."""
        super().__init__(notebook)
        self.notebook = notebook
        self.session = session
        self.on_open_in_viewer: Optional[Callable[[str], None]] = None

        self._sort_col: Optional[str] = None
        self._sort_reverse = False
        self._imported_iids: set = set()

        self._build_ui()
        self.session.on_results_changed.append(self.refresh)
        self.refresh()

    # ================= UI construction =================

    def _build_ui(self):
        """Lays out the filter checkbox, the summary line, and the scrolling
        results table, and registers the row colors and click handlers."""
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)
        self.filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Show only failed / low-confidence rows",
                        variable=self.filter_var, command=self.refresh
                        ).pack(side=tk.LEFT)
        self.summary_var = tk.StringVar(value="No results yet — run a batch first.")
        ttk.Label(top, textvariable=self.summary_var).pack(side=tk.RIGHT)

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.tree = ttk.Treeview(tree_frame, columns=DISPLAY_COLUMNS, show="headings",
                                 yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                                 selectmode="browse")
        vsb.configure(command=self.tree.yview)
        hsb.configure(command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        for col in DISPLAY_COLUMNS:
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._on_heading_click(c))
            self.tree.column(col, width=90, anchor="w", stretch=False)
        self.tree.column("Name", width=240)
        self.tree.column("Recipe ID", width=140)

        self.tree.tag_configure("failed", background="#ffd6d6")
        self.tree.tag_configure("low_conf", background="#fff3cd")
        self.tree.tag_configure("manual", background="#e6f2ff")
        # Imported (from a prior "add to existing data file" run): grayed
        # text, no background fill, and not selectable/openable — see
        # _on_selection_changed / _on_double_click. Takes priority over the
        # other tags since these rows are read-only history, not something
        # to act on in this session.
        self.tree.tag_configure("imported", foreground="#999999")

        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_changed)

    # ================= data =================

    def refresh(self):
        """Rebuilds the whole table from session.rows: tags each row by how
        much attention it needs, tallies the summary line, and restores the
        previous sort and selection so a redraw doesn't lose the user's
        place. Called on every results change."""
        rows = self.session.rows
        selected = self._selected_name()

        self.tree.delete(*self.tree.get_children())
        self._imported_iids = set()
        shown = 0
        n_failed = 0
        n_low = 0
        n_imported = 0
        for i, row in enumerate(rows):
            is_imported = bool(row.get("_imported"))
            is_failed = bool(row.get("error"))
            is_low_conf = _is_low_confidence(row)
            n_imported += int(is_imported)
            # The failed/low tallies describe THIS session's work. Imported
            # rows are history the user already reviewed elsewhere, so they
            # stay out of the "needs a look" counts even if they were low
            # confidence back when they were measured.
            if not is_imported:
                n_failed += int(is_failed)
                n_low += int(is_low_conf and not is_failed)

            # Filter mode shows only this session's problem rows.
            if self.filter_var.get() and (is_imported or not (is_failed or is_low_conf)):
                continue

            # One tag per row, most important first — imported wins because
            # those rows are read-only regardless of their old confidence.
            tags = []
            if is_imported:
                tags.append("imported")
            elif is_failed:
                tags.append("failed")
            elif is_low_conf:
                tags.append("low_conf")
            elif _is_manual(row):
                tags.append("manual")

            values = [_cell(row.get(c)) for c in DISPLAY_COLUMNS]
            # Name is the natural row id and should be unique per run. Fall
            # back to a positional id if it's missing, so a blank or repeated
            # name can't collide with — and silently hide — another row.
            iid = row.get("Name") or f"row{i}"
            if self.tree.exists(iid):
                iid = f"{iid}#{i}"
            self.tree.insert("", tk.END, iid=iid, values=values, tags=tags)
            if is_imported:
                self._imported_iids.add(iid)
            shown += 1

        suffix = f"  (showing {shown})" if self.filter_var.get() else ""
        imported_note = f", {n_imported} imported" if n_imported else ""
        summary = (f"{len(rows)} image(s) — {n_failed} failed, {n_low} low-confidence"
                   f"{imported_note}" + suffix) if rows else "No results yet — run a batch first."
        self.summary_var.set(summary)

        # Put the sort and selection back the way the user had them.
        if self._sort_col:
            self._apply_sort(self._sort_col, self._sort_reverse)
        if selected and self.tree.exists(selected):
            self.tree.selection_set(selected)

    def _selected_name(self) -> Optional[str]:
        """Returns the selected row's id (its image name), or None if nothing
        is selected."""
        sel = self.tree.selection()
        return sel[0] if sel else None

    # ================= sorting =================

    def _on_heading_click(self, col: str):
        """Handles a click on a column heading: same column flips the
        direction, a new column sorts ascending."""
        if self._sort_col == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col, self._sort_reverse = col, False
        self._apply_sort(col, self._sort_reverse)

    def _apply_sort(self, col: str, reverse: bool):
        """Reorders the visible rows by one column and moves the ▲/▼ arrow to
        that heading. Numbers sort before text so blanks and error strings
        collect at one end instead of scattering through the numbers."""
        def sort_key(iid):
            """Sorts numerically where the cell parses as a number, otherwise
            alphabetically. The leading 0/1 keeps the two kinds apart."""
            v = self.tree.set(iid, col)
            try:
                return (0, float(v))
            except ValueError:
                return (1, v.lower())

        items = list(self.tree.get_children(""))
        items.sort(key=sort_key, reverse=reverse)
        for index, iid in enumerate(items):
            self.tree.move(iid, "", index)

        # Arrow marks the active sort column; every other heading is plain.
        for c in DISPLAY_COLUMNS:
            arrow = (" ▼" if reverse else " ▲") if c == col else ""
            self.tree.heading(c, text=c + arrow)

    # ================= interaction =================

    def _on_selection_changed(self, event):
        """Imported rows belong to a prior session — no image/context is
        available for them, so block selection rather than let the user
        select a row that can never open in the Viewer."""
        sel = self.tree.selection()
        if sel and sel[0] in self._imported_iids:
            self.tree.selection_remove(sel[0])

    def _on_double_click(self, event):
        """Opens the double-clicked image in the Viewer. Imported rows are
        ignored — there's no image file behind them to show."""
        row_id = self.tree.identify_row(event.y)
        if not row_id or row_id in self._imported_iids:
            return
        name = self.tree.set(row_id, "Name") or row_id
        if self.on_open_in_viewer:
            self.on_open_in_viewer(name)
