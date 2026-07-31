"""
preferences_gui.py
------------------
The Preferences menu and its dialogs.

app.py hangs a menubar off the main window with two entries:

    Modify parameters…   the editing dialog built here
    Revert to defaults   throws away every override, after confirming

The dialog lists every parameter app_settings exposes, grouped into
sections, one row each: a label, a small entry box, and an [i] button that
pops a sentence or two explaining what the value does to your results.

Saving validates each field through its own spec before anything is
written, so a typo can't get as far as the settings file. Values land in
parameter_settings.json next to the app and apply to the next batch run —
no restart.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import app_settings as settings

# Weight groups that are meant to add up to 1.0. Not enforced — a user may
# have a reason — but worth saying out loud, since silently unnormalized
# weights make confidence scores hard to compare between runs.
WEIGHT_GROUPS = [
    ("Trend confidence",
     ["TREND_CONF_W_COVERAGE", "TREND_CONF_W_GAP", "TREND_CONF_W_JUMP"]),
    ("Calibration confidence",
     ["CONF_W_R2", "CONF_W_TICK_COVERAGE", "CONF_W_OCR_MEAN",
      "CONF_W_GRIDLINE_REG"]),
]

_ERROR_BG = "#ffd6d6"
_NORMAL_BG = "white"


def install_menubar(root: tk.Tk, on_changed=None) -> tk.Menu:
    """Attaches the app's menubar to `root` and returns it. `on_changed` is
    called with no arguments after any save or revert, so the caller can
    refresh anything showing a parameter-derived value. Sits at the top of
    the window like a normal File/Edit bar."""
    menubar = tk.Menu(root)

    prefs_menu = tk.Menu(menubar, tearoff=0)
    prefs_menu.add_command(
        label="Modify parameters…",
        command=lambda: open_parameters_dialog(root, on_changed))
    prefs_menu.add_separator()
    prefs_menu.add_command(
        label="Revert to defaults",
        command=lambda: revert_to_defaults(root, on_changed))

    menubar.add_cascade(label="Preferences", menu=prefs_menu)
    root.config(menu=menubar)
    return menubar


def revert_to_defaults(parent, on_changed=None) -> bool:
    """Asks first, then resets every parameter to its shipped default and
    rewrites the settings file. Returns True if the reset happened. Says so
    plainly when nothing was customised in the first place."""
    if not settings.is_modified():
        messagebox.showinfo(
            "Revert to defaults",
            "Every parameter is already at its default value.",
            parent=parent)
        return False

    if not messagebox.askyesno(
            "Revert to defaults",
            "Reset every parameter to its default value?\n\n"
            "This discards any changes you've saved. It takes effect on the "
            "next batch run.",
            parent=parent):
        return False

    settings.reset_to_defaults()
    if on_changed:
        on_changed()
    messagebox.showinfo("Revert to defaults",
                        "All parameters are back to their defaults.",
                        parent=parent)
    return True


def open_parameters_dialog(parent, on_changed=None) -> None:
    """Opens the modal parameter editor over `parent`. Builds one row per
    parameter from app_settings.iter_specs(), so adding a parameter there is
    all it takes to have it show up here."""
    ParametersDialog(parent, on_changed)


class _InfoPopup:
    """The little help bubble behind an [i] button. Only one is ever open —
    opening another closes the first — and it dismisses on click-away, Esc,
    or when the dialog scrolls out from under it."""

    def __init__(self):
        """Starts with no bubble on screen."""
        self._win = None

    def show(self, widget, text: str) -> None:
        """Pops the bubble next to `widget` showing `text`, replacing any
        bubble already on screen."""
        self.hide()
        win = tk.Toplevel(widget)
        # No title bar — this is a tooltip, not a window to manage.
        win.wm_overrideredirect(True)
        win.configure(background="#666666")

        frame = tk.Frame(win, background="#ffffe0", padx=8, pady=6)
        frame.pack(padx=1, pady=1)
        tk.Label(frame, text=text, background="#ffffe0", justify="left",
                 wraplength=320, font=("", 9)).pack(anchor="w")
        tk.Label(frame, text="(click anywhere to close)", background="#ffffe0",
                 foreground="#888888", font=("", 7)).pack(anchor="w",
                                                          pady=(4, 0))

        # Anchor just below-right of the button that opened it.
        win.update_idletasks()
        x = widget.winfo_rootx() + widget.winfo_width() + 4
        y = widget.winfo_rooty() - 2
        # Pull back on-screen if the bubble would run off the right edge.
        screen_w = widget.winfo_screenwidth()
        if x + win.winfo_width() > screen_w:
            x = max(0, screen_w - win.winfo_width() - 8)
        win.wm_geometry(f"+{x}+{y}")

        self._win = win
        win.bind("<Button-1>", lambda _e: self.hide())

    def hide(self) -> None:
        """Closes the bubble if one is open; safe to call any time."""
        if self._win is not None:
            self._win.destroy()
            self._win = None


class ParametersDialog:
    """Modal editor over every parameter in app_settings.

    Builds a scrollable, grouped list of rows from iter_specs(); on save,
    parses every field through its spec and writes them in one go, so either
    all the edits land or none do."""

    def __init__(self, parent, on_changed=None):
        """Builds the modal dialog over `parent`, fills every field from the
        values currently in effect, and grabs input until it closes.
        `on_changed` fires after a successful save or reset."""
        self.parent = parent
        self.on_changed = on_changed
        self.info = _InfoPopup()
        # spec name -> (StringVar or BooleanVar, entry widget for highlighting)
        self.vars: dict = {}
        self.widgets: dict = {}

        self.win = tk.Toplevel(parent)
        self.win.title("Modify parameters")
        self.win.geometry("720x640")
        self.win.minsize(600, 400)
        self.win.transient(parent)

        self._build_ui()
        self._load_current()

        self.win.protocol("WM_DELETE_WINDOW", self._close)
        self.win.bind("<Escape>", lambda _e: self._close())
        self.win.grab_set()

    # ================= UI construction =================

    def _build_ui(self):
        """Lays out the header, the scrolling body of parameter rows, and the
        button strip along the bottom."""
        header = ttk.Frame(self.win)
        header.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Label(header, text="Analysis parameters",
                  font=("", 11, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text=("Click the i next to a value to see what it does. Changes are "
                  "saved to parameter_settings.json and apply to the next "
                  "batch run."),
            wraplength=680, justify="left", foreground="#555555").pack(
            anchor="w", pady=(2, 0))

        # Scrollable body: a canvas holding one inner frame of rows.
        body = ttk.Frame(self.win)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        canvas = tk.Canvas(body, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Keep the scroll region matched to the content, and the content
        # matched to the canvas width, as either one resizes.
        self.inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window_id, width=e.width))

        # Wheel scrolling, with the platform variations tk exposes.
        def _on_wheel(event):
            """Scrolls the parameter list one notch and dismisses any open
            help bubble, which would otherwise float free of its row.
            Handles both the Windows/macOS delta and X11's button 4/5."""
            delta = 1 if getattr(event, "num", None) == 5 else -1
            if getattr(event, "delta", 0):
                delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta, "units")
            self.info.hide()

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(seq, _on_wheel)
        self._wheel_canvas = canvas

        self._build_rows()

        # Bottom button strip.
        buttons = ttk.Frame(self.win)
        buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.status_var = tk.StringVar(value="")
        ttk.Label(buttons, textvariable=self.status_var,
                  foreground="#8b0000", wraplength=420,
                  justify="left").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(buttons, text="Cancel",
                   command=self._close).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Save",
                   command=self._on_save).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Reset all to defaults",
                   command=self._on_reset).pack(side=tk.RIGHT, padx=(0, 12))

    def _build_rows(self):
        """Walks the specs in dialog order, starting a new titled section each
        time the group changes and adding one editable row per parameter."""
        current_group = None
        for spec in settings.iter_specs():
            if spec.group != current_group:
                current_group = spec.group
                self._add_group_header(current_group)
            self._add_row(spec)

    def _add_group_header(self, title: str):
        """Draws a section heading with a rule under it."""
        holder = ttk.Frame(self.inner)
        holder.pack(fill=tk.X, pady=(12, 2))
        ttk.Label(holder, text=title, font=("", 10, "bold")).pack(anchor="w")
        ttk.Separator(holder, orient="horizontal").pack(fill=tk.X, pady=(2, 0))

    def _add_row(self, spec):
        """Builds one parameter row: label on the left, an input sized to the
        value type on the right, and an [i] button holding the explanation.
        Bools get a checkbox, fixed-choice values a dropdown, everything else
        a text box."""
        row = ttk.Frame(self.inner)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text=spec.label, width=42, anchor="w").pack(
            side=tk.LEFT, padx=(4, 6))

        if spec.cast is settings._as_bool:
            var = tk.BooleanVar()
            widget = ttk.Checkbutton(row, variable=var)
            widget.pack(side=tk.LEFT)
        elif spec.choices:
            var = tk.StringVar()
            widget = ttk.OptionMenu(row, var, spec.default, *spec.choices)
            widget.configure(width=10)
            widget.pack(side=tk.LEFT)
        else:
            var = tk.StringVar()
            widget = tk.Entry(row, textvariable=var, width=14)
            widget.pack(side=tk.LEFT)

        self.vars[spec.name] = var
        self.widgets[spec.name] = widget

        info_btn = tk.Button(
            row, text="i", width=2, font=("", 8, "bold"),
            relief=tk.RIDGE, cursor="hand2",
            command=lambda s=spec, w=row: self._show_info(s, w))
        info_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Show the default inline so a user can see what they've drifted from.
        default_text = "auto" if spec.default is None else str(spec.default)
        ttk.Label(row, text=f"(default: {default_text})",
                  foreground="#888888", font=("", 8)).pack(side=tk.LEFT,
                                                           padx=(8, 0))

    def _show_info(self, spec, anchor_widget):
        """Pops this parameter's help text next to its row."""
        self.info.show(anchor_widget, spec.help)

    # ================= Values in / out =================

    def _load_current(self):
        """Fills every field from the values currently in effect."""
        active = settings.current()
        for spec in settings.iter_specs():
            value = active.get(spec.name, spec.default)
            var = self.vars[spec.name]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif value is None:
                # Blank means "unset" for the optional numerics.
                var.set("")
            else:
                var.set(str(value))
            self._clear_error(spec.name)

    def _clear_error(self, name):
        """Returns a row to its normal background after a failed save."""
        widget = self.widgets.get(name)
        if isinstance(widget, tk.Entry):
            widget.configure(background=_NORMAL_BG)

    def _mark_error(self, name):
        """Tints a row red so the offending field is obvious."""
        widget = self.widgets.get(name)
        if isinstance(widget, tk.Entry):
            widget.configure(background=_ERROR_BG)

    def _collect(self):
        """Reads and validates every field. Returns (values, errors): values
        is the parsed dict, errors is a list of '<label>: <reason>' strings.
        Nothing is written when errors is non-empty, so a bad entry can't
        leave the settings file half-updated."""
        values = {}
        errors = []
        for spec in settings.iter_specs():
            self._clear_error(spec.name)
            raw = self.vars[spec.name].get()
            try:
                values[spec.name] = spec.parse(raw)
            except (TypeError, ValueError) as e:
                errors.append(f"{spec.label}: {e}")
                self._mark_error(spec.name)
        return values, errors

    def _weight_warnings(self, values) -> list[str]:
        """Checks the weight groups that are supposed to sum to 1.0 and
        returns a readable note for each that doesn't. Advisory only — the
        save still goes through."""
        notes = []
        for title, names in WEIGHT_GROUPS:
            total = sum(float(values.get(n, 0.0)) for n in names)
            if abs(total - 1.0) > 1e-6:
                notes.append(f"{title} weights add up to {total:.3f}, not 1.0.")
        return notes

    # ================= Actions =================

    def _on_save(self):
        """Validates everything, warns about unnormalized weights, then writes
        the values and closes. Leaves the dialog open if anything failed to
        parse, with the bad rows highlighted."""
        values, errors = self._collect()
        if errors:
            shown = "\n".join(f"  • {e}" for e in errors[:6])
            more = f"\n  …and {len(errors) - 6} more" if len(errors) > 6 else ""
            self.status_var.set(f"{len(errors)} value(s) need fixing.")
            messagebox.showerror(
                "Invalid values",
                f"These entries couldn't be saved:\n\n{shown}{more}",
                parent=self.win)
            return

        warnings = self._weight_warnings(values)
        if warnings:
            body = "\n".join(f"  • {w}" for w in warnings)
            if not messagebox.askyesno(
                    "Check weights",
                    f"{body}\n\nConfidence scores are easiest to compare "
                    f"between runs when each set of weights sums to 1.0.\n\n"
                    f"Save anyway?",
                    parent=self.win):
                return

        settings.save(values)
        if self.on_changed:
            self.on_changed()
        messagebox.showinfo(
            "Parameters saved",
            "Saved to parameter_settings.json. The new values apply to the "
            "next batch run.",
            parent=self.win)
        self._close()

    def _on_reset(self):
        """Resets to defaults and reloads the fields, leaving the dialog open
        so the restored values are visible."""
        if revert_to_defaults(self.win, self.on_changed):
            self._load_current()
            self.status_var.set("")

    def _close(self):
        """Tears down the dialog, releasing the wheel bindings it installed
        app-wide so they don't outlive the window."""
        self.info.hide()
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self._wheel_canvas.unbind_all(seq)
            except tk.TclError:
                pass
        self.win.grab_release()
        self.win.destroy()
