"""
batch_tab.py
------------
The Batch tab: run the full pipeline (profile audit -> per-image analysis ->
manual correction -> results CSV) from inside the app window.

Threading model: tkinter is not thread-safe, so only Phase B (per-image
analysis, pure compute) runs on a worker thread; a queue.Queue carries
progress (and, only while that thread runs, forwarded log records) back to
the main thread, drained by a root.after() poll loop. Phases A and C are
interactive and run on the main thread — when they need the user to do
something, a banner appears first, then a transient tab opens (via
app_ui.run_tab_and_wait) hosting the same embeddable pop-up used by the CLI,
and closes itself when that one task is done.
"""
from __future__ import annotations
import logging
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, ttk
from typing import Optional

import app_engine as engine
from app_ui import run_tab_and_wait, format_hms
from plot_calibration.profiles import load_profiles
from plot_calibration.gui_calibrate import launch_calibration_gui_embedded
from manual_fit_gui import launch_manual_fit_embedded


class _QueueLogHandler(logging.Handler):
    """Forwards WARNING+ log records to the GUI thread via a queue. Attached
    to the root logger only while the Phase B worker thread runs, since
    that's the only phase whose logger.* calls happen off the main thread —
    Phase A/C run on the main thread and log to the panel directly."""

    def __init__(self, q: "queue.Queue"):
        """Points this handler at the queue the main thread drains, and keeps
        the format short since these lines land in a small panel."""
        super().__init__(level=logging.WARNING)
        self.q = q
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record):
        """Hands one formatted log line to the queue instead of touching any
        widget — Tk is not thread-safe, so the main thread does the drawing."""
        self.q.put(("log", self.format(record)))


class BatchTab(ttk.Frame):
    def __init__(self, notebook: ttk.Notebook, session):
        """Builds the Batch tab and wires it to the shared session. Sets up
        the queue used to carry progress off the worker thread, but starts no
        work — that waits for Run batch."""
        super().__init__(notebook)
        self.notebook = notebook
        self.session = session

        self._q: "queue.Queue" = queue.Queue()
        self._running = False
        self._log_handler: Optional[_QueueLogHandler] = None
        self._run_log_handlers: list = []

        # "Add to existing data file" state (Optional[dict] from
        # engine.load_existing_data_file, plus the path it was loaded from).
        self._existing_data: Optional[dict] = None
        self._add_mode = False
        self._existing_csv_path: Optional[str] = None
        self._existing_rows: list = []
        self._existing_names: set = set()

        self._build_ui()

    # ================= UI construction =================

    def _build_ui(self):
        """Lays out the tab: input folder and output unit at the top, the
        optional add-to-existing-file section, the Run button, then the
        progress bar and log panel that fill in while a batch runs."""
        pad = dict(padx=8, pady=6)

        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)

        ttk.Label(top, text="Input folder:").grid(row=0, column=0, sticky="w")
        self.input_dir_var = tk.StringVar(value=self.session.config.input_dir)
        ttk.Entry(top, textvariable=self.input_dir_var, width=60).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse…", command=self._browse_input_dir).grid(
            row=0, column=2, padx=2)
        top.grid_columnconfigure(1, weight=1)

        ttk.Label(top, text="Output unit:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.unit_var = tk.StringVar(value=self.session.config.output_unit)
        self.unit_combo = ttk.Combobox(top, textvariable=self.unit_var, values=["um", "nm"],
                                       state="readonly", width=6)
        self.unit_combo.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))

        self.run_btn = ttk.Button(top, text="Run batch", command=self._on_run_clicked)
        self.run_btn.grid(row=1, column=2, sticky="e", pady=(6, 0))

        # --- add to existing data file: merge this run into a prior CSV
        # instead of starting a fresh one, so a large results file can be
        # built up across many runs without duplicate rows ---
        add_frame = ttk.Frame(self)
        add_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.add_existing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(add_frame, text="Add to existing data file",
                        variable=self.add_existing_var,
                        command=self._on_add_existing_toggled).grid(
            row=0, column=0, sticky="w")
        self.existing_csv_var = tk.StringVar(value="")
        self.existing_csv_entry = ttk.Entry(
            add_frame, textvariable=self.existing_csv_var, width=50,
            state=tk.DISABLED)
        self.existing_csv_entry.grid(row=0, column=1, sticky="ew", padx=4)
        self.existing_csv_browse_btn = ttk.Button(
            add_frame, text="Browse…", command=self._browse_existing_csv,
            state=tk.DISABLED)
        self.existing_csv_browse_btn.grid(row=0, column=2, padx=2)
        add_frame.grid_columnconfigure(1, weight=1)
        self.existing_csv_status_var = tk.StringVar(value="")
        ttk.Label(add_frame, textvariable=self.existing_csv_status_var,
                  foreground="#555555", wraplength=560, justify="left").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # --- banner: main-screen prompt before any transient tab opens ---
        self.banner_frame = ttk.Frame(self, relief=tk.RIDGE, borderwidth=1)
        self.banner_var = tk.StringVar()
        ttk.Label(self.banner_frame, textvariable=self.banner_var,
                  wraplength=520, justify="left").pack(
            side=tk.LEFT, padx=8, pady=6, fill=tk.X, expand=True)
        self.banner_btn_frame = ttk.Frame(self.banner_frame)
        self.banner_btn_frame.pack(side=tk.RIGHT, padx=8)
        # not packed into self yet — _show_banner()/_hide_banner() control visibility

        # --- progress ---
        # progress_container holds the bar+status while a run is active or
        # idle; on a successful completion it's swapped out for
        # success_label (same prog_frame slot, see _finalize_run) so the bar
        # doesn't just sit there full/idle after the run is done.
        self.prog_frame = prog_frame = ttk.Frame(self)
        prog_frame.pack(fill=tk.X, **pad)
        self.progress_container = ttk.Frame(prog_frame)
        self.progress_container.pack(fill=tk.X)
        self.progress = ttk.Progressbar(self.progress_container, mode="determinate")
        self.progress.pack(fill=tk.X)
        status_row = ttk.Frame(self.progress_container)
        status_row.pack(fill=tk.X, pady=(4, 0))
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT)
        self.eta_var = tk.StringVar(value="")
        ttk.Label(status_row, textvariable=self.eta_var).pack(side=tk.RIGHT)

        self.success_var = tk.StringVar(value="")
        self.success_label = ttk.Label(prog_frame, textvariable=self.success_var,
                                       foreground="#0a7d2c", font=("", 10, "bold"))
        # not packed here — shown in progress_container's place on completion

        # --- log panel ---
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED,
                                yscrollcommand=log_scroll.set, wrap="word")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.configure(command=self.log_text.yview)

    # ================= small UI helpers =================

    def _browse_input_dir(self):
        """Opens a folder picker for the input images and stores the choice."""
        d = filedialog.askdirectory(initialdir=self.input_dir_var.get() or None,
                                    title="Select input images folder")
        if d:
            self.input_dir_var.set(d)

    # ---- "add to existing data file" ----

    def _on_add_existing_toggled(self):
        """Enables or disables the existing-CSV controls with the checkbox,
        reloading the chosen file when switched on and clearing the selection
        when switched off."""
        if self.add_existing_var.get():
            self.existing_csv_entry.configure(state=tk.NORMAL)
            self.existing_csv_browse_btn.configure(state=tk.NORMAL)
            if self.existing_csv_var.get():
                self._load_existing_csv(self.existing_csv_var.get())
        else:
            self.existing_csv_entry.configure(state=tk.DISABLED)
            self.existing_csv_browse_btn.configure(state=tk.DISABLED)
            self._clear_existing_csv_selection()

    def _browse_existing_csv(self):
        """Opens a file picker for the results CSV to append to, then loads
        it so its row count and unit are known before the run starts."""
        path = filedialog.askopenfilename(
            title="Select existing results CSV to add to",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.existing_csv_var.set(path)
            self._load_existing_csv(path)

    def _load_existing_csv(self, path: str):
        """Reads an existing results CSV so this run can be merged into it.
        Reports how many rows it holds and, when the file records an output
        unit, locks this run to the same one — mixing units inside a single
        file would make the numbers meaningless. An unreadable file leaves
        the selection cleared with the reason shown."""
        try:
            data = engine.load_existing_data_file(path)
        except Exception as e:
            self._existing_data = None
            self.existing_csv_status_var.set(f"Could not read this file: {e}")
            self.unit_combo.configure(state="readonly")
            return

        data["path"] = path
        self._existing_data = data
        n = len(data["rows"])
        if data["output_unit"]:
            self.unit_var.set(data["output_unit"])
            self.unit_combo.configure(state=tk.DISABLED)
            self.existing_csv_status_var.set(
                f"{n} existing row(s) loaded — output unit locked to "
                f"{data['output_unit']!r} to match this file.")
        else:
            self.unit_combo.configure(state="readonly")
            self.existing_csv_status_var.set(
                f"{n} existing row(s) loaded — file has no recorded output "
                f"unit, so you can still choose one below.")

    def _clear_existing_csv_selection(self):
        """Forgets the chosen CSV and unlocks the output unit again."""
        self._existing_data = None
        self.existing_csv_status_var.set("")
        self.unit_combo.configure(state="readonly")

    def _set_status(self, text: str):
        """Replaces the one-line status message above the log."""
        self.status_var.set(text)

    def _append_log(self, text: str):
        """Adds a line to the log panel and scrolls to it. The widget is
        read-only to the user, so it's unlocked just long enough to write."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self):
        """Empties the log panel at the start of a run."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _on_status_main(self, msg: str):
        """on_status callback for Phase A/C — both run on the main thread,
        so this can touch widgets directly (no queue needed)."""
        self._append_log(msg)
        self._set_status(msg)

    def _show_banner(self, text: str, buttons=None):
        """buttons: list of (label, callback) or None for an info-only banner."""
        self.banner_var.set(text)
        for w in self.banner_btn_frame.winfo_children():
            w.destroy()
        if buttons:
            for label, cb in buttons:
                ttk.Button(self.banner_btn_frame, text=label, command=cb).pack(
                    side=tk.LEFT, padx=4)
        self.banner_frame.pack(fill=tk.X, padx=8, pady=(0, 6), before=self.prog_frame)

    def _hide_banner(self):
        """Takes the prompt strip down once its transient tab has closed."""
        self.banner_frame.pack_forget()

    # ================= run lifecycle =================

    def _on_run_clicked(self):
        """Starts a batch. Validates the input folder and any existing-CSV
        selection, resets the log and progress display, sets up this run's
        folders and logging, then kicks off phase A (the profile audit)."""
        if self._running:
            return
        if self.add_existing_var.get() and not self._existing_data:
            self._append_log("Add to existing data file is checked, but no "
                             "valid CSV is loaded — pick one first.")
            return

        config = self.session.config
        config.input_dir = self.input_dir_var.get()

        self._add_mode = bool(self.add_existing_var.get() and self._existing_data)
        if self._add_mode:
            # Re-read from disk rather than trusting the snapshot cached at
            # Browse-time: if a previous run in this session already
            # appended into this same file (without the user re-Browsing),
            # the cached snapshot predates that append and would dedupe/
            # merge against stale contents — silently dropping the earlier
            # run's rows when this run's append overwrites the file.
            try:
                fresh = engine.load_existing_data_file(self._existing_data["path"])
            except Exception as e:
                self._append_log(f"Could not re-read the existing data file: {e}")
                return
            fresh["path"] = self._existing_data["path"]
            self._existing_data = fresh

            # Locked to the existing file's unit — never the user's combo
            # selection, even if it's still enabled (no recorded unit case).
            config.output_unit = fresh["output_unit"] or self.unit_var.get()
            self.unit_var.set(config.output_unit)
            self._existing_csv_path = fresh["path"]
            self._existing_rows = list(fresh["rows"])
            self._existing_names = set(fresh["names"])
        else:
            config.output_unit = self.unit_var.get()

        self._running = True
        self.run_btn.configure(state=tk.DISABLED)
        self._clear_log()
        self._hide_banner()
        self.success_label.pack_forget()
        self.progress_container.pack(fill=tk.X)

        self.run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._append_log(f"Batch analysis started {self.run_ts}")
        log_path, _counter, self._run_log_handlers = engine.setup_run_logging(
            config, self.run_ts)
        self._append_log(f"Log file: {log_path}")

        self._set_status("Listing images…")
        images = engine.list_images(config)
        if not images:
            self._append_log(f"No images found in {config.input_dir!r} — place "
                             f"exported plot images there and run again.")
            self._set_status("No images found.")
            self._end_run()
            return

        if self._add_mode:
            before = len(images)
            images = [p for p in images
                     if os.path.basename(p) not in self._existing_names]
            skipped_n = before - len(images)
            if skipped_n:
                self._append_log(f"{skipped_n} image(s) already present in "
                                 f"the existing data file — skipped.")
            if not images:
                self._append_log(f"All {before} image(s) are already in the "
                                 f"existing data file — nothing new to add.")
                self._set_status("Nothing new to add.")
                self._end_run()
                return

        self.images = images
        self.run_dir = os.path.join(config.image_fits_dir, self.run_ts)
        os.makedirs(self.run_dir, exist_ok=True)
        self.buffer, self.max_gap = engine.extraction_params(config)
        self.session.extraction_buffer = self.buffer
        self.session.extraction_max_gap = self.max_gap

        self._set_status(f"Checking color profiles for {len(images)} image(s)…")
        self.after(10, self._do_profile_audit)

    # ---- Phase A: profile audit (main thread, interactive) ----

    def _do_profile_audit(self):
        """Phase A: make sure every image has a matching color profile. Runs
        on the main thread because it may need to open the calibration tab and
        ask the user to pick colors. Hands off to the analysis thread when
        every image is covered."""
        config = self.session.config

        def on_unmatched(rgb, default_name):
            """Opens the color-picking tab for an image with no profile and
            returns the new profile, or None if the user cancelled."""
            self._show_banner(
                "This image's color layout doesn't match any saved profile. "
                "Pick background/plot/axis/gridline colors in the tab that "
                "just opened, then Finish (or Cancel to skip this image).")

            def build(frame, on_done):
                """Fills the transient tab with this step's form."""
                launch_calibration_gui_embedded(frame, rgb, on_done,
                                                default_name=default_name)

            result = run_tab_and_wait(self.notebook, "Create profile", build)
            self._hide_banner()
            return result

        hists, skipped, created = engine.resolve_profile_audit(
            config, self.images, self._on_status_main, on_unmatched)
        if created:
            self._append_log(f"{len(created)} new color profile(s) saved.")
        self.hists = hists
        self.skipped_set = set(skipped)
        self.profiles_list = load_profiles(config.profiles_path)
        self._start_analysis_thread()

    # ---- Phase B: per-image analysis (worker thread, pure compute) ----

    def _start_analysis_thread(self):
        """Phase B: run the per-image analysis on a worker thread so the
        window stays responsive, and start draining the progress queue. The
        worker only computes — every widget update happens on the main
        thread via that queue."""
        n = len(self.images)
        self.progress.configure(maximum=n, value=0)
        self._phase_b_start = time.monotonic()
        self._set_status(f"Analyzing 0/{n}…")
        self.eta_var.set("")

        self._log_handler = _QueueLogHandler(self._q)
        logging.getLogger().addHandler(self._log_handler)

        config, images, hists = self.session.config, self.images, self.hists
        profiles_list, skipped_set = self.profiles_list, self.skipped_set
        run_dir, buffer, max_gap = self.run_dir, self.buffer, self.max_gap

        def on_progress(i, n, name):
            """Queues a progress update before each image is analyzed."""
            self._q.put(("progress", i, n, name, time.monotonic()))

        def on_log(msg):
            """Queues an informational line from the worker for the panel."""
            self._q.put(("log", msg))

        def worker():
            """Runs the whole per-image analysis off-thread and queues the
            result. Any unexpected failure is queued as an error rather than
            dying silently on a thread nobody is watching."""
            rows, failed_items, ctx_by_path = engine.run_batch(
                config, images, hists, profiles_list, skipped_set,
                run_dir, buffer, max_gap, on_progress=on_progress,
                on_log=on_log)
            self._q.put(("batch_done", rows, failed_items, ctx_by_path))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._pump_queue)

    def _pump_queue(self):
        """Drains whatever the worker has queued — progress, log lines, the
        finished result — and updates the UI, then reschedules itself. This is
        the only place worker output reaches a widget, which is what keeps Tk
        on one thread."""
        try:
            while True:
                item = self._q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "progress":
                    _, i, n, name, ts = item
                    self.progress.configure(maximum=n, value=i)
                    elapsed = ts - self._phase_b_start
                    eta = (elapsed / i) * (n - i) if i > 0 else 0.0
                    self._set_status(f"Analyzing {i}/{n}: {name}")
                    self.eta_var.set(
                        f"Elapsed {format_hms(elapsed)}   ETA {format_hms(eta)}")
                elif kind == "batch_done":
                    logging.getLogger().removeHandler(self._log_handler)
                    self._log_handler = None
                    _, rows, failed_items, ctx_by_path = item
                    self._on_batch_done(rows, failed_items, ctx_by_path)
                    return  # next phase drives its own loop; stop polling
        except queue.Empty:
            pass
        if self._running:
            self.after(100, self._pump_queue)

    # ---- Phase C: manual correction (main thread, interactive, optional) ----

    def _on_batch_done(self, rows, failed_items, ctx_by_path):
        """Phase B finished: store the results on the session and decide what
        comes next — offer manual review when images failed, otherwise go
        straight to writing the CSV."""
        self.progress.configure(value=len(self.images))
        self.session.run_ts = self.run_ts
        self.session.run_dir = self.run_dir
        self.session.image_paths = list(self.images)
        self.session.rows = rows
        self.session.ctx_by_path = ctx_by_path
        self.session.notify_results_changed()

        if not failed_items:
            self._finalize_run(rows)
            return

        self._pending_failed_items = failed_items
        self._pending_rows = rows
        fail_dir = engine.copy_failed_images(self.session.config, failed_items,
                                             self.run_ts)
        self._append_log(f"{len(failed_items)} failed image(s) copied to {fail_dir}")
        self._show_banner(
            f"{len(failed_items)} image(s) failed automatic analysis. "
            f"Review and correct them now?",
            buttons=[("Review now", self._start_manual_review),
                    ("Skip (keep as errors)", self._skip_manual_review)])

    def _skip_manual_review(self):
        """Skips the correction pass; failed images keep their error rows."""
        self._hide_banner()
        self._append_log("Skipping manual correction — failures stay as error rows.")
        self._finalize_run(self._pending_rows)

    def _start_manual_review(self):
        """Phase C: walk the failed images one at a time, opening the
        manual-fit form for each in a transient tab and merging whatever the
        user enters back into that image's row."""
        self._hide_banner()
        failed_items, rows = self._pending_failed_items, self._pending_rows
        config = self.session.config

        def on_review_one(path, name, prefill):
            """Opens the manual-fit form for one failed image and returns its
            (action, data) result."""
            self._show_banner(
                f"Manual fit — {name}: complete it in the tab that just "
                f"opened (every field is optional).")

            def build(frame, on_done):
                """Fills the transient tab with this step's form."""
                rgb = engine.load_rgb(path)
                launch_manual_fit_embedded(frame, rgb, name, prefill,
                                           config.output_unit, on_done)

            result = run_tab_and_wait(self.notebook, f"Manual fit — {name}", build)
            self._hide_banner()
            return result if result is not None else ("skipped", None)

        engine.manual_review_phase(config, failed_items, rows, self.run_dir,
                                   self.buffer, self.max_gap,
                                   self._on_status_main, on_review_one)
        self.session.rows = rows
        self.session.notify_results_changed()
        self._finalize_run(rows)

    # ---- Phase D: results CSV ----

    def _finalize_run(self, rows):
        """Phase D: write the results CSV — either a new file for this run or
        merged into the existing one being added to — then tell the other tabs
        the results are ready and report where everything landed."""
        engine.log_lambda_c_capping(rows)
        if self._add_mode:
            csv_path = engine.append_results_to_csv(
                self._existing_csv_path, self._existing_rows, rows, backup=True)
            self._append_log(f"Merged into existing data file (backup saved "
                             f"alongside it): {csv_path}")
            # Imported rows first so the Dashboard's default order matches
            # the file's — they carry '_imported' from load_existing_data_file
            # (dashboard_tab grays them; session.image_paths, set in
            # _on_batch_done, stays just this run's images, so the Viewer
            # can't navigate to them).
            self.session.rows = self._existing_rows + rows
        else:
            csv_path = engine.write_results_csv(self.session.config, rows, self.run_ts)
            self.session.rows = rows
        self.session.csv_path = csv_path
        self.session.notify_results_changed()

        n_failed = sum(1 for r in rows
                       if isinstance(r.get("error"), str) and r["error"])
        ok = len(self.images) - n_failed
        self._set_status(f"Done: {ok}/{len(self.images)} analyzed, {n_failed} failed.")
        self._append_log(f"Results CSV: {csv_path}")
        self._append_log(f"Overlays: {self.run_dir}")

        self.progress_container.pack_forget()
        self.success_var.set(
            f"✓ Batch successful — {ok}/{len(self.images)} analyzed"
            + (f", {n_failed} failed" if n_failed else ""))
        self.success_label.pack(fill=tk.X)

        self._end_run()

    def _end_run(self):
        """Puts the tab back to its idle state: re-enable the controls, stop
        the progress bar, and detach this run's log handlers so a later run
        doesn't keep writing into the finished run's file."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        if self._run_log_handlers:
            engine.teardown_run_logging(self._run_log_handlers)
            self._run_log_handlers = []
        self._running = False
        self.run_btn.configure(state=tk.NORMAL)
