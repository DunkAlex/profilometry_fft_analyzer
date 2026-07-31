"""
tuning_gui.py

Two-tab interactive tool for the profilometry image-digitization + Gaussian
regression filtering pipeline.

Tab 1 - Calibration: set up crop area, curve color, and independent X/Y
         pixel-to-unit scales (each can have its own unit).
Tab 2 - Tuning: live diagnostic plots, filtering parameters, JSON export.

Assumes it lives in the same folder as profilometry_fft_analyzer.py and
reuses that module's corner/color/calibration selection popups directly, so
behavior (including the Confirm/Redo buttons) stays in one place.

Run directly:
    python tuning_gui.py
"""

import json
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
import cv2


from profilometry_fft_analyzer import (
    IMAGE_NAME,
    get_image,
    crop_corners,
    set_figure_bounds,
    load_default_corners,
    select_plot_color,
    load_default_calibration,
    save_default_calibration,
    pick_calibration_points,
    largest_contiguous_run,
    detrend_profile,
    lambda_c_to_sigma,
    compute_roughness_params,
    compute_rz,
)

# Module-level toggles mirroring profilometry_fft_analyzer.py's style.
USE_DEFAULT_AREA = False
USE_DEFAULT_COLOR = False

UNIT_OPTIONS = ['nm', 'um', 'mm', 'cm', 'in', 'px']


# ---------------------------------------------------------------------
# Extraction split into two stages so slider moves don't recompute the
# expensive per-pixel color-distance array every time.
# ---------------------------------------------------------------------

def compute_color_distance(image, plot_rgb):
    plot_rgb = np.array(plot_rgb, dtype=np.float64)
    diff = image.astype(np.float64) - plot_rgb
    return np.sqrt(np.sum(diff ** 2, axis=2))  # shape (h, w)


def extract_from_distance(dist, buffer, max_gap, interp_gaps=True):
    h, w = dist.shape
    y_indices = np.arange(h)

    x_vals, y_vals, skipped_x = [], [], []

    for x in range(w):
        col_dist = dist[:, x]
        mask = col_dist <= buffer

        if not np.any(mask):
            skipped_x.append(x)
            continue

        matched_y = y_indices[mask]
        matched_dist = col_dist[mask]
        matched_y, matched_dist = largest_contiguous_run(matched_y, matched_dist, max_gap=max_gap)

        weights = np.clip(buffer - matched_dist, a_min=1e-6, a_max=None)
        centroid_y = np.sum(matched_y * weights) / np.sum(weights)

        x_vals.append(x)
        y_vals.append(centroid_y)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    n_skipped = len(skipped_x)

    if interp_gaps and n_skipped > 0 and len(x_vals) > 1:
        full_x = np.arange(w)
        y_vals = np.interp(full_x, x_vals, y_vals)
        x_vals = full_x

    return x_vals, (h - 1) - y_vals, n_skipped


def _hex_from_rgb(rgb):
    r, g, b = (int(v) for v in rgb)
    return f'#{r:02x}{g:02x}{b:02x}'


class ProfilometryTunerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Profilometry Calibration + Tuning GUI")
        self.geometry("1300x920")
        self.minsize(1100, 800)

        # ---- Load image + do (one-time, possibly interactive) setup ----
        self.full_image = get_image(IMAGE_NAME)

        if USE_DEFAULT_AREA:
            corners = load_default_corners()
            if corners is None:
                corners = set_figure_bounds(self.full_image)
        else:
            corners = set_figure_bounds(self.full_image)
        self.corners = corners
        self.cropped = crop_corners(self.full_image, self.corners)

        self.plot_rgb = select_plot_color(self.cropped, load_default_color_flag=USE_DEFAULT_COLOR)

        calib = load_default_calibration()
        if calib is not None:
            self.dx_per_px = calib['dx_per_px']
            self.unit_x = calib['unit_x']
            self.dy_per_px = calib['dy_per_px']
            self.unit_y = calib['unit_y']
        else:
            self.dx_per_px = 0.1
            self.unit_x = 'um'
            self.dy_per_px = 0.1
            self.unit_y = 'um'

        self.h, self.w = self.cropped.shape[:2]
        self.dist = compute_color_distance(self.cropped, self.plot_rgb)

        # ---- Tk variables ----
        self.buffer_var = tk.DoubleVar(value=140)
        self.max_gap_var = tk.IntVar(value=5)
        self.lambda_c_log_var = tk.DoubleVar(value=np.log10(10.0))

        self.unit_x_var = tk.StringVar(value=self.unit_x)
        self.unit_y_var = tk.StringVar(value=self.unit_y)
        self.known_x_var = tk.StringVar(value="10.0")
        self.known_y_var = tk.StringVar(value="10.0")

        self._build_notebook()
        self._refresh_calibration_preview()
        self._redraw()

    # ==================================================================
    # Notebook / tab scaffolding
    # ==================================================================
    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.calib_tab = ttk.Frame(self.notebook)
        self.tuning_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.calib_tab, text="1. Calibration")
        self.notebook.add(self.tuning_tab, text="2. Tuning / Diagnostics")

        self._build_calibration_tab()
        self._build_tuning_tab()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, _event=None):
        if self.notebook.index(self.notebook.select()) == 1:
            self._redraw()

    # ==================================================================
    # Calibration tab
    # ==================================================================
    def _build_calibration_tab(self):
        frame = self.calib_tab

        area_frame = ttk.LabelFrame(frame, text="1. Crop area")
        area_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        color_frame = ttk.LabelFrame(frame, text="2. Curve color")
        color_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        x_calib_frame = ttk.LabelFrame(frame, text="3. X-axis calibration")
        x_calib_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        y_calib_frame = ttk.LabelFrame(frame, text="4. Y-axis calibration")
        y_calib_frame.grid(row=0, column=3, sticky="nsew", padx=10, pady=10)

        frame.grid_columnconfigure(0, weight=2)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        frame.grid_columnconfigure(3, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        # --- Area section ---
        self.calib_fig = Figure(figsize=(5, 4), dpi=100)
        self.calib_ax = self.calib_fig.add_subplot(1, 1, 1)
        self.calib_canvas = FigureCanvasTkAgg(self.calib_fig, master=area_frame)
        self.calib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        ttk.Button(area_frame, text="Reset crop area", command=self._reset_area).pack(pady=(0, 8))

        # --- Color section ---
        self.color_swatch = tk.Canvas(color_frame, width=70, height=70, highlightthickness=1,
                                       highlightbackground="black")
        self.color_swatch.pack(pady=(10, 6))
        self.color_label_var = tk.StringVar()
        ttk.Label(color_frame, textvariable=self.color_label_var).pack(pady=(0, 10))
        ttk.Button(color_frame, text="Reset color", command=self._reset_color).pack()
        self._refresh_color_swatch()

        # --- X-axis calibration ---
        self._build_axis_calib_section(
            x_calib_frame, self.unit_x_var, self.known_x_var, self._run_x_calibration)
        self.x_calib_status_var = tk.StringVar()
        ttk.Label(x_calib_frame, textvariable=self.x_calib_status_var, wraplength=180,
                  justify="left").pack(anchor="w", padx=10, pady=(4, 10))

        # --- Y-axis calibration ---
        self._build_axis_calib_section(
            y_calib_frame, self.unit_y_var, self.known_y_var, self._run_y_calibration)
        self.y_calib_status_var = tk.StringVar()
        ttk.Label(y_calib_frame, textvariable=self.y_calib_status_var, wraplength=180,
                  justify="left").pack(anchor="w", padx=10, pady=(4, 10))

        self._refresh_calibration_status()

    def _build_axis_calib_section(self, parent, unit_var, known_var, command):
        ttk.Label(parent, text="Unit:").pack(anchor="w", padx=10, pady=(10, 0))
        ttk.Combobox(parent, textvariable=unit_var, values=UNIT_OPTIONS,
                     state="readonly", width=10).pack(anchor="w", padx=10, pady=(0, 10))

        ttk.Label(parent, text="Known distance value:").pack(anchor="w", padx=10)
        ttk.Entry(parent, textvariable=known_var, width=12).pack(anchor="w", padx=10, pady=(0, 10))

        ttk.Button(parent, text="Pick points on raw image",
                   command=command).pack(padx=10, pady=(0, 10), fill=tk.X)

    def _refresh_calibration_preview(self):
        ax = self.calib_ax
        ax.clear()
        ax.imshow(self.full_image)
        (x1, y1), (x2, y2) = self.corners
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                          linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.set_title("Current default crop area")
        ax.axis('off')
        self.calib_canvas.draw()

    def _reset_area(self):
        self.corners = set_figure_bounds(self.full_image)
        self.cropped = crop_corners(self.full_image, self.corners)
        self.h, self.w = self.cropped.shape[:2]
        self.dist = compute_color_distance(self.cropped, self.plot_rgb)
        self._refresh_calibration_preview()
        self._redraw()

    def _refresh_color_swatch(self):
        self.color_swatch.configure(bg=_hex_from_rgb(self.plot_rgb))
        r, g, b = (int(v) for v in self.plot_rgb)
        self.color_label_var.set(f"RGB: ({r}, {g}, {b})")

    def _reset_color(self):
        self.plot_rgb = select_plot_color(self.cropped, load_default_color_flag=False)
        self.dist = compute_color_distance(self.cropped, self.plot_rgb)
        self._refresh_color_swatch()
        self._redraw()

    def _refresh_calibration_status(self):
        self.x_calib_status_var.set(f"Current: {self.dx_per_px:.5g} {self.unit_x}/px")
        self.y_calib_status_var.set(f"Current: {self.dy_per_px:.5g} {self.unit_y}/px")

    def _persist_calibration(self):
        save_default_calibration(self.dx_per_px, self.unit_x, self.dy_per_px, self.unit_y)

    def _run_x_calibration(self):
        try:
            known_value = float(self.known_x_var.get())
            if known_value <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid value", "Enter a positive numeric known distance.")
            return

        # Calibration is performed on the RAW (uncropped) image, since axis
        # ticks/gridlines used as reference are typically outside the
        # cropped curve region.
        try:
            pixel_distance = pick_calibration_points(self.full_image, axis_label="X")
        except ValueError as e:
            messagebox.showerror("Calibration cancelled", str(e))
            return

        if pixel_distance <= 0:
            messagebox.showerror("Invalid points", "The two points must not be identical.")
            return

        self.dx_per_px = known_value / pixel_distance
        self.unit_x = self.unit_x_var.get()
        self._persist_calibration()
        self._refresh_calibration_status()
        self._redraw()

    def _run_y_calibration(self):
        try:
            known_value = float(self.known_y_var.get())
            if known_value <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid value", "Enter a positive numeric known distance.")
            return

        try:
            pixel_distance = pick_calibration_points(self.full_image, axis_label="Y")
        except ValueError as e:
            messagebox.showerror("Calibration cancelled", str(e))
            return

        if pixel_distance <= 0:
            messagebox.showerror("Invalid points", "The two points must not be identical.")
            return

        self.dy_per_px = known_value / pixel_distance
        self.unit_y = self.unit_y_var.get()
        self._persist_calibration()
        self._refresh_calibration_status()
        self._redraw()

    # ==================================================================
    # Tuning tab
    # ==================================================================
    def _build_tuning_tab(self):
        frame = self.tuning_tab

        controls = ttk.Frame(frame)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.buffer_value_label = self._add_slider(
            controls, "Color buffer (match threshold)", self.buffer_var, 20, 250, row=0)
        self.maxgap_value_label = self._add_slider(
            controls, "Max gap (contiguous run, px)", self.max_gap_var, 1, 20, row=1)
        self.lambdac_value_label = self._add_log_slider(
            controls, "Lambda_c (cutoff wavelength, x-unit)", self.lambda_c_log_var,
            real_min=0.05, real_max=100, row=2)

        self.calib_display_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.calib_display_var, font=("Consolas", 9)).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        export_btn = ttk.Button(controls, text="Export tuned parameters to JSON",
                                 command=self._export_params)
        export_btn.grid(row=4, column=0, columnspan=3, pady=(10, 0))

        self.stats_var = tk.StringVar(value="")
        stats_label = ttk.Label(frame, textvariable=self.stats_var, font=("Consolas", 10))
        stats_label.pack(side=tk.BOTTOM, pady=(4, 0))

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax_overlay = self.fig.add_subplot(2, 2, 1)
        self.ax_spectrum = self.fig.add_subplot(2, 2, 2)
        self.ax_trace = self.fig.add_subplot(2, 2, 3)
        self.ax_roughness = self.fig.add_subplot(2, 2, 4)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _add_slider(self, parent, label_text, var, frm, to, row):
        ttk.Label(parent, text=label_text, width=32).grid(row=row, column=0, sticky="w")
        value_label = ttk.Label(parent, width=8)
        value_label.grid(row=row, column=2, sticky="w")

        def on_change(_event=None):
            value_label.config(text=f"{var.get():.3g}")
            self._redraw()

        scale = ttk.Scale(parent, from_=frm, to=to, orient="horizontal",
                           variable=var, command=on_change, length=400)
        scale.grid(row=row, column=1, sticky="ew", padx=8)
        parent.grid_columnconfigure(1, weight=1)
        value_label.config(text=f"{var.get():.3g}")
        return value_label

    def _add_log_slider(self, parent, label_text, log_var, real_min, real_max, row):
        ttk.Label(parent, text=label_text, width=32).grid(row=row, column=0, sticky="w")
        value_label = ttk.Label(parent, width=8)
        value_label.grid(row=row, column=2, sticky="w")

        def on_change(_event=None):
            real_val = 10 ** log_var.get()
            value_label.config(text=f"{real_val:.4g}")
            self._redraw()

        scale = ttk.Scale(parent, from_=np.log10(real_min), to=np.log10(real_max),
                           orient="horizontal", variable=log_var, command=on_change, length=400)
        scale.grid(row=row, column=1, sticky="ew", padx=8)
        parent.grid_columnconfigure(1, weight=1)
        value_label.config(text=f"{10 ** log_var.get():.4g}")
        return value_label

    def _export_params(self):
        lambda_c = 10 ** self.lambda_c_log_var.get()
        params = {
            "color_buffer": round(self.buffer_var.get(), 3),
            "max_gap_px": int(round(self.max_gap_var.get())),
            "lambda_c": round(lambda_c, 5),
            "unit_x": self.unit_x,
            "dx_per_px": round(self.dx_per_px, 6),
            "unit_y": self.unit_y,
            "dy_per_px": round(self.dy_per_px, 6),
        }

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="tuned_params.json",
            title="Export tuned parameters",
        )
        if not path:
            return

        with open(path, "w") as f:
            json.dump(params, f, indent=2)

        messagebox.showinfo("Export complete", f"Saved tuned parameters to:\n{path}")

    # ------------------------------------------------------------------
    def _redraw(self):
        buffer = self.buffer_var.get()
        max_gap = int(round(self.max_gap_var.get()))
        lambda_c = 10 ** self.lambda_c_log_var.get()
        dx = max(self.dx_per_px, 1e-9)
        dy = max(self.dy_per_px, 1e-9)
        unit_x = self.unit_x
        unit_y = self.unit_y

        self.calib_display_var.set(
            f"X: {dx:.5g} {unit_x}/px    Y: {dy:.5g} {unit_y}/px    (set on Calibration tab)"
        )

        x_px, y_px, n_skipped = extract_from_distance(self.dist, buffer, max_gap, interp_gaps=True)

        if len(x_px) < 3:
            self.stats_var.set("Not enough matched pixels - lower the color buffer.")
            self.canvas.draw()
            return

        sigma_samples = lambda_c_to_sigma(lambda_c, dx)

        waviness_px = ndimage.gaussian_filter1d(y_px, sigma=sigma_samples, mode='reflect')

        y_phys = y_px * dy
        y_detrended = detrend_profile(y_phys)
        waviness_phys = ndimage.gaussian_filter1d(y_detrended, sigma=sigma_samples, mode='reflect')

        results = compute_roughness_params(y_detrended, waviness_phys)
        Ra, Rq, roughness = results["Ra"], results["Rq"], results["roughness_profile"]
        Rz = compute_rz(roughness, dx, sampling_length_um=lambda_c * 5)

        windowed = y_detrended * np.hanning(len(y_detrended))
        Y = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), d=dx)
        mag = np.abs(Y)

        # ---------------- Plot 1: image overlay ----------------
        ax = self.ax_overlay
        ax.clear()
        ax.imshow(self.cropped)
        img_row_raw = (self.h - 1) - y_px
        img_row_wavy = (self.h - 1) - waviness_px
        ax.plot(x_px, img_row_raw, color='yellow', linewidth=1, label='Extracted trace')
        ax.plot(x_px, img_row_wavy, color='red', linewidth=1.5, label='Filtered trend')
        ax.set_title('Image overlay')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=7)

        # ---------------- Plot 2: frequency spectrum ----------------
        ax = self.ax_spectrum
        ax.clear()
        ax.plot(freqs, mag, color='steelblue', linewidth=1)
        if lambda_c > 0:
            cutoff_freq = 1 / lambda_c
            ax.axvline(cutoff_freq, color='red', linestyle='--', linewidth=1,
                       label=f'cutoff = {cutoff_freq:.3f} 1/{unit_x}')
            ax.legend(fontsize=7)
        ax.set_yscale('log')
        ax.set_title('Frequency spectrum')
        ax.set_xlabel(f'Frequency (1/{unit_x})')
        ax.set_ylabel('Magnitude')

        # ---------------- Plot 3: raw vs waviness ----------------
        ax = self.ax_trace
        ax.clear()
        x_phys = np.arange(len(y_detrended)) * dx
        ax.plot(x_phys, y_detrended, color='gray', linewidth=0.7, alpha=0.7, label='Raw (detrended)')
        ax.plot(x_phys, waviness_phys, color='red', linewidth=1.5, label='Waviness (form)')
        ax.set_title('Detrended profile vs. waviness fit')
        ax.set_xlabel(f'Position ({unit_x})')
        ax.set_ylabel(f'Height ({unit_y})')
        ax.legend(fontsize=7)

        # ---------------- Plot 4: roughness residual ----------------
        ax = self.ax_roughness
        ax.clear()
        ax.plot(x_phys, roughness, color='steelblue', linewidth=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(Ra, color='green', linestyle='--', linewidth=0.8, label=f'Ra={Ra:.3f} {unit_y}')
        ax.axhline(-Ra, color='green', linestyle='--', linewidth=0.8)
        ax.axhline(Rq, color='orange', linestyle=':', linewidth=0.8, label=f'Rq={Rq:.3f} {unit_y}')
        ax.axhline(-Rq, color='orange', linestyle=':', linewidth=0.8)
        ax.set_title('Roughness residual')
        ax.set_xlabel(f'Position ({unit_x})')
        ax.set_ylabel(f'Deviation ({unit_y})')
        ax.legend(fontsize=7)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

        self.stats_var.set(
            f"Ra = {Ra:.4f} {unit_y}   Rq = {Rq:.4f} {unit_y}   Rz = {Rz:.4f} {unit_y}   "
            f"| columns interpolated: {n_skipped}/{self.w}   | sigma = {sigma_samples:.2f} samples"
        )


if __name__ == "__main__":
    app = ProfilometryTunerApp()
    app.mainloop()