"""
tuning_gui.py

Interactive tuning GUI for the profilometry image-digitization + Gaussian
regression filtering pipeline.

Assumes it lives in the same folder as profilometry_fft_analyzer.py and
reuses that module's corner/color selection (so first run will prompt you
to click plot corners + curve color, same as your existing workflow;
subsequent runs reuse the saved defaults automatically).

Run directly:
    python tuning_gui.py
"""

import json
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage

from profilometry_fft_analyzer import (
    IMAGE_NAME,
    import_and_crop,
    select_plot_color,
    largest_contiguous_run,
    detrend_profile,
    lambda_c_to_sigma,
    compute_roughness_params,
    compute_rz,
)


# ---------------------------------------------------------------------
# Extraction split into two stages so slider moves don't recompute the
# expensive per-pixel color-distance array every time - only the cheap
# thresholding/run-filtering step re-runs live.
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

    # Return in "plot" convention (y increases upward), matching extract_trend
    return x_vals, (h - 1) - y_vals, n_skipped


class ProfilometryTunerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Profilometry Tuning GUI")
        self.geometry("1200x850")
        self.minsize(1000, 750)

        # ---- Load image + do (one-time, possibly interactive) setup ----
        self.cropped = import_and_crop(IMAGE_NAME, load_default_area_flag=True)
        self.plot_rgb = select_plot_color(self.cropped, load_default_color_flag=True)
        self.h, self.w = self.cropped.shape[:2]
        self.dist = compute_color_distance(self.cropped, self.plot_rgb)

        # ---- Tk variables backing the sliders ----
        # Buffer/max_gap stay linear - their useful range is a single order
        # of magnitude. Ranges widened (with headroom) based on observed
        # tuning: buffer settled near the top of the old 5-150 range, so the
        # floor is raised and the ceiling extended rather than just centering
        # on the observed value.
        self.buffer_var = tk.DoubleVar(value=140)
        self.max_gap_var = tk.IntVar(value=5)

        # Lambda_c and dx both span multiple orders of magnitude and, in
        # testing, pinned against the bottom of their old linear ranges
        # (lambda_c hit its floor at 0.5 um; dx sat at ~5% of its range).
        # A linear slider wastes almost all its resolution on values that
        # will never be used. Stored here as log10(value) so the ttk.Scale
        # itself is linear in log-space - equal slider travel now means
        # equal *multiplicative* change in the real parameter, giving fine
        # control near the low end without losing reach at the high end.
        self.lambda_c_log_var = tk.DoubleVar(value=np.log10(10.0))
        self.dx_log_var = tk.DoubleVar(value=np.log10(0.1))

        self._build_layout()
        self._redraw()

    # ------------------------------------------------------------------
    def _build_layout(self):
        # --- Controls frame (built and packed FIRST so it always reserves
        # its required space at the bottom, regardless of window size -
        # otherwise an expand=True canvas packed first can push the export
        # button below the visible window). ---
        controls = ttk.Frame(self)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self._add_slider(controls, "Color buffer (match threshold)", self.buffer_var, 20, 250, row=0)
        self._add_slider(controls, "Max gap (contiguous run, px)", self.max_gap_var, 1, 20, row=1)
        self._add_log_slider(controls, "Lambda_c (cutoff wavelength, um)", self.lambda_c_log_var,
                              real_min=0.05, real_max=100, row=2)
        self._add_log_slider(controls, "dx (um/pixel, placeholder)", self.dx_log_var,
                              real_min=0.005, real_max=5, row=3)

        export_btn = ttk.Button(controls, text="Export tuned parameters to JSON",
                                 command=self._export_params)
        export_btn.grid(row=4, column=0, columnspan=3, pady=(10, 0))

        # --- Stats label (packed next, also anchored to bottom, sits just
        # above the controls frame) ---
        self.stats_var = tk.StringVar(value="")
        stats_label = ttk.Label(self, textvariable=self.stats_var, font=("Consolas", 10))
        stats_label.pack(side=tk.BOTTOM, pady=(4, 0))

        # --- Figure / canvas (packed LAST, takes whatever space remains) ---
        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax_overlay = self.fig.add_subplot(2, 2, 1)
        self.ax_spectrum = self.fig.add_subplot(2, 2, 2)
        self.ax_trace = self.fig.add_subplot(2, 2, 3)
        self.ax_roughness = self.fig.add_subplot(2, 2, 4)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
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

    def _add_log_slider(self, parent, label_text, log_var, real_min, real_max, row):
        """Slider whose backing Tk variable holds log10(real value). The
        widget itself is linear in log-space, giving equal resolution per
        decade instead of concentrating almost all slider travel on values
        far from where the parameter actually needs to sit."""
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

    def _export_params(self):
        params = {
            "color_buffer": round(self.buffer_var.get(), 3),
            "max_gap_px": int(round(self.max_gap_var.get())),
            "lambda_c_um": round(10 ** self.lambda_c_log_var.get(), 5),
            "dx_um_per_px": round(10 ** self.dx_log_var.get(), 5),
        }

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="tuned_params.json",
            title="Export tuned parameters",
        )
        if not path:
            return  # user cancelled

        with open(path, "w") as f:
            json.dump(params, f, indent=2)

        messagebox.showinfo("Export complete", f"Saved tuned parameters to:\n{path}")

    # ------------------------------------------------------------------
    def _redraw(self):
        buffer = self.buffer_var.get()
        max_gap = int(round(self.max_gap_var.get()))
        lambda_c = 10 ** self.lambda_c_log_var.get()
        dx = max(10 ** self.dx_log_var.get(), 1e-6)  # guard against zero

        # --- Extraction (cheap re-threshold of cached distance array) ---
        x_px, y_px, n_skipped = extract_from_distance(self.dist, buffer, max_gap, interp_gaps=True)

        if len(x_px) < 3:
            self.stats_var.set("Not enough matched pixels - lower the color buffer.")
            self.canvas.draw()
            return

        # --- Pixel-space smoothed curve, for the image overlay ---
        sigma_samples = lambda_c_to_sigma(lambda_c, dx)
        waviness_px = ndimage.gaussian_filter1d(y_px, sigma=sigma_samples, mode='reflect')

        # --- Physical-unit pipeline, for roughness stats ---
        y_um = y_px * dx
        y_detrended = detrend_profile(y_um)
        waviness_um = ndimage.gaussian_filter1d(y_detrended, sigma=sigma_samples, mode='reflect')

        results = compute_roughness_params(y_detrended, waviness_um)
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
                       label=f'cutoff = {cutoff_freq:.3f} 1/um')
            ax.legend(fontsize=7)
        ax.set_yscale('log')
        ax.set_title('Frequency spectrum')
        ax.set_xlabel('Frequency (1/um)')
        ax.set_ylabel('Magnitude')

        # ---------------- Plot 3: raw vs waviness (um) ----------------
        ax = self.ax_trace
        ax.clear()
        x_um = np.arange(len(y_detrended)) * dx
        ax.plot(x_um, y_detrended, color='gray', linewidth=0.7, alpha=0.7, label='Raw (detrended)')
        ax.plot(x_um, waviness_um, color='red', linewidth=1.5, label='Waviness (form)')
        ax.set_title('Detrended profile vs. waviness fit')
        ax.set_xlabel('Position (um)')
        ax.set_ylabel('Height (um)')
        ax.legend(fontsize=7)

        # ---------------- Plot 4: roughness residual ----------------
        ax = self.ax_roughness
        ax.clear()
        ax.plot(x_um, roughness, color='steelblue', linewidth=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(Ra, color='green', linestyle='--', linewidth=0.8, label=f'Ra={Ra:.3f} um')
        ax.axhline(-Ra, color='green', linestyle='--', linewidth=0.8)
        ax.axhline(Rq, color='orange', linestyle=':', linewidth=0.8, label=f'Rq={Rq:.3f} um')
        ax.axhline(-Rq, color='orange', linestyle=':', linewidth=0.8)
        ax.set_title('Roughness residual')
        ax.set_xlabel('Position (um)')
        ax.set_ylabel('Deviation (um)')
        ax.legend(fontsize=7)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

        self.stats_var.set(
            f"Ra = {Ra:.4f} um   Rq = {Rq:.4f} um   Rz = {Rz:.4f} um   "
            f"| columns interpolated: {n_skipped}/{self.w}   | sigma = {sigma_samples:.2f} samples"
        )


if __name__ == "__main__":
    app = ProfilometryTunerApp()
    app.mainloop()