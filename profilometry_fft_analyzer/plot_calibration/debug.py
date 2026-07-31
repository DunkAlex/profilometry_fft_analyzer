"""
debug.py
--------
Render an annotated debug image showing what the pipeline detected,
and save it into the "image fits" folder with a timestamped filename.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from . import config


def _resolve_project_root() -> str:
    """
    Resolve the directory the debug-image folder should live in, independent
    of the process's current working directory.

    Three strategies, in order:
      1. PLOT_CALIBRATION_PROJECT_ROOT environment variable, if set — an
         explicit escape hatch for any deployment layout this heuristic
         doesn't anticipate.
      2. The __main__ module's own file location — i.e. wherever the actual
         script the person runs (`python analyze.py`) lives on disk. This is
         deliberately NOT based on where plot_calibration itself is
         installed/vendored, since that can differ from the project layout
         assumed at packaging time (nesting depth, editable installs,
         copied-in vendored copies, etc). Anchoring to the entry-point
         script matches "the same directory the calibration script is run
         from" exactly, regardless of how plot_calibration got there.
      3. The current working directory, only as a last resort when neither
         of the above is available (e.g. an interactive REPL with no
         __main__.__file__).
    """
    env_override = os.environ.get("PLOT_CALIBRATION_PROJECT_ROOT")
    if env_override:
        return os.path.abspath(env_override)

    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if main_file:
        return os.path.dirname(os.path.abspath(main_file))

    return os.getcwd()


# Debug output always lands next to the actual entry-point script (see
# _resolve_project_root), never relative to whatever directory the process
# happens to be launched from.
_PROJECT_ROOT = _resolve_project_root()
DEFAULT_DEBUG_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, config.DEBUG_OUTPUT_DIR)


def save_debug_image(image_rgb: np.ndarray,
                     input_filename: str,
                     plot_region: tuple[int, int, int, int],
                     x_gridlines: np.ndarray,
                     y_gridlines: np.ndarray,
                     x_tick_pairs: list[tuple[float, float, float]],
                     y_tick_pairs: list[tuple[float, float, float]],
                     x_tick_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
                     y_tick_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
                     x_recovered_idx: Optional[list[int]] = None,
                     y_recovered_idx: Optional[list[int]] = None,
                     x_fit: Optional[dict] = None,
                     y_fit: Optional[dict] = None,
                     x_confidence: float = 0.0,
                     y_confidence: float = 0.0,
                     x_unit: Optional[str] = None,
                     y_unit: Optional[str] = None,
                     x_unit_bbox: Optional[tuple[int, int, int, int]] = None,
                     y_unit_bbox: Optional[tuple[int, int, int, int]] = None,
                     axis_lines: Optional[dict[str, float]] = None,
                     x_inferred_idx: Optional[list[int]] = None,
                     y_inferred_idx: Optional[list[int]] = None,
                     output_dir: str = DEFAULT_DEBUG_OUTPUT_DIR,
                     draw_fit_box: bool = True
                     ) -> str:
    """
    Render overlays on a copy of the input image and save to disk.

    `axis_lines` ({"x": row_px, "y": col_px}, from
    detect_plot_region_and_gridlines) is drawn as a distinct-colored line
    separately from `plot_region`'s box: plot_region can be padded by a
    margin (see gridlines._detect_shared_axis_gridlines) when axis and
    gridline share a color, so its box edge and the true axis line are not
    always the same pixel — drawing both makes that distinction visible
    instead of leaving it to be inferred from the numbers.

    `x_inferred_idx`/`y_inferred_idx` mark tick entries that were filled in
    or overridden by sequence reconstruction (ticks.reconstruct_tick_values)
    rather than read directly from OCR — rendered in a third, distinct
    color from both normal and outlier-recovered ticks.

    `draw_fit_box` controls the single-line X/Y confidence/unit summary
    banner across the top of the image. Callers with downstream measurement
    data to show alongside it (batch_analyze) pass False here and draw one
    combined box themselves (see measurement_overlay.draw_measurements)
    instead of stacking two separate boxes. Standalone callers with no
    measurement data (the calibration GUI, run_example.py,
    sample_calibration.py, tuning_gui.py) leave this at the default True.

    Returns the absolute path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = canvas.shape[:2]

    # colors are BGR because we're drawing on a BGR canvas
    C_REGION = (0, 255, 255)      # yellow
    C_GRID = (255, 255, 255)      # white
    C_AXIS_LINE = (0, 165, 255)   # orange — true axis line (see axis_lines)
    C_TICK = (0, 255, 0)          # green — normal (confirmed OCR)
    C_TICK_RECOV = (0, 0, 255)    # red — recovered from outlier
    C_TICK_INFER = (255, 128, 0)  # blue — filled in / overridden by sequence reconstruction
    C_UNIT = (255, 0, 255)        # magenta — unit label
    C_TEXT = (255, 255, 255)

    # Plot region
    x0, y0, x1, y1 = plot_region
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), C_REGION, 1)

    # Gridlines
    for x in x_gridlines:
        cv2.line(canvas, (int(x), y0), (int(x), y1 - 1), C_GRID, 1)
    for y in y_gridlines:
        cv2.line(canvas, (x0, int(y)), (x1 - 1, int(y)), C_GRID, 1)

    # True axis lines (may differ from plot_region's padded box edge)
    if axis_lines:
        if "x" in axis_lines:
            row = int(round(axis_lines["x"]))
            cv2.line(canvas, (0, row), (w - 1, row), C_AXIS_LINE, 1)
        if "y" in axis_lines:
            col = int(round(axis_lines["y"]))
            cv2.line(canvas, (col, 0), (col, h - 1), C_AXIS_LINE, 1)

    # Tick label bboxes (if provided) + values.
    # Passing the axis's own unit bbox lets `_draw_ticks` suppress an inferred
    # label whose gridline falls inside it — the unit label already occupies
    # that region and a "100" (or whatever the sequence solved to) drawn on top
    # of it is exactly the visual confusion Fix K4 targets. Belt-and-suspenders
    # on top of Fix K1 (which skips the tick search there entirely upstream) —
    # if a future edit accidentally re-introduces an inferred entry at the unit
    # gridline the overlay still stays clean.
    _draw_ticks(canvas, x_tick_pairs, x_tick_bboxes, x_recovered_idx, x_inferred_idx,
                axis="x", color_normal=C_TICK, color_recov=C_TICK_RECOV,
                color_infer=C_TICK_INFER, unit_bbox=x_unit_bbox)
    _draw_ticks(canvas, y_tick_pairs, y_tick_bboxes, y_recovered_idx, y_inferred_idx,
                axis="y", color_normal=C_TICK, color_recov=C_TICK_RECOV,
                color_infer=C_TICK_INFER, unit_bbox=y_unit_bbox)

    # Unit label bboxes + recognized text (magenta, distinct from tick color)
    _draw_unit_label(canvas, x_unit_bbox, x_unit, C_UNIT)
    _draw_unit_label(canvas, y_unit_bbox, y_unit, C_UNIT)

    # Fit summary banner — display all available fit information on one
    # horizontal line across the top of the image.
    summaries = []

    if draw_fit_box and x_fit is not None:
        unit_str = f"  unit={x_unit!r}" if x_unit else "  unit=?"
        summaries.append(f"X: conf={x_confidence:.2f}{unit_str}")

    if draw_fit_box and y_fit is not None:
        unit_str = f"  unit={y_unit!r}" if y_unit else "  unit=?"
        summaries.append(f"Y: conf={y_confidence:.2f}{unit_str}")

    if summaries:
        # Join the X and Y summaries into a single line.
        summary_text = "    |    ".join(summaries)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        pad_x = 8
        pad_y = 6

        # Reduce the font scale if necessary so the entire summary fits
        # within the image width without wrapping or clipping.
        available_text_width = max(1, w - 2 * pad_x)
        text_size, baseline = cv2.getTextSize(
            summary_text, font, font_scale, thickness
        )

        if text_size[0] > available_text_width:
            font_scale *= available_text_width / text_size[0]
            # Avoid making the text unreasonably small.
            font_scale = max(font_scale, 0.3)
            text_size, baseline = cv2.getTextSize(
                summary_text, font, font_scale, thickness
            )

        _text_w, text_h = text_size
        banner_h = text_h + 2 * pad_y + baseline

        # Full-width dark banner at the very top of the image.
        cv2.rectangle(
            canvas,
            (0, 0),
            (w - 1, banner_h),
            (25, 25, 25),
            thickness=cv2.FILLED,
        )

        # Light border along the bottom of the banner.
        cv2.line(
            canvas,
            (0, banner_h),
            (w - 1, banner_h),
            (200, 200, 200),
            thickness=1,
        )

        # Draw the complete summary as one line.
        text_y = pad_y + text_h
        cv2.putText(
            canvas,
            summary_text,
            (pad_x, text_y),
            font,
            font_scale,
            C_TEXT,
            thickness,
            cv2.LINE_AA,
        )

    # Filename: <YYYYMMDD_HHMMSS>_<input_stem>.png
    stem = os.path.splitext(os.path.basename(input_filename))[0] or "image"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{ts}_{stem}.png")
    cv2.imwrite(out_path, canvas)
    return os.path.abspath(out_path)


def _draw_unit_label(canvas, bbox, unit_str, color):
    """Draw the unit-label bbox (if found) with its recognized text above it."""
    if bbox is None:
        return
    x, y, bw, bh = bbox
    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 1)
    label = unit_str if unit_str else "?"
    tx, ty = x, max(10, y - 4)
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def _draw_ticks(canvas, pairs, bboxes, recovered_idx, inferred_idx, axis,
                color_normal, color_recov, color_infer,
                unit_bbox: Optional[tuple[int, int, int, int]] = None):
    if not pairs:
        return
    recovered_idx = set(recovered_idx or [])
    inferred_idx = set(inferred_idx or [])
    ux0 = uy0 = ux1 = uy1 = None
    if unit_bbox is not None:
        ux, uy, uw, uh = unit_bbox
        ux0, uy0, ux1, uy1 = ux, uy, ux + uw, uy + uh
    for i, (pix, val, _conf) in enumerate(pairs):
        # Inferred (filled in / overridden by sequence reconstruction) takes
        # priority in color choice over outlier-recovered, since it's the
        # more fundamental origin of the value — an entry can technically be
        # both if reconstruction's output itself later got snapped.
        is_inferred = i in inferred_idx
        if is_inferred and ux0 is not None:
            # Skip inferred ticks whose gridline pixel falls inside the axis's
            # unit-label bbox — the unit already occupies that region visually
            # and stacking a "100" (or whatever the sequence solved to) on top
            # of the "µm" glyph is exactly the far-right overlay confusion Fix
            # K4 targets. Confirmed ticks (from OCR or J3 back-verify) still
            # render normally; only the fabricated inferred labels are hidden.
            if axis == "x" and ux0 <= pix <= ux1:
                continue
            if axis == "y" and uy0 <= pix <= uy1:
                continue
        if is_inferred:
            c = color_infer
        elif i in recovered_idx:
            c = color_recov
        else:
            c = color_normal
        # Inferred ticks have no backing OCR blob (bboxes[i] is None even
        # when the bboxes list itself is present), so fall through to the
        # axis-relative label position used when no bboxes were provided.
        if bboxes is not None and i < len(bboxes) and bboxes[i] is not None:
            x, y, bw, bh = bboxes[i]
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), c, 1)
            tx, ty = x, max(10, y - 2)
        else:
            if axis == "x":
                tx, ty = int(pix) - 10, canvas.shape[0] - 8
            else:
                tx, ty = 4, int(pix) + 4
        label = f"{val:g}"
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, c, 1, cv2.LINE_AA)
