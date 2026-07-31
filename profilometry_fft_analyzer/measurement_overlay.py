"""
measurement_overlay.py
----------------------
Draw measurement annotations onto the calibration debug overlay.
All coordinates are full-image pixel coordinates.
"""
from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

import app_settings as settings

C_TRACE = (255, 255, 0)
C_SMOOTH = (0, 69, 255)
C_HEIGHT = (255, 221, 51)
C_DISHING = (0, 0, 255)
C_WIDTH = (255, 0, 255)
C_BOX_BG = (25, 25, 25)
C_BOX_BORDER = (200, 200, 200)
C_TEXT = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 2
Y_AXIS_LABEL_CLEARANCE_PX = 46

# How many height/width markers get a printed value before the overlay falls
# back to labelling only the min/median/max. Lives in app_settings so it's
# adjustable from Preferences — dense feature arrays turn into unreadable
# label soup otherwise.
import app_settings as settings  # noqa: E402


def _put_label(canvas, text: str, x: int, y: int, color) -> None:
    """Draws `text` at (x, y), nudging it back inside the canvas if it would
    hang off an edge. Writes straight onto the canvas array."""
    h, w = canvas.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    # Clamp so the whole string stays visible, not just its anchor point.
    x = int(min(max(2, x), w - tw - 2))
    y = int(min(max(th + 2, y), h - 4))
    cv2.putText(canvas, text, (x, y), FONT, FONT_SCALE, color,
                FONT_THICK, cv2.LINE_AA)


def _draw_polyline(canvas, cols, rows, color, thickness=1) -> None:
    """Draws a curve from matching column and row arrays. Non-finite points
    (gaps in the extracted trace) are dropped first; fewer than two points
    left means there's nothing to draw and it returns quietly."""
    pts = np.column_stack([np.asarray(cols, dtype=float),
                           np.asarray(rows, dtype=float)])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 2:
        return
    cv2.polylines(canvas, [pts.astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=False, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)


def _draw_vertical_marker(canvas, col: int, row_a: float, row_b: float,
                          color, label: Optional[str],
                          min_label_y: int = 0, min_label_x: int = 0) -> None:
    """Draws a vertical measurement bar with end caps at `col`, spanning the
    two rows — this is what a height or dishing marker looks like. The label
    goes to the right of the bar, flipping to the left when there isn't room.
    `min_label_y`/`min_label_x` keep text clear of the summary strip and the
    y-axis numbers."""
    h, w = canvas.shape[:2]
    col = int(min(max(0, col), w - 1))
    r0, r1 = int(round(min(row_a, row_b))), int(round(max(row_a, row_b)))
    r0 = min(max(0, r0), h - 1)
    r1 = min(max(0, r1), h - 1)
    cv2.line(canvas, (col, r0), (col, r1), color, 1, cv2.LINE_AA)
    # End caps mark exactly which two levels were measured between.
    cap = 4
    cv2.line(canvas, (col - cap, r0), (col + cap, r0), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (col - cap, r1), (col + cap, r1), color, 1, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        # Prefer the right side; flip left only if the text would run off.
        lx = col + 6 if col + 6 + tw < w else max(col - tw - 6, min_label_x)
        ly = max((r0 + r1) // 1.8, min_label_y + th)
        _put_label(canvas, label, lx, ly, color)


def _draw_horizontal_marker(canvas, row: float, col_a: float, col_b: float,
                            color, label: Optional[str],
                            min_label_y: int = 0, min_label_x: int = 0) -> None:
    """The width counterpart of _draw_vertical_marker: a horizontal bar at
    `row` spanning the two columns, with end caps. The label sits above the
    bar's LEFT end rather than its centre, so it clears the height label,
    which is drawn at the feature's centre."""
    h, w = canvas.shape[:2]
    row_i = int(min(max(0, round(row)), h - 1))
    c0, c1 = int(round(min(col_a, col_b))), int(round(max(col_a, col_b)))
    c0 = min(max(0, c0), w - 1)
    c1 = min(max(0, c1), w - 1)
    cv2.line(canvas, (c0, row_i), (c1, row_i), color, 1, cv2.LINE_AA)
    cap = 4
    cv2.line(canvas, (c0, row_i - cap), (c0, row_i + cap), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (c1, row_i - cap), (c1, row_i + cap), color, 1, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        lx = max(min_label_x, min(c0, w - tw - 2))
        ly = max(row_i - 6, min_label_y + th)
        _put_label(canvas, label, lx, ly, color)


def build_measurement_markers(feats, cols_full, row_of_um, fmt,
                              dx_um: float) -> tuple[list[dict], list[dict], list[dict]]:
    """Convert feature-analysis results into pixel-space marker dictionaries."""
    n = len(cols_full)
    height_markers: list[dict] = []
    dishing_markers: list[dict] = []
    width_markers: list[dict] = []
    if n == 0:
        return height_markers, dishing_markers, width_markers

    for f in feats.features:
        ci = min((f.i0 + f.i1) // 2, n - 1)
        height_markers.append({
            "col": int(round(cols_full[ci])),
            "row_top": row_of_um(f.top_level, ci),
            "row_base": row_of_um(f.base_level, ci),
            "label": f"h={fmt(f.height)}",
            "sort_value": f.height,
        })
        if np.isfinite(f.dishing) and f.dishing > 0:
            j = int(round(f.dish_x / max(dx_um, 1e-12)))
            j = min(max(f.i0, j), n - 1)
            dishing_markers.append({
                "col": int(round(cols_full[j])),
                "row_top": row_of_um(f.shoulder_level, j),
                "row_base": row_of_um(f.dish_min_level, j),
                "label": f"d={fmt(f.dishing)}",
            })
        if np.isfinite(f.width) and np.isfinite(f.x_left) and np.isfinite(f.x_right):
            jl = min(max(0, int(round(f.x_left / max(dx_um, 1e-12)))), n - 1)
            jr = min(max(0, int(round(f.x_right / max(dx_um, 1e-12)))), n - 1)
            mid_level = (f.top_level + f.base_level) / 2.0
            width_markers.append({
                "row": row_of_um(mid_level, ci),
                "col_left": int(round(cols_full[jl])),
                "col_right": int(round(cols_full[jr])),
                "label": f"w={fmt(f.width)}",
                "sort_value": f.width,
            })
    return height_markers, dishing_markers, width_markers


def draw_measurements(canvas: np.ndarray,
                      trace_cols: np.ndarray,
                      trace_rows: np.ndarray,
                      smooth_rows: Optional[np.ndarray] = None,
                      height_markers: Optional[list[dict]] = None,
                      dishing_markers: Optional[list[dict]] = None,
                      summary_lines: Optional[list[str]] = None,
                      width_markers: Optional[list[dict]] = None) -> np.ndarray:
    """Draw measurement curves, markers, and a one-line top summary banner."""
    # Summary strip goes down FIRST so the trace and markers draw on top of
    # it; its height comes back so labels can be kept clear of it.
    header_clear = _draw_top_strip(canvas, summary_lines or [])

    # Raw extracted trace, then the smoothed curve the measurements came from.
    _draw_polyline(canvas, trace_cols, trace_rows, C_TRACE, 1)
    if smooth_rows is not None:
        _draw_polyline(canvas, trace_cols, smooth_rows, C_SMOOTH, 2)

    height_markers = height_markers or []
    dishing_markers = dishing_markers or []
    width_markers = width_markers or []

    def _label_set(markers, limit):
        """Takes a marker list and a cap, returns the indices that should get
        a printed value. Under the cap everything is labelled; over it, only
        the smallest, middle and largest, so a dense array stays readable."""
        if len(markers) <= limit:
            return set(range(len(markers)))
        values = [m.get("sort_value", 0.0) for m in markers]
        order = np.argsort(values)
        return {int(order[0]), int(order[len(order) // 2]), int(order[-1])}

    labeled_h = _label_set(height_markers, settings.MAX_LABELED_HEIGHTS)
    labeled_w = _label_set(width_markers, settings.MAX_LABELED_WIDTHS)

    for i, m in enumerate(height_markers):
        _draw_vertical_marker(
            canvas, m["col"], m["row_top"], m["row_base"], C_HEIGHT,
            m.get("label") if i in labeled_h else None,
            min_label_y=header_clear, min_label_x=Y_AXIS_LABEL_CLEARANCE_PX)
    for m in dishing_markers:
        _draw_vertical_marker(
            canvas, m["col"], m["row_top"], m["row_base"], C_DISHING,
            m.get("label"), min_label_y=header_clear,
            min_label_x=Y_AXIS_LABEL_CLEARANCE_PX)
    for i, m in enumerate(width_markers):
        _draw_horizontal_marker(
            canvas, m["row"], m["col_left"], m["col_right"], C_WIDTH,
            m.get("label") if i in labeled_w else None,
            min_label_y=header_clear, min_label_x=Y_AXIS_LABEL_CLEARANCE_PX)
    return canvas


def _draw_top_strip(canvas, lines: list[str]) -> int:
    """Draw every summary item on one full-width line at the image top.

    Existing entries in ``lines`` are joined with separators. The font is
    scaled down as needed so the complete summary fits without wrapping.
    Returns the bottom pixel of the banner so marker labels can avoid it.
    """
    clean_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not clean_lines:
        return 0

    _h, w = canvas.shape[:2]
    summary_text = "  |  ".join(clean_lines)
    pad_x = 6
    pad_y = 4
    available_w = max(1, w - 2 * pad_x)

    # Calculate a scale that makes the entire string fit on exactly one line.
    base_scale = 0.42
    (base_w, _), _ = cv2.getTextSize(
        summary_text, FONT, base_scale, FONT_THICK)
    font_scale = base_scale if base_w <= available_w else base_scale * available_w / max(1, base_w)

    # Re-measure after scaling; compensate for OpenCV's integer rounding if
    # the first calculation is still a pixel or two too wide.
    (text_w, text_h), baseline = cv2.getTextSize(
        summary_text, FONT, font_scale, FONT_THICK)
    if text_w > available_w:
        font_scale *= available_w / text_w
        (text_w, text_h), baseline = cv2.getTextSize(
            summary_text, FONT, font_scale, FONT_THICK)

    box_h = text_h + baseline + 2 * pad_y
    cv2.rectangle(canvas, (0, 0), (w - 1, box_h),
                  C_BOX_BG, thickness=cv2.FILLED)
    cv2.line(canvas, (0, box_h), (w - 1, box_h), C_BOX_BORDER, 1)
    cv2.putText(canvas, summary_text, (pad_x, pad_y + text_h), FONT,
                font_scale, C_TEXT, FONT_THICK, cv2.LINE_AA)
    return box_h
