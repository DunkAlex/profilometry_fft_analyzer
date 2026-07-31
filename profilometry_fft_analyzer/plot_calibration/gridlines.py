"""
gridlines.py
------------
- detect_plot_region: bounding box of the plot area via axis-line detection
- build_gridline_mask: boolean mask of gridline-colored pixels
- find_gridline_positions: peaks in 1D projections of the gridline mask
- check_gridline_regularity: coefficient-of-variation on gap sizes
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks

from . import config
from .colors import color_mask


# ---------------- Plot region ----------------

def detect_plot_region(image_rgb: np.ndarray,
                       axis_rgb: tuple[int, int, int],
                       tol: int = config.COLOR_MASK_TOLERANCE
                       ) -> tuple[int, int, int, int]:
    """
    Find bounding box of the plot region by locating long axis-colored lines.

    Strategy: build axis-color mask; project onto rows and cols; the plot
    box edges are the outermost row/col with a projection above the
    "long line" length threshold.

    Returns (x0, y0, x1, y1) inclusive-exclusive.
    Falls back to full image if detection fails.
    """
    h, w = image_rgb.shape[:2]
    mask = color_mask(image_rgb, axis_rgb, tol)

    row_proj = mask.sum(axis=1)   # per-row count -> horizontal axes
    col_proj = mask.sum(axis=0)   # per-col count -> vertical axes

    min_h_len = int(w * config.AXIS_LINE_MIN_LENGTH_FRAC)
    min_v_len = int(h * config.AXIS_LINE_MIN_LENGTH_FRAC)

    h_rows = np.where(row_proj >= min_h_len)[0]
    v_cols = np.where(col_proj >= min_v_len)[0]

    if len(h_rows) == 0 or len(v_cols) == 0:
        # fallback: whole image minus a small margin
        m = 5
        return m, m, w - m, h - m

    y0 = int(h_rows.min())
    y1 = int(h_rows.max()) + 1
    x0 = int(v_cols.min())
    x1 = int(v_cols.max()) + 1

    # sanity: enforce non-degenerate
    if x1 - x0 < 20 or y1 - y0 < 20:
        m = 5
        return m, m, w - m, h - m
    return x0, y0, x1, y1


# ---------------- Gridline mask + peaks ----------------

def build_gridline_mask(image_rgb: np.ndarray,
                        gridline_rgb: tuple[int, int, int],
                        tol: int | None = None) -> np.ndarray:
    """Boolean mask of gridline-colored pixels. `tol` of None defers to
    color_mask, which reads the current config.COLOR_MASK_TOLERANCE."""
    return color_mask(image_rgb, gridline_rgb, tol)


def find_gridline_positions(mask: np.ndarray,
                            plot_region: tuple[int, int, int, int],
                            axis: str) -> np.ndarray:
    """
    Project mask onto the axis of interest and find peaks.

    axis='x' -> looking for vertical gridlines; sum along rows within the
                plot region gives a per-column projection whose peaks are
                the x-pixel positions of vertical gridlines.
    axis='y' -> looking for horizontal gridlines; per-row projection.

    Returns 1D array of pixel positions (columns for x, rows for y).
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    x0, y0, x1, y1 = plot_region
    sub = mask[y0:y1, x0:x1]

    if axis == "x":
        proj = sub.sum(axis=0).astype(np.float32)
        offset = x0
    else:
        proj = sub.sum(axis=1).astype(np.float32)
        offset = y0

    if proj.size == 0 or proj.max() == 0:
        return np.array([], dtype=int)

    height = proj.max() * config.GRIDLINE_PEAK_MIN_HEIGHT_FRAC
    peaks, _ = find_peaks(proj,
                          height=height,
                          distance=config.GRIDLINE_MIN_SPACING_PX)
    return peaks + offset


def check_gridline_regularity(positions: np.ndarray
                              ) -> tuple[bool, dict]:
    """
    Compute spacing coefficient of variation (std / mean).

    Returns (is_regular, metrics) where metrics carries diagnostic info.
    is_regular = True iff we have >= 2 gaps and CV < tolerance.
    """
    metrics = {
        "n_gridlines": int(len(positions)),
        "cv": float("inf"),
        "mean_spacing": None,
        "std_spacing": None,
    }
    if len(positions) < 3:
        return False, metrics

    diffs = np.diff(np.sort(positions)).astype(float)
    mean = diffs.mean()
    std = diffs.std()
    cv = std / mean if mean > 0 else float("inf")

    metrics["cv"] = float(cv)
    metrics["mean_spacing"] = float(mean)
    metrics["std_spacing"] = float(std)

    is_regular = cv < config.GRIDLINE_SPACING_CV_TOLERANCE
    return bool(is_regular), metrics


# ---------------- Unified detection (handles shared axis/gridline color) ----------------
#
# Many plotting programs style the bounding frame identically to the
# interior gridlines -- there is no separate "axis color" to distinguish
# them by. When that's the case, color alone can't tell a border line from
# an interior gridline, but POSITION can: the bottom-most horizontal line
# and left-most vertical line are the x-axis and y-axis respectively (by
# convention, axes sit at the bottom and left of a plot). Every detected
# line -- including those two -- is then offered to the tick search; a
# pure frame line with no adjacent number label simply comes back "missing"
# from the per-gridline OCR search, which already handles that gracefully.
#
# This also fixes a real, separate bug that shows up specifically in the
# shared-color case: the border spans nearly the full plot width/height,
# so it dominates the row/column projections used for peak detection.
# Since the peak-height threshold is a FRACTION of the max value in that
# projection, the border's very high count can push the threshold above
# what weaker interior gridlines reach -- especially ones partly hidden
# under a trace. The two-pass approach below finds the (reliably strong)
# border first, suppresses it, then re-searches the remainder with a
# threshold based on what's left, recovering gridlines the border was
# masking out.

def _colors_match(rgb_a: tuple[int, int, int],
                  rgb_b: tuple[int, int, int],
                  tol: int | None = None) -> bool:
    """True when two RGB triples sit within `tol` on every channel. `tol` of
    None reads the current config.COLOR_MASK_TOLERANCE at call time."""
    if tol is None:
        tol = config.COLOR_MASK_TOLERANCE
    a = np.asarray(rgb_a, dtype=np.int16)
    b = np.asarray(rgb_b, dtype=np.int16)
    return bool(np.max(np.abs(a - b)) <= tol)


def _find_line_peaks(proj: np.ndarray) -> np.ndarray:
    """Peak-detect a 1D projection using the standard gridline thresholds."""
    if proj.size == 0 or proj.max() == 0:
        return np.array([], dtype=int)
    height = proj.max() * config.GRIDLINE_PEAK_MIN_HEIGHT_FRAC
    peaks, _ = find_peaks(proj, height=height, distance=config.GRIDLINE_MIN_SPACING_PX)
    return peaks


def _merge_nearby_peaks(strong_peaks: np.ndarray, weak_peaks: np.ndarray,
                        min_distance: int) -> np.ndarray:
    """
    Combine the strong-pass and weak-pass peak lists into one, collapsing
    anything within `min_distance` of each other into a single position.

    This matters because find_peaks' own `distance` parameter only
    enforces minimum spacing WITHIN one call — running two separate calls
    (strong pass, then weak pass on the remainder) and simply concatenating
    their results does nothing to stop the same physical line from being
    picked up twice: a real line's anti-aliasing halo can extend a little
    past the narrow window suppressed after pass 1, and pass 2 (searching
    for anything it can find in what's left) happily reports that halo as
    a second, slightly-offset "line" a few pixels from the real one. Left
    unmerged, this doubles the reported gridline count and corrupts the
    fit entirely.

    When a strong and a weak peak collide, the strong one wins (pass 1
    measured the true line directly; pass 2 only exists to catch lines
    pass 1 missed, not to refine ones it already found).
    """
    tagged = [(int(p), True) for p in strong_peaks] + [(int(p), False) for p in weak_peaks]
    if not tagged:
        return np.array([], dtype=int)
    tagged.sort(key=lambda t: t[0])

    merged = [tagged[0]]
    for pos, is_strong in tagged[1:]:
        last_pos, last_strong = merged[-1]
        if pos - last_pos < min_distance:
            if is_strong and not last_strong:
                merged[-1] = (pos, True)
            # else: duplicate of an already-kept peak, drop it
        else:
            merged.append((pos, is_strong))
    return np.array([p for p, _ in merged], dtype=int)


def _detect_shared_axis_gridlines(image_rgb: np.ndarray,
                                  shared_rgb: tuple[int, int, int],
                                  tol: int = config.COLOR_MASK_TOLERANCE
                                  ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray,
                                            dict[str, float]]:
    """
    Two-pass line detection for the case where axis and gridline share a
    color. See module-level note above for the reasoning.

    Returns (plot_region, x_gridline_positions, y_gridline_positions, axis_lines).
    `axis_lines` is {"x": row_px, "y": col_px} — the TRUE (unpadded) pixel
    position of the x-axis line and y-axis line respectively, distinct from
    plot_region's box edges (which are padded by a margin so the box fully
    encloses the detected lines — see below). Downstream tick/unit-label
    search must anchor on these, not on plot_region, or it leaves a gap
    between the axis and the search start.
    """
    h, w = image_rgb.shape[:2]
    mask = color_mask(image_rgb, shared_rgb, tol)

    row_proj = mask.sum(axis=1).astype(np.float32)
    col_proj = mask.sum(axis=0).astype(np.float32)

    # Pass 1: find the strong lines first. The border reliably survives
    # this pass since it's the longest, most fully-unoccluded line(s) in
    # each direction.
    row_peaks_strong = _find_line_peaks(row_proj)
    col_peaks_strong = _find_line_peaks(col_proj)

    if len(row_peaks_strong) == 0 or len(col_peaks_strong) == 0:
        m = 5
        return ((m, m, w - m, h - m), np.array([]), np.array([]),
                {"x": float(h - m), "y": float(m)})

    # Pass 2: re-search for weaker interior lines. Two sources of
    # contamination need excluding, not just one:
    #   1. Same-direction: the strong lines just found (a border row can
    #      have anti-aliased neighbors that would otherwise register as
    #      spurious extra "weak" rows right next to it).
    #   2. Cross-direction: the PERPENDICULAR border contributes a small
    #      constant baseline to every row/column of the projection it
    #      doesn't itself belong to (e.g. the left/right vertical border
    #      pixels add +2 to literally every row's sum, since they exist at
    #      every row within the plot). Once the dominant peaks are zeroed
    #      out, this flat baseline can get misread as a series of spurious
    #      low-level "peaks" spread across otherwise-empty rows. Excluding
    #      the perpendicular border's own columns/rows from the sum before
    #      re-searching removes that baseline entirely rather than fighting
    #      it with a higher threshold.
    suppress = config.GRIDLINE_MIN_SPACING_PX

    col_exclude = np.zeros(w, dtype=bool)
    for p in col_peaks_strong:
        lo, hi = max(0, p - suppress), min(w, p + suppress + 1)
        col_exclude[lo:hi] = True
    row_proj_remainder = mask[:, ~col_exclude].sum(axis=1).astype(np.float32)
    for p in row_peaks_strong:
        lo, hi = max(0, p - suppress), min(h, p + suppress + 1)
        row_proj_remainder[lo:hi] = 0
    row_peaks_weak = _find_line_peaks(row_proj_remainder)

    row_exclude = np.zeros(h, dtype=bool)
    for p in row_peaks_strong:
        lo, hi = max(0, p - suppress), min(h, p + suppress + 1)
        row_exclude[lo:hi] = True
    col_proj_remainder = mask[~row_exclude, :].sum(axis=0).astype(np.float32)
    for p in col_peaks_strong:
        lo, hi = max(0, p - suppress), min(w, p + suppress + 1)
        col_proj_remainder[lo:hi] = 0
    col_peaks_weak = _find_line_peaks(col_proj_remainder)

    # Merge rather than naively concatenate: a real line's anti-aliasing
    # can extend past the suppression window and get reported a second
    # time by the weak pass a few pixels off from where the strong pass
    # already found it. See _merge_nearby_peaks for why this matters.
    row_peaks = _merge_nearby_peaks(row_peaks_strong, row_peaks_weak,
                                    config.GRIDLINE_MIN_SPACING_PX)
    col_peaks = _merge_nearby_peaks(col_peaks_strong, col_peaks_weak,
                                    config.GRIDLINE_MIN_SPACING_PX)

    # Bottom-most horizontal line = the x-axis; left-most vertical line =
    # the y-axis. Together with the opposite extrema (top-most row,
    # right-most col -- the top/right frame if one exists, or just the
    # outermost interior gridline otherwise) these bound the plot region.
    y1 = int(row_peaks.max())
    y0 = int(row_peaks.min())
    x0 = int(col_peaks.min())
    x1 = int(col_peaks.max())

    margin = 3
    plot_region = (
        max(0, x0 - margin), max(0, y0 - margin),
        min(w, x1 + margin + 1), min(h, y1 + margin + 1),
    )

    # True (unpadded) axis-line positions for the tick/unit search to
    # anchor on -- NOT plot_region's padded edges (see docstring).
    axis_lines = {"x": float(y1), "y": float(x0)}

    # x gridlines are vertical lines (column positions); y gridlines are
    # horizontal lines (row positions). All detected lines -- border
    # included -- are passed through; see the module note on why that's
    # safe (a labelless frame line just comes back "missing" downstream).
    return plot_region, col_peaks, row_peaks, axis_lines


def detect_plot_region_and_gridlines(image_rgb: np.ndarray,
                                     axis_rgb: tuple[int, int, int],
                                     gridline_rgb: tuple[int, int, int],
                                     tol: int = config.COLOR_MASK_TOLERANCE
                                     ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray,
                                               dict[str, float]]:
    """
    Unified entry point for plot-region + gridline detection. This is what
    calibrate.py should call rather than detect_plot_region /
    build_gridline_mask / find_gridline_positions directly, since it
    transparently picks the right strategy:

      - axis_rgb and gridline_rgb are distinct colors (the common case):
        behaves exactly as the original separate functions did.
      - axis_rgb and gridline_rgb are the same (or within `tol`): uses the
        position-based two-pass approach in _detect_shared_axis_gridlines,
        since color alone can't distinguish border from gridline here.

    Returns (plot_region, x_gridline_positions, y_gridline_positions, axis_lines).
    `axis_lines` is {"x": row_px, "y": col_px}, the true pixel position of
    the x-axis and y-axis lines for the tick/unit-label search to anchor
    on. In the distinct-color path, detect_plot_region adds no padding, so
    these are simply plot_region's bottom/left edges; in the shared-color
    path they come from _detect_shared_axis_gridlines, which DOES pad
    plot_region — see that function's docstring for why the distinction
    matters.
    """
    if _colors_match(axis_rgb, gridline_rgb, tol):
        return _detect_shared_axis_gridlines(image_rgb, axis_rgb, tol)

    plot_region = detect_plot_region(image_rgb, axis_rgb, tol)
    gmask = build_gridline_mask(image_rgb, gridline_rgb, tol)
    x_grid = find_gridline_positions(gmask, plot_region, axis="x")
    y_grid = find_gridline_positions(gmask, plot_region, axis="y")
    axis_lines = {"x": float(plot_region[3]), "y": float(plot_region[0])}
    return plot_region, x_grid, y_grid, axis_lines
