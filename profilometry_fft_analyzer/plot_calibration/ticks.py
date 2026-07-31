"""
ticks.py
--------
Text blob detection via connected components on the not-bg/not-plot mask,
filtering to axis-tick-label regions, OCR of each blob, and robust linear
fit of pixel <-> unit values with outlier recovery.
"""
from __future__ import annotations
from typing import Optional
import logging

import cv2
import numpy as np
from scipy import stats as sstats

from . import config
from .colors import not_background_not_plot_mask, color_mask
from .ocr import (
    ocr_number, ocr_text, postprocess_number_string, normalize_unit_string,
    score_unit_match, extract_unit_suffix,
)

logger = logging.getLogger(__name__)


# ---------------- Per-gridline anchored tick search (primary path) ----------------
#
# Rationale: filtering a whole-image blob list down to a coarse "band near
# the axis" is fragile — a thin single-digit label can fail the area/aspect
# filters, or land just outside the assumed band, and silently vanish with
# no record of *which* tick went missing. Anchoring the search to each
# gridline's own pixel position instead guarantees one attempt per gridline,
# and expanding the search window on failure (redundancy) recovers labels
# that a fixed-size window would miss.
#
# A second benefit: using the gridline's own pixel position as the fit's
# x-coordinate (rather than the OCR'd label's bbox centroid) is more
# accurate — the label position has font-rendering jitter the gridline
# doesn't — and it removes the need for the post-hoc "match blob to
# gridline by nearest centroid" step entirely, since the pairing is
#1:1 and exact by construction.

def _local_tick_search_box(gridline_pixel: float,
                           axis_line_pos: float,
                           axis: str,
                           band_px: int,
                           perp_half_width: int
                           ) -> tuple[float, float, float, float]:
    """
    Compute the local search window for one gridline, before clamping.

    `axis_line_pos` is the TRUE pixel position of the axis line itself —
    the row for axis='x' (the horizontal x-axis line), the column for
    axis='y' (the vertical y-axis line). This is deliberately NOT
    plot_region's bounding-box edge: when axis and gridline share a color
    (see gridlines._detect_shared_axis_gridlines), plot_region is padded
    by a few px so its box fully encloses the detected lines, and
    anchoring the label search on that padded edge instead of the real
    line leaves a visible gap between the axis and where the search
    starts. Callers get this value from
    detect_plot_region_and_gridlines' `axis_lines` return.
    """
    if axis == "x":
        bx0 = gridline_pixel - perp_half_width
        bx1 = gridline_pixel + perp_half_width
        by0 = float(axis_line_pos)
        by1 = float(axis_line_pos + band_px)
    else:
        bx0 = float(axis_line_pos - band_px)
        bx1 = float(axis_line_pos)
        by0 = gridline_pixel - perp_half_width
        by1 = gridline_pixel + perp_half_width
    return bx0, by0, bx1, by1


def _cross_search_boxes(anchor_pixel: float,
                        axis_line_pos: float,
                        axis: str,
                        band_px: int,
                        perp_half_width: float,
                        parallel_half_width: float
                        ) -> list[tuple[float, float, float, float]]:
    """
    Build a redundant pair of search windows around `anchor_pixel` instead
    of a single box, per the project's requirement that localization be
    robust across generally-similar layouts rather than tuned to one exact
    label position:

      - the PERPENDICULAR arm (unchanged — what _local_tick_search_box has
        always produced): reaches away from the axis line into the margin,
        narrow along the axis direction.
      - the PARALLEL arm (new): a band running alongside the axis line
        itself — shallower reach away from the axis, but wide along the
        axis direction (sized from `parallel_half_width`, which callers
        derive from gridline spacing). This catches a label or unit string
        offset along the axis direction from its anchor rather than
        directly beneath/beside it (e.g. a unit string printed at
        axis-line height instead of down in the tick-label margin).

    Callers try both boxes and keep whatever's found; see
    find_tick_label_at_gridline / find_unit_label_at_axis.
    """
    perp_box = _local_tick_search_box(
        anchor_pixel, axis_line_pos, axis, band_px, perp_half_width
    )
    shallow_band = max(1, int(band_px * 0.6))
    if axis == "x":
        bx0 = anchor_pixel - parallel_half_width
        bx1 = anchor_pixel + parallel_half_width
        by0 = float(axis_line_pos)
        by1 = float(axis_line_pos + shallow_band)
    else:
        bx0 = float(axis_line_pos - shallow_band)
        bx1 = float(axis_line_pos)
        by0 = anchor_pixel - parallel_half_width
        by1 = anchor_pixel + parallel_half_width
    parallel_box = (bx0, by0, bx1, by1)
    return [perp_box, parallel_box]


def _recover_vertical_fragments(sub_height: list[tuple[int, int, int, int]]
                                ) -> list[tuple[int, int, int, int]]:
    """
    Merge vertically-adjacent, column-shared fragments into single-blob bboxes.

    The primary blob loop in `_local_blob_search` discards any component shorter
    than TICK_SEARCH_MIN_LABEL_HEIGHT_PX — that's what a gridline/axis-line
    anti-aliasing fleck looks like, and admitting them one-by-one would flood
    the search with garbage. But a lone "1" on this instrument is a ~2 px × 10
    px vertical stroke whose anti-aliasing splits into 2-3 stacked sub-height
    fragments, all of which the primary filter rejects — the digit vanishes.

    This pass pools those sub-height rejects and merges any two-plus that share
    a column (x centres within `TICK_SEARCH_VERT_FRAG_X_TOL_PX`) and sit near
    each other vertically (gap between successive fragments
    <= `TICK_SEARCH_VERT_FRAG_MAX_GAP_PX`). Merged bboxes are only kept if
    their combined height clears `TICK_SEARCH_MIN_LABEL_HEIGHT_PX` — the
    original threshold — so nothing that would fail the primary shape test
    can slip through here.

    Returns the merged bboxes in local (crop-relative) coordinates.
    """
    if not sub_height:
        return []
    x_tol = float(config.TICK_SEARCH_VERT_FRAG_X_TOL_PX)
    gap_tol = float(config.TICK_SEARCH_VERT_FRAG_MAX_GAP_PX)
    min_h = int(config.TICK_SEARCH_MIN_LABEL_HEIGHT_PX)

    items = sorted(
        sub_height,
        key=lambda f: (round((f[0] + f[2] / 2.0) / max(1.0, x_tol)), f[1]),
    )
    columns: list[list[tuple[int, int, int, int]]] = []
    for f in items:
        cx = f[0] + f[2] / 2.0
        placed = False
        for col in columns:
            # Compare against any fragment already in the column — column ids
            # are a simple binning by cx tolerance, and a stroke that drifts
            # slightly can still land in the same visual column.
            col_cx = col[0][0] + col[0][2] / 2.0
            if abs(cx - col_cx) <= x_tol:
                col.append(f)
                placed = True
                break
        if not placed:
            columns.append([f])

    merged: list[tuple[int, int, int, int]] = []
    for col in columns:
        if len(col) < 2:
            continue
        col.sort(key=lambda f: f[1])
        # Group by vertical adjacency inside the column.
        run: list[tuple[int, int, int, int]] = [col[0]]

        def _flush(cur_run):
            if len(cur_run) < 2:
                return
            m = _merge_bboxes(cur_run, min_h)
            if m is not None:
                merged.append(m)

        for f in col[1:]:
            prev = run[-1]
            gap = f[1] - (prev[1] + prev[3])
            if gap <= gap_tol:
                run.append(f)
            else:
                _flush(run)
                run = [f]
        _flush(run)
    return merged


def _merge_bboxes(cluster: list[tuple[int, int, int, int]],
                  min_h: int) -> Optional[tuple[int, int, int, int]]:
    xs0 = min(c[0] for c in cluster); ys0 = min(c[1] for c in cluster)
    xs1 = max(c[0] + c[2] for c in cluster); ys1 = max(c[1] + c[3] for c in cluster)
    h = ys1 - ys0
    if h < min_h:
        return None
    return (xs0, ys0, xs1 - xs0, h)


def _local_blob_search(image_rgb: np.ndarray,
                       bx0: int, by0: int, bx1: int, by1: int,
                       bg_rgb: tuple[int, int, int],
                       plot_rgb: tuple[int, int, int],
                       axis_rgb: Optional[tuple[int, int, int]] = None,
                       gridline_rgb: Optional[tuple[int, int, int]] = None,
                       exclude_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
                       axis: Optional[str] = None,
                       reference_pixel: Optional[float] = None,
                       return_all: bool = False
                       ) -> Optional[tuple[int, int, int, int]] | list[tuple[int, int, int, int]]:
    """
    Run connected components within an already-clamped full-image-coordinate
    window, excluding background/plot/gridline/axis colors and any blob that
    overlaps a bbox in `exclude_bboxes` (used to keep the unit-label search
    from re-detecting a numeric tick label it's already claimed).

    When multiple disjoint components survive, they are clustered by
    proximity (adjacent digits of the same label merge together) rather
    than blindly merged into one bbox — an expanded search window can
    otherwise sweep in unrelated nearby content (a neighboring tick's
    digits, or a unit symbol placed close to the outermost tick) and fuse
    it onto the real label. If `axis` and `reference_pixel` are given, only
    the cluster closest to that reference point along that axis is kept;
    otherwise all clusters are merged (original behavior).

    `return_all`, when True, returns every merged cluster bbox (sorted by
    distance to `reference_pixel` if given) instead of collapsing to just
    the nearest one. This exists for the unit-label search: the nearest
    cluster to an anchor point is very often the adjacent NUMERIC tick
    label, not the unit string (e.g. the x-axis unit conventionally sits
    near the origin, right next to the first tick's digits) — always
    picking "nearest" meant the unit search kept re-finding that numeric
    label and rejecting it as "just a number" without ever considering the
    farther, correct cluster. Content-based classification (in the caller)
    needs to see every candidate, not just the closest one.

    Returns the bbox (in full-image coords) of the selected cluster (or a
    list of all of them if `return_all`), or None / [] if nothing survived
    the filters.
    """
    empty = [] if return_all else None
    if bx1 <= bx0 or by1 <= by0:
        return empty

    crop = image_rgb[by0:by1, bx0:bx1]
    if crop.size == 0:
        return empty

    mask = not_background_not_plot_mask(crop, bg_rgb, plot_rgb)
    if gridline_rgb is not None:
        mask = mask & (~color_mask(crop, gridline_rgb))
    if axis_rgb is not None:
        mask = mask & (~color_mask(crop, axis_rgb))

    m8 = mask.astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)

    found = []
    halo = int(config.UNIT_EXCLUDE_HALO_PX)
    sub_height: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < config.TICK_SEARCH_LOCAL_MIN_AREA:
            continue
        lx = int(stats[i, cv2.CC_STAT_LEFT]); ly = int(stats[i, cv2.CC_STAT_TOP])
        lw = int(stats[i, cv2.CC_STAT_WIDTH]); lh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Halo widens the exclusion by UNIT_EXCLUDE_HALO_PX in every direction:
        # a tightly-measured unit bbox sits pixel-perfect on the glyph but its
        # anti-aliased skirt can extend a couple of pixels further, and the tick
        # search's expansion windows sweep past that skirt — so a naked-edge
        # exclusion still lets fragments of the same unit glyph through as
        # phantom numeric hits. Applied via bbox coordinate math rather than
        # inflating the stored bbox so the visual overlay stays truthful.
        if exclude_bboxes:
            gx0, gy0 = bx0 + lx, by0 + ly
            gx1, gy1 = gx0 + lw, gy0 + lh
            overlap = False
            for (cx0, cy0, cw, ch) in exclude_bboxes:
                ex0, ey0 = cx0 - halo, cy0 - halo
                ex1, ey1 = cx0 + cw + halo, cy0 + ch + halo
                if not (gx1 < ex0 or ex1 < gx0 or gy1 < ey0 or ey1 < gy0):
                    overlap = True
                    break
            if overlap:
                continue

        if lh < config.TICK_SEARCH_MIN_LABEL_HEIGHT_PX:
            # Individually too flat to be a real character, but a lone "1" is a
            # ~2 px × 10 px vertical stroke that anti-aliasing can chop into
            # 2-3 sub-height fragments — pool them here so vertically-adjacent
            # column-shared fragments can be merged back into one blob below.
            # The primary pass still discards a lone sub-height fragment (as a
            # gridline/axis-line fleck) — only cluster-recovered ones survive.
            sub_height.append((lx, ly, lw, lh))
            continue

        found.append((lx, ly, lw, lh))

    if sub_height:
        recovered = _recover_vertical_fragments(sub_height)
        # Halo-based exclude filter re-run against the merged fragments so a
        # column of µ-skirt splinters can't sneak back in as a "recovered" hit.
        if exclude_bboxes:
            keep = []
            for (lx, ly, lw, lh) in recovered:
                gx0, gy0 = bx0 + lx, by0 + ly
                gx1, gy1 = gx0 + lw, gy0 + lh
                overlap = False
                for (cx0, cy0, cw, ch) in exclude_bboxes:
                    ex0, ey0 = cx0 - halo, cy0 - halo
                    ex1, ey1 = cx0 + cw + halo, cy0 + ch + halo
                    if not (gx1 < ex0 or ex1 < gx0 or gy1 < ey0 or ey1 < gy0):
                        overlap = True
                        break
                if not overlap:
                    keep.append((lx, ly, lw, lh))
            recovered = keep
        found.extend(recovered)

    if not found:
        return empty

    if len(found) == 1 or axis is None or reference_pixel is None:
        xs0 = min(f[0] for f in found); ys0 = min(f[1] for f in found)
        xs1 = max(f[0] + f[2] for f in found); ys1 = max(f[1] + f[3] for f in found)
        bbox = (bx0 + xs0, by0 + ys0, xs1 - xs0, ys1 - ys0)
        return [bbox] if return_all else bbox

    # Multiple components: cluster by proximity, then keep only the
    # cluster nearest the reference pixel (the gridline or unit anchor
    # this window was built around).
    heights = [f[3] for f in found]
    med_h = float(np.median(heights))
    gap_tol = max(3.0, 0.75 * med_h)

    items = sorted(found, key=lambda f: f[0])
    clusters: list[list[tuple[int, int, int, int]]] = [[items[0]]]
    for f in items[1:]:
        last = clusters[-1][-1]
        gap = f[0] - (last[0] + last[2])
        last_cy = last[1] + last[3] / 2.0
        f_cy = f[1] + f[3] / 2.0
        same_row = abs(f_cy - last_cy) < med_h
        if gap < gap_tol and same_row:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    def _merge_global(cl):
        xs0 = min(c[0] for c in cl); ys0 = min(c[1] for c in cl)
        xs1 = max(c[0] + c[2] for c in cl); ys1 = max(c[1] + c[3] for c in cl)
        return (bx0 + xs0, by0 + ys0, xs1 - xs0, ys1 - ys0)

    merged_clusters = [_merge_global(cl) for cl in clusters]

    def _axis_dist(m):
        gx, gy, gw, gh = m
        center = (gx + gw / 2.0) if axis == "x" else (gy + gh / 2.0)
        return abs(center - reference_pixel)

    if return_all:
        return sorted(merged_clusters, key=_axis_dist)
    return min(merged_clusters, key=_axis_dist)


def find_tick_label_at_gridline(image_rgb: np.ndarray,
                                gridline_pixel: float,
                                axis_line_pos: float,
                                axis: str,
                                bg_rgb: tuple[int, int, int],
                                plot_rgb: tuple[int, int, int],
                                axis_rgb: Optional[tuple[int, int, int]] = None,
                                gridline_rgb: Optional[tuple[int, int, int]] = None,
                                base_half_width: Optional[float] = None,
                                base_parallel_half_width: Optional[float] = None,
                                exclude_bboxes: Optional[list[tuple[int, int, int, int]]] = None
                                ) -> Optional[tuple[int, int, int, int]]:
    """
    Search for the tick-label blob belonging to one specific gridline.

    `exclude_bboxes` (full-image coords) are regions the numeric search must
    ignore — used to keep the already-identified axis unit label (e.g. the
    far-end "µm" that sits in place of the last tick's number) from being
    picked up here and mis-OCR'd as a failed/garbage numeric read.

    Starts with a tight window centered on the gridline's own pixel
    position and expands it (config.TICK_SEARCH_EXPANSIONS) if nothing is
    found — this is the "redundancy" pass for labels too thin or too far
    off-center to be caught by the initial window. At each expansion level,
    both a perpendicular (into the margin) and a parallel (along the axis
    line) window are tried — see _cross_search_boxes.

    `axis_line_pos` is the true pixel row/column of the axis line (see
    _local_tick_search_box) — the band starts there, not at a possibly
    padded plot_region edge.

    Returns a bbox (x, y, w, h) in full-image coordinates, or None if no
    candidate was found after all expansions.
    """
    h, w = image_rgb.shape[:2]
    if base_half_width is None:
        base_half_width = float(config.TICK_SEARCH_PERP_HALF_WIDTH)
    if base_parallel_half_width is None:
        base_parallel_half_width = base_half_width

    for expand in config.TICK_SEARCH_EXPANSIONS:
        band = int(config.TICK_LABEL_BAND_PX * expand)
        perp = base_half_width * expand
        parallel = base_parallel_half_width * expand
        boxes = _cross_search_boxes(
            gridline_pixel, axis_line_pos, axis, band, perp, parallel
        )
        for (bx0f, by0f, bx1f, by1f) in boxes:
            bx0 = max(0, int(np.floor(bx0f))); by0 = max(0, int(np.floor(by0f)))
            bx1 = min(w, int(np.ceil(bx1f)));  by1 = min(h, int(np.ceil(by1f)))

            bbox = _local_blob_search(
                image_rgb, bx0, by0, bx1, by1,
                bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
                exclude_bboxes=exclude_bboxes,
                axis=axis, reference_pixel=gridline_pixel,
            )
            if bbox is not None:
                if config.CALIB_DEBUG_LOG:
                    logger.debug(
                        "tick search: gridline=%.1f axis=%s expand=%.1f -> bbox=%s",
                        gridline_pixel, axis, expand, bbox,
                    )
                return bbox

    if config.CALIB_DEBUG_LOG:
        logger.debug(
            "tick search: gridline=%.1f axis=%s -> no candidate after all expansions",
            gridline_pixel, axis,
        )
    return None


def _ocr_bbox_numeric(image_rgb: np.ndarray,
                      bbox: tuple[int, int, int, int],
                      pad: int = config.OCR_LABEL_PAD_PX
                      ) -> tuple[Optional[float], float]:
    """
    Crop bbox (with padding), OCR as a number, trying inversion as fallback.

    Skinny-glyph retry ladder: a lone "1" on this instrument is a ~2 px × 10 px
    vertical stroke that `ocr_number`'s digit allowlist at 4× upscale cannot
    reliably classify — EasyOCR prefers "I"/"l"/"|" and the allowlist rejects
    those. Two extra attempts run only when the bbox is narrower than
    `NARROW_NUM_WIDTH_PX` AND the primary read failed:
      (a) re-crop with `pad_x = NARROW_NUM_EXTRA_PAD_PX`, re-run the allowlisted
          OCR upright + inverted — extra whitespace context often flips the
          classifier;
      (b) if still None and `NARROW_NUM_ALLOW_TEXT_FALLBACK`, run `ocr_text`
          (no allowlist) once and coerce through `postprocess_number_string`,
          whose `_OCR_SUBS` table maps "I"/"l"/"|" -> "1".
    The retries only fire on a genuinely skinny bbox, so at most 3 extra OCR
    calls per axis in the pathological case; typical images add zero.
    """
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
    crop = image_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    value, conf = ocr_number(gray)
    if value is None:
        value, conf = ocr_number(gray, invert=True)
    if value is not None:
        return value, conf

    if bw >= int(config.NARROW_NUM_WIDTH_PX):
        return value, conf

    wide_pad = int(config.NARROW_NUM_EXTRA_PAD_PX)
    x0w = max(0, x - wide_pad); y0w = max(0, y - pad)
    x1w = min(w, x + bw + wide_pad); y1w = min(h, y + bh + pad)
    wide_crop = image_rgb[y0w:y1w, x0w:x1w]
    if wide_crop.size == 0:
        return value, conf

    wide_gray = cv2.cvtColor(wide_crop, cv2.COLOR_RGB2GRAY)
    value, conf = ocr_number(wide_gray)
    if value is None:
        value, conf = ocr_number(wide_gray, invert=True)
    if value is not None:
        return value, conf

    if not config.NARROW_NUM_ALLOW_TEXT_FALLBACK:
        return value, conf

    text, text_conf = ocr_text(wide_gray)
    if text is None:
        text, text_conf = ocr_text(wide_gray, invert=True)
    if text is None:
        return None, 0.0
    coerced = postprocess_number_string(text)
    if coerced is None:
        return None, 0.0
    return float(coerced), float(text_conf)


def _gridline_in_exclude(gridline_pixel: float,
                         axis: str,
                         exclude_bboxes: Optional[list[tuple[int, int, int, int]]]
                         ) -> bool:
    """True if a gridline's position falls within an excluded bbox's span
    along the axis direction — i.e. that gridline holds the (already-claimed)
    unit label rather than a number, so a numeric miss there is expected and
    should not be reported as a failure.

    The comparison uses a ±UNIT_EXCLUDE_HALO_PX margin. A tightly-measured unit
    bbox may sit 1-2 px inside the actual last-gridline pixel position — pixel-
    exact containment would then miss it and the expansion window would re-open
    the phantom-numeric-search path at that gridline. Symmetric to the halo the
    `_local_blob_search` exclusion uses.
    """
    if not exclude_bboxes:
        return False
    halo = float(config.UNIT_EXCLUDE_HALO_PX)
    for (ex, ey, ew, eh) in exclude_bboxes:
        if axis == "x":
            if (ex - halo) <= gridline_pixel <= (ex + ew + halo):
                return True
        else:
            if (ey - halo) <= gridline_pixel <= (ey + eh + halo):
                return True
    return False


def find_and_ocr_ticks_per_gridline(image_rgb: np.ndarray,
                                    gridline_positions: np.ndarray,
                                    axis_line_pos: float,
                                    axis: str,
                                    bg_rgb: tuple[int, int, int],
                                    plot_rgb: tuple[int, int, int],
                                    axis_rgb: Optional[tuple[int, int, int]] = None,
                                    gridline_rgb: Optional[tuple[int, int, int]] = None,
                                    exclude_bboxes: Optional[list[tuple[int, int, int, int]]] = None
                                    ) -> tuple[list[tuple[float, float, float]],
                                              list[tuple[int, int, int, int]],
                                              list[float]]:
    """
    For every gridline, run the anchored local search + OCR. This is the
    primary tick-finding path — it guarantees an attempt per gridline
    rather than depending on a blob surviving a global filter pass.

    `exclude_bboxes` (full-image coords) are regions to skip — the axis's
    already-identified unit label. A gridline whose position sits inside an
    excluded bbox is neither searched-into-that-region nor counted as a
    missing numeric read (it holds a unit, not a number).

    Returns
    -------
    (pairs, bboxes, missing)
        pairs   : list of (gridline_pixel, ocr_value, ocr_confidence).
                  The pixel coordinate is the gridline's own position, not
                  the label's bbox centroid — see module docstring.
        bboxes  : bbox aligned 1:1 with `pairs`.
        missing : gridline pixel positions where no label was found or
                  OCR failed, for the caller to report/log. Gridlines that
                  coincide with an excluded unit bbox are omitted here.
    """
    if len(gridline_positions) == 0:
        return [], [], []

    sorted_g = np.sort(np.asarray(gridline_positions, dtype=float))
    if len(sorted_g) > 1:
        spacing = float(np.median(np.diff(sorted_g)))
    else:
        spacing = config.TICK_SEARCH_PERP_HALF_WIDTH * 2.0
    base_half_width = max(float(config.TICK_SEARCH_PERP_HALF_WIDTH), spacing * 0.4)
    # Parallel-arm half-width (see _cross_search_boxes): sized from gridline
    # spacing per the project's redundant-localization requirement, wide
    # enough to bridge to a neighboring gridline's label if a tick's own
    # label is offset along the axis direction.
    base_parallel_half_width = max(base_half_width, spacing * 0.5)

    pairs: list[tuple[float, float, float]] = []
    bboxes: list[tuple[int, int, int, int]] = []
    missing: list[float] = []

    for g in gridline_positions:
        g = float(g)
        # A gridline sitting on the axis unit label (e.g. the far-end "µm"
        # that replaces the last tick's number) holds no number — skip the
        # tick search entirely. Even with `exclude_bboxes` threaded down, an
        # expansion-window sweep can still catch a fragment of the µm glyph or
        # a same-column neighbour and OCR it into a spurious value that pools
        # into `pairs` and pulls the confident-anchor span with it. There is
        # nothing legitimate to find at a unit gridline, so the correct
        # behaviour is to not look.
        if _gridline_in_exclude(g, axis, exclude_bboxes):
            continue

        bbox = find_tick_label_at_gridline(
            image_rgb, g, axis_line_pos, axis,
            bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
            base_half_width=base_half_width,
            base_parallel_half_width=base_parallel_half_width,
            exclude_bboxes=exclude_bboxes,
        )
        if bbox is None:
            missing.append(g)
            continue

        value, conf = _ocr_bbox_numeric(image_rgb, bbox)
        if config.CALIB_DEBUG_LOG:
            logger.debug(
                "numeric OCR: gridline=%.1f axis=%s bbox=%s -> value=%s conf=%.3f",
                g, axis, bbox, value, conf,
            )
        if value is None:
            missing.append(g)
            continue

        pairs.append((g, float(value), float(conf)))
        bboxes.append(bbox)

    return pairs, bboxes, missing


def backverify_inferred_tick(image_rgb: np.ndarray,
                             gridline_pixel: float,
                             axis_line_pos: float,
                             axis: str,
                             bg_rgb: tuple[int, int, int],
                             plot_rgb: tuple[int, int, int],
                             predicted_value: float,
                             step: float,
                             axis_rgb: Optional[tuple[int, int, int]] = None,
                             gridline_rgb: Optional[tuple[int, int, int]] = None,
                             exclude_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
                             base_half_width: Optional[float] = None,
                             base_parallel_half_width: Optional[float] = None,
                             ) -> Optional[tuple[float, float, tuple[int, int, int, int]]]:
    """
    Retry the anchored tick search + OCR at a gridline whose value was filled
    in by `reconstruct_tick_values`, and promote the entry to confirmed if the
    OCR read agrees with the arithmetic-progression prediction.

    Two guards:
      * `abs(ocr_value - predicted_value)` must be
         `<= INFERRED_BACKVERIFY_AGREEMENT_FRAC_OF_STEP * step` — a matching
         read is trusted because the sequence prediction already fits the
         arithmetic progression, so a value that lands within a fraction of
         one step of it can only be the true tick;
      * `ocr_conf >= INFERRED_BACKVERIFY_MIN_OCR_CONF` — deliberately below the
         primary `OCR_MIN_CONFIDENCE`; the prediction is the trust anchor, not
         the OCR confidence.

    Returns `(ocr_value, ocr_conf, bbox)` on acceptance, or None otherwise.
    Never widens `exclude_bboxes` reach or changes the fit input on failure —
    a rejected retry simply leaves the entry inferred as before.
    """
    if not config.INFERRED_BACKVERIFY_ENABLED:
        return None
    if step <= 0:
        return None

    bbox = find_tick_label_at_gridline(
        image_rgb, gridline_pixel, axis_line_pos, axis,
        bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
        base_half_width=base_half_width,
        base_parallel_half_width=base_parallel_half_width,
        exclude_bboxes=exclude_bboxes,
    )
    if bbox is None:
        return None

    value, conf = _ocr_bbox_numeric(image_rgb, bbox)
    if value is None:
        return None
    if conf < float(config.INFERRED_BACKVERIFY_MIN_OCR_CONF):
        return None

    tol = float(config.INFERRED_BACKVERIFY_AGREEMENT_FRAC_OF_STEP) * float(step)
    if abs(float(value) - float(predicted_value)) > tol:
        if config.CALIB_DEBUG_LOG:
            logger.debug(
                "backverify: gridline=%.1f axis=%s ocr=%s predicted=%s "
                "step=%s tol=%.3f -> reject (disagreement)",
                gridline_pixel, axis, value, predicted_value, step, tol,
            )
        return None

    if config.CALIB_DEBUG_LOG:
        logger.debug(
            "backverify: gridline=%.1f axis=%s ocr=%s predicted=%s "
            "step=%s tol=%.3f -> promote",
            gridline_pixel, axis, value, predicted_value, step, tol,
        )
    return float(value), float(conf), bbox


# ---------------- Unit label auto-detection (full-band scan) ----------------
#
# Unit strings ("um", "nm", "mV") are not numeric, so ocr_number always
# returns None on them. The unit can sit anywhere along an axis — on this
# instrument the x-axis unit is printed at the FAR END of the axis (in place
# of the last tick's number) while the y-axis unit sits near the origin — so
# rather than guessing anchor points, scan the entire tick-label band for the
# axis in one pass and classify every text blob found (the "find all areas of
# text, then pick the unit" workflow the user asked for). Bounded perpendicular
# to the axis by a fixed margin depth so unrelated content elsewhere in the
# image stays out of range.

def _axis_band_blobs(image_rgb: np.ndarray,
                     axis_line_pos: float,
                     cross_axis_pos: Optional[float],
                     axis: str,
                     bg_rgb: tuple[int, int, int],
                     plot_rgb: tuple[int, int, int],
                     axis_rgb: Optional[tuple[int, int, int]] = None,
                     gridline_rgb: Optional[tuple[int, int, int]] = None
                     ) -> list[tuple[int, int, int, int]]:
    """
    Return every clustered text blob (full-image coords) in the axis's
    tick-label band with a single _local_blob_search(return_all=True) pass —
    reusing the same masking / connected-components / proximity-clustering the
    numeric search uses.

    Band geometry:
      - PERPENDICULAR to the axis: reaches config.UNIT_SEARCH_BAND_DEPTH_PX
        into the margin (below the x-axis line / left of the y-axis line).
      - ALONG the axis: for the x-axis, from the cross-axis (y-axis) column to
        the right edge; for the y-axis, the full image height.

    The x-axis along-band deliberately starts at `cross_axis_pos` rather than
    column 0: the two axes share the bottom-left origin corner, where the
    y-axis unit label sits. Spanning the full width would let the x-axis unit
    search pick up the *y-axis'* unit blob in that corner (e.g. report the
    y-axis "nm" as the x-axis unit). Everything belonging to the x-axis
    (its tick labels and its own far-end unit) lies to the right of the y-axis
    column, so starting there loses nothing and removes the collision. The
    y-axis band needs no such trim: all x-axis content lies to the right of the
    y-axis column, outside this left-of-axis strip.
    """
    h, w = image_rgb.shape[:2]
    depth = int(config.UNIT_SEARCH_BAND_DEPTH_PX)
    if axis == "x":
        bx0 = 0 if cross_axis_pos is None else max(0, int(np.floor(cross_axis_pos)))
        bx1 = w
        by0 = max(0, int(np.floor(axis_line_pos)))
        by1 = min(h, by0 + depth)
    else:
        by0, by1 = 0, h
        bx1 = min(w, int(np.ceil(axis_line_pos)))
        bx0 = max(0, bx1 - depth)

    # axis + reference_pixel must be non-None so _local_blob_search returns the
    # separate per-blob clusters instead of merging the whole band into one
    # bbox; reference_pixel=0.0 just orders them along the axis (we re-rank).
    clusters = _local_blob_search(
        image_rgb, bx0, by0, bx1, by1,
        bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
        axis=axis, reference_pixel=0.0,
        return_all=True,
    )
    return clusters or []


def _wholeband_unit_ocr(image_rgb: np.ndarray,
                        axis_line_pos: float,
                        cross_axis_pos: Optional[float],
                        axis: str,
                        bg_rgb: tuple[int, int, int],
                        plot_rgb: tuple[int, int, int],
                        axis_rgb: Optional[tuple[int, int, int]] = None,
                        gridline_rgb: Optional[tuple[int, int, int]] = None
                        ) -> Optional[tuple[str, float, None]]:
    """
    Last-resort unit detection: crop the entire axis tick-label band as one
    image and run `ocr_text` on it. Sometimes EasyOCR's own detector segments
    the "µm" glyph cleanly at page scale even though our connected-components
    extractor split it into unrecognisable fragments — this pass gives EasyOCR
    that page-scale look.

    Only runs when the per-blob classification and the ASCII-only textual
    fallback both produced nothing, so it can never regress a currently-passing
    case. Accept iff there is exactly one axis-band token that matches a known
    variant via `score_unit_match`: on this instrument's tick labels
    (`nm`/`um`/`mm` never appear as a numeric label), that condition is
    genuinely met only when the band contains the unit itself.

    Returns `(canonical_unit, ocr_conf, None)` on success — bbox=None because
    the whole-band OCR doesn't localise the unit within the band, and the
    debug overlay handles bbox=None gracefully. Returns None if nothing
    matched (unit remains unrecognised — the pipeline already warns).
    """
    h, w = image_rgb.shape[:2]
    depth = int(config.UNIT_SEARCH_BAND_DEPTH_PX)
    if axis == "x":
        bx0 = 0 if cross_axis_pos is None else max(0, int(np.floor(cross_axis_pos)))
        bx1 = w
        by0 = max(0, int(np.floor(axis_line_pos)))
        by1 = min(h, by0 + depth)
    else:
        by0, by1 = 0, h
        bx1 = min(w, int(np.ceil(axis_line_pos)))
        bx0 = max(0, bx1 - depth)

    if bx1 <= bx0 or by1 <= by0:
        return None
    crop = image_rgb[by0:by1, bx0:bx1]
    if crop.size == 0:
        return None

    max_side = float(config.UNIT_WHOLEBAND_OCR_MAX_SIDE_PX)
    hh, ww = crop.shape[:2]
    scale = min(1.0, max_side / max(hh, ww))
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    text, conf = ocr_text(crop)
    if text is None:
        text, conf = ocr_text(crop, invert=True)
    if text is None:
        return None

    # A single call can return whitespace-separated tokens; scan each for a
    # variant match rather than only testing the whole string.
    matches: list[tuple[str, float]] = []
    seen = set()
    for tok in text.split():
        canonical, score = score_unit_match(tok)
        if score >= config.UNIT_MATCH_MIN_CONFIDENCE:
            key = canonical
            if key not in seen:
                seen.add(key)
                matches.append((canonical, float(conf)))
        else:
            suffix = extract_unit_suffix(tok)
            if suffix:
                s_canonical, s_score = score_unit_match(suffix)
                if s_score >= config.UNIT_MATCH_MIN_CONFIDENCE:
                    key = s_canonical
                    if key not in seen:
                        seen.add(key)
                        matches.append((s_canonical, float(conf)))

    if len(matches) != 1:
        return None
    canonical, conf_val = matches[0]
    if not canonical.isascii():
        return None
    return canonical, conf_val, None


def _ocr_bbox_text(image_rgb: np.ndarray,
                   bbox: tuple[int, int, int, int],
                   pad: int = config.OCR_LABEL_PAD_PX,
                   pad_x: Optional[int] = None,
                   pad_y: Optional[int] = None
                   ) -> tuple[Optional[str], float]:
    """
    Crop bbox (with padding), OCR as general text, trying inversion as fallback.

    `pad_x`/`pad_y` override `pad` per-direction when given. The unit search
    passes a larger `pad_x` (config.UNIT_OCR_CROP_PAD_PX): tick/unit text is
    drawn horizontally, so a micro sign that connected-components split off
    into its own blob sits directly LEFT of the "m" — widening the crop
    horizontally pulls it back in so the token reads "um" instead of a bare
    "m". Vertical padding stays tight so the crop doesn't swallow an adjacent
    text row.
    """
    h, w = image_rgb.shape[:2]
    px = pad_x if pad_x is not None else pad
    py = pad_y if pad_y is not None else pad
    x, y, bw, bh = bbox
    x0 = max(0, x - px); y0 = max(0, y - py)
    x1 = min(w, x + bw + px); y1 = min(h, y + bh + py)
    crop = image_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0

    text, conf = ocr_text(crop)
    if text is None:
        text, conf = ocr_text(crop, invert=True)
    return text, conf


def find_unit_label_at_axis(image_rgb: np.ndarray,
                            gridline_positions: np.ndarray,
                            axis_line_pos: float,
                            axis: str,
                            bg_rgb: tuple[int, int, int],
                            plot_rgb: tuple[int, int, int],
                            axis_rgb: Optional[tuple[int, int, int]] = None,
                            gridline_rgb: Optional[tuple[int, int, int]] = None,
                            cross_axis_pos: Optional[float] = None
                            ) -> tuple[Optional[str], float, Optional[tuple[int, int, int, int]]]:
    """
    Find the axis unit label by scanning the whole tick-label band once
    (_axis_band_blobs) and classifying every text blob, rather than probing a
    couple of guessed anchor points. This is robust to the unit sitting
    anywhere along the axis — near the origin (y-axis here) or at the far end
    in place of the last tick's number (x-axis here).

    `cross_axis_pos` (the OTHER axis's line position) bounds the x-axis band so
    it doesn't reach into the shared origin corner and steal the y-axis unit —
    see _axis_band_blobs.

    Classification is by CONTENT, not geometry: each blob's OCR'd text is
    scored against known unit spellings via `score_unit_match` (config's
    UNIT_CANONICAL_FORMS variant lookup — deliberately not fuzzy, since
    nm/um/mm are visually similar but 1000x apart in scale). If the whole
    string doesn't match, a MERGED number+unit token (e.g. "100um", where a
    tick label and the unit fused into one component) gets a second attempt via
    `extract_unit_suffix`. Unit-candidate blobs are OCR'd with extra horizontal
    padding (config.UNIT_OCR_CROP_PAD_PX) so a micro sign split off into its own
    connected component is pulled back into the crop and reads "um" rather than
    a bare "m".

    Among blobs that clear both UNIT_MATCH_MIN_CONFIDENCE and
    UNIT_OCR_MIN_CONFIDENCE, the winner is chosen by, in order:
      1. most OCR'd characters — a 2-char "um" beats any 1-char read; combined
         with "m"/"cm" no longer being valid units (see config), this is what
         kills the far-end "µm" -> "m" misread,
      2. higher OCR confidence,
      3. closeness to a first/last axis extreme (units conventionally sit at an
         end label) — a soft tiebreak only, never overriding 1 or 2.
    A blob that isn't a recognized unit but also isn't a clean number is kept
    as a last-resort display fallback, EXCEPT a bare single character: on this
    instrument those are always degraded unit fragments (a lone "m"/"n"/"u"
    left after a micro sign split), never a real label.

    Returns (unit_string_or_None, confidence, bbox_or_None).
    """
    if len(gridline_positions) == 0:
        return None, 0.0, None

    clusters = _axis_band_blobs(
        image_rgb, axis_line_pos, cross_axis_pos, axis,
        bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
    )
    if not clusters:
        return None, 0.0, None

    def _axis_center(b):
        bx, by, bw, bh = b
        return (bx + bw / 2.0) if axis == "x" else (by + bh / 2.0)

    centers = [_axis_center(b) for b in clusters]
    lo, hi = min(centers), max(centers)
    span = max(1.0, hi - lo)

    # (char_count, ocr_conf, position_bonus, canonical, bbox)
    unit_candidates: list[tuple[int, float, float, str, tuple[int, int, int, int]]] = []
    fallback_text, fallback_conf, fallback_bbox = None, 0.0, None

    for bbox, center in zip(clusters, centers):
        text, conf = _ocr_bbox_text(
            image_rgb, bbox,
            pad_x=config.UNIT_OCR_CROP_PAD_PX, pad_y=config.OCR_LABEL_PAD_PX,
        )
        if text is None:
            continue

        canonical, score = score_unit_match(text)
        matched_token = text.strip()
        if score < config.UNIT_MATCH_MIN_CONFIDENCE:
            # Merged number+unit token ("100um") — retry on the trailing suffix.
            suffix = extract_unit_suffix(text)
            if suffix:
                s_canonical, s_score = score_unit_match(suffix)
                if s_score > score:
                    canonical, score, matched_token = s_canonical, s_score, suffix.strip()

        if config.CALIB_DEBUG_LOG:
            logger.debug(
                "unit band: axis=%s bbox=%s raw_text=%r conf=%.3f "
                "-> canonical=%r score=%.2f numeric=%s",
                axis, bbox, text, conf, canonical, score,
                postprocess_number_string(text),
            )

        if (score >= config.UNIT_MATCH_MIN_CONFIDENCE
                and conf >= config.UNIT_OCR_MIN_CONFIDENCE):
            dist_to_extreme = min(abs(center - lo), abs(center - hi))
            position_bonus = 1.0 - (dist_to_extreme / span)  # 1.0 at an end, ->0 mid-axis
            unit_candidates.append(
                (len(matched_token), float(conf), float(position_bonus), canonical, bbox)
            )
            continue

        # Not a recognized unit. Keep as a last-resort display fallback, but
        # never a bare single character (always a degraded unit fragment here),
        # never a clean number (that's a tick value), and never a non-ASCII
        # string — every valid unit on this instrument becomes ASCII once
        # `_clean_unit_text` folds µ/μ into "u", so a non-ASCII fallback is by
        # definition a garbage OCR read that would render as `?m`/`m` under
        # Hershey Simplex and mislead the user into thinking the unit was
        # recovered as "m". Rejecting it makes the header honestly say
        # `unit=?` — the pipeline already warns "unit label not recognized"
        # in that case, so the failure mode is transparent.
        if postprocess_number_string(text) is None:
            normalized = normalize_unit_string(text)
            if (normalized and len(normalized) >= 2
                    and normalized.isascii()
                    and conf > fallback_conf):
                fallback_text, fallback_conf, fallback_bbox = normalized, conf, bbox

    if unit_candidates:
        best = max(unit_candidates, key=lambda c: (c[0], c[1], c[2]))
        _char_count, best_conf, _pos, best_canonical, best_bbox = best
        # Canonical forms in config.UNIT_CANONICAL_FORMS are ASCII by design,
        # but assert once at the return point so a future addition can't sneak
        # a unicode key back into the render path.
        assert best_canonical.isascii(), (
            f"non-ASCII canonical unit {best_canonical!r} — "
            "config.UNIT_CANONICAL_FORMS keys must stay ASCII"
        )
        return best_canonical, best_conf, best_bbox

    if fallback_text is not None:
        return fallback_text, fallback_conf, fallback_bbox

    if config.UNIT_WHOLEBAND_OCR_ENABLED:
        wb = _wholeband_unit_ocr(
            image_rgb, axis_line_pos, cross_axis_pos, axis,
            bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
        )
        if wb is not None:
            return wb

    return None, 0.0, None


# ---------------- Text blob detection (legacy utility, no longer used ----------------
# ---------------- by the main calibration pipeline) ------------------------------

def find_text_blobs(image_rgb: np.ndarray,
                    bg_rgb: tuple[int, int, int],
                    plot_rgb: tuple[int, int, int],
                    plot_region: tuple[int, int, int, int],
                    axis_rgb: Optional[tuple[int, int, int]] = None,
                    gridline_rgb: Optional[tuple[int, int, int]] = None,
                    tol: int = config.COLOR_MASK_TOLERANCE
                    ) -> list[dict]:
    """
    Build not-bg/not-plot mask, optionally subtract gridline/axis pixels,
    run connected components, filter by size and aspect ratio.

    Returns list of {'bbox': (x,y,w,h), 'centroid': (cx,cy), 'area': n}.
    Coordinates are in full-image space.

    Retained as a general utility (e.g. for ad-hoc inspection) but no
    longer called by the main pipeline — both the numeric tick search and
    the unit-label search now use anchored local windows instead of a
    whole-image blob scan.
    """
    h, w = image_rgb.shape[:2]
    mask = not_background_not_plot_mask(image_rgb, bg_rgb, plot_rgb, tol)
    if gridline_rgb is not None:
        mask = mask & (~color_mask(image_rgb, gridline_rgb, tol))
    if axis_rgb is not None:
        mask = mask & (~color_mask(image_rgb, axis_rgb, tol))

    m8 = mask.astype(np.uint8) * 255
    n, _, stats, centroids = cv2.connectedComponentsWithStats(m8, connectivity=8)

    max_area = int(h * w * config.TEXT_BLOB_MAX_AREA_FRAC)
    blobs = []
    for i in range(1, n):  # skip background (label 0)
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < config.TEXT_BLOB_MIN_AREA or area > max_area:
            continue
        if bh <= 0:
            continue
        aspect = bw / bh
        if aspect < config.TEXT_BLOB_MIN_ASPECT or aspect > config.TEXT_BLOB_MAX_ASPECT:
            continue

        blobs.append({
            "bbox": (x, y, bw, bh),
            "centroid": (float(centroids[i, 0]), float(centroids[i, 1])),
            "area": area,
        })
    return blobs


# ---------------- Robust linear fit ----------------

def fit_pixel_to_unit(pairs: list[tuple[float, float, float]],
                      method: Optional[str] = None) -> dict:
    """
    Fit line: value = slope * pixel + intercept.

    Parameters
    ----------
    pairs  : list of (pixel, value, conf); conf is unused here but preserved
             for downstream weighting/reporting.
    method : 'theil_sen' (robust), 'ransac', or 'lstsq'. None reads the
             current config.FIT_METHOD.

    Returns
    -------
    {'slope', 'intercept', 'r_squared', 'residuals', 'inlier_mask',
     'method', 'n_points'}
    """
    # Read at call time, not as a default argument — a default would be
    # evaluated once at import and ignore any later Preferences change.
    if method is None:
        method = config.FIT_METHOD

    if len(pairs) < config.MIN_TICKS_FOR_FIT:
        raise ValueError(
            f"insufficient tick pairs for fit: got {len(pairs)}, "
            f"need >= {config.MIN_TICKS_FOR_FIT}"
        )

    px = np.array([p[0] for p in pairs], dtype=float)
    val = np.array([p[1] for p in pairs], dtype=float)

    if method == "theil_sen":
        slope, _scipy_intercept, _, _ = sstats.theilslopes(val, px)
        # scipy's intercept uses y_median - slope*x_median, which is biased
        # by outliers that shift the y-median. Recompute as median(y - slope*x),
        # which is the standard robust intercept and pairs correctly with the
        # Theil-Sen slope.
        intercept = float(np.median(val - slope * px))
        predicted = slope * px + intercept
        residuals = val - predicted
        val_range = val.max() - val.min()
        tol = max(1e-9, config.OUTLIER_RESIDUAL_TOLERANCE * val_range)
        inlier_mask = np.abs(residuals) <= tol
    elif method == "ransac":
        from sklearn.linear_model import RANSACRegressor
        model = RANSACRegressor(random_state=0)
        model.fit(px.reshape(-1, 1), val)
        slope = float(model.estimator_.coef_[0])
        intercept = float(model.estimator_.intercept_)
        predicted = slope * px + intercept
        residuals = val - predicted
        inlier_mask = model.inlier_mask_
    else:  # lstsq
        slope, intercept = np.polyfit(px, val, 1)
        predicted = slope * px + intercept
        residuals = val - predicted
        val_range = val.max() - val.min()
        tol = max(1e-9, config.OUTLIER_RESIDUAL_TOLERANCE * val_range)
        inlier_mask = np.abs(residuals) <= tol

    # R^2 computed on inliers only (fair to the robust methods)
    if inlier_mask.sum() >= 2:
        y_in = val[inlier_mask]
        p_in = predicted[inlier_mask]
        ss_res = float(np.sum((y_in - p_in) ** 2))
        ss_tot = float(np.sum((y_in - y_in.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        r_squared = 0.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "residuals": residuals.tolist(),
        "inlier_mask": inlier_mask.astype(bool).tolist(),
        "method": method,
        "n_points": int(len(pairs)),
    }


# ---------------- Outlier recovery by snapping to nice steps ----------------

def recover_outliers(pairs: list[tuple[float, float, float]],
                     fit: dict) -> list[tuple[float, float, float]]:
    """
    For pairs flagged as outliers, try snapping their OCR value to the
    nearest 'nice' tick value predicted by the fit. If the corrected value
    is close enough to the prediction, keep it; otherwise drop the pair.

    Returns a new pair list (possibly shorter than input).
    """
    if not pairs:
        return pairs

    slope = fit["slope"]
    intercept = fit["intercept"]
    inlier_mask = np.array(fit["inlier_mask"])
    px = np.array([p[0] for p in pairs])
    val = np.array([p[1] for p in pairs])

    # Estimate the tick step from inlier values (fallback: from range/N)
    if inlier_mask.sum() >= 2:
        inlier_vals = np.sort(val[inlier_mask])
        step_est = float(np.median(np.diff(inlier_vals)))
        if step_est <= 0:
            step_est = abs(slope) * 20.0  # rough
    else:
        step_est = max(1e-6, (val.max() - val.min()) / max(1, len(val) - 1))

    nice_step = _nearest_nice_step(step_est)
    val_range = max(1e-9, val.max() - val.min())
    accept_tol = config.OUTLIER_RESIDUAL_TOLERANCE * val_range

    recovered: list[tuple[float, float, float]] = []
    for i, (p, v, c) in enumerate(pairs):
        if inlier_mask[i]:
            recovered.append((p, v, c))
            continue
        predicted = slope * p + intercept
        # snap prediction to nearest multiple of nice_step
        snapped = round(predicted / nice_step) * nice_step
        if abs(snapped - predicted) <= accept_tol:
            recovered.append((p, float(snapped), c))
        # else: drop this pair
    return recovered


def _nearest_nice_step(step: float) -> float:
    """Snap `step` to the nearest value in {1,2,2.5,5} * 10^k."""
    if step <= 0:
        return 1.0
    exp = np.floor(np.log10(step))
    base = step / (10 ** exp)
    nice = min(config.NICE_STEPS, key=lambda s: abs(s - base))
    return float(nice * (10 ** exp))


# ---------------- Tick-value sequence reconstruction ----------------

def reconstruct_tick_values(gridline_positions: np.ndarray,
                            cross_axis_pos: float,
                            pairs: list[tuple[float, float, float]],
                            regular: bool
                            ) -> tuple[list[tuple[float, float, float]], list[bool]]:
    """
    Reconstruct the full tick-value sequence over every REAL gridline
    (i.e. `gridline_positions` minus the cross-axis's own border line — see
    calibrate.py's axis_lines) as an arithmetic progression
    `value = v0 + i * step` over gridline index `i`, using confidently
    OCR'd reads as anchors to solve for (v0, step).

    This recovers two distinct failure modes with one mechanism:
      - a tick OCR missed entirely (a skinny "1" that never produced a
        blob at all — it's simply absent from `pairs`), and
      - a tick OCR read but got flatly wrong in a way that's invisible to
        per-point outlier rejection (e.g. two different gridlines both
        misread as the same digit — since gridlines are verified evenly
        spaced, they can't both be correct, but each individually looks
        like a plausible number).

    Gated hard on `regular` (gridline spacing must already be verified
    even — see check_gridline_regularity) and on having
    >= config.SEQ_RECONSTRUCT_MIN_CONFIDENT_ANCHORS reads whose OCR
    confidence clears config.SEQ_RECONSTRUCT_MIN_OCR_CONF: without enough
    trustworthy anchors there's nothing reliable to solve the progression
    from, and fabricating one from unreliable data would be worse than
    leaving ticks missing. On any gate failure, returns `pairs` unchanged
    (all inferred=False) — this is a pure enhancement, never a regression.

    Parameters
    ----------
    gridline_positions : ALL gridline pixel positions for this axis
                          (includes the cross-axis's own border line).
    cross_axis_pos      : the pixel position of the OTHER axis's line —
                          the one entry in gridline_positions that is a
                          border, not a real tick, and must be excluded.
    pairs               : OCR'd (pixel, value, confidence) triples from
                          find_and_ocr_ticks_per_gridline. May be missing
                          entries for gridlines OCR failed on entirely.
    regular             : from check_gridline_regularity(gridline_positions).

    Returns
    -------
    (new_pairs, inferred_mask)
        new_pairs : one (gridline_pixel, value, confidence) triple per real
                    gridline (sorted by pixel). confidence is the original
                    OCR confidence for confirmed reads, or
                    config.SEQ_RECONSTRUCT_INFERRED_CONF for values that
                    were filled in or overridden.
        inferred_mask : parallel bool list, True where the value came from
                        the sequence rather than a confirmed OCR read.
    """
    real_gridlines = sorted(
        float(g) for g in gridline_positions
        if abs(float(g) - cross_axis_pos) > 1e-6
    )
    if not regular or len(real_gridlines) < config.SEQ_RECONSTRUCT_MIN_CONFIDENT_ANCHORS:
        return list(pairs), [False] * len(pairs)

    index_of_pixel = {p: i for i, p in enumerate(real_gridlines)}

    anchors = [
        (index_of_pixel[p], v) for (p, v, c) in pairs
        if p in index_of_pixel and c >= config.SEQ_RECONSTRUCT_MIN_OCR_CONF
    ]
    if len(anchors) < config.SEQ_RECONSTRUCT_MIN_CONFIDENT_ANCHORS:
        return list(pairs), [False] * len(pairs)

    def _fit_from_anchors(anchor_list):
        """Median-of-pairwise-slopes (v0, dv), or None if degenerate."""
        pair_slopes = []
        for a in range(len(anchor_list)):
            for b in range(a + 1, len(anchor_list)):
                i1, v1 = anchor_list[a]; i2, v2 = anchor_list[b]
                if i1 != i2:
                    pair_slopes.append((v2 - v1) / (i2 - i1))
        if not pair_slopes:
            return None
        raw = float(np.median(pair_slopes))
        if raw == 0:
            return None
        step = _nearest_nice_step(abs(raw)) * (1.0 if raw > 0 else -1.0)
        intercept = float(np.median([v - i * step for i, v in anchor_list]))
        return intercept, step

    # Robust step: median of pairwise slopes (value-per-index) across all
    # anchor pairs, snapped to a "nice" step size (same snapping already
    # used for outlier recovery, applied here to the underlying axis step
    # rather than to an individual point).
    fit0 = _fit_from_anchors(anchors)
    if fit0 is None:
        return list(pairs), [False] * len(pairs)
    v0, dv = fit0

    # Refinement: an anchor confidently OCR'd but flatly wrong (e.g. a
    # crowded/dense scan's trace pixels misread as tick text near a
    # gridline) still counts toward the plain median above, and with only a
    # handful of anchors a couple of self-consistent bad ones can pull the
    # fit enough that the CORRECT anchors are the ones that look wrong next
    # pass. One refit, dropping any anchor that disagrees with the first-
    # pass line, guards against that without the cost/risk of a full
    # iterative (RANSAC-style) rejection loop.
    tol0 = config.SEQ_RECONSTRUCT_AGREEMENT_FRAC_OF_STEP * abs(dv)
    trimmed = [(i, v) for (i, v) in anchors if abs(v - (v0 + i * dv)) <= tol0]
    if (len(trimmed) < len(anchors)
            and len(trimmed) >= config.SEQ_RECONSTRUCT_MIN_CONFIDENT_ANCHORS):
        fit1 = _fit_from_anchors(trimmed)
        if fit1 is not None:
            v0, dv = fit1
            anchors = trimmed

    # Interpolate ONLY — never extrapolate. Fill-in / override is confined to
    # gridline indices BETWEEN the outermost confident anchors. A gridline
    # beyond that span (e.g. the far-end x tick that actually holds the unit
    # label rather than a number) keeps its own OCR read if it has one, and is
    # otherwise simply dropped instead of having a value fabricated for it.
    # Losing an end tick doesn't hurt the fit — the interior anchors already
    # pin the line — and it avoids inventing a value the image never showed.
    anchor_indices = [i for i, _v in anchors]
    lo_idx, hi_idx = min(anchor_indices), max(anchor_indices)

    agree_tol = config.SEQ_RECONSTRUCT_AGREEMENT_FRAC_OF_STEP * abs(dv)
    ocr_by_pixel = {p: (v, c) for (p, v, c) in pairs}

    new_pairs: list[tuple[float, float, float]] = []
    inferred_mask: list[bool] = []
    for i, p in enumerate(real_gridlines):
        entry = ocr_by_pixel.get(p)
        if not (lo_idx <= i <= hi_idx):
            # Outside the confident-anchor span: keep a real OCR read as-is,
            # but never fabricate/override (that would be extrapolation).
            if entry is not None:
                v, c = entry
                new_pairs.append((p, v, c))
                inferred_mask.append(False)
            continue

        seq_val = v0 + i * dv
        if entry is not None:
            v, c = entry
            if abs(v - seq_val) <= agree_tol:
                new_pairs.append((p, v, c))
                inferred_mask.append(False)
                continue
        # Between anchors, and either missing entirely or OCR'd but disagreeing
        # with the verified-even sequence -> interpolate / override.
        new_pairs.append((p, seq_val, config.SEQ_RECONSTRUCT_INFERRED_CONF))
        inferred_mask.append(True)

    if config.CALIB_DEBUG_LOG:
        n_inferred = sum(inferred_mask)
        if n_inferred:
            logger.debug(
                "sequence reconstruction: %d/%d tick(s) inferred/overridden "
                "(v0=%.4g, step=%.4g)", n_inferred, len(new_pairs), v0, dv,
            )

    return new_pairs, inferred_mask
