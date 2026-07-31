"""
calibrate.py
------------
Top-level orchestrator. Drop-in entry point:

    from plot_calibration.calibrate import calibrate_image
    result = calibrate_image(image_rgb, "myfile.png", "profiles.json")
    if result is None:
        # calibration failed — log a warning, skip this image
        ...
    else:
        px_to_um_x = 1.0 / result.x_px_per_unit
        px_to_um_y = 1.0 / result.y_px_per_unit

`calibrate_image` NEVER raises on expected failure modes (too few ticks,
missing profile with no GUI available, OCR unavailable). It logs and
returns None so the caller's larger script keeps running.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging
import warnings

import numpy as np

from . import config
from .profiles import load_profiles, match_profile, save_profile
from .colors import infer_background, validate_profile_colors
from .gridlines import (
    detect_plot_region_and_gridlines, check_gridline_regularity,
)
from .ticks import (
    find_and_ocr_ticks_per_gridline, find_unit_label_at_axis,
    fit_pixel_to_unit, recover_outliers, reconstruct_tick_values,
    backverify_inferred_tick,
)
from .debug import save_debug_image


logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """
    Calibration output. slope is `value_per_pixel`, so:
        world_value = slope * pixel + intercept
    Reciprocal `_px_per_unit` fields exposed for convenience.
    """
    x_slope: float
    y_slope: float
    x_intercept: float
    y_intercept: float
    x_px_per_unit: float
    y_px_per_unit: float
    x_confidence: float          # [0, 1]
    y_confidence: float
    profile_used: Optional[str]
    x_fit: dict = field(default_factory=dict)
    y_fit: dict = field(default_factory=dict)
    plot_region: tuple = (0, 0, 0, 0)
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None
    x_unit_confidence: float = 0.0
    y_unit_confidence: float = 0.0
    x_unit_bbox: Optional[tuple] = None
    y_unit_bbox: Optional[tuple] = None
    warnings: list = field(default_factory=list)
    debug_path: Optional[str] = None   # where the debug overlay was saved (if it was)


# ---------------- Confidence scoring ----------------

def compute_confidence(fit: dict,
                       ocr_confidences: list[float],
                       n_expected_ticks: int,
                       gridline_regularity_cv: float) -> float:
    """
    confidence = W_R2 * r2
               + W_TICK_COVERAGE * min(1, n_valid / n_expected)
               + W_OCR_MEAN * mean(ocr_conf)
               + W_GRIDLINE_REG * max(0, 1 - cv / CV_TOL)
    Clipped to [0, 1].
    """
    r2 = max(0.0, min(1.0, fit.get("r_squared", 0.0)))

    n_valid = int(sum(fit.get("inlier_mask", [])))
    tick_cov = min(1.0, n_valid / max(1, n_expected_ticks))

    ocr_mean = float(np.mean(ocr_confidences)) if ocr_confidences else 0.0
    ocr_mean = max(0.0, min(1.0, ocr_mean))

    cv_tol = config.GRIDLINE_SPACING_CV_TOLERANCE
    if np.isfinite(gridline_regularity_cv):
        grid_score = max(0.0, 1.0 - gridline_regularity_cv / cv_tol)
        grid_score = min(1.0, grid_score)
    else:
        grid_score = 0.0

    conf = (
        config.CONF_W_R2 * r2
        + config.CONF_W_TICK_COVERAGE * tick_cov
        + config.CONF_W_OCR_MEAN * ocr_mean
        + config.CONF_W_GRIDLINE_REG * grid_score
    )
    return float(max(0.0, min(1.0, conf)))


# ---------------- Per-axis pipeline ----------------

def _calibrate_axis(image_rgb, axis, axis_lines,
                    bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
                    gridline_positions
                    ) -> tuple[Optional[dict], list, list[int], list[int], list, float,
                              list[str], Optional[str], float, Optional[tuple]]:
    """
    Run the full axis pipeline. Returns:
        (fit, pairs_final, recovered_idx, inferred_idx, bboxes, confidence,
         warnings, unit_string, unit_confidence, unit_bbox)
    fit is None if calibration failed for this axis. `recovered_idx` marks
    entries snapped by outlier recovery; `inferred_idx` marks entries
    filled in / overridden by sequence reconstruction (see
    ticks.reconstruct_tick_values) — the two are tracked separately since
    they come from different mechanisms and the debug overlay renders them
    distinctly.

    `axis_lines` is the {"x": row_px, "y": col_px} dict from
    detect_plot_region_and_gridlines — the true pixel positions of both
    axis lines, NOT plot_region (see ticks._local_tick_search_box for why
    the distinction matters when axis and gridline share a color). This
    axis's own entry anchors the tick/unit search band; the OTHER axis's
    entry (`cross_line_pos`) is the one entry in `gridline_positions` that
    is a border, not a real tick — used both to exclude it from sequence
    reconstruction and from the expected-tick count.
    """
    axis_line_pos = axis_lines[axis]
    cross_line_pos = axis_lines["y" if axis == "x" else "x"]
    warns: list[str] = []
    fail = (None, [], [], [], [], 0.0, warns, None, 0.0, None)

    # regularity
    is_reg, grid_metrics = check_gridline_regularity(gridline_positions)
    if not is_reg:
        warns.append(
            f"{axis}-axis gridlines not regular "
            f"(n={grid_metrics['n_gridlines']}, cv={grid_metrics['cv']:.3f})"
        )

    # Unit label: a full-band scan of the axis, classified by content (see
    # find_unit_label_at_axis). Run this FIRST so its bbox can be excluded from
    # the numeric search below — otherwise the axis unit (e.g. the far-end "µm"
    # that sits in place of the last x tick's number) gets re-detected there and
    # counted as a missing/garbage numeric read. `cross_line_pos` keeps the
    # x-axis band from reaching into the shared origin corner and stealing the
    # y-axis unit.
    unit_str, unit_conf, unit_bbox = find_unit_label_at_axis(
        image_rgb, gridline_positions, axis_line_pos, axis, bg_rgb, plot_rgb,
        axis_rgb=axis_rgb, gridline_rgb=gridline_rgb, cross_axis_pos=cross_line_pos,
    )
    if unit_str is None:
        warns.append(f"{axis}-axis: unit label not recognized")

    # Primary tick search: one attempt per gridline, anchored to the
    # gridline's own pixel position, with expanding redundancy on miss. The
    # already-identified unit bbox is excluded so it can't be mistaken for a
    # numeric tick.
    pairs_ocr, bboxes, missing = find_and_ocr_ticks_per_gridline(
        image_rgb, gridline_positions, axis_line_pos, axis,
        bg_rgb, plot_rgb, axis_rgb, gridline_rgb,
        exclude_bboxes=[unit_bbox] if unit_bbox is not None else None,
    )
    if missing:
        warns.append(
            f"{axis}-axis: no numeric label recovered for gridline(s) at "
            f"pixel {[round(m, 1) for m in missing]}"
        )

    # Sequence reconstruction: gridlines are evenly spaced by construction,
    # so tick VALUES form an arithmetic progression. Fill in ticks OCR
    # missed entirely and override ones OCR read but that disagree with the
    # verified-even sequence, using confident reads as anchors. Gated hard
    # (regularity + enough confident anchors) — see reconstruct_tick_values;
    # on any gate failure this is a no-op and `pairs` == `pairs_ocr`.
    pairs, inferred_mask = reconstruct_tick_values(
        gridline_positions, cross_line_pos, pairs_ocr, is_reg,
    )

    # Back-verify inferred entries against the arithmetic-progression prediction:
    # the sequence-solved value tells us what the true tick label *should* read,
    # so a retry OCR at that gridline whose read agrees within the tolerance
    # gate has to be the true glyph. Promoting the entry attaches a bbox (so the
    # debug overlay renders a green box, not just an inferred label) and swaps
    # the sequence-solved value for the confirmed OCR value. A rejected retry
    # is a no-op — the entry stays inferred as reconstructed.
    promoted_bbox_by_pixel: dict = {}
    if config.INFERRED_BACKVERIFY_ENABLED and any(inferred_mask):
        pair_values = [v for (_p, v, _c) in pairs]
        if len(pair_values) >= 2:
            step_est = float(np.median(np.abs(np.diff(pair_values))))
        else:
            step_est = 0.0
        if step_est > 0:
            base_hw = max(
                float(config.TICK_SEARCH_PERP_HALF_WIDTH),
                float(np.median(np.diff(np.sort(gridline_positions)))) * 0.4
                if len(gridline_positions) > 1 else float(config.TICK_SEARCH_PERP_HALF_WIDTH),
            )
            base_par = max(
                base_hw,
                float(np.median(np.diff(np.sort(gridline_positions)))) * 0.5
                if len(gridline_positions) > 1 else base_hw,
            )
            excludes = [unit_bbox] if unit_bbox is not None else None
            promoted_pairs = []
            promoted_mask = []
            for (pix, val, conf), inferred in zip(pairs, inferred_mask):
                if not inferred:
                    promoted_pairs.append((pix, val, conf))
                    promoted_mask.append(False)
                    continue
                promoted = backverify_inferred_tick(
                    image_rgb, pix, axis_line_pos, axis,
                    bg_rgb, plot_rgb, val, step_est,
                    axis_rgb=axis_rgb, gridline_rgb=gridline_rgb,
                    exclude_bboxes=excludes,
                    base_half_width=base_hw,
                    base_parallel_half_width=base_par,
                )
                if promoted is None:
                    promoted_pairs.append((pix, val, conf))
                    promoted_mask.append(True)
                else:
                    v_new, c_new, bbox_new = promoted
                    promoted_pairs.append((pix, v_new, c_new))
                    promoted_mask.append(False)
                    promoted_bbox_by_pixel[pix] = bbox_new
            pairs = promoted_pairs
            inferred_mask = promoted_mask

    inferred_by_pixel = {p: inf for (p, _v, _c), inf in zip(pairs, inferred_mask)}

    if len(pairs) < config.MIN_TICKS_FOR_FIT:
        warns.append(
            f"{axis}-axis: only {len(pairs)} usable tick labels found, "
            f"need >= {config.MIN_TICKS_FOR_FIT}"
        )
        return fail

    # Initial fit
    try:
        fit = fit_pixel_to_unit(pairs)
    except ValueError as e:
        warns.append(f"{axis}-axis fit failed: {e}")
        return fail

    # Outlier recovery pass
    pre_len = len(pairs)
    pairs_rec = recover_outliers(pairs, fit)
    recovered_idx: list[int] = []
    for i, (p_new, v_new, c_new) in enumerate(pairs_rec):
        for (p_old, v_old, c_old) in pairs:
            if abs(p_new - p_old) < 1e-6 and abs(v_new - v_old) > 1e-9:
                recovered_idx.append(i)
                break

    if len(pairs_rec) < config.MIN_TICKS_FOR_FIT:
        warns.append(
            f"{axis}-axis: after outlier removal only {len(pairs_rec)} "
            f"points remain (need >= {config.MIN_TICKS_FOR_FIT})"
        )
        return fail

    # Re-fit on cleaned data if anything changed
    if len(pairs_rec) != pre_len or recovered_idx:
        try:
            fit = fit_pixel_to_unit(pairs_rec)
        except ValueError as e:
            warns.append(f"{axis}-axis re-fit failed: {e}")
            return fail

    # Degenerate-fit guard: tick values that collapse to (near-)one
    # distinct value, or a near-zero slope, are not a real linear
    # relationship — they're misread OCR data that happened to average out
    # flat (Theil-Sen reports slope~=0 and, since ss_tot~=0, r_squared=1.0:
    # a deceptively "perfect" fit to nothing). Reject rather than silently
    # emitting px_per_unit = inf. Sequence reconstruction above should have
    # already prevented most of these; this is the safety net for whatever
    # slips through the reconstruction gate (e.g. axis wasn't regular
    # enough, or too few confident anchors).
    distinct_values = len({round(v, 9) for (_p, v, _c) in pairs_rec})
    if (distinct_values < config.MIN_DISTINCT_TICK_VALUES
            or abs(fit["slope"]) < config.MIN_ABS_SLOPE):
        warns.append(
            f"{axis}-axis: fit is degenerate (slope={fit['slope']:.3g}, "
            f"{distinct_values} distinct tick value(s) across "
            f"{len(pairs_rec)} ticks) — rejecting rather than reporting an "
            f"unreliable px/unit"
        )
        return fail

    ocr_confs = [p[2] for p in pairs_rec]
    # Exclude the cross-axis's own border line from the expected-tick count:
    # when axis and gridline share a color, `gridline_positions` includes
    # that border (e.g. x_grid, the column positions searched for x-axis
    # ticks, includes the y-axis's own vertical line at its column) — see
    # gridlines._detect_shared_axis_gridlines. That entry structurally never
    # carries a numeric label, so counting it as "expected" caps tick
    # coverage below 100% even on a perfect detection. In the distinct-color
    # path this cross-axis position never coincides with a real gridline, so
    # the filter is a no-op there.
    n_expected = sum(
        1 for g in gridline_positions if abs(float(g) - cross_line_pos) > 1e-6
    )
    conf = compute_confidence(
        fit=fit,
        ocr_confidences=ocr_confs,
        n_expected_ticks=max(1, n_expected),
        gridline_regularity_cv=grid_metrics["cv"],
    )

    # bboxes are aligned to the pre-reconstruction, pre-recovery `pairs_ocr`
    # list by construction (find_and_ocr_ticks_per_gridline returns them
    # 1:1). Neither reconstruction nor recovery change a CONFIRMED pair's
    # pixel position, so re-align by pixel position for those. An inferred
    # entry gets its bbox from `promoted_bbox_by_pixel` if back-verify (Fix J3)
    # confirmed it, and None otherwise — deliberately NOT `bbox_by_pixel`,
    # which for an OVERRIDDEN entry (reconstruction replaced a disagreeing OCR
    # read, not just filled a gap) would still hold the bbox of that WRONG
    # read. Falling through to it would draw a colored box around the
    # misread digits while showing the corrected reconstructed value next to
    # it — exactly the "overlay confused about the grid mark value" mismatch
    # reported; None correctly falls back to the axis-relative label position.
    bbox_by_pixel = {p: b for (p, _v, _c), b in zip(pairs_ocr, bboxes)}
    aligned_bboxes = []
    for (p, _v, _c) in pairs_rec:
        if p in promoted_bbox_by_pixel:
            aligned_bboxes.append(promoted_bbox_by_pixel[p])
        elif inferred_by_pixel.get(p, False):
            aligned_bboxes.append(None)
        else:
            aligned_bboxes.append(bbox_by_pixel.get(p))
    inferred_idx = [
        i for i, (p, _v, _c) in enumerate(pairs_rec)
        if inferred_by_pixel.get(p, False)
    ]

    # (unit_str/unit_conf/unit_bbox were resolved up front, before the numeric
    # search, so the unit bbox could be excluded from it — see above.)
    return (fit, pairs_rec, recovered_idx, inferred_idx, aligned_bboxes, conf,
            warns, unit_str, unit_conf, unit_bbox)


# ---------------- Main entry point ----------------

def calibrate_image(image_rgb: np.ndarray,
                    image_filename: str,
                    profiles_path: str,
                    on_no_match: str = "return_none",
                    picked_colors: Optional[dict] = None,
                    save_debug: Optional[bool] = None,
                    output_dir: Optional[str] = None,
                    draw_fit_box: bool = True
                    ) -> Optional[CalibrationResult]:
    """
    Full pipeline. Returns CalibrationResult, or None on unrecoverable failure.

    Parameters
    ----------
    image_rgb       : (H, W, 3) uint8 RGB image
    image_filename  : original filename, used for the debug image name
    profiles_path   : JSON file storing known profiles
    on_no_match     : 'gui'         -> launch calibration GUI (blocking)
                      'return_none' -> log and return None (default; safe for batch)
                      'raise'       -> raise RuntimeError
    picked_colors   : optional dict of pre-picked colors to bypass profile lookup,
                      e.g. from an active GUI session. Keys as in build_profile.
    save_debug      : whether to write the debug overlay image
    output_dir      : explicit override for where the debug image is saved.
                      If None (default), uses debug.DEFAULT_DEBUG_OUTPUT_DIR,
                      which resolves next to the entry-point script regardless
                      of the current working directory — see debug.py. Pass
                      this (or set the PLOT_CALIBRATION_PROJECT_ROOT env var)
                      if that heuristic doesn't match your deployment layout.
    draw_fit_box    : whether the saved debug image includes the X/Y fit
                      summary box. Callers that draw their own combined box
                      with additional (e.g. measurement) data — see
                      batch_analyze.py — pass False here; see debug.py's
                      save_debug_image docstring for details.

    Returns
    -------
    CalibrationResult or None
    """
    # None means "use whatever DEBUG_SAVE says right now" — resolved here
    # rather than as a default argument, since a default is evaluated once
    # at import and would ignore a later Preferences change.
    if save_debug is None:
        save_debug = config.DEBUG_SAVE

    all_warns: list[str] = []
    profile_name: Optional[str] = None
    colors: Optional[dict] = None

    # Source colors: explicit override > profile match > GUI (optional)
    if picked_colors is not None:
        required = {"background", "plot", "axis", "gridline"}
        if not required.issubset(picked_colors):
            logger.error("picked_colors missing required keys: %s",
                         required - set(picked_colors))
            return None
        colors = {k: tuple(picked_colors[k]) for k in required}
    else:
        profiles = load_profiles(profiles_path)
        matched = match_profile(image_rgb, profiles)
        if matched is not None:
            profile_name = matched["name"]
            colors = {k: tuple(v) for k, v in matched["colors"].items()}
            logger.info("Matched profile %r (dist=%.3f)",
                        profile_name, matched.get("_match_distance", -1))
        else:
            if on_no_match == "gui":
                try:
                    from .gui_calibrate import launch_calibration_gui
                except ImportError as e:
                    warns_msg = f"GUI unavailable: {e}"
                    logger.warning(warns_msg)
                    all_warns.append(warns_msg)
                    return None
                new_profile = launch_calibration_gui(image_rgb)
                if new_profile is None:
                    logger.warning("Calibration GUI cancelled by user")
                    return None
                save_profile(new_profile, profiles_path)
                profile_name = new_profile["name"]
                colors = {k: tuple(v) for k, v in new_profile["colors"].items()}
            elif on_no_match == "raise":
                raise RuntimeError(
                    "No matching profile and on_no_match='raise'"
                )
            else:  # return_none
                logger.warning(
                    "No matching profile for %r and no GUI requested — "
                    "returning None", image_filename
                )
                return None

    # Fallback: infer background if the profile version differs from actual image
    if colors["background"] is None:
        colors["background"] = infer_background(image_rgb)

    # Fail fast on an ambiguous profile (most commonly axis == gridline from
    # a click that missed a thin line) rather than letting it silently
    # corrupt plot-region/gridline detection and surface later as a
    # confusing "no gridlines found" error far from the actual cause.
    color_problems = validate_profile_colors(colors)
    if color_problems:
        for problem in color_problems:
            logger.error("Profile %r has ambiguous colors: %s",
                        profile_name or "(explicit picked_colors)", problem)
        all_warns.extend(color_problems)
        return None

    bg_rgb = colors["background"]
    plot_rgb = colors["plot"]
    axis_rgb = colors["axis"]
    grid_rgb = colors["gridline"]

    # ---------------- Plot region + gridlines ----------------
    # Handles both the common case (distinct axis/gridline colors) and the
    # case where a plotting program styles the frame identically to
    # interior gridlines — see detect_plot_region_and_gridlines.
    plot_region, x_grid, y_grid, axis_lines = detect_plot_region_and_gridlines(
        image_rgb, axis_rgb, grid_rgb
    )

    # ---------------- Per-axis calibration ----------------
    (x_fit, x_pairs, x_recov, x_infer, x_bboxes, x_conf, xw,
     x_unit, x_unit_conf, x_unit_bbox) = _calibrate_axis(
        image_rgb, "x", axis_lines, bg_rgb, plot_rgb, axis_rgb, grid_rgb,
        x_grid,
    )
    all_warns.extend(xw)
    (y_fit, y_pairs, y_recov, y_infer, y_bboxes, y_conf, yw,
     y_unit, y_unit_conf, y_unit_bbox) = _calibrate_axis(
        image_rgb, "y", axis_lines, bg_rgb, plot_rgb, axis_rgb, grid_rgb,
        y_grid,
    )
    all_warns.extend(yw)

    # pm/um cross-axis disambiguation: pm and um are visually confusable in
    # OCR, and per the user's domain knowledge pm only ever occurs on the
    # Y-axis (height), never X (lateral distance is always nm/um/mm). If the
    # X-axis unit already resolved to one of those, a Y-axis "pm" read is
    # presumed a misread "um" and corrected — see config.Y_PM_DISAMBIGUATION_
    # ENABLED / Y_PM_SAFE_XUNITS. Left as "pm" only when the X-axis unit is
    # itself unresolved or genuinely "pm" — no evidence to rule it out then.
    if (config.Y_PM_DISAMBIGUATION_ENABLED and y_unit == "pm"
            and x_unit in config.Y_PM_SAFE_XUNITS):
        if config.CALIB_DEBUG_LOG:
            logger.debug(
                "y-unit pm/um disambiguation: x_unit=%r -> correcting "
                "y_unit 'pm' to 'um'", x_unit,
            )
        y_unit = "um"

    if x_fit is None or y_fit is None:
        for w in all_warns:
            logger.warning(w)
        if save_debug:
            try:
                debug_kwargs = {}
                if output_dir is not None:
                    debug_kwargs["output_dir"] = output_dir
                save_debug_image(
                    image_rgb, image_filename, plot_region,
                    x_grid, y_grid,
                    x_pairs or [], y_pairs or [],
                    x_bboxes or None, y_bboxes or None,
                    x_recov or None, y_recov or None,
                    x_fit, y_fit, x_conf, y_conf,
                    x_unit, y_unit,
                    x_unit_bbox, y_unit_bbox,
                    axis_lines=axis_lines,
                    x_inferred_idx=x_infer or None, y_inferred_idx=y_infer or None,
                    draw_fit_box=draw_fit_box,
                    **debug_kwargs,
                )
            except Exception as e:
                logger.warning("debug image save failed: %s", e)
        return None

    # ---------------- Assemble result ----------------
    # slope is value_per_pixel; px_per_unit = 1/slope (guard against zero)
    def _px_per_unit(slope):
        if abs(slope) < 1e-12:
            return float("inf")
        return abs(1.0 / slope)

    result = CalibrationResult(
        x_slope=x_fit["slope"],
        y_slope=y_fit["slope"],
        x_intercept=x_fit["intercept"],
        y_intercept=y_fit["intercept"],
        x_px_per_unit=_px_per_unit(x_fit["slope"]),
        y_px_per_unit=_px_per_unit(y_fit["slope"]),
        x_confidence=x_conf,
        y_confidence=y_conf,
        profile_used=profile_name,
        x_fit=x_fit,
        y_fit=y_fit,
        plot_region=plot_region,
        x_unit=x_unit,
        y_unit=y_unit,
        x_unit_confidence=x_unit_conf,
        y_unit_confidence=y_unit_conf,
        x_unit_bbox=x_unit_bbox,
        y_unit_bbox=y_unit_bbox,
        warnings=all_warns,
    )

    # ---------------- Debug image ----------------
    if save_debug:
        try:
            debug_kwargs = {}
            if output_dir is not None:
                debug_kwargs["output_dir"] = output_dir
            result.debug_path = save_debug_image(
                image_rgb, image_filename, plot_region,
                x_grid, y_grid,
                x_pairs, y_pairs,
                x_bboxes, y_bboxes,
                x_recov, y_recov,
                x_fit, y_fit, x_conf, y_conf,
                x_unit, y_unit,
                x_unit_bbox, y_unit_bbox,
                axis_lines=axis_lines,
                x_inferred_idx=x_infer or None, y_inferred_idx=y_infer or None,
                draw_fit_box=draw_fit_box,
                **debug_kwargs,
            )
        except Exception as e:
            logger.warning("debug image save failed: %s", e)

    for w in all_warns:
        logger.info(w)
    return result
