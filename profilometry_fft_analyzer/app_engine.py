"""
app_engine.py
-------------
Headless, callback-driven core of the batch pipeline. Single source of truth
for both the terminal CLI (batch_analyze.py) and the GUI app (app.py +
batch_tab.py / viewer_tab.py): all the actual image analysis, profile
matching, overlay rendering, and CSV assembly lives here as plain functions
over a BatchConfig, with GUI/terminal-specific behavior (prompts, progress
display, how a color profile or a manual fit gets collected from the user)
injected as callbacks by the caller.

Phase shape (matches the original batch_analyze.py, now split so a caller
can drive it interactively or from a worker thread):
    1. resolve_profile_audit  — match/create color profiles (interactive)
    2. run_batch              — per-image analysis (pure compute; thread-safe)
    3. manual_review_phase    — manual correction of failures (interactive)
    4. write_results_csv      — results CSV for the run

update_results_csv_row lets the session viewer amend an already-written CSV
row after a later manual correction, using the same merge logic as a live run.
"""
from __future__ import annotations
import logging
import os
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

import app_settings as settings
from plot_calibration.calibrate import calibrate_image
from plot_calibration.profiles import (
    load_profiles, save_profile, compute_histogram, match_histograms,
)
from plot_calibration.gridlines import detect_plot_region_and_gridlines
from plot_calibration import config as calib_config
from manual_merge import merge_manual_row

from profilometry_fft_analyzer import (
    SUPPORTED_IMAGE_EXTENSIONS, extract_trend_with_info, import_tuning_param,
    natural_key,
)
from get_sample_data import get_sample_info, SAMPLE_FIELDS
import feature_analysis as fa
import measurement_overlay as mo

logger = logging.getLogger("app_engine")


# ================= App directory (frozen-safe) =================

# Defined in app_settings (which is stdlib-only, so config modules can import
# it without dragging cv2/pandas along) and re-exported here, since callers,
# app.spec and the README all refer to app_engine.get_app_dir().
get_app_dir = settings.get_app_dir


# ================= Configuration =================

@dataclass
class BatchConfig:
    """Everything one batch run needs to know: where its folders are, and the
    handful of measurement settings the run reads.

    The settings-backed fields use default_factory rather than a plain
    default, because a plain default is evaluated once when the class is
    created and would pin whatever was in Preferences at import time. Going
    through a factory means each default_config() call picks up the current
    value, so an edit applies to the very next run."""
    base_dir: str
    input_dir: str
    output_dir: str
    image_fits_dir: str
    error_log_dir: str
    failed_dir: str
    profiles_path: str

    # Unit for EVERY length in the CSV and on the overlays: 'nm' or 'um'.
    output_unit: str = field(default_factory=lambda: settings.OUTPUT_UNIT)

    # Fixed roughness cutoff in um; None means ISO 4288 picks one per image.
    lambda_c_override_um: Optional[float] = field(
        default_factory=lambda: settings.LAMBDA_C_OVERRIDE_UM)

    # Measure Ra/Rq/Rz on the flat surface between features rather than the
    # whole profile, so step heights don't swamp the texture numbers.
    roughness_on_baseline: bool = field(
        default_factory=lambda: settings.ROUGHNESS_ON_BASELINE)

    # Allow the calibration flow to run for images with no matching color
    # profile. False = fully headless; unmatched images are skipped.
    # Not user-facing: batch_analyze/app set this themselves per run mode.
    profile_audit_gui: bool = True

    # Trace-extraction fallbacks. tuned_params.json (written by the tuning
    # GUI) still wins for these two when present — see extraction_params().
    color_buffer: float = field(default_factory=lambda: settings.COLOR_BUFFER)
    max_gap_px: int = field(default_factory=lambda: settings.MAX_GAP_PX)

    # Pixels shaved off each plot_region edge before extraction, so the
    # border/axis lines can't contaminate the color match.
    crop_inset_px: int = field(default_factory=lambda: settings.CROP_INSET_PX)

    # Feature count at which a profile counts as a periodic array, switching
    # roughness cutoff selection to the ISO 4288 RSm rule instead of Ra.
    periodic_min_features: int = field(
        default_factory=lambda: settings.PERIODIC_MIN_FEATURES)


def default_config(app_dir: Optional[str] = None) -> BatchConfig:
    """Standard directory layout, rooted at app_dir (default: get_app_dir())."""
    base = app_dir or get_app_dir()
    return BatchConfig(
        base_dir=base,
        input_dir=os.path.join(base, "input_files"),
        output_dir=os.path.join(base, "data_files"),
        image_fits_dir=os.path.join(base, "image_fits"),
        error_log_dir=os.path.join(base, "session_logs"),
        failed_dir=os.path.join(base, "unprocessed_images"),
        profiles_path=os.path.join(base, "calibration_profiles.json"),
    )


class AnalysisError(RuntimeError):
    """Per-image failure with a short, CSV-friendly message."""


MEASUREMENT_COLUMNS = [
    # User-set review marker (Viewer tab checkbox) — first measurement
    # column so it reads right next to the sample-metadata columns, not
    # buried after every numeric measurement.
    "flagged",
    "feature_type", "n_features",
    "height_avg", "height_min", "height_max", "height_median",
    "width_avg", "width_min", "width_max", "width_median",
    "dishing_avg", "dishing_max", "dishing_raw_min",
    "Ra", "Rq", "Rz",
    "lambda_c", "lambda_c_capped",
    "x_unit_detected", "y_unit_detected", "output_unit",
    "cal_conf_x", "cal_conf_y", "trend_conf",
    "profile", "overlay_file", "error",
]
CSV_COLUMNS = SAMPLE_FIELDS + MEASUREMENT_COLUMNS

# Columns that must be stored as float, not string/object, so Excel doesn't
# treat them as text (which adds a leading ' when it opens the CSV).
NUMERIC_CSV_COLUMNS = [
    "height_avg", "height_min", "height_max", "height_median",
    "width_avg", "width_min", "width_max", "width_median",
    "dishing_avg", "dishing_max", "dishing_raw_min",
    "Ra", "Rq", "Rz", "lambda_c",
    "cal_conf_x", "cal_conf_y", "trend_conf",
]

# Truthy string forms accepted when re-reading "flagged" back out of a CSV
# (pandas round-trips a bool column as the literal text "True"/"False", but
# an older file, a hand-edited one, or a re-imported one might spell it any
# of these ways) — anything else, including blank/NaN, reads back False.
_FLAG_TRUE_STRINGS = {"true", "1", "yes", "y"}


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force NUMERIC_CSV_COLUMNS to float dtype in place; returns df."""
    for col in NUMERIC_CSV_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coerce_flagged(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 'flagged' column to real bools in place; returns df.
    Creates the column (all False) if a pre-existing CSV predates this
    feature, so older result files still round-trip through
    update_results_csv_row / load_existing_data_file without a KeyError."""
    if "flagged" not in df.columns:
        df["flagged"] = False
        return df

    def _to_bool(v):
        """Turns whatever pandas read out of the cell into a real bool —
        already-bool passes through, NaN (blank cell) reads False, and
        anything else is matched against the accepted truthy spellings."""
        if isinstance(v, bool):
            return v
        if isinstance(v, float) and v != v:  # NaN
            return False
        return str(v).strip().lower() in _FLAG_TRUE_STRINGS

    df["flagged"] = df["flagged"].map(_to_bool)
    return df


# ================= Setup helpers =================

def setup_run_logging(config: BatchConfig, run_ts: str
                      ) -> tuple[str, logging.Handler, list[logging.Handler]]:
    """Root-logger file handler so plot_calibration's logs are captured too.
    Returns (log_path, counting_handler, handlers): `handlers` is every
    handler this call added to the root logger. A one-shot CLI process can
    ignore it; a long-lived process that runs more than one batch (the GUI
    app) must pass it to teardown_run_logging() when the run ends, or a
    later run keeps writing into — and counting warnings/errors against —
    an earlier run's log file."""
    os.makedirs(config.error_log_dir, exist_ok=True)
    log_path = os.path.join(config.error_log_dir, f"{run_ts}.txt")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(fh)

    class _Counting(logging.Handler):
        """Counts warnings and errors as they're logged, so the run can print
        a tally at the end without re-reading its own log file."""

        def __init__(self):
            """Starts both counters at zero and ignores anything below
            WARNING — routine info lines aren't worth counting."""
            super().__init__(level=logging.WARNING)
            self.warnings = 0
            self.errors = 0

        def emit(self, record):
            """Files each record into the error or warning tally by level."""
            if record.levelno >= logging.ERROR:
                self.errors += 1
            else:
                self.warnings += 1

    counter = _Counting()
    root.addHandler(counter)
    return log_path, counter, [fh, counter]


def teardown_run_logging(handlers: list[logging.Handler]) -> None:
    """Remove handlers previously returned by setup_run_logging from the
    root logger — see that function's docstring for why this matters in a
    long-lived process."""
    root = logging.getLogger()
    for h in handlers:
        root.removeHandler(h)


def list_images(config: BatchConfig) -> list[str]:
    """Returns the full paths of every supported image in the input folder,
    in natural order so 'img2' sorts before 'img10'. Creates the folder and
    returns an empty list on a first run where it doesn't exist yet."""
    if not os.path.isdir(config.input_dir):
        os.makedirs(config.input_dir, exist_ok=True)
        return []
    files = [
        os.path.join(config.input_dir, f) for f in os.listdir(config.input_dir)
        if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
    ]
    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    return files


def load_rgb(path: str) -> np.ndarray:
    """Reads an image file and returns it as an RGB array. OpenCV hands back
    BGR, so this flips the channel order — every other module in the pipeline
    assumes RGB. Raises AnalysisError if the file can't be decoded."""
    image = cv2.imread(path)
    if image is None:
        raise AnalysisError("could not read image file")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extraction_params(config: BatchConfig) -> tuple[float, int]:
    """Works out the (color_buffer, max_gap) this run should extract with.

    A tuned_params.json written by the tuning GUI wins over the config
    values, since it was measured against real images from this instrument;
    without one, the config's own settings (themselves user-editable in
    Preferences) are used. Which source won is logged, because "I changed it
    in Preferences and nothing happened" is otherwise a confusing few
    minutes."""
    params = import_tuning_param(config.base_dir)
    if params:
        buffer = float(params.get("color_buffer", config.color_buffer))
        max_gap = int(params.get("max_gap_px", config.max_gap_px))
        logger.info("extraction params from tuned_params.json (overriding "
                    "Preferences): buffer=%.1f max_gap=%d", buffer, max_gap)
    else:
        buffer, max_gap = config.color_buffer, config.max_gap_px
        logger.info("extraction params from Preferences: buffer=%.1f "
                    "max_gap=%d", buffer, max_gap)
    return buffer, max_gap


# ================= Phase 1: profile audit =================

def best_profile(hist: np.ndarray, profiles: list[dict]):
    """Best-matching profile for a precomputed histogram, or None."""
    best, best_d = None, float("inf")
    for p in profiles:
        if "histogram" not in p:
            continue
        ref = np.asarray(p["histogram"], dtype=np.float32)
        if ref.size != hist.size:
            continue
        d = match_histograms(hist, ref)
        if d < best_d:
            best, best_d = p, d
    if best is not None and best_d < calib_config.HIST_MATCH_THRESHOLD:
        return best
    return None


def compute_profile_audit_state(config: BatchConfig, image_paths: list[str]):
    """Pure-compute half of the profile audit: histograms + unmatched list.
    No GUI, no prompts. Returns (hist_cache, skipped, unmatched, profiles)."""
    profiles = load_profiles(config.profiles_path)
    skipped: list[str] = []
    hist_cache: dict[str, np.ndarray] = {}
    for p in image_paths:
        try:
            hist_cache[p] = compute_histogram(load_rgb(p))
        except Exception as e:
            logger.error("%s: unreadable during profile audit (%s) — "
                         "image skipped", os.path.basename(p), e)
            skipped.append(p)

    unmatched = [p for p in hist_cache
                 if best_profile(hist_cache[p], profiles) is None]
    return hist_cache, skipped, unmatched, profiles


def resolve_profile_audit(
    config: BatchConfig,
    image_paths: list[str],
    on_status: Callable[[str], None],
    on_unmatched: Callable[[np.ndarray, str], Optional[dict]],
) -> tuple[dict, list[str], list[str]]:
    """
    Ensure every image has a matching color profile.

    on_status(message): human-readable progress text (CLI: print; GUI: a
        status banner / log line).
    on_unmatched(image_rgb, default_name) -> profile_dict | None: collect one
        new profile for an image with no match (CLI: the blocking
        gui_calibrate popup; GUI app: an embedded "Create profile" tab).
        Called synchronously, once per image that still needs a profile
        after each new one is saved and the remainder re-matched.

    Returns (hist_cache, skipped, created):
        hist_cache : {path: histogram} so analysis doesn't recompute
        skipped    : images left without a profile (declined/failed/unreadable)
        created    : names of profiles created during the audit
    """
    hist_cache, skipped, unmatched, profiles = compute_profile_audit_state(
        config, image_paths)
    created: list[str] = []

    if not unmatched:
        on_status(f"Profile audit: all {len(hist_cache)} readable image(s) "
                  f"matched a saved color profile. [OK]")
        return hist_cache, skipped, created

    on_status(f"Profile audit: {len(unmatched)}/{len(image_paths)} image(s) "
              f"have no matching color profile.")
    logger.info("profile audit: %d unmatched image(s)", len(unmatched))

    if not config.profile_audit_gui:
        for p in unmatched:
            logger.error("no color profile for %r and profile_audit_gui is "
                         "off — image skipped", os.path.basename(p))
        return hist_cache, skipped + list(unmatched), created

    remaining = list(unmatched)
    while remaining:
        path = remaining[0]
        name = os.path.basename(path)
        on_status(f"Creating a profile for: {name}")
        try:
            new_profile = on_unmatched(
                load_rgb(path), f"profile{len(profiles) + len(created)}")
        except Exception as e:
            logger.error("calibration GUI failed for %r (%s) — image will "
                         "be skipped this run", name, e)
            skipped.append(path)
            remaining.pop(0)
            continue
        if new_profile is None:
            logger.warning("calibration GUI cancelled for %r — image will be "
                           "skipped this run", name)
            skipped.append(path)
            remaining.pop(0)
            continue

        save_profile(new_profile, config.profiles_path)
        created.append(new_profile["name"])
        profiles = load_profiles(config.profiles_path)
        # One new profile usually covers many images — re-match the rest.
        remaining = [p for p in remaining
                    if best_profile(hist_cache[p], profiles) is None]

    on_status(f"Profile audit complete: {len(created)} profile(s) created"
              f"{', ' + str(len(skipped)) + ' image(s) skipped' if skipped else ''}."
              f" All remaining images matched. [OK]")
    return hist_cache, skipped, created


# ================= Phase 2: per-image analysis =================

def _fmt(config: BatchConfig, value_um: float) -> str:
    """Format a um-value in config.output_unit for overlay labels ('-' if absent)."""
    if value_um is None or not np.isfinite(value_um):
        return "-"
    v = fa.convert_length(value_um, "um", config.output_unit)
    return f"{v:.4g}{config.output_unit}"


def _empty_row(config: BatchConfig, meta: dict, error: str) -> dict:
    """Builds the CSV row for an image that failed, so a failure still gets a
    line in the results rather than vanishing. Every measurement is NaN; the
    filename metadata and the error message are what carry the information."""
    row = {c: np.nan for c in MEASUREMENT_COLUMNS}
    row.update(meta)
    row["flagged"] = False
    row["output_unit"] = config.output_unit
    row["error"] = error
    return row


def analyze_one(config: BatchConfig, path: str, run_dir: str, profile: dict,
                buffer: float, max_gap: int,
                context_out: Optional[dict] = None) -> dict:
    """Full pipeline for one image. Raises AnalysisError on failure.

    `profile` is the histogram-matched color profile for this image (audit
    guarantees one exists for every non-skipped image). `context_out`, when
    given, is filled progressively with whatever succeeded (profile, plot
    color, partial calibration result) so a manual-correction UI can be
    prefilled after a failure — or reopened later for any image, success or
    not, from a session viewer."""
    name = os.path.basename(path)
    meta = get_sample_info(name)
    if meta["DIEX"] is None or meta["DIEY"] is None:
        logger.warning("%s: filename doesn't follow the naming convention — "
                       "metadata columns will be partially empty", name)
    rgb = load_rgb(path)

    ctx = context_out if context_out is not None else {}
    ctx["profile"] = profile
    plot_rgb = tuple(profile["colors"]["plot"])
    ctx["plot_rgb"] = plot_rgb

    # ---- axis calibration (also writes the base debug overlay) ----
    # draw_fit_box=False: the calibration box is combined with the
    # measurement summary into one top-strip box, drawn later in
    # _render_overlay (see measurement_overlay.draw_measurements).
    result = calibrate_image(
        rgb, name, config.profiles_path,
        on_no_match="return_none",
        save_debug=True, output_dir=run_dir,
        draw_fit_box=False,
    )
    if result is None:
        raise AnalysisError("axis calibration failed (see log for details)")
    ctx["result"] = result

    try:
        if result.x_unit is None or result.y_unit is None:
            raise AnalysisError(
                f"axis unit unresolved (x={result.x_unit!r}, y={result.y_unit!r})"
                " — refusing to guess a scale")
        try:
            fx = fa.unit_factor(result.x_unit, "um")
            fy = fa.unit_factor(result.y_unit, "um")
        except ValueError as e:
            raise AnalysisError(str(e))

        # ---- trace extraction inside the detected plot region ----
        px0, py0, px1, py1 = result.plot_region
        ix0, iy0 = px0 + config.crop_inset_px, py0 + config.crop_inset_px
        ix1, iy1 = px1 - config.crop_inset_px, py1 - config.crop_inset_px
        if ix1 - ix0 < 20 or iy1 - iy0 < 20:
            raise AnalysisError("detected plot region is degenerate")
        crop = rgb[iy0:iy1, ix0:ix1]
        crop_h = crop.shape[0]

        x_cols, y_flip, ext_info = extract_trend_with_info(
            crop, plot_rgb, buffer=buffer, max_gap=max_gap, interp_gaps=True)
        if len(x_cols) < 16:
            raise AnalysisError("trace extraction failed (too few matched columns)")

        # Interpolated (gap-filled) column count — surfaced to the batch log
        # by run_batch as normal (non-warning) info, since gap-filling is
        # expected behavior, not an error.
        ctx["n_interpolated"] = len(ext_info.get("skipped_x", []))
        ctx["n_columns"] = int(ext_info.get("n_columns", 0))

        rows_crop = (crop_h - 1) - np.asarray(y_flip, dtype=float)
        cols_full = np.asarray(x_cols, dtype=float) + ix0
        rows_full = rows_crop + iy0

        # ---- pixel -> physical units (per-image calibration) ----
        x_axis_vals = result.x_slope * cols_full + result.x_intercept
        y_axis_vals = result.y_slope * rows_full + result.y_intercept
        x_um = x_axis_vals * fx
        y_um = y_axis_vals * fy

        if x_um[-1] < x_um[0]:  # keep x increasing; keep pixel arrays aligned
            cols_full = cols_full[::-1]
            rows_full = rows_full[::-1]
            rows_crop = rows_crop[::-1]
            x_um = x_um[::-1]
            y_um = y_um[::-1]

        dx_um = float(np.median(np.diff(x_um)))
        if not np.isfinite(dx_um) or dx_um <= 0:
            raise AnalysisError("x-axis calibration produced a non-positive "
                                "sample spacing")

        # ---- leveling, features, roughness (shared pipeline) ----
        pipe = fa.run_measurement_pipeline(
            y_um, dx_um, rows_crop=rows_crop, ext_info=ext_info,
            lambda_c_override_um=config.lambda_c_override_um,
            periodic_min_features=config.periodic_min_features,
            roughness_on_baseline=config.roughness_on_baseline)
        y_lev, feats = pipe["y_lev"], pipe["feats"]
        lc, rough, tconf = pipe["lc"], pipe["rough"], pipe["trend_conf"]

        # lambda_c capping is the norm (not the exception) on these short
        # exports — tallied and reported once per run instead of per image;
        # see the lambda_c_capped column and log_lambda_c_capping().
        if rough["rz_n_chunks"] < 5:
            logger.warning("%s: Rz averaged over %d sampling length(s) instead "
                           "of the standard 5 (short profile)",
                           name, rough["rz_n_chunks"])

        # ---- review overlay ----
        overlay_rel = _render_overlay(
            config, path, rgb, result, run_dir, cols_full, rows_full,
            x_um, y_um, y_lev, feats, lc, rough, tconf, fy)

        # ---- CSV row (lengths converted um -> config.output_unit) ----
        cv = lambda v: fa.convert_length(v, "um", config.output_unit)
        row = dict(meta)
        row.update({
            "flagged": False,
            "feature_type": feats.feature_type,
            "n_features": feats.n_features,
            "height_avg": cv(feats.height_avg),
            "height_min": cv(feats.height_min),
            "height_max": cv(feats.height_max),
            "height_median": cv(feats.height_median),
            "width_avg": cv(feats.width_avg),
            "width_min": cv(feats.width_min),
            "width_max": cv(feats.width_max),
            "width_median": cv(feats.width_median),
            "dishing_avg": cv(feats.dishing_avg),
            "dishing_max": cv(feats.dishing_max),
            "dishing_raw_min": cv(feats.dishing_raw_min),
            "Ra": cv(rough["Ra"]),
            "Rq": cv(rough["Rq"]),
            "Rz": cv(rough["Rz"]),
            "lambda_c": cv(lc["lambda_c_um"]),
            "lambda_c_capped": bool(lc["capped"]),
            "x_unit_detected": result.x_unit,
            "y_unit_detected": result.y_unit,
            "output_unit": config.output_unit,
            "cal_conf_x": round(result.x_confidence, 3),
            "cal_conf_y": round(result.y_confidence, 3),
            "trend_conf": round(tconf, 3),
            "profile": result.profile_used,
            "overlay_file": overlay_rel,
            "error": "",
        })
        return row
    except AnalysisError as e:
        # Calibration succeeded but a later stage didn't. The debug PNG was
        # saved with draw_fit_box=False (no box baked in — see the
        # calibrate_image call above), on the assumption _render_overlay
        # would draw the combined box; since that step was never reached,
        # draw it now (calibration facts + the failure reason) so the image
        # isn't left with no summary box at all.
        _render_failed_overlay(result, name, str(e))
        raise


def _calib_summary_lines(result) -> list[str]:
    """X/Y calibration facts (confidence + detected unit) as ONE combined
    line for the top-strip box (see measurement_overlay.draw_measurements).
    Slope/R2 were dropped from the overlay to keep the box compact — they're
    still available in the per-run log if needed."""
    if result is None:
        return []
    parts = []
    if getattr(result, "x_fit", None) is not None:
        unit_str = f"unit={result.x_unit!r}" if result.x_unit else "unit=?"
        parts.append(f"X: conf={result.x_confidence:.2f} {unit_str}")
    if getattr(result, "y_fit", None) is not None:
        unit_str = f"unit={result.y_unit!r}" if result.y_unit else "unit=?"
        parts.append(f"Y: conf={result.y_confidence:.2f} {unit_str}")
    return ["   ".join(parts)] if parts else []


def _overlay_base_path(debug_path: str) -> str:
    """Sidecar path holding the box-FREE calibration overlay for `debug_path`
    — kept in a `.bases/` subfolder so the run's overlay folder still shows
    exactly one review PNG per image."""
    d, fn = os.path.split(debug_path)
    return os.path.join(d, ".bases", fn)


def _load_overlay_base(debug_path: str):
    """Return the box-FREE base overlay (BGR) that measurement annotations
    should be drawn onto, so every render — batch, failed, or any later
    manual re-fit — starts from the clean calibration image instead of one
    that already has a (possibly larger) summary box baked in. Without this,
    re-rendering a smaller box over a bigger one leaves the old box's text
    peeking out (there's no way to erase baked-in pixels back to the clean
    overlay after the fact).

    The first render for an image snapshots the still-box-free debug PNG
    (calibration writes it with draw_fit_box=False) into the sidecar; every
    render reads the sidecar. Returns None if nothing is readable, so the
    caller can fall back to the raw image.
    """
    base_path = _overlay_base_path(debug_path)
    if os.path.isfile(base_path):
        return cv2.imread(base_path)
    canvas = cv2.imread(debug_path)
    if canvas is None:
        return None
    try:
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        shutil.copy2(debug_path, base_path)
    except OSError as e:
        logger.warning("could not save box-free overlay base for %s: %s",
                       os.path.basename(debug_path), e)
    return canvas


def _render_failed_overlay(result, name: str, error_msg: str) -> None:
    """When axis calibration succeeded but a later stage raised
    AnalysisError, the debug PNG was saved with draw_fit_box=False on the
    assumption _render_overlay would draw the combined box — since that
    never happened, draw it now (calibration facts + the failure reason)
    so the image isn't left with no summary box at all."""
    debug_path = getattr(result, "debug_path", None) if result is not None else None
    if not debug_path or not os.path.isfile(debug_path):
        return
    canvas = _load_overlay_base(debug_path)
    if canvas is None:
        return
    lines = _calib_summary_lines(result) + [f"FAILED: {error_msg}"]
    mo.draw_measurements(canvas, [], [], None, None, None, lines)
    cv2.imwrite(debug_path, canvas)


def _render_overlay(config: BatchConfig, path, rgb, result, run_dir, cols_full,
                    rows_full, x_um, y_um, y_lev, feats, lc, rough, tconf,
                    fy) -> str:
    """Draw measurement annotations onto the calibration debug overlay and
    save it (same file the calibration step wrote, so there's exactly one
    review image per input). Returns the path relative to
    config.image_fits_dir."""
    name = os.path.basename(path)
    canvas = None
    out_path = result.debug_path
    if out_path and os.path.isfile(out_path):
        canvas = _load_overlay_base(out_path)
    if canvas is None:
        stem = os.path.splitext(name)[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(run_dir, f"{ts}_{stem}.png")
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    baseline_um = y_um - y_lev  # the removed leveling line, per sample

    def row_of_um(v_um_leveled: float, idx: int) -> float:
        """Leveled um value at sample idx -> full-image pixel row."""
        v_axis = (v_um_leveled + baseline_um[idx]) / fy
        return (v_axis - result.y_intercept) / result.y_slope

    smooth_rows = None
    if feats.y_smooth is not None and len(feats.y_smooth) == len(cols_full):
        v_axis = (feats.y_smooth + baseline_um) / fy
        smooth_rows = (v_axis - result.y_intercept) / result.y_slope

    fmt = lambda v: _fmt(config, v)

    n = len(cols_full)
    dx_um_local = float(x_um[1] - x_um[0]) if n > 1 else 1.0
    height_markers, dishing_markers, width_markers = mo.build_measurement_markers(
        feats, cols_full, row_of_um, fmt, dx_um_local)

    summary = _calib_summary_lines(result) + [
        f"type={feats.feature_type}  n={feats.n_features}  "
        f"trend_conf={tconf:.2f}",
        f"h_avg={fmt(feats.height_avg)}  "
        f"w_avg={fmt(feats.width_avg)}  "
        f"dish_max={fmt(feats.dishing_max)}",
        f"Ra={fmt(rough['Ra'])}  Rq={fmt(rough['Rq'])}  "
        f"Rz={fmt(rough['Rz'])}",
        f"lambda_c={fmt(lc['lambda_c_um'])}"
        + ("  (capped)" if lc["capped"] else ""),
    ]

    mo.draw_measurements(canvas, cols_full, rows_full, smooth_rows,
                         height_markers, dishing_markers, summary,
                         width_markers=width_markers)
    cv2.imwrite(out_path, canvas)
    return os.path.relpath(out_path, config.image_fits_dir)


def run_batch(config: BatchConfig, image_paths: list[str], hist_cache: dict,
             profiles_list: list[dict], skipped_set: set,
             run_dir: str, buffer: float, max_gap: int,
             on_progress: Optional[Callable[[int, int, str], None]] = None,
             on_log: Optional[Callable[[str], None]] = None
             ) -> tuple[list[dict], list[dict], dict]:
    """Phase B: per-image analysis. Pure compute — no GUI calls, safe to run
    on a worker thread.

    on_progress(i, n, name), if given, is called just before each image
    starts (i is 1-based).

    on_log(message), if given, receives plain informational lines (currently
    the per-image interpolated-column count) meant for a user-facing log —
    NOT warnings/errors, so a GUI can show them without an alarming level.
    The GUI's queue-based log handler only forwards WARNING+ off the worker
    thread, so this expected-behavior note would otherwise be invisible.

    Returns (rows, failed_items, ctx_by_path):
        rows         : list[dict] results, one per image, in image_paths order
        failed_items : [{'path', 'idx', 'ctx'}, ...] for manual review
        ctx_by_path  : {path: ctx} for EVERY attempted image (success or
                       fail) — lets a session viewer reopen any processed
                       image for manual adjustment later, not just failures.
    """
    rows: list[dict] = []
    failed_items: list[dict] = []
    ctx_by_path: dict[str, dict] = {}

    def record_failure(path, name, error, ctx):
        """Logs one image's failure, appends its error row, and remembers it
        (with whatever context got collected) for the manual-review phase."""
        logger.error("%s: %s", name, error)
        rows.append(_empty_row(config, get_sample_info(name), error))
        failed_items.append({"path": path, "idx": len(rows) - 1, "ctx": ctx})

    n = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        name = os.path.basename(path)
        if on_progress:
            on_progress(i, n, name)
        if path in skipped_set:
            rows.append(_empty_row(config, get_sample_info(name),
                                   "skipped during profile audit (see log)"))
            failed_items.append({"path": path, "idx": len(rows) - 1, "ctx": {}})
            ctx_by_path[path] = {}
            continue
        profile = best_profile(hist_cache[path], profiles_list)
        if profile is None:
            record_failure(path, name, "no matching color profile", {})
            ctx_by_path[path] = {}
            continue
        ctx: dict = {}
        try:
            rows.append(analyze_one(config, path, run_dir, profile,
                                    buffer, max_gap, context_out=ctx))
            n_interp = ctx.get("n_interpolated")
            if on_log and n_interp:
                on_log(f"{name}: {n_interp}/{ctx.get('n_columns', '?')} "
                       f"columns interpolated (gap-filled — normal)")
        except AnalysisError as e:
            record_failure(path, name, str(e), ctx)
        except Exception as e:  # unexpected — log traceback, keep batch alive
            logger.error("%s: unexpected %s: %s\n%s", name,
                         type(e).__name__, e, traceback.format_exc())
            rows.append(_empty_row(config, get_sample_info(name),
                                   f"unexpected {type(e).__name__}: {e}"))
            failed_items.append({"path": path, "idx": len(rows) - 1,
                                 "ctx": ctx})
        ctx_by_path[path] = ctx

    return rows, failed_items, ctx_by_path


# ================= Phase 3: manual correction =================

def build_prefill(config: BatchConfig, rgb, ctx: dict, error,
                  buffer: float, max_gap: int) -> dict:
    """Assemble the manual-fit GUI prefill from whatever the automatic
    pipeline managed for this image before failing (or completed, for a
    session-viewer re-edit of a successful image)."""
    prefill = {"error": error or None, "buffer": buffer, "max_gap": max_gap,
               "crop_inset": config.crop_inset_px}
    if ctx.get("plot_rgb") is not None:
        prefill["plot_rgb"] = ctx["plot_rgb"]

    result = ctx.get("result")
    if result is not None:
        # calibration fit succeeded (failure, if any, was downstream, e.g.
        # units) — slope magnitude is the axis-unit-per-pixel scale
        prefill["plot_region"] = result.plot_region
        prefill["x_unit_per_px"] = abs(result.x_slope)
        prefill["x_unit"] = result.x_unit
        prefill["x_conf"] = result.x_confidence
        prefill["y_unit_per_px"] = abs(result.y_slope)
        prefill["y_unit"] = result.y_unit
        prefill["y_conf"] = result.y_confidence
    elif ctx.get("profile") is not None:
        # calibration failed outright (usually OCR) — the plot region can
        # still be detected from the profile colors so the auto re-fit works
        colors = ctx["profile"]["colors"]
        try:
            region, _xg, _yg, _al = detect_plot_region_and_gridlines(
                rgb, tuple(colors["axis"]), tuple(colors["gridline"]))
            prefill["plot_region"] = region
        except Exception as e:
            logger.warning("plot-region detection for manual prefill "
                           "failed: %s", e)
    return prefill


def render_manual_overlay(config: BatchConfig, rgb, ctx: dict, data: dict,
                          run_dir: str, name: str, row: dict) -> str:
    """Render the final review overlay for a manually-corrected image onto
    the prior calibration debug PNG (when one exists) or the raw image.
    Returns the path relative to config.image_fits_dir."""
    result = ctx.get("result")
    out_path = getattr(result, "debug_path", None) if result is not None else None
    canvas = None
    if out_path and os.path.isfile(out_path):
        canvas = _load_overlay_base(out_path)
    if canvas is None:
        stem = os.path.splitext(name)[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(run_dir, f"{ts}_{stem}.png")
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    trace = data.get("trace") or {}
    cols = np.asarray(trace.get("cols") or [], dtype=float)
    rws = np.asarray(trace.get("rows") or [], dtype=float)
    smooth = trace.get("smooth_rows")
    smooth = np.asarray(smooth, dtype=float) if smooth else None

    def _sv(v):
        """Formats one summary value for the overlay box: numbers to 4
        significant digits, None and NaN both as a dash."""
        if v is None:
            return "-"
        if isinstance(v, (int, float)):
            return f"{v:.4g}" if np.isfinite(v) else "-"
        return str(v)

    summary = _calib_summary_lines(result) + [
        "entry=MANUAL"
        + ("  (auto trend accepted)" if data.get("auto_accepted") else ""),
        f"type={_sv(row.get('feature_type'))}  n={_sv(row.get('n_features'))}"
        f"  trend_conf={_sv(row.get('trend_conf'))}",
        f"h_avg={_sv(row.get('height_avg'))}  "
        f"w_avg={_sv(row.get('width_avg'))}  "
        f"dish_max={_sv(row.get('dishing_max'))}  ({config.output_unit})",
        f"Ra={_sv(row.get('Ra'))}  Rq={_sv(row.get('Rq'))}  "
        f"Rz={_sv(row.get('Rz'))}  ({config.output_unit})",
    ]

    mo.draw_measurements(canvas, cols, rws, smooth,
                         data["markers"]["height"],
                         data["markers"]["dishing"], summary,
                         width_markers=data["markers"].get("width"))
    cv2.imwrite(out_path, canvas)
    return os.path.relpath(out_path, config.image_fits_dir)


def copy_failed_images(config: BatchConfig, failed_items: list[dict],
                       run_ts: str) -> str:
    """Copy failed images into failed_images/<run_ts>/. Returns that dir."""
    fail_dir = os.path.join(config.failed_dir, run_ts)
    os.makedirs(fail_dir, exist_ok=True)
    for item in failed_items:
        try:
            shutil.copy2(item["path"], fail_dir)
        except OSError as e:
            logger.warning("could not copy %s to failed folder: %s",
                           os.path.basename(item["path"]), e)
    return fail_dir


def manual_review_phase(
    config: BatchConfig,
    failed_items: list[dict],
    rows: list[dict],
    run_dir: str,
    buffer: float,
    max_gap: int,
    on_status: Callable[[str], None],
    on_review_one: Callable[[str, str, dict], tuple[str, Optional[dict]]],
) -> None:
    """Drive manual correction over failed_items, mutating `rows` in place.

    on_status(message): progress text.
    on_review_one(path, name, prefill) -> (action, data): action is
        'saved' | 'skipped' | 'quit'; data is the manual_fit_gui payload
        (CLI: the blocking manual_fit_gui popup; GUI app: an embedded
        "Manual fit" tab).
    """
    for item in failed_items:
        path, idx, ctx = item["path"], item["idx"], item["ctx"]
        name = os.path.basename(path)
        try:
            rgb = load_rgb(path)
        except Exception as e:
            logger.error("%s: unreadable during manual review (%s)", name, e)
            continue
        prefill = build_prefill(config, rgb, ctx, rows[idx].get("error"),
                                buffer, max_gap)
        on_status(f"Manual fit: {name}")
        action, data = on_review_one(path, name, prefill)
        if action == "quit":
            on_status("Manual review stopped.")
            break
        if action != "saved" or not data:
            logger.info("manual fit skipped for %s", name)
            continue
        rows[idx] = merge_manual_row(rows[idx], data, config.output_unit)
        try:
            rel = render_manual_overlay(config, rgb, ctx, data, run_dir, name,
                                        rows[idx])
            rows[idx]["overlay_file"] = rel
        except Exception as e:
            logger.warning("manual overlay render failed for %s: %s", name, e)
        logger.info("manual entry saved for %s", name)


def log_lambda_c_capping(rows: list[dict]) -> int:
    """Counts how many rows had their roughness cutoff capped and, if any
    did, says so once for the whole run instead of per image — on these short
    profiles capping is the norm, so a per-image warning would be noise.
    Returns the count."""
    n_capped = sum(1 for r in rows if r.get("lambda_c_capped") is True)
    if n_capped:
        logger.warning(
            "%d/%d image(s) had lambda_c capped below the ISO 4288 "
            "recommendation — profiles are shorter than the standard's "
            "evaluation length (see the per-row lambda_c_capped column)",
            n_capped, len(rows))
    return n_capped


# ================= Phase 4: results CSV =================

def write_results_csv(config: BatchConfig, rows: list[dict], run_ts: str) -> str:
    """Writes the run's rows to a timestamped CSV in the output folder and
    returns its path. Columns are forced into a fixed order and their proper
    types first, so Excel opens the numbers as numbers."""
    os.makedirs(config.output_dir, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    _coerce_numeric_columns(df)
    _coerce_flagged(df)

    csv_path = os.path.join(config.output_dir, f"analysis_results_{run_ts}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def read_results_csv(csv_path: str) -> pd.DataFrame:
    """Reads a results CSV back into a DataFrame with the numeric columns as
    floats and `flagged` as real bools — including for files written before
    the flag column existed, which get it added as all-False."""
    df = pd.read_csv(csv_path)
    df = _coerce_numeric_columns(df)
    return _coerce_flagged(df)


def update_results_csv_row(csv_path: str, image_name: str, data: dict,
                           output_unit: str) -> bool:
    """Amend one row of an already-written results CSV, merging a manual-fit
    `data` payload with the same logic a live run uses
    (manual_merge.merge_manual_row). Matches by the 'Name' column.

    Used by the session viewer's "Adjust parameters" flow to push a
    correction made after the batch already wrote its CSV back into that
    file, so the review/verification pass amends the actual data file.

    Returns True if a matching row was found and updated, False otherwise.
    Rewrites the whole file — these are one-row-per-image run CSVs, so a
    full read/write is cheap.
    """
    df = read_results_csv(csv_path)
    mask = df["Name"] == image_name
    if not mask.any():
        return False
    for idx in df.index[mask]:
        merged = merge_manual_row(df.loc[idx].to_dict(), data, output_unit)
        df.loc[idx, df.columns] = [merged[c] for c in df.columns]

    _coerce_numeric_columns(df)
    _coerce_flagged(df)
    df.to_csv(csv_path, index=False)
    return True


def update_results_csv_flag(csv_path: str, image_name: str, flagged: bool) -> bool:
    """Amend just the 'flagged' cell of one row in an already-written
    results CSV. Matches by the 'Name' column — same pattern as
    update_results_csv_row, kept separate since toggling the review flag
    has nothing to do with a manual-fit payload and shouldn't run through
    merge_manual_row.

    Returns True if a matching row was found and updated, False otherwise.
    """
    df = read_results_csv(csv_path)
    mask = df["Name"] == image_name
    if not mask.any():
        return False
    df.loc[mask, "flagged"] = bool(flagged)
    df.to_csv(csv_path, index=False)
    return True


# ================= "Add to existing data file" =================

def load_existing_data_file(csv_path: str) -> dict:
    """
    Read a previously-written results CSV so a new batch run can be merged
    into it instead of starting a fresh file (see append_results_to_csv).

    Returns dict:
        rows        : list[dict], one per existing row, aligned to
                       CSV_COLUMNS and each tagged '_imported': True so the
                       Dashboard can gray them out and the Viewer can refuse
                       to open them (they belong to a prior session, not this
                       one — no image files or ctx are available for them).
        names       : set of the 'Name' column — the dedup key against the
                       new batch's input images.
        output_unit : the file's output unit ('um'/'nm'), taken from the
                       first non-null value of the 'output_unit' column, or
                       None if the column is absent/entirely empty. The
                       caller uses this to lock the new run's output unit so
                       every row in the merged file shares one unit.
    """
    df = read_results_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        row = {c: r[c] if c in df.columns else np.nan for c in CSV_COLUMNS}
        row["_imported"] = True
        rows.append(row)

    # isinstance guard (not just truthiness): float('nan') is truthy in
    # Python, so a CSV with no usable 'Name' values would otherwise leave
    # NaN floats in this set — harmless against string basenames, but noise.
    names = {r["Name"] for r in rows if isinstance(r.get("Name"), str) and r["Name"]}

    output_unit = None
    if "output_unit" in df.columns:
        non_null = df["output_unit"].dropna()
        non_null = non_null[non_null.astype(str).str.strip() != ""]
        if len(non_null):
            output_unit = str(non_null.iloc[0])

    return {"rows": rows, "names": names, "output_unit": output_unit}


def append_results_to_csv(csv_path: str, existing_rows: list[dict],
                          new_rows: list[dict], backup: bool = True) -> str:
    """
    Merge new_rows into the existing data file at csv_path and rewrite it in
    place, so a results CSV can be built up across many batch runs without
    duplicating images (existing_rows must already exclude anything also
    present in new_rows — the caller filters input images against
    load_existing_data_file()'s 'names' before running the batch).

    backup: when True (default), copy the file to
    '<name>.<YYYYmmdd_HHMMSS>.bak' next to it before overwriting, so the
    prior state of a growing historical dataset is always recoverable.

    Rows carrying '_imported' (from load_existing_data_file) are dropped
    silently since it isn't a CSV_COLUMNS field — the merged file has no
    per-row memory of "imported vs. new" on disk, only in the live session
    (Dashboard graying is a runtime-only view of that same distinction).

    Returns csv_path.
    """
    if backup and os.path.isfile(csv_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem, ext = os.path.splitext(csv_path)
        shutil.copy2(csv_path, f"{stem}.{ts}.bak{ext or '.csv'}")

    df = pd.DataFrame(list(existing_rows) + list(new_rows), columns=CSV_COLUMNS)
    _coerce_numeric_columns(df)
    _coerce_flagged(df)
    df.to_csv(csv_path, index=False)
    return csv_path
