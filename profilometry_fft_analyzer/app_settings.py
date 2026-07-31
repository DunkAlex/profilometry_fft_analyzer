"""
app_settings.py
---------------
One place for every tunable the pipeline exposes to the user, plus the JSON
files that persist them between sessions.

Two files live next to the app (see get_app_dir, so this works the same
frozen into an .exe as it does from source):

    parameter_defaults.json   pristine values, regenerated from PARAM_SPECS
    parameter_settings.json   what's actually in effect right now

Every parameter is also set as an attribute on this module, so the rest of
the codebase reads `settings.DISHING_NOISE_K` and picks up a change made in
the Preferences dialog on the next batch run — no restart needed. That only
holds for attribute reads at call time; a value captured once (a function
default argument, a dataclass field default) freezes at import and won't
follow along, which is why those few spots resolve their value inside the
function body instead.

Stdlib only, on purpose: feature_analysis and plot_calibration/config both
import this, and either of them pulling in cv2/pandas through app_engine
would be a circular import.
"""
from __future__ import annotations

import json
import logging
import os
import sys

logger = logging.getLogger("app_settings")

DEFAULTS_FILENAME = "parameter_defaults.json"
SETTINGS_FILENAME = "parameter_settings.json"


def get_app_dir() -> str:
    """Where the app keeps its writable files. Frozen by PyInstaller, that's
    the folder holding the .exe; running from source, it's this file's own
    folder. Everything user-editable (settings, profiles, output dirs) hangs
    off this so a frozen build stays configurable."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


# ================= Value casting =================

def _as_bool(v):
    """Coerce a JSON value or a typed-in string to a real bool. Accepts the
    spellings a user might reasonably type ('true', 'yes', '1') and treats
    anything else as False, so a stray entry can't crash a batch run."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y", "on")


def _as_opt_float(v):
    """Float, or None for a blank/'none' entry. Used by the parameters whose
    'unset' state is meaningful — lambda_c override being the main one, where
    None means 'let ISO 4288 pick it' rather than 'use 0'."""
    if v is None:
        return None
    text = str(v).strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


class ParamSpec:
    """One tunable: what it's called, what it defaults to, how to parse a
    typed-in value, which dialog section it belongs under, and the sentence
    or two of help text behind its [i] button. Optional `choices` turns the
    dialog row into a dropdown; optional lo/hi bound a numeric entry."""

    __slots__ = ("name", "default", "cast", "group", "label", "help",
                 "choices", "lo", "hi")

    def __init__(self, name, default, cast, group, label, help,
                 choices=None, lo=None, hi=None):
        """Records one parameter's identity, its default, how to parse it,
        and how the dialog should present it."""
        self.name = name
        self.default = default
        self.cast = cast
        self.group = group
        self.label = label
        self.help = help
        self.choices = choices
        self.lo = lo
        self.hi = hi

    def parse(self, raw):
        """Turn a raw dialog entry into a stored value, or raise ValueError
        with a message worth showing the user. Casts first, then range-checks
        numerics and membership-checks dropdowns."""
        value = self.cast(raw)
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"must be one of: {', '.join(map(str, self.choices))}")
        if value is not None and self.lo is not None and value < self.lo:
            raise ValueError(f"must be at least {self.lo}")
        if value is not None and self.hi is not None and value > self.hi:
            raise ValueError(f"must be at most {self.hi}")
        return value


# Dialog section order.
GROUPS = [
    "Feature detection",
    "Dishing",
    "Roughness",
    "Trace extraction",
    "Trend confidence",
    "Profile matching & color",
    "Gridlines & plot region",
    "OCR",
    "Axis fitting",
    "Calibration confidence",
    "Review & diagnostics",
]

# The tunables the Preferences dialog exposes. Deliberately curated: colors,
# fonts, CSV column lists and the OCR spelling-variant tables are all left
# out, since changing them by hand breaks things rather than tuning them.
PARAM_SPECS = [
    # --- Feature detection ---
    ParamSpec(
        "FEATURE_SMOOTH_LENGTH_FRAC", 1.0 / 50.0, float, "Feature detection",
        "Smoothing length (fraction of profile)",
        "How much the trace is smoothed before features are found, as a "
        "fraction of the profile length. Larger values suppress more noise "
        "but start rounding off real plateau shape.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "FEATURE_SMOOTH_MIN_SAMPLES", 5, int, "Feature detection",
        "Smoothing floor (samples)",
        "Lower limit on the smoothing window in samples, so very short "
        "profiles still get some noise suppression. Raise it if short scans "
        "come out jagged.",
        lo=1),
    ParamSpec(
        "NOISE_GATE_K", 5.0, float, "Feature detection",
        "Feature/noise gate (x noise Rq)",
        "How far apart the high and low levels must be, in multiples of the "
        "trace's own noise, before the profile counts as having real "
        "features. Raise it to stop noise being reported as features.",
        lo=0.0),
    ParamSpec(
        "SQUARE_MAX_TRANSITION_FRAC", 0.35, float, "Feature detection",
        "Square edge sharpness limit",
        "A feature is called square only if its rise and fall take up less "
        "than this fraction of its width. Raise it to classify more "
        "gently-sloped features as square rather than sine.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "SQUARE_MIN_DWELL_FRAC", 0.50, float, "Feature detection",
        "Square minimum flat-top fraction",
        "A feature is called square only if at least this fraction of it sits "
        "flat near its own median height. Lower it if genuinely square "
        "features with short tops are being misread as sine.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "DWELL_BAND_FRAC_OF_HEIGHT", 0.15, float, "Feature detection",
        "Flat-top tolerance band",
        "How close to the median height a sample must be to count as part of "
        "the flat top, as a fraction of feature height. Widen it for noisy "
        "or slightly domed tops.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "STEP_CENTRAL_FRAC", 0.60, float, "Feature detection",
        "Step-height central portion",
        "Step heights are measured over this central fraction of each "
        "segment, keeping the sloped edges out of the number. ISO 5436-1 "
        "uses the central third; 0.60 suits these shorter profiles.",
        lo=0.05, hi=1.0),

    # --- Dishing ---
    ParamSpec(
        "DISHING_PLATEAU_TOP_FRAC", 0.85, float, "Dishing",
        "Plateau top threshold",
        "Height fraction that marks where the flat top starts, used to trim "
        "the sloped flanks off before dishing is measured. Lower it if "
        "plateaus with rounded shoulders are being cut too short.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "DISHING_CENTER_SEARCH_FRAC", 0.50, float, "Dishing",
        "Center search width",
        "Fraction of the span between the two rim peaks that's searched for "
        "the deepest sag. Keeping it near the middle avoids the flat region "
        "right next to each peak; widen it to catch off-center dips.",
        lo=0.05, hi=1.0),
    ParamSpec(
        "DISHING_NOISE_K", 2.0, float, "Dishing",
        "Dishing noise floor (x noise Rq)",
        "A sag must exceed this multiple of the trace noise before it's "
        "reported as real dishing; anything smaller reports zero. Lower it "
        "to catch shallow dishing, at the risk of reporting noise.",
        lo=0.0),
    ParamSpec(
        "DISHING_MIN_PLATEAU_SAMPLES", 5, int, "Dishing",
        "Minimum plateau samples",
        "Plateaus narrower than this many samples are skipped rather than "
        "measured. Raise it if narrow features are producing unstable "
        "dishing numbers.",
        lo=2),

    # --- Roughness ---
    ParamSpec(
        "LAMBDA_C_MAX_PROFILE_FRAC", 0.20, float, "Roughness",
        "Roughness cutoff cap (fraction of profile)",
        "Caps the roughness cutoff so at least five sampling lengths fit in "
        "the profile. These scans are short, so this cap usually binds — "
        "rows where it did are flagged in the CSV.",
        lo=0.01, hi=1.0),
    ParamSpec(
        "ROUGHNESS_BASELINE_MIN_SAMPLES", 16, int, "Roughness",
        "Minimum baseline samples",
        "How many reference-surface samples are needed before roughness is "
        "measured on the baseline alone instead of the whole profile. Below "
        "it there isn't enough flat ground to say anything about texture.",
        lo=2),
    ParamSpec(
        "LAMBDA_C_OVERRIDE_UM", None, _as_opt_float, "Roughness",
        "Fixed roughness cutoff (um, blank = auto)",
        "Forces a specific roughness cutoff in micrometers instead of letting "
        "ISO 4288 pick one per image. Leave blank for automatic selection; "
        "set it when you need every image measured on identical terms.",
        lo=0.0),
    ParamSpec(
        "ROUGHNESS_ON_BASELINE", True, _as_bool, "Roughness",
        "Measure roughness on baseline only",
        "When on, Ra/Rq/Rz come from the flat surface between features, so "
        "step heights don't dominate the numbers. Turn it off to measure "
        "texture across the whole profile including the features."),

    # --- Trace extraction ---
    ParamSpec(
        "OUTPUT_UNIT", "um", str, "Trace extraction",
        "Output unit",
        "Unit for every length in the results CSV and on the overlays. "
        "Changing it rescales what's reported; it doesn't change what was "
        "measured.",
        choices=("um", "nm")),
    ParamSpec(
        "COLOR_BUFFER", 120.0, float, "Trace extraction",
        "Trace color tolerance",
        "How far a pixel's color may sit from the profile's trace color and "
        "still be treated as part of the curve. Raise it if the trace comes "
        "out patchy, lower it if background is being picked up. Note: a saved "
        "tuned_params.json from the tuning GUI overrides this.",
        lo=0.0),
    ParamSpec(
        "MAX_GAP_PX", 5, int, "Trace extraction",
        "Max interpolated gap (px)",
        "Largest run of trace-free columns that gets filled in by "
        "interpolation; wider gaps are left empty. Raise it for dashed or "
        "broken traces. Also overridden by tuned_params.json.",
        lo=0),
    ParamSpec(
        "CROP_INSET_PX", 2, int, "Trace extraction",
        "Plot border inset (px)",
        "Pixels trimmed off each edge of the detected plot area before the "
        "trace is read, so the axis frame can't be mistaken for data. Raise "
        "it if thick borders are bleeding into the trace.",
        lo=0),
    ParamSpec(
        "PERIODIC_MIN_FEATURES", 3, int, "Trace extraction",
        "Periodic-array feature count",
        "How many features an image needs before it's treated as a periodic "
        "array, which switches roughness cutoff selection to the ISO spacing "
        "rule instead of the Ra rule.",
        lo=1),

    # --- Trend confidence ---
    ParamSpec(
        "TREND_CONF_W_COVERAGE", 0.50, float, "Trend confidence",
        "Weight: trace coverage",
        "Share of the extraction confidence score that comes from how much "
        "of the plot width the trace actually covers. The three trend "
        "weights should add up to 1.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "TREND_CONF_W_GAP", 0.25, float, "Trend confidence",
        "Weight: gap penalty",
        "Share of the extraction confidence score driven by the largest gap "
        "that had to be interpolated. The three trend weights should add up "
        "to 1.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "TREND_CONF_W_JUMP", 0.25, float, "Trend confidence",
        "Weight: jump penalty",
        "Share of the extraction confidence score driven by sudden "
        "column-to-column jumps in the trace. The three trend weights should "
        "add up to 1.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "TREND_CONF_GAP_ZERO_FRAC", 0.10, float, "Trend confidence",
        "Gap size that zeroes the score",
        "A single interpolated gap this wide, as a fraction of plot width, "
        "drives the gap part of the confidence score to zero.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "TREND_CONF_JUMP_HEIGHT_FRAC", 0.05, float, "Trend confidence",
        "Jump size counted as a glitch",
        "A column-to-column step larger than this fraction of the plot "
        "height is counted as an extraction glitch rather than real signal.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "TREND_CONF_JUMP_ZERO_FRAC", 0.10, float, "Trend confidence",
        "Glitch rate that zeroes the score",
        "Once this fraction of columns are glitches, the jump part of the "
        "confidence score goes to zero.",
        lo=0.0, hi=1.0),

    # --- Profile matching & color ---
    ParamSpec(
        "HIST_MATCH_THRESHOLD", 0.35, float, "Profile matching & color",
        "Color profile match distance",
        "How different an image's colors may be from a saved profile and "
        "still match it. Raise it if familiar images keep prompting for a new "
        "profile; lower it if the wrong profile is being picked.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "COLOR_MASK_TOLERANCE", 15, int, "Profile matching & color",
        "Color mask tolerance (per channel)",
        "Per-channel RGB slack when selecting pixels of a known color, used "
        "for axis and gridline masks. Raise it for compressed or anti-aliased "
        "images where colors are smeared.",
        lo=0, hi=255),
    ParamSpec(
        "PROFILE_COLOR_MIN_SEPARATION", 20, int, "Profile matching & color",
        "Minimum separation between picked colors",
        "How distinguishable the four picked colors must be from each other. "
        "If two are too close, calibration refuses to run rather than "
        "silently producing garbage gridlines.",
        lo=0, hi=255),

    # --- Gridlines & plot region ---
    ParamSpec(
        "AXIS_LINE_MIN_LENGTH_FRAC", 0.3, float, "Gridlines & plot region",
        "Minimum axis line length",
        "How much of the image width or height a line must span to count as "
        "a plot axis. Lower it if the plot frame is short or partly broken.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "GRIDLINE_PEAK_MIN_HEIGHT_FRAC", 0.3, float, "Gridlines & plot region",
        "Gridline detection strength",
        "How strong a candidate gridline must be relative to the strongest "
        "one found. Lower it to pick up faint gridlines, at the risk of "
        "catching texture.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "GRIDLINE_MIN_SPACING_PX", 15, int, "Gridlines & plot region",
        "Minimum gridline spacing (px)",
        "Closest two gridlines may sit and still be counted separately. Raise "
        "it if a single thick gridline is being detected twice.",
        lo=1),
    ParamSpec(
        "GRIDLINE_SPACING_CV_TOLERANCE", 0.08, float, "Gridlines & plot region",
        "Gridline regularity tolerance",
        "How uneven gridline spacing may be before the grid is judged "
        "irregular. Regularity is what lets missed tick values be "
        "reconstructed, so loosening this affects OCR recovery.",
        lo=0.0, hi=1.0),

    # --- OCR ---
    ParamSpec(
        "OCR_UPSCALE_FACTOR", 4, int, "OCR",
        "OCR upscale factor",
        "How much tick labels are enlarged before character recognition. "
        "Higher is more accurate on small text but slower; 4 is a good "
        "balance for typical exports.",
        lo=1, hi=16),
    ParamSpec(
        "OCR_MIN_CONFIDENCE", 0.3, float, "OCR",
        "Minimum tick-number confidence",
        "Recognition results below this confidence are discarded rather than "
        "trusted as axis values. Raise it if misread numbers are corrupting "
        "the axis fit.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "UNIT_OCR_MIN_CONFIDENCE", 0.15, float, "OCR",
        "Minimum unit-label confidence",
        "Same idea as the tick threshold but for unit strings like 'um', "
        "which read lower than clean digits. Raise it if units are being "
        "misdetected.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "TICK_LABEL_BAND_PX", 60, int, "OCR",
        "Tick label search band (px)",
        "How far outside the plot area to look for tick labels. Raise it when "
        "labels sit unusually far from the axis.",
        lo=1),

    # --- Axis fitting ---
    ParamSpec(
        "FIT_METHOD", "theil_sen", str, "Axis fitting",
        "Axis fit method",
        "How the pixel-to-units axis line is fitted. theil_sen resists bad "
        "OCR reads best, ransac also rejects outliers, lstsq is a plain fit "
        "that trusts every point.",
        choices=("theil_sen", "ransac", "lstsq")),
    ParamSpec(
        "OUTLIER_RESIDUAL_TOLERANCE", 0.15, float, "Axis fitting",
        "Outlier rejection tolerance",
        "How far a tick value may sit from the fitted line, as a fraction of "
        "the axis range, before it's dropped as an outlier. Lower it to "
        "reject misreads more aggressively.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "MIN_TICKS_FOR_FIT", 3, int, "Axis fitting",
        "Minimum ticks to fit an axis",
        "Fewer readable tick labels than this and calibration fails outright "
        "instead of fitting a line through too little data.",
        lo=2),
    ParamSpec(
        "MIN_DISTINCT_TICK_VALUES", 2, int, "Axis fitting",
        "Minimum distinct tick values",
        "Guards against a fit through ticks that all read the same value, "
        "which looks mathematically perfect but describes nothing. Leave at "
        "2 unless you have a reason.",
        lo=2),

    # --- Calibration confidence ---
    ParamSpec(
        "CONF_W_R2", 0.40, float, "Calibration confidence",
        "Weight: fit quality",
        "Share of the calibration confidence score from how well the axis fit "
        "matches its tick values. These four weights should add up to 1.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "CONF_W_TICK_COVERAGE", 0.25, float, "Calibration confidence",
        "Weight: tick coverage",
        "Share of the calibration confidence score from how many gridlines "
        "got a readable label. These four weights should add up to 1.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "CONF_W_OCR_MEAN", 0.20, float, "Calibration confidence",
        "Weight: OCR confidence",
        "Share of the calibration confidence score from average character "
        "recognition confidence. These four weights should add up to 1.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "CONF_W_GRIDLINE_REG", 0.15, float, "Calibration confidence",
        "Weight: gridline regularity",
        "Share of the calibration confidence score from how evenly spaced the "
        "gridlines are. These four weights should add up to 1.",
        lo=0.0, hi=1.0),

    # --- Review & diagnostics ---
    ParamSpec(
        "LOW_CONFIDENCE_THRESHOLD", 0.5, float, "Review & diagnostics",
        "Dashboard low-confidence highlight",
        "Rows whose calibration or extraction confidence falls below this get "
        "highlighted for review. Raise it to be shown more images to check.",
        lo=0.0, hi=1.0),
    ParamSpec(
        "MAX_LABELED_HEIGHTS", 8, int, "Review & diagnostics",
        "Max height labels per overlay",
        "Above this many features, only the smallest, median and largest get "
        "a printed height on the overlay — the bars are still drawn for all. "
        "Keeps dense arrays readable.",
        lo=1),
    ParamSpec(
        "MAX_LABELED_WIDTHS", 8, int, "Review & diagnostics",
        "Max width labels per overlay",
        "Same limit as heights, applied to the width bars.",
        lo=1),
    ParamSpec(
        "CALIB_DEBUG_LOG", False, _as_bool, "Review & diagnostics",
        "Verbose calibration logging",
        "Writes detailed OCR and tick-search decisions to the run log. Useful "
        "when chasing down why one image calibrated badly; noisy otherwise."),
    ParamSpec(
        "DEBUG_SAVE", True, _as_bool, "Review & diagnostics",
        "Save calibration overlay images",
        "Saves the annotated calibration image for each run. Turning it off "
        "speeds up large batches but leaves you nothing to review visually."),
]

SPEC_BY_NAME = {spec.name: spec for spec in PARAM_SPECS}
MANAGED_NAMES = frozenset(SPEC_BY_NAME)


# ================= File paths =================

def defaults_path() -> str:
    """Full path to the pristine-defaults JSON."""
    return os.path.join(get_app_dir(), DEFAULTS_FILENAME)


def settings_path() -> str:
    """Full path to the active-settings JSON."""
    return os.path.join(get_app_dir(), SETTINGS_FILENAME)


def default_values() -> dict:
    """The built-in defaults straight from PARAM_SPECS, as a fresh dict."""
    return {spec.name: spec.default for spec in PARAM_SPECS}


def _read_json(path: str) -> dict:
    """Load a JSON dict from disk, returning {} if it's missing or unreadable
    — a corrupt settings file should fall back to defaults, not stop the app
    from starting."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("could not read %s (%s) — using defaults",
                       os.path.basename(path), e)
        return {}


def _write_json(path: str, values: dict) -> None:
    """Write a values dict to disk as sorted, indented JSON so it stays
    readable and diffable if someone edits it by hand."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(values, fh, indent=2, sort_keys=True)
        fh.write("\n")


def ensure_files() -> None:
    """Make sure both JSON files exist and cover every known parameter.
    Defaults are rewritten whenever a new parameter appears in PARAM_SPECS;
    the active file is only topped up with the missing keys, so a user's
    existing choices survive an upgrade."""
    defaults = default_values()
    on_disk_defaults = _read_json(defaults_path())
    if on_disk_defaults != defaults:
        _write_json(defaults_path(), defaults)

    active = _read_json(settings_path())
    merged = dict(defaults)
    merged.update({k: v for k, v in active.items() if k in MANAGED_NAMES})
    if merged != active:
        _write_json(settings_path(), merged)


# ================= Active values =================

_values: dict = {}


def _apply(values: dict) -> None:
    """Publish a values dict onto this module as plain attributes, so every
    `settings.NAME` read elsewhere sees it immediately. Values are pushed
    through each spec's cast so a hand-edited JSON string still lands as the
    right type."""
    _values.clear()
    for spec in PARAM_SPECS:
        raw = values.get(spec.name, spec.default)
        try:
            value = spec.cast(raw) if raw is not None or spec.default is None else spec.default
        except (TypeError, ValueError):
            logger.warning("bad value %r for %s — using default %r",
                           raw, spec.name, spec.default)
            value = spec.default
        _values[spec.name] = value
        globals()[spec.name] = value


def load() -> dict:
    """Read defaults, overlay whatever the user has saved, publish the result
    as module attributes, and hand back the active values. Unknown keys in
    the settings file are ignored and missing ones fall back to their
    default, so an old file from a previous version still works."""
    ensure_files()
    values = default_values()
    stored = _read_json(settings_path())
    unknown = [k for k in stored if k not in MANAGED_NAMES]
    if unknown:
        logger.info("ignoring %d unrecognized setting(s) in %s: %s",
                    len(unknown), SETTINGS_FILENAME, ", ".join(sorted(unknown)))
    values.update({k: v for k, v in stored.items() if k in MANAGED_NAMES})
    _apply(values)
    return dict(_values)


def save(values: dict) -> dict:
    """Merge a dict of new values over the current ones, write them to the
    active settings file, and publish them so the change takes effect on the
    next run. Returns the full active set."""
    merged = dict(_values)
    merged.update({k: v for k, v in values.items() if k in MANAGED_NAMES})
    _write_json(settings_path(), merged)
    _apply(merged)
    logger.info("saved %d parameter(s) to %s", len(merged), SETTINGS_FILENAME)
    return dict(_values)


def reset_to_defaults() -> dict:
    """Throw away the user's overrides and go back to the shipped defaults,
    rewriting the active file so the reset sticks across restarts."""
    defaults = default_values()
    _write_json(settings_path(), defaults)
    _apply(defaults)
    logger.info("parameters reset to defaults")
    return dict(_values)


def current() -> dict:
    """Snapshot of the active values, safe for the dialog to mutate."""
    return dict(_values)


def is_modified() -> bool:
    """True when any active value differs from its default — lets the UI say
    whether a reset would actually change anything."""
    return _values != default_values()


def iter_specs():
    """Walk the specs in dialog order: grouped by GROUPS, in the order they
    were declared within each group."""
    for group in GROUPS:
        for spec in PARAM_SPECS:
            if spec.group == group:
                yield spec


# Populate on import so `import app_settings as settings` is enough to read
# any parameter, without every caller having to remember to call load().
load()
