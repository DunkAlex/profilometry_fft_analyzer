"""
config.py
---------
Tunable constants for the calibration pipeline.

Values split two ways. The ones an engineer is likely to want to change
(OCR confidence gates, gridline thresholds, the fit method, confidence
weights) now live in app_settings.py so the Preferences dialog can edit
them and persist the change — see MANAGED below. The rest stay here as
plain constants, either because they're structural (unit spelling tables,
"nice" step sizes) or because getting them wrong breaks the pipeline
rather than tuning it.

Nothing changes for callers: `config.OCR_MIN_CONFIDENCE` still works for
either kind. Managed names resolve through the module __getattr__ at the
bottom, which forwards to app_settings, so a Preferences edit takes effect
on the next run without a restart. The rationale comments for managed
values are kept here — that reasoning is worth more next to the pipeline
than squeezed into a tooltip.

One catch worth knowing: module __getattr__ only fires for attribute
access from *outside* the module. A bare reference to a managed name
inside this file would raise NameError, so derived values (see
UNIT_SEARCH_BAND_DEPTH_PX) are computed in __getattr__ too.
"""
import app_settings as _settings

# --- Profile matching ---
HIST_BITS_PER_CHANNEL = 4              # 4 bits/channel -> 16^3 = 4096 bins
HIST_MATCH_METHOD = "bhattacharyya"    # cv2.HISTCMP_BHATTACHARYYA
# HIST_MATCH_THRESHOLD (managed): distance below this = profile match.

# --- Color picking / masks ---
COLOR_PICK_PATCH_SIZE = 3              # NxN patch median (no bg exclusion)
COLOR_PICK_PATCH_SIZE_MASKED = 5       # NxN patch when excluding background
COLOR_PICK_EXCLUDE_TOL = 15            # per-channel tol for "is background"
COLOR_PICK_MIN_NON_BG = 3              # need >= this many non-bg pixels for median
BG_QUANTIZE_STEP = 8                   # quantization step for background mode
# COLOR_MASK_TOLERANCE (managed): per-channel RGB tolerance for masks.

# --- Profile color validation ---
# PROFILE_COLOR_MIN_SEPARATION (managed). The four picked colors
# (background, plot, axis, gridline) must be distinguishable from each
# other, or plot-region/gridline detection silently produces garbage (e.g.
# axis and gridline picked as the same color makes it impossible to tell
# the plot border from an internal gridline). This is the minimum
# per-channel max-abs-difference required between every pair; below it, the
# pipeline refuses to proceed and reports which pair is the problem rather
# than cascading into a confusing downstream failure like "no gridlines
# found".

# --- Plot region detection ---
# AXIS_LINE_MIN_LENGTH_FRAC (managed): axis line must span >= this frac of image.

# --- Gridline detection ---
# All three managed:
#   GRIDLINE_PEAK_MIN_HEIGHT_FRAC  frac of max projection value
#   GRIDLINE_MIN_SPACING_PX        min separation between gridlines
#   GRIDLINE_SPACING_CV_TOLERANCE  std/mean of gap sizes for regularity

# --- OCR ---
OCR_LABEL_PAD_PX = 4                   # padding around blob bbox before crop
OCR_LANGUAGES = ["en"]
# OCR_UPSCALE_FACTOR (managed): upscale before OCR (huge accuracy win).
# OCR_MIN_CONFIDENCE (managed): drop OCR results below this.

# --- Text blob filtering (used by global unit-label scan) ---
TEXT_BLOB_MIN_AREA = 8                 # px, reject noise
TEXT_BLOB_MAX_AREA_FRAC = 0.02         # frac of image area, reject huge blobs
TEXT_BLOB_MIN_ASPECT = 0.1             # w/h min (reject thin lines)
TEXT_BLOB_MAX_ASPECT = 10.0            # w/h max
# TICK_LABEL_BAND_PX (managed): search band beyond plot region edge.

# --- Per-gridline anchored tick search (primary path) ---
# Each gridline gets its own local search window, sized from that gridline's
# own neighborhood, with expanding redundancy if nothing is found initially.
# This is what catches thin single-digit labels (e.g. a lone "1") that a
# global min-area filter would otherwise drop, and it guarantees every
# gridline gets an attempt rather than relying on a blob surviving a
# whole-image filter pass.
TICK_SEARCH_EXPANSIONS = [1.0, 1.6, 2.5, 4.0]   # successive window growth factors
TICK_SEARCH_PERP_HALF_WIDTH = 14                # px, half-width across the axis direction
TICK_SEARCH_LOCAL_MIN_AREA = 2                  # relaxed vs TEXT_BLOB_MIN_AREA: window is tight,
                                                 # so false-positive risk from noise is low
TICK_SEARCH_MIN_LABEL_HEIGHT_PX = 5             # reject blobs shorter than this — real digits run
                                                 # ~8-14px tall; anti-aliasing artifacts where a
                                                 # gridline crosses the axis line are 1-2px flecks
                                                 # that would otherwise satisfy the area filter and
                                                 # get accepted before the search ever reaches the
                                                 # real (slightly farther) digit. Width is NOT
                                                 # constrained here — a thin "1" is still fine.

# --- Vertical-fragment recovery (skinny-"1" fix) ---
# A lone "1" is a ~2 px × 10 px vertical stroke; anti-aliasing can chop it into
# 2-3 sub-height fragments (top serif, stem, baseline) that individually fail
# TICK_SEARCH_MIN_LABEL_HEIGHT_PX and get dropped, so no blob is ever found for
# that gridline. _local_blob_search pools the sub-height rejects and, if two or
# more of them share a column (x centres within VERT_FRAG_X_TOL_PX) and are near-
# adjacent vertically (gap <= VERT_FRAG_MAX_GAP_PX), merges them into one bbox
# whose combined height clears the min. Symmetric counterpart to the existing
# horizontal proximity clustering. Only activates on blobs the primary pass
# would have discarded, so it cannot regress any currently-passing case.
TICK_SEARCH_VERT_FRAG_X_TOL_PX = 2
TICK_SEARCH_VERT_FRAG_MAX_GAP_PX = 3

# --- Narrow-glyph numeric OCR retry (skinny-"1" fix) ---
# Even with the correct blob found, ocr_number's digit allowlist at 4× upscale
# often can't classify a ~2 px-wide vertical stroke as "1" — EasyOCR prefers
# "I"/"l"/"|" and the allowlist rejects them. Two extra attempts run only when
# the bbox is narrower than NARROW_NUM_WIDTH_PX AND the primary read failed:
# (a) re-crop with pad_x = NARROW_NUM_EXTRA_PAD_PX and re-run the allowlisted
# OCR — extra whitespace context helps classification; (b) if still nothing and
# NARROW_NUM_ALLOW_TEXT_FALLBACK, run ocr_text (no allowlist) and coerce through
# postprocess_number_string, which maps "I"/"l"/"|" -> "1" via _OCR_SUBS.
NARROW_NUM_WIDTH_PX = 4
NARROW_NUM_EXTRA_PAD_PX = 10
NARROW_NUM_ALLOW_TEXT_FALLBACK = True

# --- Inferred-entry back-verification ---
# reconstruct_tick_values fills gridlines OCR missed with values solved from the
# arithmetic progression, but the debug overlay draws them as inferred (no green
# bbox) which the user reads (correctly) as "detection failed". For each inferred
# entry, retry the anchored tick search + narrow-glyph-tolerant OCR at that
# gridline; if the read agrees with the predicted value (|ocr - predicted| <=
# AGREEMENT_FRAC_OF_STEP * step) and clears MIN_OCR_CONF, promote the entry to
# a confirmed detection (bbox attached, inferred flag cleared, value replaced).
# The agreement gate uses the sequence prediction as the trust anchor, so
# MIN_OCR_CONF is deliberately below the primary OCR threshold — a matching
# retry is trustworthy on its own if the number came out right. A mismatched
# retry cannot change the fit input; it just leaves the entry inferred as before.
INFERRED_BACKVERIFY_ENABLED = True
INFERRED_BACKVERIFY_AGREEMENT_FRAC_OF_STEP = 0.3
INFERRED_BACKVERIFY_MIN_OCR_CONF = 0.15

# --- Unit label OCR (e.g. "um", "mV") ---
# UNIT_OCR_MIN_CONFIDENCE (managed). Units are short, sometimes stylized
# strings; EasyOCR confidence tends to run lower on them than on clean
# digits, so a separate, lower threshold than OCR_MIN_CONFIDENCE.

# --- Unit vs. number classification (unified per-axis label pass) ---
# When classifying an axis's OCR candidates, a blob whose text matches a
# known unit variant (see UNIT_CANONICAL_FORMS) at or above this score wins
# unit classification EVEN IF it also happens to parse as a number (e.g. a
# stray digit picked up alongside the glyph). Only the single
# highest-scoring unit candidate per axis is removed from the numeric pool;
# match_score is binary (1.0 = known variant, 0.0 = unrecognized) by design
# — see score_unit_match in ocr.py — so this threshold is really an on/off
# gate, not a tunable fuzzy cutoff.
UNIT_MATCH_MIN_CONFIDENCE = 0.6

# --- Unit label full-band search ---
# The unit string can sit anywhere along an axis — on this instrument the
# x-axis unit is printed at the FAR END of the axis (in place of the last
# tick's number), while the y-axis unit sits near the origin. Rather than
# guessing anchor points, scan the whole tick-label band for the axis in one
# pass (find_unit_label_at_axis -> _axis_band_blobs) and classify every text
# blob found. This is the "identify all areas of text, then pick the unit"
# workflow, scoped to the axis band so unrelated content stays out of range.
# UNIT_SEARCH_BAND_DEPTH_PX is derived from the managed TICK_LABEL_BAND_PX
# (x3) and so is computed in __getattr__ rather than assigned here — how far
# into the margin, perpendicular to the axis, the unit-band scan reaches.
UNIT_OCR_CROP_PAD_PX = 12             # padding around a unit-candidate bbox before
                                      # OCR — deliberately larger than the numeric
                                      # OCR_LABEL_PAD_PX so a micro sign that got
                                      # split off into its own connected component
                                      # (leaving a bare "m") is pulled back into the
                                      # crop and the token reads "um", not "m". This
                                      # is the fix for the far-end x-axis unit, whose
                                      # glyph is identical to the reliably-read
                                      # y-axis one but degrades in isolation.

# Extra axis-parallel margin around the identified unit bbox when excluding it
# from the numeric tick search / from the missing-number bookkeeping. A tightly
# measured unit bbox may sit 1-2 px inside the actual last-gridline pixel
# position, which would let the numeric search re-open a phantom read at that
# gridline; adding a small halo makes the exclusion positionally forgiving
# without changing what "the unit is here" means.
UNIT_EXCLUDE_HALO_PX = 4

# --- Whole-band unit OCR safety net ---
# Terminal fallback for find_unit_label_at_axis: when the per-blob classification
# and the ASCII-only textual fallback both yield nothing, run a single ocr_text
# over the entire axis band as one image (downscaled if larger than
# UNIT_WHOLEBAND_OCR_MAX_SIDE_PX so EasyOCR handles it happily) and accept
# whichever unique returned token matches a known variant. This catches the case
# where EasyOCR's own detector segments "µm" cleanly at page scale even though
# our connected-components extractor split the glyph. Bounded to activate only
# after the primary + ASCII paths both fail, so it cannot regress a currently-
# passing case; disable via the flag if it ever misbehaves on new data.
UNIT_WHOLEBAND_OCR_ENABLED = True
UNIT_WHOLEBAND_OCR_MAX_SIDE_PX = 600

# --- Unit string normalization ---
# Canonical ASCII form -> known raw OCR strings/unicode spellings that mean
# the same unit. Keys are what the pipeline always reports (per convention,
# always ASCII "um", never the unicode micro sign, so downstream code never
# has to deal with encoding surprises in filenames/CSVs/etc).
#
# Deliberately conservative: this is an explicit lookup, not fuzzy/edit-
# distance matching. Units that are genuinely different scales (nm vs um vs
# mm) must never be auto-corrected into each other even if an edit-distance
# heuristic would consider them "close" — a silent 1000x scale error is far
# worse than an unrecognized unit. Extend the variant sets below as new OCR
# misreads are observed for this instrument/font; each addition should be a
# specific confirmed misread, not a guessed pattern.
#
# NOTE: bare "m" (meters) and "cm" (centimeters) are DELIBERATELY absent.
# This instrument only ever emits micro/nano/milli-scale units
# (µm/nm/mm, occasionally pm — confirmed by the user against the full test
# set), so a bare "m" is ALWAYS a degraded read of "µm" whose micro sign was
# lost, never a real meters label. Registering "m" as a valid single-char
# unit made that garbage read score a perfect match and win over the correct
# "µm" — the exact far-end x-axis misread this round fixes. Do not re-add "m"
# or "cm" unless this instrument starts producing them.
UNIT_CANONICAL_FORMS = {
    "um": {"um", "µm", "μm", "hm", "hrn", "urn", "jum", "wn", "wrn"},
    "pm": {"pm"},
    "nm": {"nm", "nrn", "nmn", "nn"},
    "mm": {"mm", "rnm"},
    "mV": {"mv", "rnv"},
    "V":  {"v"},
    "s":  {"s"},
    "ms": {"ms", "rns"},
}

# --- Y-axis pm/um cross-axis disambiguation ---
# "pm" and "um" are visually confusable in OCR — the µ glyph can misread as
# "p" at small font sizes, in either direction (a true "um" label can read as
# "pm", and vice versa). Per the user's confirmed domain knowledge: pm, when
# it occurs at all, is only ever a Y-axis (height) unit on this instrument —
# the X-axis (lateral distance) is always nm/um/mm, never pm. So whenever the
# X-axis unit has already resolved to one of those, a Y-axis reading of "pm"
# is presumed to be a misread "um" and gets corrected. The one case left as
# "pm" is when the X-axis unit is itself unresolved (None) or genuinely reads
# "pm" — there isn't enough cross-axis evidence to rule pm out then, and the
# raw OCR match is trusted as-is.
Y_PM_DISAMBIGUATION_ENABLED = True
Y_PM_SAFE_XUNITS = {"nm", "um", "mm"}

# --- Fitting / outlier handling (all three managed) ---
#   FIT_METHOD                  "theil_sen" | "ransac" | "lstsq"
#   OUTLIER_RESIDUAL_TOLERANCE  frac of axis value range
#   MIN_TICKS_FOR_FIT           hard fail below this

# "Nice" step sizes used when snapping recovered OCR values
NICE_STEPS = [1, 2, 2.5, 5]            # multiplied by powers of 10

# --- Tick-value sequence reconstruction ---
# Gridlines are drawn evenly spaced by the plotting software (verified via
# check_gridline_regularity), so tick VALUES must form an arithmetic
# progression `value = v0 + i * step` over gridline index `i`. When OCR
# misses a tick entirely (a skinny "1" that never produced a blob) or
# misreads one badly enough that two different gridlines end up with the
# same value (collapsing the fit toward a near-zero slope), solving for
# (v0, step) from a few trustworthy reads and filling in the rest recovers
# both failure modes with one mechanism. Gated hard on regularity plus a
# minimum number of confident anchor reads — with too few or too weak
# anchors, reconstruction is skipped and the input is returned unchanged
# rather than fabricating a sequence from unreliable data.
SEQ_RECONSTRUCT_MIN_CONFIDENT_ANCHORS = 2     # need >= this many trustworthy OCR reads to solve v0/step
SEQ_RECONSTRUCT_MIN_OCR_CONF = 0.5            # "trustworthy" bar for an anchor — stricter than
                                               # OCR_MIN_CONFIDENCE, since an anchor error propagates
                                               # to every reconstructed tick, not just itself
SEQ_RECONSTRUCT_AGREEMENT_FRAC_OF_STEP = 0.3  # an OCR'd value within this fraction of one step of the
                                               # predicted sequence value is "confirmed" and kept as-is;
                                               # otherwise it's treated as a misread and overridden
SEQ_RECONSTRUCT_INFERRED_CONF = 0.5           # confidence assigned to filled-in/overridden values —
                                               # moderate: not as trusted as a real high-confidence OCR
                                               # read, but not zero, since the sequence is well-grounded

# --- Degenerate-fit rejection ---
# A fit whose tick values collapse to (near-)one distinct value despite
# spanning >= MIN_TICKS_FOR_FIT gridlines is not a real linear relationship
# — it's misread OCR data that happened to average out flat. Theil-Sen
# reports slope~=0 and, since ss_tot~=0, r_squared=1.0: a deceptively
# "perfect" fit to nothing. Reject rather than silently emitting
# px_per_unit = inf (or a wildly unphysical value) — see calibrate.py's
# degenerate-fit guard.
# MIN_DISTINCT_TICK_VALUES (managed).
MIN_ABS_SLOPE = 1e-6                   # value units per pixel; below this, treat the fit as degenerate

# --- Diagnostics ---
# CALIB_DEBUG_LOG (managed). Opt-in verbose logging (via logger.debug) of
# OCR candidate/rejection traces: unit-search candidates and their
# score_unit_match / postprocess_number_string results, and per-gridline
# tick search hits/misses. Off by default — this is for root-causing a
# specific misdetection, not routine operation.

# --- Confidence weights (managed; should sum to 1.0) ---
#   CONF_W_R2, CONF_W_TICK_COVERAGE, CONF_W_OCR_MEAN, CONF_W_GRIDLINE_REG

# --- Debug output ---
DEBUG_OUTPUT_DIR = "image_fits"
# DEBUG_SAVE (managed): whether calibration overlay PNGs are written.


# ================= Managed-value forwarding =================

# Names that live in app_settings.py instead of this file. Kept as a frozen
# set so __getattr__ can tell "user-tunable, go look it up" from a genuine
# typo and still raise AttributeError for the latter.
MANAGED = frozenset({
    "HIST_MATCH_THRESHOLD",
    "COLOR_MASK_TOLERANCE",
    "PROFILE_COLOR_MIN_SEPARATION",
    "AXIS_LINE_MIN_LENGTH_FRAC",
    "GRIDLINE_PEAK_MIN_HEIGHT_FRAC",
    "GRIDLINE_MIN_SPACING_PX",
    "GRIDLINE_SPACING_CV_TOLERANCE",
    "OCR_UPSCALE_FACTOR",
    "OCR_MIN_CONFIDENCE",
    "UNIT_OCR_MIN_CONFIDENCE",
    "TICK_LABEL_BAND_PX",
    "FIT_METHOD",
    "OUTLIER_RESIDUAL_TOLERANCE",
    "MIN_TICKS_FOR_FIT",
    "MIN_DISTINCT_TICK_VALUES",
    "CONF_W_R2",
    "CONF_W_TICK_COVERAGE",
    "CONF_W_OCR_MEAN",
    "CONF_W_GRIDLINE_REG",
    "CALIB_DEBUG_LOG",
    "DEBUG_SAVE",
})


def __getattr__(name):
    """Resolve the user-tunable constants that live in app_settings, plus the
    one value derived from them. Python calls this only for names this module
    doesn't define itself, so unmanaged constants above are untouched and a
    real typo still raises AttributeError. Looking the value up per access
    (rather than copying it in at import) is what lets a Preferences edit
    apply without restarting the app."""
    if name in MANAGED:
        return getattr(_settings, name)
    # Scan depth for the unit-label band, three tick-bands deep.
    if name == "UNIT_SEARCH_BAND_DEPTH_PX":
        return _settings.TICK_LABEL_BAND_PX * 3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
