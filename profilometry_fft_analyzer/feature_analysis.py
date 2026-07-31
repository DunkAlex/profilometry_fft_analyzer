"""
feature_analysis.py
-------------------
Standalone measurement engine for calibrated profilometry traces.
Everything here operates on physical-unit arrays (canonically micrometers)
with uniform sample spacing — no image/OCR/GUI dependencies, so the whole
module is unit-testable with synthetic profiles (see test_round5.py).

Provides:
  - unit conversion helpers (pm/nm/um/mm), ASCII unit names per project
    convention
  - profile leveling by fitting the baseline through low-state segments
    (ISO 5436-1 spirit: a linear detrend through a mesa profile tilts the
    baseline; fitting only the base regions doesn't)
  - ISO 4288 cutoff (lambda_c) auto-selection, iterated on measured Ra
    (or picked from RSm for periodic profiles), capped so at least ~5
    sampling lengths fit the available profile
  - ISO 16610-21 Gaussian waviness/roughness split + Ra/Rq/Rz
  - feature segmentation (Otsu two-level threshold), square/sine
    classification, per-feature step heights (central-portion medians,
    ISO 5436-1 style) and plateau dishing (peak-to-line depth: max deviation
    from the chord between the plateau's two rim peaks, searched over the
    central half of the span between them)
  - extraction ("trend") confidence scoring for the results CSV
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage


# ---------------- Tunables ----------------

# The numbers an engineer might want to change now live in app_settings.py so
# the Preferences dialog can edit and persist them. They're read as
# `settings.NAME` at the point of use, which means a change applies to the
# next run without restarting the app. The notes below explain what each one
# actually does to a measurement — worth keeping here next to the code that
# uses them, rather than only in a tooltip.
#
# Feature-scale smoothing
#   FEATURE_SMOOTH_LENGTH_FRAC, FEATURE_SMOOTH_MIN_SAMPLES
#   Cutoff = max(profile_length * FRAC, MIN_SAMPLES * dx). Knocks down
#   extraction noise before segmentation and dishing without flattening the
#   feature itself — the committed examples are ~60 um features on ~100 um
#   profiles, where L/50 = 2 um leaves plateau curvature intact.
#
# Feature/noise gate
#   NOISE_GATE_K — the two-level split has to beat this multiple of the
#   extraction noise Rq to count as real features; under it the profile is
#   classified 'none'.
#
# Square vs sine classification, per feature. Square requires BOTH
#   transition_frac < SQUARE_MAX_TRANSITION_FRAC  (summed 10-90% rise/fall
#     width over feature width — i.e. how sharp the edges are), and
#   dwell_frac >= SQUARE_MIN_DWELL_FRAC  (fraction of samples sitting within
#     DWELL_BAND_FRAC_OF_HEIGHT of the feature's median level — i.e. how long
#     it dwells at the plateau).
#
# Step heights
#   STEP_CENTRAL_FRAC — heights come from this central slice of each segment,
#   keeping sloped edges out of the number. ISO 5436-1 says central third;
#   0.60 is used here because these profiles are short.
#
# Dishing on square ridges — the peak-to-line method
#   Isolate the flank-free plateau top, take its two rim peaks (highest point
#   on each half), draw the chord between them, then search the central
#   DISHING_CENTER_SEARCH_FRAC of that span for the point of MAXIMUM
#   DEVIATION from the chord — not the raw lowest point.
#
#   That distinction matters whenever the chord is tilted: the lowest
#   absolute point and the point that has sagged furthest below the trendline
#   can be two different places, and dishing means the latter. Searching only
#   the middle avoids the near-zero-deviation region beside each peak, and
#   taking the max (rather than a fit or an average) reports the full depth
#   of a sharply-bottomed sag instead of under-reading it.
#
#   DISHING_PLATEAU_TOP_FRAC brackets the flat top by height fraction of the
#   base->top rise, so sloped flanks drop out while a dished centre stays in.
#   DISHING_NOISE_K is why a flat top reads 0 rather than a few nm of noise:
#   even a perfectly flat but noisy top has a peak-to-valley span of roughly
#   1.3x Rq on the smoothed curve, so the floor has to clear that first.
#   DISHING_MIN_PLATEAU_SAMPLES skips plateaus too narrow to measure.
#
#   Reported `dishing` is never negative — the floor clamp guarantees it
#   regardless of what the search finds. `dishing_raw` is the unfloored
#   deviation and CAN come out slightly negative (noise, or a top that bulges
#   above the chord inside the search window); it's kept as a diagnostic, and
#   a noticeably negative value is worth a manual look.
#
# Roughness
#   LAMBDA_C_MAX_PROFILE_FRAC caps the cutoff so at least ~5 sampling lengths
#   fit, matching the standard's 5-sampling-length evaluation.
#   ROUGHNESS_BASELINE_MIN_SAMPLES is the point below which the baseline
#   slice is too short to say anything about texture, so the whole-profile
#   fallback is used instead.
#
# Trend confidence
#   TREND_CONF_W_* are the weights (should sum to 1) and the *_FRAC values
#   are the normalizers — see trend_confidence for how they combine.
import app_settings as settings

# Not user-tunable: the ISO 4288 cutoff series and its selection tables are
# fixed by the standard, so they stay here as plain constants.
ISO_CUTOFFS_UM = (80.0, 250.0, 800.0, 2500.0, 8000.0)
# Non-periodic profiles, by Ra (um): bounds are the table's Ra upper limits.
_ISO_RA_BOUNDS_UM = (0.02, 0.1, 2.0, 10.0)
# Periodic profiles, by RSm (um): bounds are the table's RSm upper limits.
_ISO_RSM_BOUNDS_UM = (40.0, 130.0, 400.0, 1300.0)


# ---------------- Unit conversion ----------------

UM_PER_UNIT = {
    "pm": 1e-6,
    "nm": 1e-3,
    "um": 1.0,
    "mm": 1e3,
}


def unit_factor(from_unit: str, to_unit: str) -> float:
    """Multiplicative factor converting lengths in `from_unit` to `to_unit`.
    Units are the pipeline's canonical ASCII strings (pm/nm/um/mm).
    Raises ValueError on an unknown unit — callers must not guess scales."""
    try:
        f = UM_PER_UNIT[str(from_unit).strip().lower()]
        t = UM_PER_UNIT[str(to_unit).strip().lower()]
    except KeyError as e:
        raise ValueError(f"unknown length unit: {e.args[0]!r} "
                         f"(known: {sorted(UM_PER_UNIT)})") from None
    return f / t


def convert_length(values, from_unit: str, to_unit: str):
    """Convert scalar or array lengths between units. NaNs pass through."""
    factor = unit_factor(from_unit, to_unit)
    if np.isscalar(values) or values is None:
        return values if values is None else float(values) * factor
    return np.asarray(values, dtype=float) * factor


# ---------------- Gaussian filtering (ISO 16610-21) ----------------

def gaussian_sigma_samples(cutoff: float, dx: float) -> float:
    """
    Sigma (in samples) for scipy's gaussian_filter1d such that amplitude
    transmission at wavelength `cutoff` is 50%, per ISO 16610-21:
    the standard's weighting function s(x) = 1/(a*lc) * exp(-pi*(x/(a*lc))^2)
    with a = sqrt(ln2/pi) = 0.4697 has standard deviation a*lc/sqrt(2*pi).
    """
    alpha = 0.4697
    return (alpha * cutoff / np.sqrt(2.0 * np.pi)) / dx


def gaussian_lowpass(y: np.ndarray, cutoff: float, dx: float,
                     mode: str = "reflect") -> np.ndarray:
    """Low-pass (waviness) filter with 50% transmission at `cutoff`."""
    sigma = gaussian_sigma_samples(cutoff, dx)
    return ndimage.gaussian_filter1d(np.asarray(y, dtype=float),
                                     sigma=max(sigma, 1e-9), mode=mode)


# ---------------- Otsu threshold ----------------

def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """1D Otsu threshold: maximizes between-class variance of a histogram."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin <= 0:
        return vmin
    hist, edges = np.histogram(v, bins=bins, range=(vmin, vmax))
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2.0

    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    total_mean = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (total_mean - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centers[int(np.argmax(between))])


# ---------------- Leveling ----------------

def level_profile(y: np.ndarray, dx: float) -> tuple[np.ndarray, dict]:
    """
    Remove the baseline tilt by least-squares fitting a line through the
    BASELINE-state samples only (identified via Otsu on a feature-smoothed
    copy, with the baseline state itself chosen by `_baseline_state` — the
    state shared by the profile's first/last samples, or the majority state
    when they disagree), then subtracting it and shifting so the baseline
    sits near zero.

    Baseline is NOT assumed to be the low state: a trench array (mostly-high
    surface with cut trenches) has its baseline in the HIGH state, and
    fitting through the low (trench) samples there would tilt the result
    exactly the way a plain linear detrend does for a mesa. A plain linear
    detrend through a mesa/trench profile splits the step between both
    levels and tilts the baseline; fitting only the base regions is the
    ISO 5436-1-style leveling for step samples. When no meaningful two-level
    split exists (flat or pure-roughness profiles), this degrades gracefully
    to a standard linear detrend (all samples weighted).

    Returns (y_leveled, info) with info keys:
        slope, intercept  : the removed line (units of y per unit of x-index*dx)
        baseline_fraction : fraction of samples treated as baseline
        mode              : 'baseline_segments' or 'linear_detrend'
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(n) * dx
    if n < 8:
        return y - np.mean(y), {"slope": 0.0, "intercept": float(np.mean(y)),
                                "baseline_fraction": 1.0, "mode": "linear_detrend"}

    smooth_cutoff = max(n * dx * settings.FEATURE_SMOOTH_LENGTH_FRAC,
                        settings.FEATURE_SMOOTH_MIN_SAMPLES * dx)
    y_f = gaussian_lowpass(y, smooth_cutoff, dx)

    t = otsu_threshold(y_f)
    high = y_f > t
    baseline_is_high = _baseline_state(high, _runs(high))
    baseline = high if baseline_is_high else ~high
    baseline_frac = float(np.mean(baseline))

    # Guard: the baseline set must be a real baseline, not a degenerate
    # sliver, and the split must be larger than the local noise to mean
    # anything.
    noise = estimate_noise_rq(y, y_f)
    split = (float(np.median(y_f[~baseline]) - np.median(y_f[baseline]))
             if 0 < baseline.sum() < n else 0.0)
    if baseline.sum() < max(4, int(0.05 * n)) or abs(split) < settings.NOISE_GATE_K * noise:
        coeff = np.polyfit(x, y, 1)
        mode = "linear_detrend"
        fit_line = np.polyval(coeff, x)
        leveled = y - fit_line
        return leveled, {"slope": float(coeff[0]), "intercept": float(coeff[1]),
                         "baseline_fraction": 1.0, "mode": mode}

    coeff = np.polyfit(x[baseline], y[baseline], 1)
    fit_line = np.polyval(coeff, x)
    leveled = y - fit_line
    return leveled, {"slope": float(coeff[0]), "intercept": float(coeff[1]),
                     "baseline_fraction": baseline_frac, "mode": "baseline_segments"}


def estimate_noise_rq(y: np.ndarray, y_smooth: np.ndarray) -> float:
    """
    Robust RMS of the extraction noise: 1.4826 * MAD of (y - y_smooth).
    MAD (vs plain std) keeps the estimate honest when the residual contains
    a few large transition-edge spikes.
    """
    resid = np.asarray(y, dtype=float) - np.asarray(y_smooth, dtype=float)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return 1.4826 * mad


# ---------------- ISO 4288 lambda_c selection ----------------

def _iso_lambda_c_from_ra(ra_um: float) -> float:
    """ISO 4288 table 1 (non-periodic profiles): cutoff from measured Ra."""
    for bound, cutoff in zip(_ISO_RA_BOUNDS_UM, ISO_CUTOFFS_UM):
        if ra_um <= bound:
            return cutoff
    return ISO_CUTOFFS_UM[-1]


def _iso_lambda_c_from_rsm(rsm_um: float) -> float:
    """ISO 4288 table 3 (periodic profiles): cutoff from mean spacing RSm."""
    for bound, cutoff in zip(_ISO_RSM_BOUNDS_UM, ISO_CUTOFFS_UM):
        if rsm_um <= bound:
            return cutoff
    return ISO_CUTOFFS_UM[-1]


def select_lambda_c(y_leveled_um: np.ndarray,
                    dx_um: float,
                    rsm_um: Optional[float] = None,
                    override_um: Optional[float] = None,
                    max_iter: int = 5) -> dict:
    """
    Choose the roughness cutoff lambda_c (um).

    Order of precedence:
      1. `override_um` — explicit fixed cutoff (config), used as-is.
      2. Periodic rule: if `rsm_um` is given (profile recognized as a
         periodic feature array), pick from the ISO 4288 RSm table.
      3. Non-periodic rule: iterate the ISO 4288 Ra procedure — filter with
         a guess, measure Ra of the residual, look up the table cutoff,
         repeat until stable.

    In all non-override cases the result is capped at
    LAMBDA_C_MAX_PROFILE_FRAC * profile_length so at least ~5 sampling
    lengths fit; these microscope exports (~100 um) are far shorter than the
    standard's preferred evaluation lengths, and the cap is reported via
    'capped'=True + the uncapped 'iso_recommended_um' so the CSV/log can
    flag it.

    Returns dict: lambda_c_um, capped, iso_recommended_um, rule.
    """
    y = np.asarray(y_leveled_um, dtype=float)
    profile_len = len(y) * dx_um
    cap = profile_len * settings.LAMBDA_C_MAX_PROFILE_FRAC

    if override_um is not None:
        return {"lambda_c_um": float(override_um), "capped": False,
                "iso_recommended_um": float(override_um), "rule": "override"}

    if rsm_um is not None and np.isfinite(rsm_um) and rsm_um > 0:
        iso = _iso_lambda_c_from_rsm(rsm_um)
        lam = min(iso, cap)
        lam = max(lam, 4 * dx_um)
        return {"lambda_c_um": float(lam), "capped": iso > cap,
                "iso_recommended_um": float(iso), "rule": "periodic_rsm"}

    # Non-periodic: iterate Ra -> cutoff until stable.
    lam = min(ISO_CUTOFFS_UM[2], cap)  # 0.8 mm is the standard default guess
    lam = max(lam, 4 * dx_um)
    iso = lam
    for _ in range(max_iter):
        waviness = gaussian_lowpass(y, lam, dx_um)
        ra = float(np.mean(np.abs(y - waviness)))
        iso = _iso_lambda_c_from_ra(ra)
        new = max(min(iso, cap), 4 * dx_um)
        if abs(new - lam) < 1e-9:
            break
        lam = new
    return {"lambda_c_um": float(lam), "capped": iso > cap,
            "iso_recommended_um": float(iso), "rule": "nonperiodic_ra"}


# ---------------- Roughness parameters ----------------

def roughness_params(y_leveled: np.ndarray, dx: float, lambda_c: float,
                     mask: Optional[np.ndarray] = None) -> dict:
    """
    ISO waviness/roughness split and amplitude parameters.
    Returns dict: waviness, roughness, Ra, Rq, Rz, rz_n_chunks.
    Rz: mean peak-to-valley over up to 5 chunks of one sampling length
    (= lambda_c) each; if fewer fit, the available chunks are used.

    `mask`, when given, restricts the computation to those samples only
    (e.g. the baseline/reference-surface samples between features, so a
    scan's step heights don't dominate the "roughness" numbers) — the
    selected samples are concatenated into one 1D array before filtering,
    so a mask that excludes several separate baseline stretches introduces
    a seam at each junction; acceptable for a short-profile surface-texture
    estimate, not appropriate for a rigorous ISO evaluation length.
    """
    y = np.asarray(y_leveled, dtype=float)
    if mask is not None:
        y = y[np.asarray(mask, dtype=bool)]
    waviness = gaussian_lowpass(y, lambda_c, dx)
    roughness = y - waviness
    Ra = float(np.mean(np.abs(roughness)))
    Rq = float(np.sqrt(np.mean(roughness ** 2)))

    chunk = max(2, int(round(lambda_c / dx)))
    n_chunks = min(5, max(1, len(roughness) // chunk))
    trimmed = roughness[: n_chunks * chunk] if len(roughness) >= chunk else roughness
    chunks = np.array_split(trimmed, n_chunks)
    Rz = float(np.mean([c.max() - c.min() for c in chunks]))

    return {"waviness": waviness, "roughness": roughness,
            "Ra": Ra, "Rq": Rq, "Rz": Rz, "rz_n_chunks": int(n_chunks)}


# ---------------- Feature segmentation & measurement ----------------

@dataclass
class Feature:
    """One measured feature (interior segment of the two-level split)."""
    kind: str                 # 'ridge' | 'trench' | 'step'
    shape: str                # 'square' | 'sine'
    i0: int                   # start sample index (inclusive)
    i1: int                   # end sample index (exclusive)
    height: float             # feature height (input y-units)
    top_level: float          # representative feature level
    base_level: float         # representative surrounding level
    center_x: float           # x of feature center (input x-units)
    transition_frac: float
    dwell_frac: float
    # Physical width (50%-height crossing to 50%-height crossing; NaN for
    # the single-edge 'step' pseudo-feature, which has no second boundary)
    width: float = float("nan")
    # The two 50%-height crossings themselves (input x-units), so the width
    # can be drawn as a horizontal overlay bar spanning the actual feature
    # rather than just centered on center_x. NaN alongside width.
    x_left: float = float("nan")
    x_right: float = float("nan")
    # Dishing (square ridges only; NaN otherwise). `dishing` is floored at 0
    # (physically non-negative); `dishing_raw` keeps the signed peak-to-line
    # measurement (negative only in the degenerate case of an inverted chord)
    # as a diagnostic.
    dishing: float = float("nan")
    dishing_raw: float = float("nan")
    shoulder_level: float = float("nan")
    dish_min_level: float = float("nan")
    dish_x: float = float("nan")


@dataclass
class FeatureAnalysis:
    feature_type: str          # image-level: 'square' | 'sine' | 'none'
    features: list = field(default_factory=list)
    n_features: int = 0
    height_avg: float = float("nan")
    height_min: float = float("nan")
    height_max: float = float("nan")
    height_median: float = float("nan")
    width_avg: float = float("nan")
    width_min: float = float("nan")
    width_max: float = float("nan")
    width_median: float = float("nan")
    dishing_avg: float = float("nan")
    dishing_max: float = float("nan")
    # Most-negative per-feature raw (unfloored) dishing — a diagnostic for
    # spotting questionable measurements (image level). The reported/floored
    # `dishing` is always >= 0; `dishing_raw` is usually ~0/positive too but,
    # unlike `dishing`, is not guaranteed so (see _peak_line_dishing) — a
    # noticeably negative value here is worth a manual look.
    dishing_raw_min: float = float("nan")
    rsm: Optional[float] = None      # mean spacing between feature centers
    threshold: float = float("nan")
    noise_rq: float = float("nan")
    y_smooth: Optional[np.ndarray] = None
    baseline_mask: Optional[np.ndarray] = None  # True = reference-surface sample


def _central_slice(i0: int, i1: int, frac: Optional[float] = None) -> slice:
    """Takes a sample range and returns the slice covering its middle `frac`,
    always at least one sample wide. `frac` of None means the current
    STEP_CENTRAL_FRAC — read here rather than as a default argument, since a
    default is evaluated once at import and would ignore a Preferences edit.
    Used to keep sloped edges out of level/height measurements."""
    if frac is None:
        frac = settings.STEP_CENTRAL_FRAC
    m = i1 - i0
    trim = int(round(m * (1.0 - frac) / 2.0))
    lo, hi = i0 + trim, i1 - trim
    if hi <= lo:
        mid = (i0 + i1) // 2
        lo, hi = mid, mid + 1
    return slice(lo, hi)


def _plateau_top_bounds(y_f: np.ndarray, i0: int, i1: int,
                        base_level: float, top_level: float) -> tuple[int, int]:
    """
    The plateau top within [i0, i1): bracketed by the OUTERMOST samples whose
    smoothed height is at/above DISHING_PLATEAU_TOP_FRAC of the base->top rise.
    This strips the sloped flanks (below the threshold, outside the brackets)
    while keeping a dished/sagged center in between (which may itself dip
    below the threshold on a deeply dished feature). Returns (p0, p1), p1
    exclusive; falls back to (i0, i1) if nothing clears the threshold
    (degenerate feature).
    """
    rise = top_level - base_level
    if rise <= 0:
        return i0, i1
    t_top = base_level + settings.DISHING_PLATEAU_TOP_FRAC * rise
    above = np.where(y_f[i0:i1] >= t_top)[0]
    if above.size == 0:
        return i0, i1
    return i0 + int(above[0]), i0 + int(above[-1]) + 1


def _peak_line_dishing(y_f: np.ndarray, x: np.ndarray, i0: int, i1: int,
                       base_level: float, top_level: float,
                       noise_rq: float) -> tuple[float, float, float, float, float]:
    """
    Dishing via the peak-to-line method: on the isolated (flank-free) plateau
    top, find the two RIM PEAKS (the highest sample on each half of the
    plateau), draw the straight chord (trendline) between them, then search
    the central DISHING_CENTER_SEARCH_FRAC of the span BETWEEN the two peaks
    for the point of MAXIMUM DEVIATION from that chord — chord(x) - y(x),
    maximized, not the raw lowest y. This matters whenever the chord itself
    is tilted: the lowest absolute point and the point that has sagged
    furthest below the (tilted) trendline are not necessarily the same point,
    and dishing means the latter. Restricting the search to the center half
    (rather than the whole peak-to-peak span) avoids the near-zero-deviation
    region right next to either peak.

    The REPORTED `dishing` is non-negative unconditionally — that guarantee
    comes from the floor clamp at the end (raw if raw > floor else 0.0, with
    floor >= 0), independent of the search itself. `dishing_raw` is the
    unfloored deviation and, unlike the old plain-interior-minimum approach,
    is not provably bounded below by 0 in every edge case (e.g. noise, or a
    top that bulges above the chord within the search window) — it can come
    out slightly negative, which is fine: it only makes the floor clamp to 0
    either way. Tilt-absorbing for realistic tilt (a purely tilted, undished
    segment is exactly reproduced by its own chord, so deviation is ~0
    everywhere between the peaks, not just at the search point). Reports the
    FULL sag depth, unlike an average/fit-based estimate which under-
    approximates a sharply-bottomed dish.

    Returns (dishing, dishing_raw, shoulder_level, dish_min_level, dish_x):
      - dishing        : floored at 0 and at the noise level
      - dishing_raw    : unfloored max-deviation-from-chord (see above)
      - shoulder_level : chord height at the max-deviation point (bar top)
      - dish_min_level : plateau height at the max-deviation point (bar base)
      - dish_x         : x of the max-deviation point (overlay marker column)
    All NaN if the isolated plateau is too short to measure.
    """
    nan = float("nan")
    p0, p1 = _plateau_top_bounds(y_f, i0, i1, base_level, top_level)
    m = p1 - p0
    if m < settings.DISHING_MIN_PLATEAU_SAMPLES:
        return nan, nan, nan, nan, nan

    # Rim peaks: highest sample on the left half and on the right half of the
    # isolated plateau (so each peak is found on "its own" side, even if the
    # global max happens to sit near the middle of a mostly-flat top).
    mid = (p0 + p1) // 2
    if mid <= p0 or p1 <= mid:
        return nan, nan, nan, nan, nan
    iL = p0 + int(np.argmax(y_f[p0:mid]))
    iR = mid + int(np.argmax(y_f[mid:p1]))
    if iR - iL < 2:
        # Peaks adjacent (or misordered on a very short/degenerate plateau):
        # no interior to measure.
        lvl = 0.5 * (float(y_f[iL]) + float(y_f[iR]))
        return 0.0, 0.0, lvl, lvl, float(x[(p0 + p1) // 2])

    xL, xR = float(x[iL]), float(x[iR])
    L, R = float(y_f[iL]), float(y_f[iR])
    denom = xR - xL   # > 0: x is strictly increasing and iR - iL >= 2 here

    # Search the CENTRAL DISHING_CENTER_SEARCH_FRAC of the span between the
    # two peaks for the point of maximum deviation from the chord (not the
    # raw minimum height) — see docstring above.
    csl = _central_slice(iL, iR + 1, settings.DISHING_CENTER_SEARCH_FRAC)
    xs = x[csl]
    ys = y_f[csl]
    chord = L + (R - L) * (xs - xL) / denom
    dev = chord - ys
    k = int(np.argmax(dev))
    im = csl.start + k

    dish_min_level = float(ys[k])
    shoulder_level = float(chord[k])
    raw = float(dev[k])
    dish_x = float(x[im])

    floor = settings.DISHING_NOISE_K * max(noise_rq, 0.0)
    dishing = raw if raw > floor else 0.0
    return dishing, raw, shoulder_level, dish_min_level, dish_x


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    """Contiguous runs of a boolean mask as (state, i0, i1) with i1 exclusive."""
    runs = []
    n = len(mask)
    if n == 0:
        return runs
    start = 0
    for i in range(1, n):
        if mask[i] != mask[start]:
            runs.append((bool(mask[start]), start, i))
            start = i
    runs.append((bool(mask[start]), start, n))
    return runs


def _baseline_state(mask: np.ndarray, runs: list[tuple[bool, int, int]]) -> bool:
    """
    Which of the two run states is the baseline (reference surface) rather
    than the feature (ridge/trench): the state shared by the profile's first
    and last run when they agree — the profile starts and ends on baseline,
    the common case for a feature array — else whichever state holds the
    majority of samples. Returns the boolean run-state value (matching
    `mask`'s True/False) that represents baseline.

    Baseline is deliberately NOT assumed to be the low state: a trench array
    (mostly-high surface with narrow cut trenches) has baseline in the HIGH
    state, and a ridge array (mostly-low surface with narrow raised ridges)
    has it in the LOW state — either is common in real feature-array scans.
    """
    if not runs:
        return True
    if runs[0][0] == runs[-1][0]:
        return runs[0][0]
    n_true = int(np.count_nonzero(mask))
    return n_true >= (len(mask) - n_true)


def _transition_width(y_f: np.ndarray, x: np.ndarray,
                      lo_idx: int, hi_idx: int,
                      base_level: float, top_level: float) -> float:
    """
    10%-90% transition width between two sample indices (window endpoints
    at the centers of the adjacent segments). Direction-agnostic: works for
    rising and falling edges, ridges and trenches. Falls back to the window
    width if the crossings can't be located (very degraded edges).
    """
    a, b = (lo_idx, hi_idx) if lo_idx <= hi_idx else (hi_idx, lo_idx)
    if b - a < 2:
        return abs(x[min(b, len(x) - 1)] - x[a])
    seg = y_f[a:b + 1]
    h = top_level - base_level
    if abs(h) < 1e-12:
        return abs(x[b] - x[a])
    # Normalized crossing positions (fraction through the transition);
    # direction-agnostic since frac runs 0 -> 1 whichever way the edge goes.
    frac = (seg - base_level) / h
    idx10 = np.where(np.diff(np.sign(frac - 0.10)) != 0)[0]
    idx90 = np.where(np.diff(np.sign(frac - 0.90)) != 0)[0]
    if len(idx10) == 0 or len(idx90) == 0:
        return abs(x[b] - x[a])
    return abs(x[a + int(idx90[0])] - x[a + int(idx10[0])])


def _half_height_crossing_x(y_f: np.ndarray, x: np.ndarray,
                            lo_idx: int, hi_idx: int,
                            level_lo: float, level_hi: float,
                            frac: float = 0.5) -> Optional[float]:
    """
    x-coordinate where the smoothed curve crosses `frac` of the way from
    level_lo to level_hi, within [lo_idx, hi_idx] (direction-agnostic: works
    for rising or falling transitions), linearly interpolated between the
    bracketing samples. Returns None if no crossing is found (degenerate/
    flat transition) — callers fall back to a raw index-based width.

    Used for feature width: the 50%-height crossing on each side of a
    feature is a shape-agnostic boundary definition — it works the same way
    for a sharp square edge (where it lands near the plateau's true edge)
    and a sloped/asymmetric sine flank (where a hard Otsu-run boundary would
    be arbitrary).
    """
    a, b = (lo_idx, hi_idx) if lo_idx <= hi_idx else (hi_idx, lo_idx)
    if b - a < 1:
        return None
    seg = y_f[a:b + 1]
    h = level_hi - level_lo
    if abs(h) < 1e-12:
        return None
    frac_arr = (seg - level_lo) / h
    idx = np.where(np.diff(np.sign(frac_arr - frac)) != 0)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    f0, f1 = frac_arr[i], frac_arr[i + 1]
    t = (frac - f0) / (f1 - f0) if abs(f1 - f0) > 1e-12 else 0.0
    x0, x1 = x[a + i], x[a + i + 1]
    return float(x0 + t * (x1 - x0))


def analyze_features(y_leveled: np.ndarray, dx: float) -> FeatureAnalysis:
    """
    Segment a leveled profile into two-level features and measure them.

    Pipeline:
      1. Feature-scale smoothing (noise suppression only).
      2. Otsu threshold -> high/low state per sample -> contiguous runs.
      3. Noise gate: median(high) - median(low) must exceed
         NOISE_GATE_K * noise Rq, else feature_type='none'.
      4. Features = interior runs (bounded by a transition on both sides).
         With exactly two runs (a single edge), one 'step' pseudo-feature is
         reported so a lone step height is still measured.
      5. Per feature: square/sine classification (transition sharpness +
         plateau dwell), then height —
           square: |median(central 60% of feature) - mean of adjacent
                   segments' central medians|          (ISO 5436-1 style)
           sine:   |feature extremum - mean of adjacent segments' opposing
                   extrema|                             (full peak-to-valley)
      6. Dishing for square RIDGES: the peak-to-line depth on the isolated
         (flank-free) plateau top of the smoothed curve — the point of MAXIMUM
         DEVIATION from the chord between the plateau's two rim peaks (not the
         raw lowest point), searched over the central half of the span between
         them. Non-negative by construction (dishing is physically non-
         negative); floored at 0 and at the noise level. See
         _peak_line_dishing.

    All x/y values keep the caller's units (batch pipeline passes um).
    """
    y = np.asarray(y_leveled, dtype=float)
    n = len(y)
    result = FeatureAnalysis(feature_type="none")
    if n < 16:
        return result

    x = np.arange(n) * dx
    smooth_cutoff = max(n * dx * settings.FEATURE_SMOOTH_LENGTH_FRAC,
                        settings.FEATURE_SMOOTH_MIN_SAMPLES * dx)
    y_f = gaussian_lowpass(y, smooth_cutoff, dx)
    result.y_smooth = y_f

    noise_rq = estimate_noise_rq(y, y_f)
    result.noise_rq = noise_rq

    t = otsu_threshold(y_f)
    result.threshold = t
    high = y_f > t
    if high.all() or (~high).all():
        return result

    lo_med = float(np.median(y_f[~high]))
    hi_med = float(np.median(y_f[high]))
    if (hi_med - lo_med) < settings.NOISE_GATE_K * max(noise_rq, 1e-12):
        return result

    runs = _runs(high)
    if len(runs) < 2:
        return result

    # Which state is the reference surface vs. the feature (ridge/trench) —
    # see _baseline_state. Interior runs matching this state are gaps/base
    # between features, not features themselves, and must not be measured
    # as pseudo-features (they previously were, inflating n_features and
    # placing height markers on flat ground).
    baseline_state = _baseline_state(high, runs)
    result.baseline_mask = high if baseline_state else ~high

    # Representative level of each run: median of its central portion.
    run_levels = [float(np.median(y_f[_central_slice(i0, i1)]))
                  for (_s, i0, i1) in runs]

    features: list[Feature] = []

    if len(runs) == 2:
        # Single edge: report one 'step' pseudo-feature at the boundary.
        (_s0, a0, a1), (_s1, b0, b1) = runs
        base_level, top_level = sorted([run_levels[0], run_levels[1]])
        height = top_level - base_level
        w = _transition_width(y_f, x, (a0 + a1) // 2, (b0 + b1) // 2,
                              run_levels[0], run_levels[1])
        span = max(x[-1] - x[0], dx)
        tf = w / span
        shape = "square" if tf < settings.SQUARE_MAX_TRANSITION_FRAC else "sine"
        features.append(Feature(
            kind="step", shape=shape, i0=a1, i1=a1,
            height=abs(height), top_level=top_level, base_level=base_level,
            center_x=float(x[a1]), transition_frac=tf, dwell_frac=1.0,
        ))
    else:
        for k in range(1, len(runs) - 1):
            state, i0, i1 = runs[k]
            if state == baseline_state:
                # Baseline/gap run between features — not a feature itself.
                continue
            kind = "ridge" if state else "trench"
            feat_level = run_levels[k]
            left_level = run_levels[k - 1]
            right_level = run_levels[k + 1]
            base_level = (left_level + right_level) / 2.0
            h_est = abs(feat_level - base_level)
            if h_est <= 0:
                continue

            # -- classification --
            lc = (runs[k - 1][1] + runs[k - 1][2]) // 2
            rc = (runs[k + 1][1] + runs[k + 1][2]) // 2
            w_left = _transition_width(y_f, x, lc, (i0 + i1) // 2,
                                       left_level, feat_level)
            w_right = _transition_width(y_f, x, (i0 + i1) // 2, rc,
                                        feat_level, right_level)
            feat_width = max((i1 - i0) * dx, dx)
            tf = (w_left + w_right) / feat_width
            dwell_band = settings.DWELL_BAND_FRAC_OF_HEIGHT * h_est
            dwell = float(np.mean(np.abs(y_f[i0:i1] - feat_level) <= dwell_band))
            shape = ("square"
                     if (tf < settings.SQUARE_MAX_TRANSITION_FRAC
                         and dwell >= settings.SQUARE_MIN_DWELL_FRAC)
                     else "sine")

            # -- height --
            if shape == "square":
                height = h_est
                top_level, base_lv = feat_level, base_level
            else:
                # sine: full peak-to-valley against neighboring opposing extrema
                if state:  # ridge
                    peak = float(np.max(y_f[i0:i1]))
                    neigh = np.concatenate([y_f[runs[k - 1][1]:runs[k - 1][2]],
                                            y_f[runs[k + 1][1]:runs[k + 1][2]]])
                    base_lv = float(np.min(neigh))
                    height = peak - base_lv
                    top_level = peak
                else:      # trench
                    valley = float(np.min(y_f[i0:i1]))
                    neigh = np.concatenate([y_f[runs[k - 1][1]:runs[k - 1][2]],
                                            y_f[runs[k + 1][1]:runs[k + 1][2]]])
                    base_lv = float(np.max(neigh))
                    height = base_lv - valley
                    top_level = valley

            feat = Feature(
                kind=kind, shape=shape, i0=i0, i1=i1,
                height=abs(float(height)),
                top_level=float(top_level), base_level=float(base_lv),
                center_x=float(x[(i0 + i1) // 2]),
                transition_frac=float(tf), dwell_frac=float(dwell),
            )

            # -- width: 50%-height crossing to 50%-height crossing, same
            # definition for square and sine so the two shapes are
            # comparable; falls back to the raw run span if a crossing
            # can't be located (degenerate edge). --
            x_left = _half_height_crossing_x(y_f, x, lc, (i0 + i1) // 2,
                                             left_level, feat_level)
            x_right = _half_height_crossing_x(y_f, x, (i0 + i1) // 2, rc,
                                              feat_level, right_level)
            if x_left is not None and x_right is not None:
                feat.width = abs(x_right - x_left)
                feat.x_left, feat.x_right = min(x_left, x_right), max(x_left, x_right)
            else:
                feat.width = (i1 - i0) * dx
                feat.x_left = float(x[i0])
                feat.x_right = float(x[min(i1, n - 1)])

            # -- dishing (square ridges only): peak-to-line depth, tilt-
            # immune and non-negative by construction. See _peak_line_dishing
            # and the DISHING_* constants. --
            if shape == "square" and kind == "ridge":
                (feat.dishing, feat.dishing_raw, feat.shoulder_level,
                 feat.dish_min_level, feat.dish_x) = _peak_line_dishing(
                    y_f, x, i0, i1, base_level, top_level, noise_rq)

            features.append(feat)

    if not features:
        return result

    result.features = features
    result.n_features = len(features)

    heights = np.array([f.height for f in features], dtype=float)
    result.height_avg = float(np.mean(heights))
    result.height_min = float(np.min(heights))
    result.height_max = float(np.max(heights))
    result.height_median = float(np.median(heights))

    widths = np.array([f.width for f in features
                       if np.isfinite(f.width)], dtype=float)
    if widths.size:
        result.width_avg = float(np.mean(widths))
        result.width_min = float(np.min(widths))
        result.width_max = float(np.max(widths))
        result.width_median = float(np.median(widths))

    dishings = np.array([f.dishing for f in features
                         if np.isfinite(f.dishing)], dtype=float)
    if dishings.size:
        # f.dishing is floored at 0, so these are non-negative by construction.
        result.dishing_avg = float(np.mean(dishings))
        result.dishing_max = float(np.max(dishings))
    raws = np.array([f.dishing_raw for f in features
                     if np.isfinite(f.dishing_raw)], dtype=float)
    if raws.size:
        result.dishing_raw_min = float(np.min(raws))

    # RSm: mean spacing between consecutive retained feature centers — all
    # retained features share the same state (ridge-only or trench-only, by
    # construction), so this is correct regardless of which state is baseline.
    centers = [f.center_x for f in features]
    if len(centers) >= 2:
        result.rsm = float(np.mean(np.diff(centers)))

    # Image-level type: majority vote over real features; ties favor square
    # (the user's stated priority for mixed feature sets).
    n_square = sum(1 for f in features if f.shape == "square")
    result.feature_type = "square" if n_square >= len(features) - n_square else "sine"
    return result


# ---------------- Full measurement pipeline ----------------

def run_measurement_pipeline(y_um: np.ndarray,
                             dx_um: float,
                             rows_crop=None,
                             ext_info: Optional[dict] = None,
                             lambda_c_override_um: Optional[float] = None,
                             periodic_min_features: int = 3,
                             roughness_on_baseline: bool = True) -> dict:
    """
    The complete measurement chain from a calibrated profile (um) to
    parameters: leveling -> feature segmentation/classification ->
    ISO 4288 lambda_c selection -> roughness params -> trend confidence.

    Shared by app_engine.analyze_one and the manual-fit GUI's auto
    re-fit, so the two paths can never drift apart.

    rows_crop/ext_info are optional — when absent (no extraction-quality
    info available), trend_conf comes back NaN.

    roughness_on_baseline: when True (default) and real features were
    found, Ra/Rq/Rz are measured over the baseline/reference-surface
    samples only (feats.baseline_mask) instead of the whole profile, so a
    scan's step heights don't dominate the "roughness" numbers on a feature
    array. Falls back to the whole profile when there aren't enough
    baseline samples to say anything, or when no features were found.

    Returns dict: y_lev, lev_info, feats (FeatureAnalysis), lc (dict from
    select_lambda_c), rough (dict from roughness_params), trend_conf.
    """
    y_lev, lev_info = level_profile(y_um, dx_um)
    feats = analyze_features(y_lev, dx_um)

    rsm = feats.rsm if feats.n_features >= periodic_min_features else None
    lc = select_lambda_c(y_lev, dx_um, rsm_um=rsm,
                         override_um=lambda_c_override_um)

    rough_mask = None
    if (roughness_on_baseline and feats.n_features > 0
            and feats.baseline_mask is not None
            and int(np.count_nonzero(feats.baseline_mask))
            >= settings.ROUGHNESS_BASELINE_MIN_SAMPLES):
        rough_mask = feats.baseline_mask
    rough = roughness_params(y_lev, dx_um, lc["lambda_c_um"], mask=rough_mask)

    if rows_crop is not None and ext_info:
        tconf = trend_confidence(rows_crop, ext_info)
    else:
        tconf = float("nan")

    return {"y_lev": y_lev, "lev_info": lev_info, "feats": feats,
            "lc": lc, "rough": rough, "trend_conf": tconf}


# ---------------- Trend (extraction) confidence ----------------

def trend_confidence(y_rows_crop: np.ndarray, extraction_info: dict) -> float:
    """
    Score extraction quality in [0, 1] from three components:
      coverage : fraction of columns with a real color match (not interpolated)
      gap      : penalty for the longest contiguous unmatched run
                 (>= TREND_CONF_GAP_ZERO_FRAC of the width zeroes the term)
      jump     : penalty for column-to-column jumps larger than
                 TREND_CONF_JUMP_HEIGHT_FRAC of the crop height (extraction
                 glitches — the trace teleporting between features)
    conf = 0.50*coverage + 0.25*gap + 0.25*jump
    """
    n_cols = max(1, int(extraction_info.get("n_columns", 1)))
    n_skipped = len(extraction_info.get("skipped_x", []))
    coverage = 1.0 - n_skipped / n_cols

    max_gap = float(extraction_info.get("max_gap_run", 0))
    gap_term = 1.0 - min(1.0, max_gap / max(1.0, settings.TREND_CONF_GAP_ZERO_FRAC * n_cols))

    crop_h = max(1.0, float(extraction_info.get("crop_height", 1)))
    y = np.asarray(y_rows_crop, dtype=float)
    if len(y) > 1:
        jumps = np.abs(np.diff(y))
        jump_frac = float(np.mean(jumps > settings.TREND_CONF_JUMP_HEIGHT_FRAC * crop_h))
    else:
        jump_frac = 1.0
    jump_term = 1.0 - min(1.0, jump_frac / settings.TREND_CONF_JUMP_ZERO_FRAC)

    conf = (settings.TREND_CONF_W_COVERAGE * coverage
            + settings.TREND_CONF_W_GAP * gap_term
            + settings.TREND_CONF_W_JUMP * jump_term)
    return float(max(0.0, min(1.0, conf)))
