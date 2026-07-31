"""
colors.py
---------
Color picking (patch-median), background inference, and boolean masks.

Convention: all functions expect images in RGB order (H, W, 3), uint8.
If you're loading with cv2.imread, convert with cv2.cvtColor(..., cv2.COLOR_BGR2RGB)
before calling into this module.
"""
from __future__ import annotations
import numpy as np

from . import config


def pick_color_at(image_rgb: np.ndarray, x: int, y: int,
                  patch: int | None = None,
                  exclude_rgb: tuple[int, int, int] | None = None,
                  exclude_tol: int | None = None
                  ) -> tuple[int, int, int]:
    """
    Median RGB of an NxN patch centered on (x, y). Robust to anti-aliased
    edge pixels compared with sampling a single pixel.

    When `exclude_rgb` is provided (typically the background color picked
    earlier), pixels within `exclude_tol` per channel of that color are
    dropped from the median computation. This prevents the surrounding
    background from swamping the signal when the user clicks on a thin
    feature (single-pixel gridline, axis line, etc.). If too few non-bg
    pixels remain, the patch expands progressively (9 -> 15 -> 25) before
    falling back to the raw pixel at (x, y).

    Parameters
    ----------
    x, y        : click coordinates in image space
    patch       : NxN sampling window; defaults to COLOR_PICK_PATCH_SIZE_MASKED
                  when exclude_rgb is given, else COLOR_PICK_PATCH_SIZE
    exclude_rgb : RGB triple to exclude from the median (typically background)
    exclude_tol : per-channel tolerance for exclusion
    """
    h, w = image_rgb.shape[:2]

    if patch is None:
        patch = (config.COLOR_PICK_PATCH_SIZE_MASKED
                 if exclude_rgb is not None
                 else config.COLOR_PICK_PATCH_SIZE)
    if exclude_tol is None:
        exclude_tol = config.COLOR_PICK_EXCLUDE_TOL

    def _grab(p):
        r = p // 2
        x0 = max(0, x - r); x1 = min(w, x + r + 1)
        y0 = max(0, y - r); y1 = min(h, y + r + 1)
        return image_rgb[y0:y1, x0:x1].reshape(-1, 3)

    # Simple case: no background exclusion
    if exclude_rgb is None:
        region = _grab(patch)
        if len(region) == 0:
            return tuple(int(v) for v in image_rgb[y, x])
        med = np.median(region, axis=0).astype(int)
        return int(med[0]), int(med[1]), int(med[2])

    # Background-aware: filter out bg pixels, expand patch if needed
    t = np.asarray(exclude_rgb, dtype=np.int16)
    for p in (patch, max(patch * 2 - 1, patch + 6), max(patch * 3 - 2, patch + 16)):
        region = _grab(p)
        if len(region) == 0:
            continue
        diff = np.abs(region.astype(np.int16) - t)
        keep = np.any(diff > exclude_tol, axis=-1)
        non_bg = region[keep]
        if len(non_bg) >= config.COLOR_PICK_MIN_NON_BG:
            med = np.median(non_bg, axis=0).astype(int)
            return int(med[0]), int(med[1]), int(med[2])

    # Fallback: single pixel at click point
    return tuple(int(v) for v in image_rgb[y, x])


def infer_background(image_rgb: np.ndarray,
                     quantize: int = config.BG_QUANTIZE_STEP
                     ) -> tuple[int, int, int]:
    """
    Mode of the (coarsely quantized) image. Anti-aliased near-background
    pixels vote for the true background this way.
    """
    q = (image_rgb.astype(np.int32) // quantize) * quantize
    flat = q.reshape(-1, 3)
    # unique + counts on rows
    view = np.ascontiguousarray(flat).view(
        np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
    )
    _, idx, counts = np.unique(view, return_index=True, return_counts=True)
    top = flat[idx[np.argmax(counts)]]
    return int(top[0]), int(top[1]), int(top[2])


def color_mask(image_rgb: np.ndarray,
               target_rgb: tuple[int, int, int],
               tol: int | None = None) -> np.ndarray:
    """Boolean mask: True where |pixel - target| <= tol in all channels.
    `tol` of None means the current config.COLOR_MASK_TOLERANCE — resolved
    here rather than as a default argument, which Python would freeze at
    import and so ignore any later Preferences change."""
    if tol is None:
        tol = config.COLOR_MASK_TOLERANCE
    t = np.asarray(target_rgb, dtype=np.int16)
    diff = np.abs(image_rgb.astype(np.int16) - t)
    return np.all(diff <= tol, axis=-1)


def validate_profile_colors(colors: dict[str, tuple[int, int, int]],
                            min_separation: int | None = None
                            ) -> list[str]:
    """
    Check that background/plot/axis/gridline colors are all distinguishable
    from each other, EXCEPT axis vs gridline, which many plotting programs
    legitimately style identically (the bounding frame drawn in the same
    color as interior gridlines) — that pair is handled by
    gridlines.detect_plot_region_and_gridlines, not flagged as an error.

    `min_separation` of None means the current
    config.PROFILE_COLOR_MIN_SEPARATION (resolved at call time so a
    Preferences change applies without a restart).

    Returns a list of human-readable problem descriptions (empty if
    everything is fine).

    This exists because a too-similar pair among the OTHER combinations
    (most commonly background vs plot, from a click that missed the trace
    entirely) doesn't fail loudly — it silently corrupts detection,
    surfacing later as a confusing downstream error that's hard to trace
    back to the actual cause. Catching it here, right where the colors are
    known, lets the caller fail fast with a specific, actionable message.
    """
    if min_separation is None:
        min_separation = config.PROFILE_COLOR_MIN_SEPARATION
    problems = []
    names = list(colors.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_i, name_j = names[i], names[j]
            if {name_i, name_j} == {"axis", "gridline"}:
                continue
            a = np.asarray(colors[name_i], dtype=np.int16)
            b = np.asarray(colors[name_j], dtype=np.int16)
            max_diff = int(np.max(np.abs(a - b)))
            if max_diff < min_separation:
                problems.append(
                    f"{name_i!r} and {name_j!r} are nearly identical "
                    f"({tuple(int(v) for v in a)} vs {tuple(int(v) for v in b)}, "
                    f"max channel diff={max_diff}, need >= {min_separation}) — "
                    f"these need to be visually distinct colors; try zooming in "
                    f"and re-picking the one more likely to have been mis-clicked."
                )
    return problems


def not_background_not_plot_mask(image_rgb: np.ndarray,
                                 bg_rgb: tuple[int, int, int],
                                 plot_rgb: tuple[int, int, int],
                                 tol: int = config.COLOR_MASK_TOLERANCE
                                 ) -> np.ndarray:
    """
    Everything that isn't background and isn't plot color.
    Contains: axis lines, gridlines, tick labels, unit labels, title text.
    """
    bg = color_mask(image_rgb, bg_rgb, tol)
    pl = color_mask(image_rgb, plot_rgb, tol)
    return ~(bg | pl)
