"""
manual_merge.py
---------------
Pure-stdlib logic for merging a manual-fit GUI result into a batch results
row. Deliberately free of numpy/pandas so it can be unit-tested anywher
(see test_round5.py) and reasoned about in isolation.

Confidence convention (user decision): any USER-entered group gets the
numeric sentinel -1.0 in its confidence column — columns stay numeric, and
-1.0 is an unambiguous outlier meaning "human entered, no automatic
confidence". An auto-derived value the user merely accepted keeps its real
computed confidence.
"""
from __future__ import annotations
import statistics

USER_CONFIDENCE = -1.0


def _agg(prefix: str, values: list[float]) -> dict:
    """avg/min/max/median over a non-empty list, keyed '<prefix>_*'."""
    return {
        f"{prefix}_avg": statistics.fmean(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_median": statistics.median(values),
    }


def _agg_blank(prefix: str) -> dict:
    """NaN counterpart of _agg, for a group the user deliberately emptied
    (see the `edited` handling in merge_manual_row) — same four keys, so a
    "clear this group" edit blanks the row's aggregates instead of leaving
    a stale non-empty value behind."""
    nan = float("nan")
    return {f"{prefix}_avg": nan, f"{prefix}_min": nan,
            f"{prefix}_max": nan, f"{prefix}_median": nan}


def merge_manual_row(row: dict, data: dict, output_unit: str) -> dict:
    """
    Merge a manual-fit result `data` into a copy of `row` and return it.
    Only fields the user actually provided are touched; everything else
    keeps its existing (usually NaN) value. `error` is cleared only when at
    least one thing was entered.

    Expected `data` layout (all keys optional):
        x_cal / y_cal : {'unit_per_px', 'unit', 'source': 'user'|'auto',
                         'conf': float|None}
        feature_type  : str
        heights       : [float, ...]   (already in output_unit)
        widths        : [float, ...]
        dishings      : [float, ...]
        groups_edited : {'height': bool, 'width': bool, 'dishing': bool}
                        True for a group the caller actually looked at this
                        session (typed/clicked/deleted an entry, or accepted
                        an auto fit that filled or explicitly left it empty).
                        Distinguishes "the user emptied this group on
                        purpose" (its aggregates are blanked to NaN below)
                        from "the user never went near this group" (the
                        corresponding *s list above is empty merely because
                        nothing was ever entered — the row's existing value,
                        from the original batch run, is left alone). Absent
                        entirely (older callers / existing tests): every
                        group behaves the old way — non-empty list updates,
                        empty list is always a no-op.
        Ra, Rq, Rz    : float
        lambda_c      : float          lambda_c_capped : bool
        trend_conf    : float          (>=0 real auto confidence,
                                        USER_CONFIDENCE for manual entry)
        auto_accepted : bool
    """
    row = dict(row)
    entered = False
    edited = data.get("groups_edited") or {}

    for axis, conf_col, unit_col in (("x_cal", "cal_conf_x", "x_unit_detected"),
                                     ("y_cal", "cal_conf_y", "y_unit_detected")):
        cal = data.get(axis)
        if not cal:
            continue
        row[unit_col] = cal.get("unit")
        if cal.get("source") == "user":
            row[conf_col] = USER_CONFIDENCE
            entered = True
        else:  # auto prefill the user kept — preserve its real confidence
            conf = cal.get("conf")
            row[conf_col] = (round(float(conf), 3) if conf is not None
                             else USER_CONFIDENCE)

    if data.get("feature_type"):
        row["feature_type"] = data["feature_type"]
        entered = True

    heights = [float(h) for h in (data.get("heights") or [])]
    if heights:
        row.update(_agg("height", heights))
        row["n_features"] = len(heights)
        entered = True
    elif edited.get("height"):
        # Deliberately emptied (every height was removed, or an accepted
        # auto fit found none) — the row really has zero features now, not
        # "unknown"; blank the aggregates rather than keep a stale count.
        row.update(_agg_blank("height"))
        row["n_features"] = 0
        entered = True

    widths = [float(w) for w in (data.get("widths") or [])]
    if widths:
        row.update(_agg("width", widths))
        entered = True
    elif edited.get("width"):
        row.update(_agg_blank("width"))
        entered = True

    dishings = [float(d) for d in (data.get("dishings") or [])]
    if dishings:
        row["dishing_avg"] = statistics.fmean(dishings)
        row["dishing_max"] = max(dishings)
        # Manual dishings are hand-measured spans (already non-negative), so
        # the raw diagnostic is simply their minimum.
        row["dishing_raw_min"] = min(dishings)
        entered = True
    elif edited.get("dishing"):
        nan = float("nan")
        row["dishing_avg"] = nan
        row["dishing_max"] = nan
        row["dishing_raw_min"] = nan
        entered = True

    for key in ("Ra", "Rq", "Rz"):
        if data.get(key) is not None:
            row[key] = float(data[key])
            entered = True

    if data.get("lambda_c") is not None:
        row["lambda_c"] = float(data["lambda_c"])
        row["lambda_c_capped"] = bool(data.get("lambda_c_capped", False))
        entered = True

    if data.get("trend_conf") is not None:
        tc = float(data["trend_conf"])
        row["trend_conf"] = round(tc, 3) if tc >= 0 else USER_CONFIDENCE

    if data.get("auto_accepted"):
        entered = True

    row["output_unit"] = output_unit
    if entered:
        row["error"] = ""
    return row
