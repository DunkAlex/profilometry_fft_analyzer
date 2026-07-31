"""
profiles.py
-----------
Load/save calibration profiles as JSON. Profile fingerprint is a
quantized 3D RGB histogram; matching uses Bhattacharyya distance.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from . import config


# ---------------- Histogram fingerprint ----------------

def compute_histogram(image_rgb: np.ndarray,
                      bits: int = config.HIST_BITS_PER_CHANNEL) -> np.ndarray:
    """
    Quantized 3D RGB histogram, flattened and normalized to sum=1.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 image in RGB order
    bits      : bits per channel (4 -> 16 bins/channel -> 4096 total)

    Returns
    -------
    (bins^3,) float32 array, sum = 1.0
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("expected (H, W, 3) RGB image")
    bins = 1 << bits
    # cv2.calcHist works on multi-channel; ranges are [0, 256) per channel
    hist = cv2.calcHist(
        [image_rgb], channels=[0, 1, 2], mask=None,
        histSize=[bins, bins, bins],
        ranges=[0, 256, 0, 256, 0, 256],
    )
    hist = hist.astype(np.float32).flatten()
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def match_histograms(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """
    Bhattacharyya distance in [0, 1]. Lower = more similar.
    """
    a = hist_a.astype(np.float32).reshape(-1, 1)
    b = hist_b.astype(np.float32).reshape(-1, 1)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


# ---------------- Profile I/O ----------------

def load_profiles(json_path: str) -> list[dict]:
    """Returns list of profile dicts, or [] if file missing/empty."""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # tolerate a single-profile file
        if isinstance(data, dict):
            return [data]
        return []
    except (json.JSONDecodeError, OSError):
        return []


def save_profile(profile: dict, json_path: str) -> None:
    """
    Append profile, or replace-by-name if a profile with the same name exists.
    Creates the file if it doesn't exist.
    """
    profiles = load_profiles(json_path)
    name = profile.get("name")
    replaced = False
    for i, p in enumerate(profiles):
        if p.get("name") == name:
            profiles[i] = profile
            replaced = True
            break
    if not replaced:
        profiles.append(profile)

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(profiles, f, indent=2)


def build_profile(name: str,
                  picked_colors: dict,
                  histogram: np.ndarray,
                  unit_label_clicks: Optional[dict] = None) -> dict:
    """
    Assemble a profile dict. `picked_colors` must contain keys:
        'background', 'plot', 'axis', 'gridline'   (RGB triples)
    """
    required = {"background", "plot", "axis", "gridline"}
    if not required.issubset(picked_colors):
        missing = required - set(picked_colors)
        raise ValueError(f"picked_colors missing keys: {missing}")

    return {
        "schema_version": 1,
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "colors": {k: list(map(int, picked_colors[k])) for k in required},
        "unit_label_clicks": unit_label_clicks or {"x": [], "y": []},
        "histogram": histogram.astype(float).tolist(),
        "histogram_bits": config.HIST_BITS_PER_CHANNEL,
    }


# ---------------- Matching ----------------

def match_profile(image_rgb: np.ndarray,
                  profiles: list[dict],
                  threshold: float = config.HIST_MATCH_THRESHOLD
                  ) -> Optional[dict]:
    """
    Find the best-matching profile for `image_rgb` by histogram distance.
    Returns the profile dict if the best distance is below `threshold`,
    otherwise None.
    """
    if not profiles:
        return None

    query = compute_histogram(image_rgb)
    best = None
    best_dist = float("inf")
    for p in profiles:
        if "histogram" not in p:
            continue
        ref = np.asarray(p["histogram"], dtype=np.float32)
        # sanity: bin counts must match
        if ref.size != query.size:
            continue
        d = match_histograms(query, ref)
        if d < best_dist:
            best_dist = d
            best = p

    if best is not None and best_dist < threshold:
        # attach the score so caller can log/inspect
        best = dict(best)
        best["_match_distance"] = best_dist
        return best
    return None
