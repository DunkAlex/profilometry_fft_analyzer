"""
run_example.py
---------------
Runnable entry point exercising `plot_calibration` against the two example
images committed in `example_data/`. Unlike sample_calibration.py (which
points at a real, gitignored instrument-data directory), this script works
on a fresh clone with no external data required.

First run: no profile exists yet for this layout, so `on_no_match="gui"`
launches the interactive calibration window (plot_calibration/gui_calibrate.py)
— pick background, plot, axis, and gridline colors once and name the
profile; it's saved to `calibration_profiles.json` next to this script.
Both example images share the same layout, so one profile covers both;
subsequent runs match automatically with no GUI needed.

A debug overlay image is written to "image_fits/" next to this script for
each run (see plot_calibration/debug.py) showing detected gridlines, tick
labels, and unit-label bounding boxes.
"""
from __future__ import annotations
import os
import sys

import cv2

from plot_calibration import calibrate_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(BASE_DIR, "example_data")
PROFILES_PATH = os.path.join(BASE_DIR, "calibration_profiles.json")

# Works on a fresh clone: exercises the two committed example images.
EXAMPLE_IMAGES = sorted(
    f for f in os.listdir(EXAMPLE_DIR)
    if f.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"))
)


def load_image(name: str):
    path = os.path.join(EXAMPLE_DIR, name)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run(name: str) -> bool:
    image_rgb = load_image(name)
    result = calibrate_image(
        image_rgb,
        image_filename=name,
        profiles_path=PROFILES_PATH,
        on_no_match="gui",
    )
    if result is None:
        print(f"[{name}] calibration failed — see logs above.")
        return False

    print(f"[{name}] profile={result.profile_used}")
    print(f"  X: {result.x_px_per_unit:.4f} px/{result.x_unit or '?'}  "
          f"(conf={result.x_confidence:.2f})")
    print(f"  Y: {result.y_px_per_unit:.4f} px/{result.y_unit or '?'}  "
          f"(conf={result.y_confidence:.2f})")
    if result.warnings:
        print("  warnings:")
        for w in result.warnings:
            print(f"    - {w}")
    return True


if __name__ == "__main__":
    ok = True
    for name in EXAMPLE_IMAGES:
        ok = run(name) and ok
    sys.exit(0 if ok else 1)
