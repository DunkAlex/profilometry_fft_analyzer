"""
plot_calibration
================
Drop-in module for automated calibration of plot images:
color profile capture, gridline detection, OCR of axis labels,
pixel-to-unit conversion factor derivation with confidence scoring.

Entry point:
    from plot_calibration.calibrate import calibrate_image
"""
from .calibrate import calibrate_image, CalibrationResult  # noqa: F401
