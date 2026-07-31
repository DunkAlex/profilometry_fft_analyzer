"""
batch_analyze.py
----------------
Terminal entry point for the batch pipeline. All the actual work (profile
matching, per-image analysis, overlay rendering, CSV assembly) lives in
app_engine.py, shared with the GUI app (app.py); this file just drives those
phases with print()-based status, input()-based prompts, and the standalone
(Tk-root-owning) gui_calibrate / manual_fit_gui popups.

  1. PROFILE AUDIT — every image in input_files/ is matched against the
     saved color profiles (calibration_profiles.json). For any image with
     no match the interactive calibration GUI opens so you can create one;
     matching then re-runs until everything is covered (or a GUI cancel
     skips that image). If everything already matches, a success message
     prints and no GUI ever appears.
  2. PER-IMAGE ANALYSIS — automatic axis calibration (plot_calibration:
     gridlines + OCR + confidence), trace extraction inside the detected
     plot region, conversion to physical units, ISO 4288 cutoff selection,
     feature segmentation/classification (square vs sine), step heights,
     feature widths, plateau dishing, and ISO roughness parameters
     (Ra/Rq/Rz, measured on the baseline surface when features are found).
  3. REVIEW OVERLAYS — one PNG per image combining the calibration overlay
     with the extracted trace, filtered curve, and height/dishing markers,
     stored in image_fits/<run-timestamp>/.
  4. RESULTS CSV — one row per image (filename metadata + measurements +
     confidence scores), written to data_files/. All length outputs are in
     OUTPUT_UNIT ('nm' or 'um').
  5. ERROR LOG — warnings/errors for the whole run land in
     error_logs/<run-timestamp>.txt; per-image failures never stop the batch.
  6. MANUAL CORRECTION (optional) — failed images are copied to
     failed_images/<run-timestamp>/ and this prompts for a manual fit via
     manual_fit_gui.py; corrected rows are merged into the same CSV.

Run from a terminal (or an IDE python window):
    python batch_analyze.py

For an all-in-one application window (progress bar + ETA, session viewer/
verifier, results dashboard) see app.py instead — same engine, same output
layout.
"""
from __future__ import annotations
import logging
import os
from datetime import datetime

from plot_calibration.profiles import load_profiles

import app_engine as engine

# After the batch, offer the manual-correction GUI for failed images.
# Set False for fully headless runs (failures stay as error rows).
MANUAL_REVIEW_PROMPT = True

logger = logging.getLogger("batch_analyze")


def _on_status(msg: str) -> None:
    """Prints a progress line. This is the terminal's version of the status
    callback the engine expects; the GUI passes one that writes to a banner."""
    print(msg)


def _on_unmatched(image_rgb, default_name):
    """Called when an image matches no saved color profile. Opens the
    blocking calibration GUI so the user can pick the four colors, and
    returns the new profile dict (or None if they cancelled)."""
    print("  (pick background, plot, axis, gridline colors, then Finish)")
    from plot_calibration.gui_calibrate import launch_calibration_gui
    return launch_calibration_gui(image_rgb, default_name=default_name)


def main() -> int:
    """Runs one full batch from the terminal: reads every image in
    input_files/, analyzes it, optionally walks the user through fixing the
    failures, and writes the results CSV plus overlays and a log.

    Returns a shell exit code — 0 if every image succeeded, 1 if there was
    nothing to process, 2 if the run finished with failures."""
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = engine.default_config()
    log_path, counter, _log_handlers = engine.setup_run_logging(config, run_ts)
    print(f"Batch analysis started {run_ts}")
    print(f"Log file: {log_path}")

    # Nothing to do is a clean early exit, not an error worth a traceback.
    images = engine.list_images(config)
    if not images:
        print(f"No images found in {config.input_dir!r} — place exported plot "
              f"images there and re-run.")
        logger.error("no input images found in %s", config.input_dir)
        return 1

    # This run's overlays get their own timestamped folder.
    run_dir = os.path.join(config.image_fits_dir, run_ts)
    os.makedirs(run_dir, exist_ok=True)

    buffer, max_gap = engine.extraction_params(config)

    # ---- Phase 1: make sure every image has a color profile ----
    hists, skipped, _created = engine.resolve_profile_audit(
        config, images, _on_status, _on_unmatched)
    skipped_set = set(skipped)
    profiles_list = load_profiles(config.profiles_path)

    # ---- Phase 2: analyze each image ----
    def _progress(i, n, name):
        """Prints the [n/total] line before each image starts."""
        print(f"[{i}/{n}] {name}")

    rows, failed_items, _ctx_by_path = engine.run_batch(
        config, images, hists, profiles_list, skipped_set,
        run_dir, buffer, max_gap, on_progress=_progress)

    # ---- Phase 3: offer to hand-fit whatever failed ----
    if failed_items:
        fail_dir = engine.copy_failed_images(config, failed_items, run_ts)
        print(f"\n{len(failed_items)} failed image(s) copied to {fail_dir}")

        # Ask before opening GUIs; EOF (piped/headless run) counts as "no".
        run_review = MANUAL_REVIEW_PROMPT
        if run_review:
            try:
                ans = input(f"Manually fit the {len(failed_items)} failed "
                            f"image(s) now? [y/N] ").strip().lower()
            except EOFError:
                ans = ""
            run_review = ans in ("y", "yes")

        if run_review:
            def _on_review_one(path, name, prefill):
                """Opens the manual-fit window for one failed image and
                returns its (action, data) result. A missing GUI toolkit
                ends the review rather than killing the run."""
                try:
                    from manual_fit_gui import launch_manual_fit
                except Exception as e:
                    logger.error("manual-fit GUI unavailable: %s", e)
                    return "quit", None
                rgb = engine.load_rgb(path)
                return launch_manual_fit(rgb, name, prefill, config.output_unit)

            engine.manual_review_phase(
                config, failed_items, rows, run_dir, buffer, max_gap,
                _on_status, _on_review_one)
        else:
            print("Skipping manual correction — failures stay as error rows.")

    engine.log_lambda_c_capping(rows)

    # ---- Phase 4: write the results CSV ----
    csv_path = engine.write_results_csv(config, rows, run_ts)

    # Failed images still have a row; the error column is what marks them.
    n_failed = sum(1 for r in rows
                   if isinstance(r.get("error"), str) and r["error"])
    ok = len(images) - n_failed
    print(f"\nDone: {ok}/{len(images)} image(s) analyzed, {n_failed} failed.")
    print(f"Results CSV : {csv_path}")
    print(f"Overlays    : {run_dir}")
    print(f"Log ({counter.warnings} warning(s), {counter.errors} error(s)): "
          f"{log_path}")
    logger.info("run complete: %d ok, %d failed, %d warnings, %d errors",
               ok, n_failed, counter.warnings, counter.errors)
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
