Profilometry FFT (Fast Fourier Transform) Analyzer
===================================================

GUI application (app.py)
-------------------------
One window, no terminal required:

    python app.py

A Preferences menu sits at the top of the window (preferences_gui.py):
  Modify parameters…  Opens a grouped list of every tunable the pipeline
                      exposes — feature detection, dishing, roughness, trace
                      extraction, OCR, axis fitting, confidence weights and
                      review thresholds. Each row has the value in a text box
                      and a small [i] button explaining in a sentence or two
                      what changing it does to your results. Values are
                      checked when you save (a bad entry is rejected with the
                      reason, not silently stored) and take effect on the next
                      batch run — no restart.
  Revert to defaults  Puts every parameter back to its shipped value.

The values live in two JSON files next to the app (or next to the .exe, so a
frozen build stays configurable — see app_settings.get_app_dir):
  parameter_defaults.json   the shipped defaults, for reference and reverting
  parameter_settings.json   what's actually in effect; edited by the dialog
Both are plain JSON and can be edited by hand or copied between machines to
share a tuned setup. An older settings file still works after an upgrade:
unrecognized entries are ignored and anything missing falls back to its
default. Note that a tuned_params.json from the tuning GUI still overrides
the trace color tolerance and max gap — that's called out in those two
parameters' help text.

Three tabs, sharing one session (app_session.py):
  Batch      Pick the input folder and output unit, then Run batch — the
             same four phases as batch_analyze.py below, but in-window: a
             progress bar with an elapsed/ETA estimate during per-image
             analysis, and — instead of a terminal prompt and a separate
             pop-up window — a banner explaining what's needed followed by
             a transient tab (e.g. "Create profile", "Manual fit — <name>")
             where you do that one thing; it closes itself when you're done
             and the batch continues.
  Dashboard  A sortable table of the run's results, with failed rows and
             low-confidence rows (calibration or trend confidence below
             0.5) highlighted so they're easy to find without scrolling the
             whole run. A "flagged" column shows True/False for every image
             checked in the Viewer (see below). Double-click a row to open
             that image in Viewer.
  Viewer     Click through this session's processed images (Prev/Next, or
             jump in from a Dashboard double-click) and see each one's
             measurements and confidence values next to its review overlay.
             A "Flag for follow-up review" checkbox under the measurements
             marks the current image — it writes True/False into that
             image's "flagged" column immediately (in-session and in the
             results CSV on disk), so flagged images can be filtered or
             sorted out once the run becomes a CSV. "Adjust parameters"
             opens the very same manual-fit window used for failed images
             in the Batch tab — for ANY image, not just failed ones — and
             re-runs the automatic fit to seed it with that image's current
             measurements (so removing or adding a feature starts from what
             was actually measured, not an empty form). Saving amends that
             image's row in the results CSV on disk and re-renders its
             overlay with the updated markers.

This is a GUI layer over the same engine batch_analyze.py uses
(app_engine.py) — results are identical either way. The Viewer only shows
images processed in the current app session (this window, since it was
opened or since the last batch run) — it doesn't reopen an old CSV from a
previous session.

Batch pipeline (batch_analyze.py)
---------------------------------
Automated analysis of exported profilometry plot images:

    python batch_analyze.py

  1. Put exported plot images in input_files/ (created on first run).
  2. Profile audit: every image is matched against calibration_profiles.json.
     Images with no matching color profile open the calibration GUI once so
     you can create one (background, plot, axis, gridline). If everything
     already matches, no GUI appears — just a success message.
  3. Each image is then automatically calibrated (gridline detection + OCR
     of tick labels and units, with confidence scores), the plot trace is
     extracted, features are segmented into ridges/trenches against their
     surrounding baseline surface and classified (square vs sine, majority
     vote when a scan mixes both), and measurements are computed:
       - step heights and physical widths (avg/min/max/median across
         features); width is the 50%-height crossing to 50%-height
         crossing distance, defined the same way for square and sine
         features
       - dishing on square feature tops: the peak-to-line depth (max
         deviation from the chord between the plateau's two rim peaks,
         searched over the central half of the span between them) — never
         reported negative. A signed diagnostic column dishing_raw_min flags
         questionable measurements for review
       - ISO roughness parameters Ra, Rq, Rz, measured over the baseline/
         reference-surface samples when real features are present (so
         step heights don't dominate the "roughness" numbers), with an
         ISO 4288 auto-selected cutoff lambda_c (capped on short profiles;
         capping is flagged per row and summarized once per run)
  4. Outputs:
       - data_files/analysis_results_<timestamp>.csv — one row per image,
         filename metadata (Lot/Wafer/MS/Tool/DIEX/DIEY/Date/Time/Channel)
         plus a "flagged" column (True/False; set from the GUI app's Viewer
         tab — see below, always False for a batch_analyze.py-only run)
         and measurement/confidence columns. All lengths are in OUTPUT_UNIT
         ('nm' or 'um', set at the top of batch_analyze.py).
       - image_fits/<run-timestamp>/ — one review overlay per image: axis
         calibration + extracted trace + filtered curve + height/dishing
         markers with values, all summarized in one compact strip pinned
         to the top of the image (calibration facts and measurement facts
         together, drawn first so the trace/markers always render on top
         of it, never the other way around).
       - error_logs/<run-timestamp>.txt — warnings/errors for the run.
         A failed image never stops the batch; its CSV row carries the
         error message.
  5. Manual correction (optional): failed images are copied to
     failed_images/<run-timestamp>/ and a terminal prompt offers a manual
     fit. For each failed image a minimal GUI opens: calibrate the axes by
     clicking two points spanning a known distance (or typing unit/px),
     then either run the AUTO trend re-fit with your calibration (most
     failures are OCR-only — accept the proposal or override it), or
     measure manually: step heights and dishing by two clicks (drawn live
     over the image) or typed values, feature type, and Ra/Rq/Rz. Enter as
     much or as little as you want; saved values go into the same CSV row.
     User-entered groups carry the numeric confidence sentinel -1.0 (an
     accepted auto re-fit keeps its real computed trend confidence).

Interactive tools
-----------------
  tuning_gui.py        two-tab GUI for trace-extraction tuning (color buffer,
                       max gap, lambda_c preview). Its tuned_params.json
                       still feeds the batch pipeline's EXTRACTION settings;
                       axis scaling/units now come from plot_calibration
                       per image.
  run_example.py       exercises plot_calibration on the committed
                       example_data images (works on a fresh clone).
  sample_calibration.py  single-image calibration example.

Tests
-----
  test_round5.py       synthetic-profile tests for feature_analysis and
                       get_sample_data (needs numpy/scipy installed).
  test_round4.py       earlier stub-based calibration tests.

Building a standalone executable (Windows, PyInstaller)
---------------------------------------------------------
app.spec builds app.py into a one-folder Windows app — no Python install
needed on the machine that runs it. PyInstaller is not a cross-compiler:
build ON Windows, FOR Windows.

    cd "path\to\profilometry_fft_analyzer(nonlocal)"
    pip install -r requirements.txt pyinstaller
    pyinstaller app.spec

Output is dist\ProfilometryFFTAnalyzer\ — run ProfilometryFFTAnalyzer.exe
from inside that folder (don't copy just the .exe elsewhere: it creates
input_files\, data_files\, image_fits\, error_logs\, failed_images\, and
calibration_profiles.json next to itself on first run — see
app_engine.get_app_dir() — so the whole folder needs to travel together).

Known gotchas:
  - One-folder, not one-file. easyOCR pulls in PyTorch, which is large; a
    one-file build would re-extract a GB-scale payload to a temp directory
    on every launch. One-folder starts fast and still zips up as one unit.
  - EasyOCR's recognition models are NOT bundled by app.spec — by default
    plot_calibration/ocr.py's easyocr.Reader(...) downloads them to
    %USERPROFILE%\.EasyOCR\model\ the first time OCR runs, same as an
    unfrozen install. On a machine with internet access this just works
    once, on first use. For an offline/air-gapped lab PC, either (a) run
    the app once on that machine while it briefly has internet/a hotspot
    to trigger the download, or (b) copy an already-populated
    %USERPROFILE%\.EasyOCR\model\ folder over from a machine that has run
    it before.
  - The folder name profilometry_fft_analyzer(nonlocal) has parentheses —
    fine for PyInstaller itself (it works with paths, not package names),
    just remember to quote it in shell commands.
  - app.spec builds with console=True (a console window shows alongside
    the app) so a startup problem — e.g. a missing hidden import, the most
    common PyInstaller+scipy/sklearn issue — prints somewhere visible
    instead of the app just failing to appear. Once a build runs cleanly,
    flip console to False in app.spec and rebuild for a normal windowed
    app. If you do hit a missing-module error, add that module name to
    hiddenimports in app.spec and rebuild; needing a couple of rounds of
    this is normal for PyInstaller + this dependency stack, not a sign the
    spec is broken.
