"""
ocr.py
------
Single reusable OCR entry point: `ocr_number(image_region)`.

Wraps EasyOCR with:
    - upscaling (3-4x) for tiny axis-tick text (biggest accuracy win)
    - optional inversion
    - numeric post-processing with common substitutions
"""
from __future__ import annotations
from typing import Optional
import re

import cv2
import numpy as np

from . import config


# Lazy singleton — EasyOCR init is expensive (~seconds + GPU alloc)
_reader = None
_reader_langs = None


def _get_reader(langs=None):
    global _reader, _reader_langs
    langs = langs or config.OCR_LANGUAGES
    if _reader is None or _reader_langs != langs:
        try:
            import easyocr  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "easyocr not installed. `pip install easyocr` to enable OCR."
            ) from e
        # gpu=False by default; user can flip in easyocr.Reader kwargs if wanted
        _reader = easyocr.Reader(langs, gpu=False, verbose=False)
        _reader_langs = langs
    return _reader


# ---------------- Public entry point ----------------

def ocr_number(image_region: np.ndarray,
               upscale: Optional[int] = None,
               invert: bool = False,
               min_conf: Optional[float] = None
               ) -> tuple[Optional[float], float]:
    """
    OCR a small crop and return (parsed_float_or_None, ocr_confidence).

    Parameters
    ----------
    image_region : (h, w) or (h, w, 3) uint8
    upscale      : integer scale factor for cubic interpolation
                   (None = current config.OCR_UPSCALE_FACTOR)
    invert       : negate the image before OCR (try both if unsure)
    min_conf     : discard OCR results below this confidence
                   (None = current config.OCR_MIN_CONFIDENCE)

    Returns
    -------
    (value, conf)
        value : parsed float, or None if OCR failed / no numeric text
        conf  : EasyOCR confidence [0, 1], or 0.0 on failure
    """
    # Resolved here rather than as default arguments: Python evaluates those
    # once at import, which would pin the values before the user has had any
    # chance to change them in Preferences.
    if upscale is None:
        upscale = config.OCR_UPSCALE_FACTOR
    if min_conf is None:
        min_conf = config.OCR_MIN_CONFIDENCE

    if image_region is None or image_region.size == 0:
        return None, 0.0

    img = image_region
    if upscale and upscale > 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC)

    if invert:
        img = 255 - img

    try:
        reader = _get_reader()
    except RuntimeError:
        return None, 0.0

    # allowlist keeps EasyOCR focused on digits + common numeric chars
    try:
        results = reader.readtext(img,
                                  allowlist="0123456789.-+eE",
                                  detail=1,
                                  paragraph=False)
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    # results: list of (bbox, text, confidence) — take the highest-conf match
    best_text = None
    best_conf = 0.0
    for entry in results:
        # entry structure: [bbox, text, conf]
        if len(entry) < 3:
            continue
        text, conf = entry[1], float(entry[2])
        if conf > best_conf:
            best_text = text
            best_conf = conf

    if best_text is None or best_conf < min_conf:
        # try post-processing anyway if we got any text — sometimes confidence
        # is low but the text is fine
        if best_text is None:
            return None, 0.0

    value = postprocess_number_string(best_text)
    return value, best_conf


def ocr_text(image_region: np.ndarray,
             upscale: Optional[int] = None,
             invert: bool = False,
             min_conf: Optional[float] = None
             ) -> tuple[Optional[str], float]:
    """
    OCR a small crop expecting general text (unit labels like 'um', 'nm',
    'mV'), without the digit-only allowlist used by ocr_number. This is
    the function to use for axis unit strings, which ocr_number will
    always fail on (they're not numeric).

    `upscale`/`min_conf` default to the current config values when None.

    Returns (cleaned_text_or_None, confidence).
    """
    # See ocr_number: resolved at call time so Preferences edits apply.
    if upscale is None:
        upscale = config.OCR_UPSCALE_FACTOR
    if min_conf is None:
        min_conf = config.UNIT_OCR_MIN_CONFIDENCE

    if image_region is None or image_region.size == 0:
        return None, 0.0

    img = image_region
    if upscale and upscale > 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC)
    if invert:
        img = 255 - img

    try:
        reader = _get_reader()
    except RuntimeError:
        return None, 0.0

    try:
        results = reader.readtext(img, detail=1, paragraph=False)
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    best_text, best_conf = None, 0.0
    for entry in results:
        if len(entry) < 3:
            continue
        text, conf = entry[1], float(entry[2])
        if conf > best_conf:
            best_text, best_conf = text, conf

    if best_text is None or best_conf < min_conf:
        return None, best_conf if best_text is not None else 0.0

    cleaned = best_text.strip()
    return (cleaned if cleaned else None), best_conf


def _clean_unit_text(raw_text: Optional[str]) -> Optional[str]:
    """Unicode-fold micro signs and strip whitespace. Shared by
    normalize_unit_string and score_unit_match so both use the exact same
    cleaning before doing the variant lookup."""
    if not raw_text:
        return None
    cleaned = raw_text.strip().replace("µ", "u").replace("μ", "u")
    return cleaned or None


def score_unit_match(raw_text: Optional[str]) -> tuple[Optional[str], float]:
    """
    Test whether `raw_text` matches a known unit spelling, without falling
    back to "cleaned but unrecognized" the way normalize_unit_string does —
    this is the classifier used to decide whether an OCR candidate IS a
    unit at all (see ticks.py's unified per-axis label pass), so it needs a
    clean yes/no rather than a display string.

    Returns (canonical_unit, 1.0) on a known-variant match (case-
    insensitive, via config.UNIT_CANONICAL_FORMS), or (None, 0.0)
    otherwise. Deliberately binary, not fuzzy: nm/um/mm are one character
    apart but 1000x apart in scale, so this never guesses a "close enough"
    unit — see the module-level warning on UNIT_CANONICAL_FORMS in
    config.py for why fuzzy matching is unsafe here.
    """
    cleaned = _clean_unit_text(raw_text)
    if not cleaned:
        return None, 0.0

    lookup_key = cleaned.lower().replace(" ", "")
    for canonical, variants in config.UNIT_CANONICAL_FORMS.items():
        if lookup_key in {v.lower() for v in variants}:
            return canonical, 1.0

    return None, 0.0


_LEADING_NUMBER_RE = re.compile(r"^\s*[-+]?\d[\d.,]*\s*")


def extract_unit_suffix(raw_text: Optional[str]) -> Optional[str]:
    """
    Strip a leading numeric run (e.g. the "100" in "100um") and return the
    remaining trailing text, or None if there's no leading digit or nothing
    is left after stripping.

    Exists for merged number+unit OCR tokens: when a tick label and the
    axis unit string sit close enough to fuse into one connected component
    (e.g. "100" immediately followed by "um" with no gap), the whole
    string never matches a known unit spelling via score_unit_match on its
    own — but the trailing suffix alone often does. Callers should treat
    this as a secondary attempt after a whole-string score_unit_match
    fails, not a replacement for it.
    """
    if not raw_text:
        return None
    m = _LEADING_NUMBER_RE.match(raw_text)
    if not m:
        return None
    suffix = raw_text[m.end():].strip()
    return suffix or None


def normalize_unit_string(raw_text: Optional[str]) -> Optional[str]:
    """
    Normalize a raw OCR'd unit string to its canonical ASCII form.

    Two things happen, in order:
      1. Unicode "micro" variants are always folded to ASCII 'u' — µ (micro
         sign, U+00B5) and μ (Greek mu, U+03BC) are visually identical but
         different codepoints, and either can show up depending on how the
         source rendered the label. Per convention, the pipeline always
         reports "um", never a unicode mu, so downstream code (filenames,
         CSVs, comparisons) never has to deal with encoding surprises.
      2. The cleaned string is matched (case-insensitively) against
         config.UNIT_CANONICAL_FORMS via score_unit_match, which maps a
         canonical unit to the specific OCR misreadings observed for it
         (e.g. "um" is frequently misread as "Hm" when the micro sign's
         glyph gets mistaken for an H at small font sizes). This is a
         conservative, explicit lookup rather than fuzzy/edit-distance
         matching — see score_unit_match for why.

    Returns the canonical form if matched, the unicode-folded/cleaned
    string if not matched (still useful to display even if unrecognized),
    or None if raw_text is empty.
    """
    cleaned = _clean_unit_text(raw_text)
    if not cleaned:
        return None

    canonical, score = score_unit_match(cleaned)
    return canonical if score >= config.UNIT_MATCH_MIN_CONFIDENCE else cleaned


# ---------------- Numeric post-processing ----------------

# Common OCR letter->digit confusions. Applied only if a plain float() parse
# fails, so we never corrupt clean input.
_OCR_SUBS = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
    "T": "7",
    "‐": "-", "–": "-", "—": "-", "−": "-",  # unicode dashes
}


def postprocess_number_string(text: str) -> Optional[float]:
    """
    Try increasingly aggressive parsing:
      1. strip whitespace and try float()
      2. strip non-numeric/period/minus and try float()
      3. apply common OCR substitutions and retry
      4. extract first numeric token via regex and try float()
    Returns float or None.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    # Attempt 1: raw
    try:
        return float(s)
    except ValueError:
        pass

    # Attempt 2: substitute common OCR letter->digit confusions FIRST,
    # then strip anything else. Doing substitutions before stripping is
    # critical: 'l0' should become '10', not '0'.
    subbed = "".join(_OCR_SUBS.get(c, c) for c in s)
    stripped = re.sub(r"[^\d\.\-\+eE]", "", subbed)
    if stripped and stripped not in {"-", "+", ".", "-.", "+."}:
        try:
            return float(stripped)
        except ValueError:
            pass

    # Attempt 3: regex extract first numeric token from the substituted string
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", subbed)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None

    return None
