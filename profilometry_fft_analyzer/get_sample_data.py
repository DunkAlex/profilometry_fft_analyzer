"""
get_sample_data.py
------------------
Parse sample metadata out of instrument-export image filenames.

Expected naming convention (underscore-separated), e.g.:

    token 0            : Recipe ID (kept verbatim)
    token 1            : Lot       (may contain dots)
    token 2            : Wafer
    token 'MS...'      : MS        (measurement site, kept verbatim, e.g. 'MS01')
    tokens MS -> DIEX  : Site Name      (joined with '_', e.g. 'New_AFM_1')
    token 'DIEX<n>'    : DIEX      (int)
    token 'DIEY<n>'    : DIEY      (int)
    next two all-digit
    tokens after DIEY  : Date, Time (Excel-readable, e.g. '02/22', '02:49:49')
    token 'Ch...'      : Chamber   (e.g. 'Ch1.001')

`get_sample_info` NEVER raises on a non-conforming name (e.g. test files):
missing fields simply come back None so the batch pipeline can log a
warning and keep the row.
"""
import os
import re


# Column order used by both the dict and the DataFrame helpers, and by
# batch_analyze when assembling the results CSV.
SAMPLE_FIELDS = ['Name', 'Lot', 'Wafer', 'Recipe ID', 'MS', 'Site Name',
                 'DIEX', 'DIEY', 'Date', 'Time', 'Chamber']


_DIE_RE = re.compile(r'^DIE([XY])(-?\d+)$', re.IGNORECASE)


def _format_date(token):
    """Format an MMDD digit token as 'MM/DD' so Excel reads it as a date."""
    if len(token) == 4:
        return f'{token[0:2]}/{token[2:4]}'
    return token


def _format_time(token):
    """Format an HHMMSS digit token as 'HH:MM:SS' so Excel reads it as a time."""
    if len(token) == 6:
        return f'{token[0:2]}:{token[2:4]}:{token[4:6]}'
    return token


def get_sample_info(image_name):
    """
    Parse `image_name` into a dict with keys SAMPLE_FIELDS.
    Fields that can't be resolved are None. Never raises.
    """
    info = {k: None for k in SAMPLE_FIELDS}
    info['Name'] = image_name

    stem = os.path.splitext(os.path.basename(image_name))[0]
    tokens = stem.split('_')
    if len(tokens) < 3:
        return info

    if 'ampm' in tokens[0].lower():
        info['Lot'] = tokens[0] or None
        info['Wafer'] = tokens[1] or None

    else:
        info['Recipe ID'] = tokens[0] or None
        info['Lot'] = tokens[1] or None
        info['Wafer'] = tokens[2] or None

    ms_idx = diex_idx = diey_idx = None
    for i, tok in enumerate(tokens):
        m = _DIE_RE.match(tok)
        if m:
            axis, val = m.group(1).upper(), int(m.group(2))
            if axis == 'X' and diex_idx is None:
                info['DIEX'] = val
                diex_idx = i
            elif axis == 'Y' and diey_idx is None:
                info['DIEY'] = val
                diey_idx = i
        elif ms_idx is None and i >= 3 and tok.upper().startswith('MS'):
            info['MS'] = tok
            ms_idx = i
        elif tok.upper().startswith('CH'):
            info['Chamber'] = tok

    # Site Name: everything strictly between the MS token and the DIEX token
    if ms_idx is not None and diex_idx is not None and diex_idx > ms_idx + 1:
        info['Site Name'] = '_'.join(tokens[ms_idx + 1:diex_idx])

    # Date/Time: the first two all-digit tokens after DIEY
    if diey_idx is not None:
        numeric_after = [t for t in tokens[diey_idx + 1:] if t.isdigit()]
        if len(numeric_after) >= 1:
            info['Date'] = _format_date(numeric_after[0])
        if len(numeric_after) >= 2:
            info['Time'] = _format_time(numeric_after[1])

    return info


def get_sample_df(image_name):
    """Single-row DataFrame version of get_sample_info (legacy interface)."""
    import pandas as pd
    return pd.DataFrame([get_sample_info(image_name)], columns=SAMPLE_FIELDS)


def compile_sample_info(image_name_list):
    """Parse a list of filenames into one DataFrame (one row per file)."""
    import pandas as pd
    rows = [get_sample_info(name) for name in image_name_list]
    return pd.DataFrame(rows, columns=SAMPLE_FIELDS)
