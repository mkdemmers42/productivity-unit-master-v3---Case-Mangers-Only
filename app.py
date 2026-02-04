import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# MASTER v3 Configuration
# -----------------------------
VALID_CODES = {
    "TCM/ICC",
    "Psychosocial Rehab - Individual",
    "Psychosocial Rehabilitation Group",
    "Non-billable Attempted Contact",
    "Client Non Billable Srvc Must Document",
    "Crisis Intervention",
    "Plan Development, non-physician",
}

BILLABLE_FACE_TO_FACE_CODES = {
    "TCM/ICC",
    "Psychosocial Rehab - Individual",
    "Psychosocial Rehabilitation Group",
    "Crisis Intervention",
    "Plan Development, non-physician",
}

NON_BILLABLE_FTF_CODES = {
    "Non-billable Attempted Contact",
    "Client Non Billable Srvc Must Document",
}

REQUIRED_COLS_CANONICAL = [
    "Procedure Code Name",
    "Travel Time",
    "Documentation Time",
    "Face-to-Face Time",
]

TOL_MINUTES_WORKED = 0.1
TOL_PERCENT = 0.01


# -----------------------------
# Data Structures
# -----------------------------
@dataclass(frozen=True)
class Results:
    hours_worked: float
    minutes_worked: float
    minutes_billed: int
    units_billed: int
    non_billable_total: int
    documentation_total: int
    travel_total: int
    billable_minutes_pct: float
    billable_units_pct: float
    non_billable_pct: float
    documentation_pct: float
    travel_pct: float


# -----------------------------
# Utility functions
# -----------------------------
def normalize_header(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s.strip()


def canonicalize_headers(cols: List[Any]) -> Dict[Any, str]:
    """
    MASTER v3 Header Normalization:
    - Trim whitespace
    - Convert line breaks to spaces
    - Standardize face-to-face variants to canonical
    """
    mapping: Dict[Any, str] = {}
    for c in cols:
        n = normalize_header(c)
        low = n.lower()

        if low == "procedure code name":
            mapping[c] = "Procedure Code Name"
            continue

        if low.replace("-", " ") == "travel time":
            mapping[c] = "Travel Time"
            continue

        if low.replace("-", " ") == "documentation time":
            mapping[c] = "Documentation Time"
            continue

        # Face-to-face variants
        ftf = low.replace("face to face", "face-to-face")
        ftf = ftf.replace("–", "-").replace("—", "-")
        if ftf == "face-to-face time":
            mapping[c] = "Face-to-Face Time"
            continue

        mapping[c] = n
    return mapping


# -----------------------------
# NEW: Auto-header detection (fixes "blank row above headers" issue)
# -----------------------------
def find_header_row_index_0_based(file_bytes: bytes, scan_rows: int = 40) -> int:
    """
    Finds the header row by scanning the first N rows for the presence of
    'Procedure Code Name' (normalized). Returns 0-based row index.
    """
    bio = io.BytesIO(file_bytes)
    preview = pd.read_excel(bio, header=None, nrows=scan_rows, dtype=object)

    for i in range(len(preview)):
        row = preview.iloc[i].tolist()
        normalized = [normalize_header(x).lower() for x in row]
        if "procedure code name" in normalized:
            return i

    raise ValueError(
        "Could not locate the header row. Expected to find 'Procedure Code Name' "
        "within the first rows of the spreadsheet."
    )


def load_excel_auto_header(file_bytes: bytes, dtype=object) -> Tuple[pd.DataFrame, int]:
    """
    Loads Excel using auto-detected header row. Returns (df, header_row_index_0_based).
    """
    header_idx = find_header_row_index_0_based(file_bytes)

    bio = io.BytesIO(file_bytes)
    df = pd.read_excel(bio, header=header_idx, dtype=dtype)
    return df, header_idx


def unit_grid(minutes: float) -> int:
    """
    MASTER v3 unit grid. Ceiling at 16 for >248.
    """
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        m = 0.0
    else:
        m = float(minutes)

    if m <= 7:
        return 0
    if m <= 22:
        return 1
    if m <= 37:
        return 2
    if m <= 52:
        return 3
    if m <= 67:
        return 4
    if m <= 82:
        return 5
    if m <= 97:
        return 6
    if m <= 112:
        return 7
    if m <= 127:
        return 8
    if m <= 142:
        return 9
    if m <= 157:
        return 10
    if m <= 172:
        return 11
    if m <= 187:
        return 12
    if m <= 202:
        return 13
    if m <= 217:
        return 14
    if m <= 232:
        return 15
    return 16


def round_minutes_worked(m: float) -> float:
    return round(m, 1)


def round_pct(p: float) -> float:
    return round(p, 2)


def compute_pass(
    hours_worked: float,
    file_bytes: bytes,
    audit: Optional[Dict[str, Any]] = None,
) -> Results:
    """
    FULL WORKFLOW PASS: load -> clean -> compute
    Header now auto-detected (supports both clean exports and ones with blank rows).
    Hidden math: return results only; optionally record audit details.
    """
    minutes_worked_raw = hours_worked * 60.0

    # Load (AUTO header detection)
    df, header_idx = load_excel_auto_header(file_bytes, dtype=object)

    # Normalize / canonicalize headers
    original_cols = list(df.columns)
    mapping = canonicalize_headers(original_cols)
    df = df.rename(columns=mapping)

    # Confirm required columns exist
    for col in REQUIRED_COLS_CANONICAL:
        if col not in df.columns:
            raise ValueError(f"MISSING REQUIRED COLUMN: {col}")

    #
