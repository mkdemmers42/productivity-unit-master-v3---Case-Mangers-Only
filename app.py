import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# App Config (MUST be first Streamlit call)
# -----------------------------
APP_TITLE = "Mike's Productivity/Unit Machine (v3) - Case Managers Only"

st.set_page_config(
    page_title=APP_TITLE,
    layout="centered"
)

# -----------------------------
# UI: RED Blueprint Skin
# -----------------------------
def apply_red_blueprint_skin():
    st.markdown("""

<style>

/* Import Modern Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Apply Font Globally */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #0b1220;
    background-image:
      linear-gradient(rgba(239,68,68,0.07) 1px, transparent 1px),
      linear-gradient(90deg, rgba(239,68,68,0.07) 1px, transparent 1px),
      radial-gradient(circle at 20% 10%, rgba(239,68,68,0.12), transparent 35%),
      radial-gradient(circle at 80% 30%, rgba(248,113,113,0.10), transparent 40%);
    background-size: 34px 34px, 34px 34px, 100% 100%, 100% 100%;
    background-attachment: fixed;
}

      section[data-testid="stSidebar"] {
        background: #091022;
        border-right: 1px solid rgba(239,68,68,0.18);
      }
      section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
      }

      .bp-header {
        padding: 18px 18px 14px 18px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(17,26,46,0.92), rgba(9,16,34,0.92));
        border: 1px solid rgba(239,68,68,0.24);
        box-shadow: 0 10px 28px rgba(0,0,0,0.45);
        margin: 8px 0 14px 0;
      }
      .bp-header h1 {
        margin: 0;
        font-size: 28px;
        letter-spacing: 0.5px;
        color: #e6edf3;
      }
      .bp-header p {
        margin: 6px 0 0 0;
        opacity: 0.85;
        color: rgba(254,202,202,0.95);
      }

    div[data-testid="stMetric"] {
    background: rgba(17,26,46,0.90);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 18px;
    padding: 14px 14px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.30);
}

/* FIX TEXT INSIDE METRIC CARDS */
div[data-testid="stMetric"] * {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 600;
}

      div[data-testid="stFileUploader"],
      div[data-testid="stSelectbox"],
      div[data-testid="stTextInput"],
      div[data-testid="stNumberInput"] {
        background: rgba(17,26,46,0.60);
        border: 1px solid rgba(239,68,68,0.14);
        border-radius: 16px;
        padding: 10px 10px 6px 10px;
      }

        .stButton > button {
        border-radius: 14px;
        padding: 0.65rem 1rem;
        border: 1px solid rgba(239,68,68,0.38);
        background: linear-gradient(180deg, rgba(239,68,68,0.22), rgba(17,26,46,0.80));
        color: #e6edf3;
        box-shadow: 0 10px 22px rgba(0,0,0,0.35);
      }
      .stButton > button:hover {
        border: 1px solid rgba(239,68,68,0.58);
        filter: brightness(1.05);
      }

    .block-container {
    padding-top: 5rem;
}

/* FIX GENERAL TEXT (BUT NOT BUTTONS) */
label,
p,
div[data-testid="stMarkdownContainer"],
div[data-testid="stWidgetLabel"],
div[data-testid="stCheckbox"],
div[data-testid="stRadio"] {
    color: rgba(226,232,240,0.80) !important;
    opacity: 1 !important;
}

/* FIX FILE UPLOADER BUTTON TEXT */
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button *,
div[data-testid="stFileUploader"] [role="button"],
div[data-testid="stFileUploader"] [role="button"] * {
    color: #111827 !important;
    fill: #111827 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* FIX FORM LABELS ABOVE INPUTS */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stTextInput"] label,
div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label {
    color: rgba(255,255,255,0.88) !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

/* FIX INPUT TEXT */
input,
textarea {
    color: #111827 !important;
    opacity: 1 !important;
}

    </style>
    """, unsafe_allow_html=True)

apply_red_blueprint_skin()

st.markdown("""
<div class="bp-header">
  <h1>Mike’s Productivity / Unit Machine</h1>
  <p>Life Made Easy • Case Managers Only</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# MASTER v3 Configuration (Case Manager)
# -----------------------------
VALID_CODES = {
    "TCM/ICC",
    "Psychosocial Rehab - Individual",
    "Psychosocial Rehabilitation Group",
    "Non-billable Attempted Contact",
    "Client Non Billable Srvc Must Document",
    "Crisis Intervention",
    "Plan Development, non-physician",
    "Brief Contact Note",
    "Targeted Outreach",
}

BILLABLE_FACE_TO_FACE_CODES = {
    "TCM/ICC",
    "Psychosocial Rehab - Individual",
    "Psychosocial Rehabilitation Group",
    "Crisis Intervention",
    "Plan Development, non-physician",
    "Brief Contact Note",
    "Targeted Outreach",
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
    minutes_worked_raw: float   # full-precision minutes for internal math
    minutes_worked: float       # rounded display minutes
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

        ftf = low.replace("face to face", "face-to-face")
        ftf = ftf.replace("–", "-").replace("—", "-")
        if ftf == "face-to-face time":
            mapping[c] = "Face-to-Face Time"
            continue

        mapping[c] = n
    return mapping

def find_header_row_index_0_based(file_bytes: bytes, scan_rows: int = 40) -> int:
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
    header_idx = find_header_row_index_0_based(file_bytes)
    bio = io.BytesIO(file_bytes)
    df = pd.read_excel(bio, header=header_idx, dtype=dtype)
    return df, header_idx

def unit_grid(minutes: float) -> int:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        m = 0.0
    else:
        m = float(minutes)

    if m <= 7: return 0
    if m <= 22: return 1
    if m <= 37: return 2
    if m <= 52: return 3
    if m <= 67: return 4
    if m <= 82: return 5
    if m <= 97: return 6
    if m <= 112: return 7
    if m <= 127: return 8
    if m <= 142: return 9
    if m <= 157: return 10
    if m <= 172: return 11
    if m <= 187: return 12
    if m <= 202: return 13
    if m <= 217: return 14
    if m <= 232: return 15
    return 16

def round_minutes_worked(m: float) -> float:
    return round(m, 1)

def round_pct(p: float) -> float:
    return round(p, 2)

def compute_pass(hours_worked: float, file_bytes: bytes, audit: Optional[Dict[str, Any]] = None) -> Results:
    minutes_worked_raw = hours_worked * 60.0

    df, header_idx = load_excel_auto_header(file_bytes, dtype=object)

    original_cols = list(df.columns)
    row_count_loaded = int(len(df))

    mapping = canonicalize_headers(original_cols)
    df = df.rename(columns=mapping)

    for col in REQUIRED_COLS_CANONICAL:
        if col not in df.columns:
            raise ValueError(f"MISSING REQUIRED COLUMN: {col}")

    df["Procedure Code Name"] = df["Procedure Code Name"].astype(str).str.strip()

    total_row_mask = df["Procedure Code Name"].str.contains("total", case=False, na=False)
    total_rows_removed = int(total_row_mask.sum())
    df = df[~total_row_mask].copy()

    minute_cols = ["Travel Time", "Documentation Time", "Face-to-Face Time"]
    for c in minute_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    invalid = sorted(set(df["Procedure Code Name"].unique()) - VALID_CODES)
    if invalid:
        raise ValueError("INVALID PROCEDURE CODE(S) FOUND:\n" + "\n".join(invalid))

    # Create audit-facing helper columns. These do NOT change the math; they only explain it.
    df["App Category"] = "Other/Valid Code Not Counted as Units"
    df.loc[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES), "App Category"] = "Billable / Unit-Producing"
    df.loc[df["Procedure Code Name"].isin(NON_BILLABLE_FTF_CODES), "App Category"] = "Non-Billable / No Units"

    df["Units Counted By App"] = 0
    billable_mask = df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES)
    df.loc[billable_mask, "Units Counted By App"] = df.loc[billable_mask, "Face-to-Face Time"].apply(unit_grid).astype(int)

    # round totals instead of truncating
    non_billable_total = int(round(df.loc[df["Procedure Code Name"].isin(NON_BILLABLE_FTF_CODES), "Face-to-Face Time"].sum()))
    documentation_total = int(round(df["Documentation Time"].sum()))
    travel_total = int(round(df["Travel Time"].sum()))

    minutes_billed = int(round(df.loc[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES), "Face-to-Face Time"].sum()))
    units_billed = int(df["Units Counted By App"].sum())

    if minutes_worked_raw == 0:
        billable_minutes_pct = billable_units_pct = non_billable_pct = documentation_pct = travel_pct = 0.0
    else:
        billable_minutes_pct = (minutes_billed / minutes_worked_raw) * 100.0
        billable_units_pct = ((units_billed * 15.0) / minutes_worked_raw) * 100.0
        non_billable_pct = (non_billable_total / minutes_worked_raw) * 100.0
        documentation_pct = (documentation_total / minutes_worked_raw) * 100.0
        travel_pct = (travel_total / minutes_worked_raw) * 100.0

    res = Results(
        hours_worked=float(hours_worked),
        minutes_worked_raw=float(minutes_worked_raw),
        minutes_worked=round_minutes_worked(minutes_worked_raw),
        minutes_billed=minutes_billed,
        units_billed=units_billed,
        non_billable_total=non_billable_total,
        documentation_total=documentation_total,
        travel_total=travel_total,
        billable_minutes_pct=round_pct(billable_minutes_pct),
        billable_units_pct=round_pct(billable_units_pct),
        non_billable_pct=round_pct(non_billable_pct),
        documentation_pct=round_pct(documentation_pct),
        travel_pct=round_pct(travel_pct),
    )

    if audit is not None:
        per_code = (
            df.groupby(["Procedure Code Name", "App Category"], dropna=False)
            .agg(
                Rows=("Procedure Code Name", "size"),
                Face_To_Face_Minutes=("Face-to-Face Time", "sum"),
                Units_Counted_By_App=("Units Counted By App", "sum"),
                Documentation_Minutes=("Documentation Time", "sum"),
                Travel_Minutes=("Travel Time", "sum"),
            )
            .reset_index()
            .sort_values(["App Category", "Procedure Code Name"])
        )

        per_category = (
            df.groupby("App Category", dropna=False)
            .agg(
                Rows=("Procedure Code Name", "size"),
                Face_To_Face_Minutes=("Face-to-Face Time", "sum"),
                Units_Counted_By_App=("Units Counted By App", "sum"),
                Documentation_Minutes=("Documentation Time", "sum"),
                Travel_Minutes=("Travel Time", "sum"),
            )
            .reset_index()
            .sort_values("App Category")
        )

        row_columns = [
            "Procedure Code Name",
            "App Category",
            "Face-to-Face Time",
            "Units Counted By App",
            "Documentation Time",
            "Travel Time",
        ]
        optional_cols = [
            "Staff Name", "StaffName", "Staff", "Client Name", "ClientName", "Client or Group Name",
            "Date of Service", "DateOfService", "Status", "Program Name", "ProgramName"
        ]
        row_columns = [c for c in optional_cols if c in df.columns] + row_columns
        row_columns = list(dict.fromkeys([c for c in row_columns if c in df.columns]))

        audit["header_row_1_indexed"] = int(header_idx + 1)
        audit["original_columns"] = [str(c) for c in original_cols]
        audit["renamed_columns"] = list(df.columns)
        audit["row_count_loaded"] = row_count_loaded
        audit["total_rows_removed"] = total_rows_removed
        audit["row_count_after_clean"] = int(len(df))
        audit["unique_codes"] = sorted(df["Procedure Code Name"].unique().tolist())
        audit["per_category_breakdown"] = per_category.round(2).to_dict(orient="records")
        audit["per_code_breakdown"] = per_code.round(2).to_dict(orient="records")
        audit["row_level_breakdown"] = df[row_columns].round(2).to_dict(orient="records")
        audit["intermediate"] = {
            "hours_worked_entered": hours_worked,
            "minutes_worked_raw": minutes_worked_raw,
            "minutes_billed": minutes_billed,
            "units_billed": units_billed,
            "unit_minutes_equivalent": units_billed * 15.0,
            "non_billable_total": non_billable_total,
            "documentation_total": documentation_total,
            "travel_total": travel_total,
            "billable_minutes_pct_raw": billable_minutes_pct,
            "billable_units_pct_raw": billable_units_pct,
            "non_billable_pct_raw": non_billable_pct,
            "documentation_pct_raw": documentation_pct,
            "travel_pct_raw": travel_pct,
        }

    return res

def compare_results(p1: Results, p2: Results) -> Tuple[bool, List[str]]:
    mismatches: List[str] = []

    def mm(name: str, a: Any, b: Any) -> None:
        mismatches.append(f"{name}: Pass1={a} Pass2={b}")

    if p1.hours_worked != p2.hours_worked:
        mm("Hours_Worked", p1.hours_worked, p2.hours_worked)

    # compare rounded display minutes, plus keep your tolerance
    if abs(p1.minutes_worked - p2.minutes_worked) > TOL_MINUTES_WORKED:
        mm("Minutes_Worked", p1.minutes_worked, p2.minutes_worked)

    if p1.minutes_billed != p2.minutes_billed:
        mm("Minutes_Billed", p1.minutes_billed, p2.minutes_billed)
    if p1.units_billed != p2.units_billed:
        mm("Units_Billed", p1.units_billed, p2.units_billed)
    if p1.non_billable_total != p2.non_billable_total:
        mm("Non_Billable_Total", p1.non_billable_total, p2.non_billable_total)
    if p1.documentation_total != p2.documentation_total:
        mm("Documentation_Time_Total", p1.documentation_total, p2.documentation_total)
    if p1.travel_total != p2.travel_total:
        mm("Travel_Time_Total", p1.travel_total, p2.travel_total)

    pct_fields = [
        ("Billable_Minutes_Percentage", p1.billable_minutes_pct, p2.billable_minutes_pct),
        ("Billable_Units_Percentage", p1.billable_units_pct, p2.billable_units_pct),
        ("Non_Billable_Percentage", p1.non_billable_pct, p2.non_billable_pct),
        ("Documentation_Percentage", p1.documentation_pct, p2.documentation_pct),
        ("Travel_Percentage", p1.travel_pct, p2.travel_pct),
    ]
    for name, a, b in pct_fields:
        if abs(a - b) > TOL_PERCENT:
            mm(name, a, b)

    return (len(mismatches) == 0, mismatches)

# -----------------------------
# Pie Chart (based on existing computed Results only)
# -----------------------------
def render_time_pie(res: Results) -> None:
    total_minutes = float(res.minutes_worked_raw)

    units_minutes = float(res.units_billed) * 15.0
    non_billable_minutes = float(res.non_billable_total)
    travel_minutes = float(res.travel_total)
    documentation_minutes = float(res.documentation_total)

    accounted_minutes = (
        units_minutes
        + non_billable_minutes
        + travel_minutes
        + documentation_minutes
    )

    unaccounted_minutes = max(0.0, total_minutes - accounted_minutes)

    df = pd.DataFrame({
        "Category": [
            "Units Billed",
            "Non-Billable",
            "Drive Time",
            "Documentation",
            "Unaccounted Time",
        ],
        "Minutes": [
            units_minutes,
            non_billable_minutes,
            travel_minutes,
            documentation_minutes,
            unaccounted_minutes,
        ],
        "Color": [
            "#16a34a",  # Units Billed (green)
            "#f97316",  # Non-Billable (orange)
            "#2563eb",  # Drive Time (blue)
            "#eab308",  # Documentation (yellow)
            "#dc2626",  # Unaccounted Time (red)
        ],
    })

    df = df[df["Minutes"] > 0].copy()

    if df.empty:
        st.info("No time data available to chart.")
        return

    fig = px.pie(
        df,
        names="Category",
        values="Minutes",
        hole=0.0,
    )

    fig.update_traces(
        marker=dict(colors=df["Color"].tolist()),
        textinfo="percent+label",
        textfont=dict(size=16)
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3", size=16),
        legend=dict(font=dict(color="#e6edf3", size=16)),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)



# -----------------------------
# County Cross-Check Helpers
# -----------------------------
def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized_lookup = {normalize_header(c).lower(): c for c in df.columns}
    for candidate in candidates:
        key = normalize_header(candidate).lower()
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None

def normalize_proc_for_crosscheck(value: Any) -> str:
    s = normalize_header(value).lower()
    s = s.replace("–", "-").replace("—", "-")
    s = " ".join(s.split())

    if s in {"tcm/icc", "targeted case management"}:
        return "TCM/ICC"

    if s in {"psychosocial rehab - individual", "psychosocial rehabilitation"}:
        return "Psychosocial Rehab - Individual"

    if s in {
        "plan development, non-physician",
        "mental health service plan developed by non-physician",
    }:
        return "Plan Development, non-physician"

    if s in {"psychosocial rehabilitation group"}:
        return "Psychosocial Rehabilitation Group"

    if s in {"crisis intervention", "crisis intervention services"}:
        return "Crisis Intervention"

    if s in {
        "non-billable attempted contact",
        "non billable attempted contact",
        "client non billable srvc must document",
        "client non-billable srvc must document",
    }:
        return "IGNORE_NON_BILLABLE"

    return normalize_header(value)

def clean_crosscheck_id(value: Any) -> str:
    """Normalize ID fields so 12345 and 12345.0 can still match."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = normalize_header(value)
    if s.lower() in {"nan", "nat", "none", "", "clientid", "client id", "client or group id"}:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

def clean_crosscheck_date(value: Any) -> str:
    """Normalize dates to YYYY-MM-DD and ignore headers/blanks."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = normalize_header(value)
    if s.lower() in {"", "nan", "nat", "none", "date", "service date", "dateofservice", "date of service", "dos"}:
        return ""
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d")

def clean_crosscheck_duration(value: Any) -> str:
        """Normalize time duration so 30, 30.0, and 00:30 can match."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""

        s = normalize_header(value)

        if s.lower() in {"", "nan", "nat", "none", "time duration", "duration"}:
            return ""

        # Try timedelta (Excel time format)
        td = pd.to_timedelta(value, errors="coerce")
        if not pd.isna(td):
            return str(int(round(td.total_seconds() / 60.0)))

        # Try numeric minutes
        try:
            f = float(s)
            if math.isfinite(f):
                return str(int(round(f)))
        except Exception:
            pass

        # Try HH:MM format
        if ":" in s:
            parts = s.split(":")
            try:
                if len(parts) >= 2:
                    hours = int(float(parts[0]))
                    minutes = int(float(parts[1]))
                    return str(hours * 60 + minutes)
            except Exception:
                pass

        return s    

def build_crosscheck_key_from_clean(
    clean_id: pd.Series,
    clean_date: pd.Series,
    clean_procedure: pd.Series,
    clean_duration: pd.Series
) -> pd.Series:
    return (
        clean_id.astype(str).str.strip()
        + "|"
        + clean_date.astype(str).str.strip()
        + "|"
        + clean_procedure.astype(str).str.strip()
        + "|"
        + clean_duration.astype(str).str.strip()
    )

def load_sdr_for_crosscheck(file_bytes: bytes) -> pd.DataFrame:
    """
    SDR positional matching rule from Mike:
    - SDR Service Date is Column B (0-based index 1)
    - SDR Client or Group Id is Column C (0-based index 2)

    Column names, spacing, dots, and headers are intentionally ignored for matching.
    """
    header_idx = find_header_row_index_0_based(file_bytes)
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None, dtype=object)
    raw_data = raw.iloc[header_idx + 1:].reset_index(drop=True).copy()

    df, _ = load_excel_auto_header(file_bytes, dtype=object)
    df = df.rename(columns=canonicalize_headers(list(df.columns)))
    df.columns = [normalize_header(c) for c in df.columns]
    df = df.reset_index(drop=True)

    if len(raw_data) < len(df):
        raise ValueError("SDR cross-check could not align raw positional rows with the cleaned SDR rows.")
    raw_data = raw_data.iloc[:len(df)].reset_index(drop=True)

    if raw_data.shape[1] < 3:
        raise ValueError("SDR cross-check needs at least columns B through C.")
    if "Procedure Code Name" not in df.columns or "Face-to-Face Time" not in df.columns:
        raise ValueError("SDR cross-check needs Procedure Code Name and Face-to-Face Time for the display breakdown.")

    # Mike's strict positional rule:
    # SDR Client ID is Column C and SDR Service Date is Column B.
    # If a row's Column C ID and Column B date match the county service, it counts as a match.
    # We do not use names, procedure wording, minutes, units, punctuation, or headers for matching.
    sdr_dates = raw_data.iloc[:, 1].ffill()
    sdr_client_ids = raw_data.iloc[:, 2]
    sdr_time_durations = raw_data.iloc[:, 12]

    df["Procedure Code Name"] = df["Procedure Code Name"].astype(str).str.strip()
    df = df[~df["Procedure Code Name"].str.contains("total", case=False, na=False)].copy()

    # Keep the same row mask for the raw positional fields.
    raw_data = raw_data.loc[df.index].reset_index(drop=True)
    sdr_dates = sdr_dates.loc[df.index].reset_index(drop=True)
    sdr_client_ids = sdr_client_ids.loc[df.index].reset_index(drop=True)
    sdr_time_durations = sdr_time_durations.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)

    df["Face-to-Face Time"] = pd.to_numeric(df["Face-to-Face Time"], errors="coerce").fillna(0)
    df["Crosscheck Procedure"] = df["Procedure Code Name"].apply(normalize_proc_for_crosscheck)
    df["Crosscheck ClientId"] = sdr_client_ids.map(clean_crosscheck_id)
    df["Crosscheck Date"] = sdr_dates.map(clean_crosscheck_date)
    df["Crosscheck Duration"] = sdr_time_durations.map(clean_crosscheck_duration)
    df["Match Key"] = build_crosscheck_key_from_clean(
    df["Crosscheck ClientId"],
    df["Crosscheck Date"],
    df["Crosscheck Procedure"],
    df["Crosscheck Duration"]
)

    df["App Units"] = 0
    mask = df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES)
    df.loc[mask, "App Units"] = df.loc[mask, "Face-to-Face Time"].apply(unit_grid).astype(int)

    # Remove junk positional rows that do not have both key pieces.
    df = df[(df["Crosscheck ClientId"] != "") & (df["Crosscheck Date"] != "")].copy()

    # County Cross-Check should compare billable/unit-producing SDR rows only.
    # Non-billables are still counted in the main app/audit mode, but ignored here.
    df = df[~df["Procedure Code Name"].isin(NON_BILLABLE_FTF_CODES)].copy()
    df = df[df["App Units"] > 0].copy()

    return df

def load_county_for_crosscheck(file_bytes: bytes) -> pd.DataFrame:
    """
    # County positional matching rule from Mike:
    # - County Client ID is Column D (0-based index 3)
    # - County Date of Service is Column O (0-based index 14)

    Column names, spacing, dots, and headers are intentionally ignored for matching.
    """
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None, dtype=object)
    if raw.shape[1] < 24:
        raise ValueError("County cross-check needs at least columns D through X.")

    # Also load the county file with headers for optional units/procedure display.
    county_named = pd.read_excel(io.BytesIO(file_bytes), dtype=object)
    county_named.columns = [normalize_header(c) for c in county_named.columns]
    county = county_named.reset_index(drop=True).copy()

    # Positional rows usually start after the header row.
    raw_data = raw.iloc[1:].reset_index(drop=True).copy()
    if len(raw_data) < len(county):
        raw_data = raw.iloc[:len(county)].reset_index(drop=True).copy()
    else:
        raw_data = raw_data.iloc[:len(county)].reset_index(drop=True).copy()

    county_ids = raw_data.iloc[:, 3]
    county_dates = raw_data.iloc[:, 14].ffill()

    proc_col = find_first_existing_col(county, ["ProcedureCodeName", "Procedure Code Name"])
    units_col = find_first_existing_col(county, ["Charge Units", "ChargeUnits", "Units"])
    minutes_col = find_first_existing_col(county, ["Minutes", "Minutes2"])

    if proc_col:
        county["Crosscheck Procedure"] = county[proc_col].apply(normalize_proc_for_crosscheck)
    else:
        county["Crosscheck Procedure"] = "County Service"

    if units_col:
        county["County Charge Units"] = pd.to_numeric(county[units_col], errors="coerce").fillna(0)
    else:
        county["County Charge Units"] = 0

    if minutes_col:
        county["County Minutes"] = pd.to_numeric(county[minutes_col], errors="coerce").fillna(0)
    else:
        county["County Minutes"] = 0

    county["Crosscheck ClientId"] = county_ids.map(clean_crosscheck_id)
    county["Crosscheck Date"] = county_dates.map(clean_crosscheck_date)
    county_time_durations = raw_data.iloc[:, 23]

    county["Crosscheck Duration"] = county_time_durations.map(clean_crosscheck_duration)

    county["Match Key"] = build_crosscheck_key_from_clean(
    county["Crosscheck ClientId"],
    county["Crosscheck Date"],
    county["Crosscheck Procedure"],
    county["Crosscheck Duration"]
)

    county = county[(county["Crosscheck ClientId"] != "") & (county["Crosscheck Date"] != "")].copy()

    # County Cross-Check should compare billable/unit-producing county rows only.
    # Ignore any county-side non-billable rows before matching.
    county = county[county["Crosscheck Procedure"] != "IGNORE_NON_BILLABLE"].copy()

    county = county[county["County Charge Units"] > 0].copy()

    return county

def run_county_crosscheck(sdr_file_bytes: bytes, county_file_bytes: bytes) -> Dict[str, Any]:
    sdr = load_sdr_for_crosscheck(sdr_file_bytes)
    county = load_county_for_crosscheck(county_file_bytes)

    # Big-picture totals by procedure code. This is informational only.
    app_summary = (
        sdr.groupby("Crosscheck Procedure", dropna=False)
        .agg(SDR_Rows=("Crosscheck Procedure", "size"), 
             SDR_Minutes=("Face-to-Face Time", "sum"), 
             SDR_Units=("App Units", "sum"))
        .reset_index()
    )
    county_summary = (
        county.groupby("Crosscheck Procedure", dropna=False)
        .agg(County_Rows=("Crosscheck Procedure", "size"), County_Minutes=("County Minutes", "sum"), County_Charge_Units=("County Charge Units", "sum"))
        .reset_index()
    )
    summary = app_summary.merge(county_summary, on="Crosscheck Procedure", how="outer").fillna(0)
    summary = summary.rename(columns={
    "County_Charge_Units": "County Units"
    })
    summary["Unit Difference"] = summary["SDR_Units"] - summary["County Units"]
    summary = summary.sort_values("Crosscheck Procedure")

    # Exact positional matching rule: County Column A + Column L against SDR Column C + Column B.
    # Occurrence prevents one county record from matching endless SDR rows with the same key.
    sdr = sdr.copy()
    county = county.copy()
    sdr["Match Occurrence"] = sdr.groupby("Match Key").cumcount()
    county["Match Occurrence"] = county.groupby("Match Key").cumcount()

    matched = sdr.merge(
        county[["Match Key", "Match Occurrence", "County Charge Units", "County Minutes", "Crosscheck Procedure"]].rename(columns={"Crosscheck Procedure": "County Procedure"}),
        on=["Match Key", "Match Occurrence"],
        how="left",
        indicator=True,
    )
    matched["County Charge Units"] = pd.to_numeric(matched["County Charge Units"], errors="coerce")
    matched["County Minutes"] = pd.to_numeric(matched["County Minutes"], errors="coerce")
    matched["Unit Difference"] = matched["App Units"] - matched["County Charge Units"].fillna(0)

    matched_rows = matched[matched["_merge"] == "both"].copy()
    exact = matched_rows[matched_rows["Unit Difference"] == 0].copy()
    adjusted = matched_rows[matched_rows["Unit Difference"] != 0].copy()
    missing = matched[matched["_merge"] == "left_only"].copy()

    # REMOVE non-billable noise from missing list
    missing = missing[~missing["Procedure Code Name"].isin([
        "Non-billable Attempted Contact",
        "Client Non Billable Srvc Must Document"
    ])]

    show_cols = [
        "Crosscheck Date", "Crosscheck ClientId", "Procedure Code Name", "Crosscheck Procedure", "County Procedure",
        "Face-to-Face Time", "App Units", "County Minutes", "County Charge Units", "Unit Difference"
    ]

    def clean_view(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        cols = [c for c in show_cols if c in df.columns]
        return df[cols].round(2)

    return {
        "summary": summary.round(2),
        "exact": clean_view(exact),
        "adjusted": clean_view(adjusted),
        "missing": clean_view(missing),
        "matched_row_count": int(len(matched_rows)),
        "exact_row_count": int(len(exact)),
        "adjusted_row_count": int(len(adjusted)),
        "missing_row_count": int(len(missing)),
        "submitted_row_count": int(len(sdr)),
        "county_row_count": int(len(county)),
    }

def render_county_crosscheck_report(report: Dict[str, Any]) -> None:
    st.markdown("---")
    st.markdown("### County Cross-Check Mode")
    st.info("This compares the files by the exact position rule: County Column D + Column O must match SDR Column C + Column B.")

    c1, c2, c3 = st.columns(3)
    c1.metric("SDR Rows", report.get("submitted_row_count", 0))
    c2.metric("County Rows", report.get("county_row_count", 0))
    c3.metric("Matched Rows", report.get("matched_row_count", 0))

    c4, c5, c6 = st.columns(3)
    c4.metric("Exact Unit Matches", report.get("exact_row_count", 0))
    c5.metric("Matched, Units Differ", report.get("adjusted_row_count", 0))
    c6.metric("No County Match", report.get("missing_row_count", 0))

    st.markdown("#### Side-by-Side by Procedure")
    st.dataframe(report["summary"], use_container_width=True, hide_index=True)

    adjusted = report.get("adjusted")
    st.markdown("#### Matched, But Units Differ")
    if adjusted is None or adjusted.empty:
        st.success("No matched rows with unit differences found.")
    else:
        st.dataframe(adjusted, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Matched Rows With Unit Differences CSV",
            data=adjusted.to_csv(index=False).encode("utf-8"),
            file_name="sdr_matched_units_differ_county.csv",
            mime="text/csv",
        )

    missing = report.get("missing")
    st.markdown("#### No County Match — SDR Rows Not Found by Client ID + Date")
    if missing is None or missing.empty:
        st.success("No unmatched SDR rows found with the positional Client ID + Date rule.")
    else:
        st.dataframe(missing, use_container_width=True, hide_index=True)
        st.download_button(
            "Download No County Match Rows CSV",
            data=missing.to_csv(index=False).encode("utf-8"),
            file_name="sdr_no_county_match.csv",
            mime="text/csv",
        )

# -----------------------------
# Results Display (Metric Cards + Pie)
# -----------------------------
def print_final(res: Results) -> None:
    st.success("VERIFICATION PASSED ✅")

    c1, c2 = st.columns(2)
    c1.metric("Hours Worked", f"{res.hours_worked:.2f}")
    c2.metric("Minutes Worked", f"{res.minutes_worked:.1f}")

    st.markdown("")

    c3, c4 = st.columns(2)
    c3.metric("Minutes Billed", f"{res.minutes_billed}")
    c4.metric("Billable Minutes %", f"{res.billable_minutes_pct}%")

    st.markdown("")

    c5, c6 = st.columns(2)
    c5.metric("Units Billed", f"{res.units_billed}")
    c6.metric("Billable Units %", f"{res.billable_units_pct}%")

    st.markdown("")

    c7, c8 = st.columns(2)
    c7.metric("Non-Billable Total", f"{res.non_billable_total}")
    c8.metric("Non-Billable %", f"{res.non_billable_pct}%")

    st.markdown("")

    c9, c10 = st.columns(2)
    c9.metric("Documentation Total", f"{res.documentation_total}")
    c10.metric("Documentation %", f"{res.documentation_pct}%")

    st.markdown("")

    c11, c12 = st.columns(2)
    c11.metric("Travel Total", f"{res.travel_total}")
    c12.metric("Travel %", f"{res.travel_pct}%")

    st.markdown("---")
    st.markdown("### Time Breakdown (Based on Hours Worked)")
    render_time_pie(res)


# -----------------------------
# Audit Mode Display
# -----------------------------
def render_audit_mode(audit_payload: Dict[str, Any], county_productivity: Optional[float] = None) -> None:
    if not audit_payload or "pass1" not in audit_payload:
        return

    audit = audit_payload["pass1"]
    final = audit_payload.get("final", {})
    intermediate = audit.get("intermediate", {})

    st.markdown("---")
    st.markdown("### Audit Mode — Calculation Transcript")
    st.warning(
        "Audit Mode is for internal review only. Normal users should leave this off. "
        "This section explains what the app counted and what it did not count."
    )

    st.markdown("#### 1. File Intake")
    st.write(f"Header row found: **Row {audit.get('header_row_1_indexed', 'Unknown')}**")
    st.write(f"Rows loaded: **{audit.get('row_count_loaded', 'Unknown')}**")
    st.write(f"Total rows removed: **{audit.get('total_rows_removed', 'Unknown')}**")
    st.write(f"Rows after cleaning: **{audit.get('row_count_after_clean', 'Unknown')}**")

    st.markdown("#### 2. Core Calculation")
    calc_rows = [
        {"Step": "Hours worked entered", "Value": final.get("hours_worked")},
        {"Step": "Minutes worked", "Value": final.get("minutes_worked_raw")},
        {"Step": "Billable face-to-face minutes", "Value": final.get("minutes_billed")},
        {"Step": "Units counted by app", "Value": final.get("units_billed")},
        {"Step": "Unit-minute equivalent", "Value": intermediate.get("unit_minutes_equivalent")},
        {"Step": "Billable units percentage", "Value": final.get("billable_units_pct")},
    ]
    st.dataframe(pd.DataFrame(calc_rows), use_container_width=True, hide_index=True)

    if county_productivity is not None:
        app_units = float(final.get("units_billed", 0) or 0)
        county_effective_units_actual_hours = (county_productivity / 100.0) * float(final.get("hours_worked", 0) or 0) * 4.0
        county_effective_units_173 = (county_productivity / 100.0) * 173.33 * 4.0

        comparison_df = pd.DataFrame([
            {"Comparison": "App units counted", "Units": round(app_units, 2)},
            {"Comparison": "County effective units if using actual hours", "Units": round(county_effective_units_actual_hours, 2)},
            {"Comparison": "County effective units if using 173.33 standard hours", "Units": round(county_effective_units_173, 2)},
            {"Comparison": "Gap vs county effective units using actual hours", "Units": round(app_units - county_effective_units_actual_hours, 2)},
            {"Comparison": "Gap vs county effective units using 173.33", "Units": round(app_units - county_effective_units_173, 2)},
        ])
        st.markdown("#### 3. County Comparison Helper")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("#### 4. Breakdown by Category")
    st.dataframe(pd.DataFrame(audit.get("per_category_breakdown", [])), use_container_width=True, hide_index=True)

    st.markdown("#### 5. Breakdown by Procedure Code")
    st.dataframe(pd.DataFrame(audit.get("per_code_breakdown", [])), use_container_width=True, hide_index=True)

    with st.expander("Show Row-Level Audit Details"):
        st.dataframe(pd.DataFrame(audit.get("row_level_breakdown", [])), use_container_width=True, hide_index=True)

# -----------------------------
# Streamlit UI (Hidden Math)
# -----------------------------
st.markdown("---")
st.subheader("Need help running this app?")

need_howto = st.radio(
    "Would you like the How-To document?",
    ["No", "Yes"],
    horizontal=True
)

if need_howto == "Yes":
    DOC_PATH = "How_to_Run_Mikes_Productivity_App.docx"
    try:
        with open(DOC_PATH, "rb") as f:
            st.download_button(
                label="Download How-To Guide (Word)",
                data=f.read(),
                file_name="How_to_Run_Mikes_Productivity_App.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    except FileNotFoundError:
        st.warning("How-To document file not found in the app repo. Tell Mike.")

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_audit_payload" not in st.session_state:
    st.session_state["last_audit_payload"] = None
if "last_crosscheck_report" not in st.session_state:
    st.session_state["last_crosscheck_report"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None
    st.session_state["last_crosscheck_report"] = None
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0

def do_reset() -> None:
    st.session_state["reset_counter"] += 1
    st.session_state["last_result"] = None
    st.session_state["last_audit_payload"] = None
    st.session_state["last_crosscheck_report"] = None
    st.session_state["last_error"] = None

k = st.session_state["reset_counter"]
hours_key = f"hours_{k}"
file_key = f"uploaded_file_{k}"
county_file_key = f"county_file_{k}"

hours = st.text_input("Please insert **Hours Worked**", placeholder="Example: 148.13", key=hours_key)

st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    'Upload the **Excel (.xlsx) file** exported from **"Staff Service Detail Report"** from SmartCare',
    type=["xlsx"],
    key=file_key
)

st.markdown("---")
audit_mode = st.checkbox(
    "Supervisor Audit Mode — show calculation transcript",
    value=False,
    help="Shows the detailed internal breakdown used to reconcile app totals against county productivity. Leave off for normal staff-facing use."
)

county_productivity_input = None
if audit_mode:
    county_productivity_input = st.text_input(
        "Optional: Enter county-reported productivity % for comparison",
        placeholder="Example: 42.81"
    )

county_crosscheck_mode = st.checkbox(
    "County Cross-Check Mode — compare SDR to county billed file",
    value=False,
    help="Optional. Use only when the county productivity/billed file is available."
)

county_uploaded = None
if county_crosscheck_mode:
    county_uploaded = st.file_uploader(
        "Optional: Upload the county productivity/billed Excel file for cross-check",
        type=["xlsx"],
        key=county_file_key
    )

col_run, col_reset = st.columns([1, 1])
with col_run:
    run = st.button("Calculate Productivity", type="primary")
with col_reset:
    reset = st.button("Run Another Staff Member", on_click=do_reset)

st.divider()

def fail(msg: str) -> None:
    msg = (msg or "").strip()
    if not msg.endswith("Tell Mike."):
        msg = f"{msg}\n\nTell Mike."
    st.session_state["last_error"] = msg
    st.error(msg)
    st.stop()

if run:
    st.session_state["last_error"] = None

    try:
        hours_worked = float((hours or "").strip())
    except Exception:
        fail("Please enter Hours Worked as a number only (example: 128.4).")

    if uploaded is None:
        fail("Please upload the Excel spreadsheet containing staff's 'Billed' and 'Non-Billable' numbers.")

    file_bytes = uploaded.getvalue()

    audit1: Dict[str, Any] = {}
    try:
        pass1 = compute_pass(hours_worked, file_bytes, audit=audit1)
    except ValueError as e:
        fail(str(e))
    except Exception as e:
        fail(f"ERROR LOADING/PROCESSING FILE: {e}")

    audit2: Dict[str, Any] = {}
    try:
        pass2 = compute_pass(hours_worked, file_bytes, audit=audit2)
    except Exception as e:
        fail(f"VERIFICATION FAILED — RESULTS NOT TRUSTWORTHY\n\nReason: {e}")

    ok, mismatches = compare_results(pass1, pass2)
    if not ok:
        fail(
            "VERIFICATION FAILED — RESULTS NOT TRUSTWORTHY\n\nMetric(s) mismatched:\n"
            + "\n".join(mismatches)
        )

    st.session_state["last_result"] = pass1
    st.session_state["last_audit_payload"] = {
        "pass1": audit1,
        "pass2": audit2,
        "final": {
            "hours_worked": pass1.hours_worked,
            "minutes_worked_raw": pass1.minutes_worked_raw,
            "minutes_worked": pass1.minutes_worked,
            "minutes_billed": pass1.minutes_billed,
            "units_billed": pass1.units_billed,
            "non_billable_total": pass1.non_billable_total,
            "documentation_total": pass1.documentation_total,
            "travel_total": pass1.travel_total,
            "billable_minutes_pct": pass1.billable_minutes_pct,
            "billable_units_pct": pass1.billable_units_pct,
            "non_billable_pct": pass1.non_billable_pct,
            "documentation_pct": pass1.documentation_pct,
            "travel_pct": pass1.travel_pct,
        },
    }
    if county_crosscheck_mode and county_uploaded is not None:
        try:
            st.session_state["last_crosscheck_report"] = run_county_crosscheck(file_bytes, county_uploaded.getvalue())
        except Exception as e:
            fail(f"COUNTY CROSS-CHECK FAILED: {e}")

    st.rerun()

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])

if st.session_state["last_result"] is not None:
    print_final(st.session_state["last_result"])

    county_productivity_value = None
    if audit_mode and county_productivity_input:
        try:
            county_productivity_value = float(county_productivity_input.strip())
        except Exception:
            st.warning("County productivity comparison was skipped because the county % was not entered as a number.")

    if audit_mode and st.session_state["last_audit_payload"] is not None:
        render_audit_mode(st.session_state["last_audit_payload"], county_productivity=county_productivity_value)

    if st.session_state.get("last_crosscheck_report") is not None:
        render_county_crosscheck_report(st.session_state["last_crosscheck_report"])

    if st.session_state["last_audit_payload"] is not None:
        st.download_button(
            "Download Audit JSON (internal math, not displayed)",
            data=json.dumps(st.session_state["last_audit_payload"], indent=2).encode("utf-8"),
            file_name="productivity_audit.json",
            mime="application/json",
        )

    st.write("")
    st.info("Ready for the next staff member? Click **Run Another Staff Member**.")
