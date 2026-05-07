import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# App Config (MUST be first Streamlit call)
# -----------------------------

APP_TITLE = "Mike's Productivity/Unit Machine (v3) - Case Managers Only"

st.set_page_config(
    page_title=APP_TITLE,
    layout="centered",
)


# -----------------------------
# UI: RED Blueprint Skin
# -----------------------------

def apply_red_blueprint_skin() -> None:
    st.markdown(
        """
        <style>
        /* ---------- Main red blueprint background ---------- */
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(239, 68, 68, 0.22), transparent 30%),
                radial-gradient(circle at bottom right, rgba(37, 99, 235, 0.18), transparent 35%),
                linear-gradient(rgba(15, 23, 42, 0.90), rgba(15, 23, 42, 0.94)),
                repeating-linear-gradient(
                    0deg,
                    rgba(255,255,255,0.055) 0px,
                    rgba(255,255,255,0.055) 1px,
                    transparent 1px,
                    transparent 28px
                ),
                repeating-linear-gradient(
                    90deg,
                    rgba(255,255,255,0.055) 0px,
                    rgba(255,255,255,0.055) 1px,
                    transparent 1px,
                    transparent 28px
                ),
                #0f172a;
            color: #e6edf3;
        }

        /* ---------- Page width and main panel ---------- */
        section.main > div.block-container {
            max-width: 980px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
        }

        .block-container {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 24px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
            backdrop-filter: blur(6px);
        }

        /* ---------- Text ---------- */
        h1 {
            color: #ffffff !important;
            font-weight: 900 !important;
            letter-spacing: 0.3px;
            text-shadow: 0 2px 18px rgba(239, 68, 68, 0.30);
        }

        h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            font-weight: 800 !important;
        }

        p, label, span, div {
            color: #e6edf3;
        }

        [data-testid="stCaptionContainer"] p {
            color: #fecaca !important;
            font-weight: 700;
            letter-spacing: 0.4px;
        }

        /* ---------- Inputs ---------- */
        .stTextInput input {
            background: rgba(2, 6, 23, 0.78) !important;
            color: #ffffff !important;
            border: 1px solid rgba(147, 197, 253, 0.42) !important;
            border-radius: 12px !important;
        }

        .stTextInput input:focus {
            border-color: rgba(248, 113, 113, 0.90) !important;
            box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.40) !important;
        }

        /* ---------- Upload box ---------- */
        div[data-testid="stFileUploader"] section {
            background: rgba(30, 41, 59, 0.84) !important;
            border: 1.5px dashed rgba(147, 197, 253, 0.75) !important;
            border-radius: 18px !important;
            box-shadow: inset 0 0 28px rgba(37, 99, 235, 0.10);
        }

        div[data-testid="stFileUploader"] section:hover {
            border-color: rgba(248, 113, 113, 0.90) !important;
            box-shadow: inset 0 0 30px rgba(239, 68, 68, 0.12);
        }

        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
            color: #ffffff !important;
            border: 1px solid rgba(191, 219, 254, 0.50) !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
        }

        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] p,
        div[data-testid="stFileUploader"] span {
            color: #dbeafe !important;
        }

        /* ---------- Buttons ---------- */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 14px !important;
            font-weight: 850 !important;
            letter-spacing: 0.2px;
            border: 1px solid rgba(255,255,255,0.18) !important;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #dc2626, #ef4444) !important;
            color: white !important;
        }

        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {
            background: rgba(30, 41, 59, 0.92) !important;
            color: #e6edf3 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(248, 113, 113, 0.75) !important;
        }

        /* ---------- Metric cards ---------- */
        [data-testid="stMetric"] {
            background:
                linear-gradient(145deg, rgba(15, 23, 42, 0.93), rgba(30, 41, 59, 0.82));
            border: 1px solid rgba(248, 113, 113, 0.34);
            padding: 1.05rem;
            border-radius: 18px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.26);
        }

        [data-testid="stMetricLabel"] p {
            color: #fecaca !important;
            font-weight: 800 !important;
            font-size: 0.98rem !important;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 900 !important;
        }

        /* ---------- Alerts/cards ---------- */
        [data-testid="stAlert"] {
            border-radius: 16px !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.20);
        }

        hr {
            border-color: rgba(148, 163, 184, 0.30) !important;
        }

        /* ---------- Radio buttons ---------- */
        div[role="radiogroup"] label {
            background: rgba(30, 41, 59, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 999px;
            padding: 0.25rem 0.70rem;
            margin-right: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_red_blueprint_skin()

st.title(APP_TITLE)
st.caption("Life Made Easy • Case Managers Only")


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
    minutes_worked_raw: float
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


def compute_pass(hours_worked: float, file_bytes: bytes, audit: Optional[Dict[str, Any]] = None) -> Results:
    minutes_worked_raw = hours_worked * 60.0

    df, header_idx = load_excel_auto_header(file_bytes, dtype=object)

    original_cols = list(df.columns)
    mapping = canonicalize_headers(original_cols)
    df = df.rename(columns=mapping)

    for col in REQUIRED_COLS_CANONICAL:
        if col not in df.columns:
            raise ValueError(f"MISSING REQUIRED COLUMN: {col}")

    df["Procedure Code Name"] = df["Procedure Code Name"].astype(str).str.strip()
    df = df[~df["Procedure Code Name"].str.contains("total", case=False, na=False)].copy()

    minute_cols = ["Travel Time", "Documentation Time", "Face-to-Face Time"]
    for c in minute_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    invalid = sorted(set(df["Procedure Code Name"].unique()) - VALID_CODES)
    if invalid:
        raise ValueError("INVALID PROCEDURE CODE(S) FOUND:\n" + "\n".join(invalid))

    non_billable_total = int(
        round(df.loc[df["Procedure Code Name"].isin(NON_BILLABLE_FTF_CODES), "Face-to-Face Time"].sum())
    )
    documentation_total = int(round(df["Documentation Time"].sum()))
    travel_total = int(round(df["Travel Time"].sum()))

    minutes_billed = int(
        round(df.loc[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES), "Face-to-Face Time"].sum())
    )
    billable_ftf = df.loc[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES), "Face-to-Face Time"]
    units_billed = int(billable_ftf.apply(unit_grid).sum())

    if minutes_worked_raw == 0:
        billable_minutes_pct = 0.0
        billable_units_pct = 0.0
        non_billable_pct = 0.0
        documentation_pct = 0.0
        travel_pct = 0.0
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
        audit["header_row_1_indexed"] = int(header_idx + 1)
        audit["original_columns"] = [str(c) for c in original_cols]
        audit["renamed_columns"] = list(df.columns)
        audit["row_count_after_clean"] = int(len(df))
        audit["unique_codes"] = sorted(df["Procedure Code Name"].unique().tolist())
        audit["intermediate"] = {
            "minutes_worked_raw": minutes_worked_raw,
            "minutes_billed": minutes_billed,
            "units_billed": units_billed,
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
# Pie Chart
# -----------------------------

def render_time_pie(res: Results) -> None:
    total_minutes = float(res.minutes_worked_raw)

    units_minutes = float(res.units_billed) * 15.0
    non_billable_minutes = float(res.non_billable_total)
    travel_minutes = float(res.travel_total)
    documentation_minutes = float(res.documentation_total)

    accounted_minutes = units_minutes + non_billable_minutes + travel_minutes + documentation_minutes
    unaccounted_minutes = max(0.0, total_minutes - accounted_minutes)

    df = pd.DataFrame(
        {
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
                "#16a34a",
                "#f97316",
                "#2563eb",
                "#eab308",
                "#dc2626",
            ],
        }
    )

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
        textfont=dict(size=16),
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
# Results Display
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
# Streamlit UI (Hidden Math)
# -----------------------------

st.markdown("---")
st.subheader("Need help running this app?")

need_howto = st.radio(
    "Would you like the How-To document?",
    ["No", "Yes"],
    horizontal=True,
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
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0


def do_reset() -> None:
    st.session_state["reset_counter"] += 1
    st.session_state["last_result"] = None
    st.session_state["last_audit_payload"] = None
    st.session_state["last_error"] = None


k = st.session_state["reset_counter"]
hours_key = f"hours_{k}"
file_key = f"uploaded_file_{k}"

hours = st.text_input(
    "Please insert Hours Worked",
    placeholder="Example: 148.13",
    key=hours_key,
)

st.markdown("", unsafe_allow_html=True)

uploaded = st.file_uploader(
    'Upload the Excel (.xlsx) file exported from "Staff Service Detail Report" from SmartCare',
    type=["xlsx"],
    key=file_key,
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
    st.rerun()

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])

if st.session_state["last_result"] is not None:
    print_final(st.session_state["last_result"])

if st.session_state["last_audit_payload"] is not None:
    st.download_button(
        "Download Audit JSON (internal math, not displayed)",
        data=json.dumps(st.session_state["last_audit_payload"], indent=2).encode("utf-8"),
        file_name="productivity_audit.json",
        mime="application/json",
    )

st.write("")
st.info("Ready for the next staff member? Click **Run Another Staff Member**.")
