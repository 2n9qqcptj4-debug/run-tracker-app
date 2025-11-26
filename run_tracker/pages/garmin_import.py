import streamlit as st
import pandas as pd
from datetime import datetime

from db import fetch_runs, insert_run
from metrics import (
    duration_to_minutes,
    prepare_metrics_df,
    calculate_prs,
    detect_pr_improvements,
)


# --------------------------
# GARMIN CSV PARSER
# --------------------------

def build_run_from_garmin_df(df: pd.DataFrame):
    if df.empty:
        return None, "CSV appears to be empty."

    row = df.iloc[0]
    cols = list(df.columns)

    # Helper to safely extract values from multiple possible Garmin headers
    def get_val(*names):
        for n in names:
            if n in cols and pd.notna(row[n]):
                return row[n]
        return None

    # ---- Date ----
    date_val = get_val("Start Time", "Start", "Date", "Timestamp")
    date_iso = datetime.today().date().isoformat()
    if date_val is not None:
        dt = pd.to_datetime(str(date_val), errors="coerce")
        if pd.notna(dt):
            date_iso = dt.date().isoformat()

    # ---- Distance ----
    distance = None
    for c in cols:
        if "distance" in c.lower():
            raw = row[c]
            try:
                val = float(str(raw).replace(",", ""))
                distance = val
            except Exception:
                pass
            break

    # ---- Duration ----
    duration_minutes = None
    for c in cols:
        if any(x in c.lower() for x in ["elapsed", "duration", "time"]):
            raw = row[c]
            s = str(raw)
            if ":" in s:
                duration_minutes = duration_to_minutes(s)
            else:
                try:
                    num = float(s)
                    # Some Garmin exports store seconds; some store minutes.
                    duration_minutes = num / 60.0 if num > 200 else num
                except Exception:
                    pass
            break

    # ---- Pace ----
    avg_pace = None
    for c in cols:
        if "pace" in c.lower():
            avg_pace = str(row[c])
            break

    # ---- HR ----
    avg_hr_raw = get_val("Average Heart Rate", "Avg HR", "Average HR")
    try:
        avg_hr = int(float(avg_hr_raw)) if avg_hr_raw is not None else None
    except Exception:
        avg_hr = None

    max_hr_raw = get_val("Maximum Heart Rate", "Max HR")
    try:
        max_hr = int(float(max_hr_raw)) if max_hr_raw is not None else None
    except Exception:
        max_hr = None

    # ---- Cadence ----
    cadence_raw = get_val("Average Run Cadence", "Avg Run Cadence", "Run Cadence")
    try:
        cadence = int(float(cadence_raw)) if cadence_raw is not None else None
    except Exception:
        cadence = None

    # ---- Elevation ----
    elev_raw = get_val("Elevation Gain", "Total Ascent", "Ascent")
    try:
        elevation_gain = int(float(elev_raw)) if elev_raw is not None else None
    except Exception:
        elevation_gain = None

    # ---- VO2 Max ----
    vo2_raw = get_val("VO2 Max", "VO2Max")
    try:
        vo2 = float(vo2_raw) if vo2_raw is not None else None
    except Exception:
        vo2 = None

    # ---- HRV ----
    hrv_raw = get_val("HRV", "HRV (ms)")
    try:
        hrv = int(hrv_raw) if hrv_raw is not None else None
    except Exception:
        hrv = None

    # ---- Build run dictionary ----
    data = {
        "date": date_iso,
        "run_type": "Easy",
        "distance": distance,
        "duration_minutes": duration_minutes,
        "avg_pace": avg_pace,
        "splits": "",
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "hr_by_segment": "",
        "cadence": cadence,
        "elevation_gain": elevation_gain,
        "effort": 5,
        "terrain": "",
        "weather": "",
        "how_felt": "",
        "pain": "",
        "sleep_hours": None,
        "stress": None,
        "nutrition_notes": "",
        "vo2max": vo2,
        "hrv": hrv,
        "shoe_id": None,
    }

    # Warnings
    warnings = []
    if distance is None:
        warnings.append("Distance not detected.")
    if duration_minutes is None:
        warnings.append("Duration not detected.")
    if avg_hr is None:
        warnings.append("Avg HR missing.")
    if cadence is None:
        warnings.append("Cadence missing.")

    msg = "Parsed CSV."
    if warnings:
        msg += " " + " ".join(warnings)

    return data, msg


# --------------------------
# PAGE RENDER
# --------------------------

def render():
    st.title("📥 Garmin CSV Import")

    uploaded = st.file_uploader("Upload Garmin CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)

        # parse CSV
        data, msg = build_run_from_garmin_df(df)
        st.info(msg)

        # import button
        if st.button("Import Run"):
            df_before = fetch_runs()
            metrics_before = prepare_metrics_df(df_before) if not df_before.empty else df_before
            prs_before = calculate_prs(metrics_before) if not df_before.empty else {}

            insert_run(data)
            st.success("Garmin run imported! ✅")

            df_after = fetch_runs()
            metrics_after = prepare_metrics_df(df_after)
            prs_after = calculate_prs(metrics_after)
            improved = detect_pr_improvements(prs_before, prs_after)

            if improved:
                st.success("🎉 New PRs: " + " | ".join(improved))

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
