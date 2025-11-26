import streamlit as st
from datetime import datetime
import pandas as pd

from db import fetch_runs, fetch_shoes, insert_run
from metrics import (
    duration_to_minutes,
    prepare_metrics_df,
    calculate_prs,
    detect_pr_improvements,
)
    

def render():
    st.title("📝 Log a Run")

    # ------ SHOE DROPDOWN -------
    shoes_df = fetch_shoes()
    shoe_options = ["None"] + [
        f"{row['name']} ({row['brand']})"
        for _, row in shoes_df.iterrows()
    ]

    # ------ FORM START -------
    with st.form("log_run"):
        date_val = st.date_input("Date", datetime.today().date())
        run_type = st.selectbox(
            "Run Type",
            ["Easy", "LongRun", "Tempo", "Intervals", "Recovery", "Race", "Other"],
        )

        distance = st.number_input("Distance (mi)", min_value=0.0, step=0.01)
        duration_str = st.text_input("Duration (MM:SS or HH:MM:SS)")
        avg_pace = st.text_input("Average Pace (optional, MM:SS)")
        splits = st.text_area("Splits (optional)")

        avg_hr = st.number_input("Average HR", min_value=0, max_value=240, value=0)
        max_hr = st.number_input("Max HR", min_value=0, max_value=240, value=0)
        cadence = st.number_input("Cadence (spm)", min_value=0, max_value=300, value=0)
        elevation_gain = st.number_input("Elevation Gain (ft)", min_value=0, value=0)

        effort = st.slider("Effort (1–10)", 1, 10, 5)
        terrain = st.text_input("Terrain")
        weather = st.text_input("Weather")
        how_felt = st.text_input("How I Felt")
        pain = st.text_input("Any Pain or Tightness")

        sleep = st.number_input("Sleep (hrs)", min_value=0.0, step=0.1)
        stress = st.slider("Stress (1–5)", 1, 5, 1)
        nutrition = st.text_area("Nutrition / Hydration Notes")

        vo2 = st.number_input("VO2 Max", min_value=0.0, step=0.1)
        hrv = st.number_input("HRV", min_value=0, max_value=300, value=0)

        shoe_choice = st.selectbox("Shoe", shoe_options)

        submitted = st.form_submit_button("Save Run")

        # ------ ON SUBMIT -------
        if submitted:
            # PR comparison BEFORE saving
            df_before = fetch_runs()
            metrics_before = prepare_metrics_df(df_before) if not df_before.empty else df_before
            prs_before = calculate_prs(metrics_before) if not df_before.empty else {}

            # convert duration
            minutes = duration_to_minutes(duration_str)
            if minutes is None:
                st.error("Invalid duration format. Use MM:SS or HH:MM:SS.")
                return

            # Shoe ID
            shoe_id = None
            if shoe_choice != "None" and not shoes_df.empty:
                idx = shoe_options.index(shoe_choice) - 1
                shoe_id = int(shoes_df.iloc[idx]["id"])

            # DB row data
            data = {
                "date": date_val.isoformat(),
                "run_type": run_type,
                "distance": distance,
                "duration_minutes": minutes,
                "avg_pace": avg_pace,
                "splits": splits,
                "avg_hr": avg_hr or None,
                "max_hr": max_hr or None,
                "hr_by_segment": "",
                "cadence": cadence or None,
                "elevation_gain": elevation_gain or None,
                "effort": effort,
                "terrain": terrain,
                "weather": weather,
                "how_felt": how_felt,
                "pain": pain,
                "sleep_hours": sleep or None,
                "stress": stress or None,
                "nutrition_notes": nutrition,
                "vo2max": vo2 or None,
                "hrv": hrv or None,
                "shoe_id": shoe_id,
            }

            insert_run(data)
            st.success("Run saved! ✅")

            # Recompute PRs AFTER saving
            df_after = fetch_runs()
            metrics_after = prepare_metrics_df(df_after)
            prs_after = calculate_prs(metrics_after)

            improved = detect_pr_improvements(prs_before, prs_after)
            if improved:
                st.success("🎉 New PRs: " + " | ".join(improved))
