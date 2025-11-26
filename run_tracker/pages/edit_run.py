import streamlit as st
from datetime import datetime
import pandas as pd

from db import (
    fetch_runs,
    fetch_shoes,
    update_run,
    delete_run,
)

from metrics import (
    minutes_to_hms,
    duration_to_minutes,
)


def render():
    st.title("✏️ Edit Run")

    run_id = st.session_state.get("edit_run_id")
    if run_id is None:
        st.error("No run selected to edit.")
        return

    # ------- FETCH THE RUN -------
    df = fetch_runs()
    row = df[df["id"] == run_id]
    if row.empty:
        st.error("Run not found.")
        return

    row = row.iloc[0]

    # ------- SHOES -------
    shoes_df = fetch_shoes()
    shoe_options = ["None"] + [
        f"{r['name']} ({r['brand']})"
        for _, r in shoes_df.iterrows()
    ]

    current_shoe = "None"
    if row["shoe_id"] and not pd.isna(row["shoe_id"]) and not shoes_df.empty:
        if row["shoe_id"] in shoes_df["id"].values:
            srow = shoes_df[shoes_df["id"] == row["shoe_id"]].iloc[0]
            current_shoe = f"{srow['name']} ({srow['brand']})"

    # ------- FORM -------
    with st.form("edit_run"):
        date_val = st.date_input("Date", datetime.fromisoformat(row["date"]))

        run_types = ["Easy", "LongRun", "Tempo", "Intervals", "Recovery", "Race", "Other"]
        try:
            default_rt_index = run_types.index(row["run_type"])
        except Exception:
            default_rt_index = 0
        run_type = st.selectbox("Run Type", run_types, index=default_rt_index)

        distance_val = 0.0 if pd.isna(row["distance"]) else float(row["distance"])
        distance = st.number_input("Distance (mi)", min_value=0.0, step=0.01, value=distance_val)

        # duration
        duration_minutes = None if pd.isna(row["duration_minutes"]) else float(row["duration_minutes"])
        duration_str = minutes_to_hms(duration_minutes)
        duration_str = st.text_input("Duration (MM:SS or HH:MM:SS)", value=duration_str)

        avg_pace = st.text_input("Average Pace (MM:SS)", value=row["avg_pace"] or "")
        splits = st.text_area("Splits", value=row["splits"] or "")

        # HR + metrics
        avg_hr_val = 0 if pd.isna(row["avg_hr"]) else int(row["avg_hr"])
        max_hr_val = 0 if pd.isna(row["max_hr"]) else int(row["max_hr"])
        cadence_val = 0 if pd.isna(row["cadence"]) else int(row["cadence"])
        elev_val = 0 if pd.isna(row["elevation_gain"]) else int(row["elevation_gain"])
        effort_val = 5 if pd.isna(row["effort"]) else int(row["effort"])
        sleep_val = 0.0 if pd.isna(row["sleep_hours"]) else float(row["sleep_hours"])
        stress_val = 1 if pd.isna(row["stress"]) else int(row["stress"])
        vo2_val = 0.0 if pd.isna(row["vo2max"]) else float(row["vo2max"])
        hrv_val = 0 if pd.isna(row["hrv"]) else int(row["hrv"])

        avg_hr = st.number_input("Average HR", min_value=0, max_value=240, value=avg_hr_val)
        max_hr = st.number_input("Max HR", min_value=0, max_value=240, value=max_hr_val)
        cadence = st.number_input("Cadence (spm)", min_value=0, max_value=300, value=cadence_val)
        elevation_gain = st.number_input("Elevation Gain (ft)", min_value=0, value=elev_val)

        effort = st.slider("Effort (1–10)", 1, 10, value=effort_val)
        terrain = st.text_input("Terrain", value=row["terrain"] or "")
        weather = st.text_input("Weather", value=row["weather"] or "")
        how_felt = st.text_input("How I Felt", value=row["how_felt"] or "")
        pain = st.text_input("Any Pain or Tightness", value=row["pain"] or "")

        sleep = st.number_input("Sleep (hrs)", min_value=0.0, step=0.1, value=sleep_val)
        stress = st.slider("Stress (1–5)", 1, 5, value=stress_val)
        nutrition = st.text_area("Nutrition / Hydration Notes", value=row["nutrition_notes"] or "")

        vo2 = st.number_input("VO2 Max", min_value=0.0, step=0.1, value=vo2_val)
        hrv = st.number_input("HRV", min_value=0, max_value=300, value=hrv_val)

        shoe_choice = st.selectbox("Shoe", shoe_options, index=shoe_options.index(current_shoe))

        save_btn = st.form_submit_button("Save Changes")
        delete_btn = st.form_submit_button("Delete Run")

        # ---------- SAVE ----------
        if save_btn:
            minutes = duration_to_minutes(duration_str)
            if minutes is None:
                st.error("Invalid duration format.")
                return

            shoe_id = None
            if shoe_choice != "None" and not shoes_df.empty:
                idx = shoe_options.index(shoe_choice) - 1
                shoe_id = int(shoes_df.iloc[idx]["id"])

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

            update_run(run_id, data)
            st.success("Run updated! ✅")

        # ---------- DELETE ----------
        if delete_btn:
            delete_run(run_id)
            st.warning("Run deleted.")
            st.session_state["page"] = "Feed"
            st.session_state["edit_run_id"] = None
            st.experimental_rerun()
