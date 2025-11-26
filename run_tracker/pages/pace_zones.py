import streamlit as st
import pandas as pd

from db import fetch_runs
from metrics import (
    prepare_metrics_df,
    calculate_prs,
    estimate_threshold_pace,
    estimate_easy_pace,
    estimate_tempo_pace,
    estimate_interval_pace,
    estimate_rep_pace,
    calculate_hr_zones,
)
from utils import convert_distance_for_display


def render():
    st.title("📏 Pace & HR Zones")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs to generate pace zones.")
        return

    metrics = prepare_metrics_df(df)
    prs = calculate_prs(metrics)

    # ----------------------------------------
    # Extract best paces from PRs
    # ----------------------------------------
    best_pace = prs.get("fastest_pace")
    if best_pace is None:
        st.info("Need more runs with distance + duration to calculate pace zones.")
        return

    threshold_pace = estimate_threshold_pace(prs)
    easy_min, easy_max = estimate_easy_pace(best_pace)
    tempo = estimate_tempo_pace(threshold_pace) if threshold_pace else None
    interval = estimate_interval_pace(best_pace)
    rep = estimate_rep_pace(best_pace)

    hr_max = st.session_state.get("hr_max", 190)
    hr_zones = calculate_hr_zones(hr_max)

    # ----------------------------------------
    # PACE ZONE TABLE
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pace Zones (min/mi)")

    pace_data = []

    def add_zone(label, pr):
        if pr and pr[0] and pr[1]:
            pace_data.append(
                {
                    "Zone": label,
                    "Min Pace (min/mi)": f"{pr[0]:.2f}",
                    "Max Pace (min/mi)": f"{pr[1]:.2f}",
                }
            )

    # Easy, recovery, aerobic
    if easy_min and easy_max:
        add_zone("Recovery", (easy_min + 0.4, easy_max + 0.6))
        add_zone("Easy", (easy_min, easy_max))
        add_zone("Aerobic Endurance", (easy_min - 0.3, easy_min + 0.3))

    # Tempo / threshold / VO2 / rep
    if tempo:
        add_zone("Tempo", tempo)
    if threshold_pace:
        add_zone("Threshold", (threshold_pace * 0.98, threshold_pace * 1.02))
    if interval:
        add_zone("Interval / VO₂ Max", interval)
    if rep:
        add_zone("Repetition", rep)

    if pace_data:
        st.table(pd.DataFrame(pace_data))
    else:
        st.info("Not enough data to calculate pace zones.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # HEART RATE ZONES
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Heart Rate Zones")

    hr_data = []
    for zone, (low, high) in hr_zones.items():
        hr_data.append(
            {"Zone": zone, "Low (bpm)": int(low), "High (bpm)": int(high)}
        )

    st.table(pd.DataFrame(hr_data))
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # AI ZONE WORKOUT GENERATOR (placeholder)
    # Will activate when AI module is added
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Zone Workout Generator")

    zone_choice = st.selectbox(
        "Select a Training Zone",
        [
            "Recovery",
            "Easy",
            "Aerobic Endurance",
            "Tempo",
            "Threshold",
            "Interval / VO₂ Max",
            "Repetition",
        ],
    )

    st.info(
        "AI workout generation will be enabled once the AI module is added in the next steps."
    )

    st.markdown("</div>", unsafe_allow_html=True)
