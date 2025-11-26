import streamlit as st
import pandas as pd

from db import fetch_runs
from utils import convert_distance_for_display
from metrics import (
    prepare_metrics_df,
    calculate_prs,
    pace_to_float,
)


def render():
    st.title("📜 Training Feed")

    df = fetch_runs()
    if df.empty:
        st.info("No runs yet — log one or import from Garmin.")
        return

    df = df.sort_values("date", ascending=False)

    metrics = prepare_metrics_df(df)
    prs = calculate_prs(metrics)

    for _, row in df.iterrows():
        rt = row["run_type"] or "Other"
        distance, dunit = convert_distance_for_display(row["distance"])
        pace = row["avg_pace"] or "—"
        hr = row["avg_hr"] or "—"
        elev = row["elevation_gain"] or "—"

        # PR badge detection
        badges = []
        if prs:
            if row["distance"] == prs.get("longest_distance"):
                badges.append("🔥 Longest Run PR")
            if pace_to_float(row["avg_pace"]) == prs.get("fastest_pace"):
                badges.append("⚡ Fastest Pace PR")

        badge_text = " | ".join(badges) if badges else ""

        st.markdown(
            f"""
            <div class='card feed-card'>
                <div class='feed-header-line'>
                    <span><strong>{row['date']}</strong></span>
                    <span class='tag tag-{rt}'>{rt}</span>
                </div>

                <div class='feed-main-metrics'>
                    <span class='big-distance'>{distance:.2f} {dunit}</span>
                    <span class='muted'>Pace: {pace}</span>
                    <span class='muted'>HR: {hr}</span>
                    <span class='muted'>Elev: {elev} ft</span>
                </div>

                <span class='muted'>Effort: {row['effort'] or "—"} / 10</span><br>
                <span class='muted'>Felt: {row['how_felt'] or "—"}</span>

                {f"<div class='pr-badge'>{badge_text}</div>" if badge_text else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Edit button for this run
        if st.button(f"Edit Run {row['id']}", key=f"edit_feed_{row['id']}"):
            st.session_state["page"] = "Edit Run"
            st.session_state["edit_run_id"] = int(row["id"])
            st.experimental_rerun()
