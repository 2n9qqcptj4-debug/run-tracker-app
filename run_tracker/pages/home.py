import streamlit as st
import pandas as pd

from db import fetch_runs
from utils import convert_distance_for_display
from metrics import (
    prepare_metrics_df,
    compute_efficiency_score,
    calculate_prs,
    compute_streaks,
    pace_to_float,
)


def render():
    st.title("🏠 Home")

    df = fetch_runs()
    if df.empty:
        st.info("You have no runs logged yet. Start by logging a run or importing from Garmin.")
        return

    # ---- Prepare data ----
    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    prs = calculate_prs(metrics)

    total_miles = metrics["distance"].sum(skipna=True)
    last7 = metrics[metrics["date_dt"] >= pd.Timestamp.today() - pd.Timedelta(days=7)]
    last7_miles = last7["distance"].sum(skipna=True)
    avg_hr = metrics["avg_hr"].mean(skipna=True)

    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)

    current_streak, longest_streak = compute_streaks(metrics)

    # ---- Summary metrics ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    col3.metric("Avg HR", f"{avg_hr:.1f}" if not pd.isna(avg_hr) else "—")
    col4.metric("Current Streak", f"{current_streak} days")

    st.markdown("<h4>Recent Runs</h4>", unsafe_allow_html=True)

    # ---- Display last 5 runs ----
    recent = df.tail(5).iloc[::-1]

    for _, row in recent.iterrows():
        rt = row["run_type"] or "Other"
        distance, dunit = convert_distance_for_display(row["distance"])
        pace = row["avg_pace"] or "—"
        hr = row["avg_hr"] or "—"
        elev = row["elevation_gain"] or "—"

        # PR badges
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

                <span class='muted'>Effort: {row['effort'] or '—'} / 10</span><br>
                <span class='muted'>Felt: {row['how_felt'] or '—'}</span>

                {f"<div class='pr-badge'>{badge_text}</div>" if badge_text else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Edit button
        if st.button(f"Edit Run {row['id']}", key=f"edit_home_{row['id']}"):
            st.session_state["page"] = "Edit Run"
            st.session_state["edit_run_id"] = int(row["id"])
            st.experimental_rerun()

