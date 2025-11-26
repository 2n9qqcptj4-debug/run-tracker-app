import streamlit as st
import pandas as pd
import altair as alt

from db import fetch_runs
from utils import convert_distance_for_display
from metrics import prepare_metrics_df, pace_to_float, minutes_to_hms


def _run_display_name(row):
    """Helper: Format run label in dropdown."""
    date = row['date']
    dist = row['distance']
    rt = row['run_type'] or "Run"
    try:
        dist_f = f"{float(dist):.2f}"
    except:
        dist_f = "—"
    return f"{date} — {dist_f} mi {rt}"


def _compare_card(label, val1, val2):
    """Helper to produce a nice comparison row."""
    if val1 is None and val2 is None:
        return f"**{label}:** — / —"

    return f"**{label}:** {val1 if val1 not in (None, '') else '—'} | {val2 if val2 not in (None, '') else '—'}"


def render():
    st.title("📊 Compare Runs")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs first.")
        return

    df = df.sort_values("date", ascending=False)
    metrics = prepare_metrics_df(df)

    # -------------------------------------------
    # Run selection
    # -------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Select Two Runs to Compare")

    run_options = { _run_display_name(r): int(r["id"]) for _, r in df.iterrows() }

    colA, colB = st.columns(2)

    with colA:
        run1_name = st.selectbox("Run 1", list(run_options.keys()))
        run1_id = run_options[run1_name]

    with colB:
        run2_name = st.selectbox("Run 2", list(run_options.keys()))
        run2_id = run_options[run2_name]

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------
    # Fetch run details
    # -------------------------------------------
    run1 = df[df["id"] == run1_id].iloc[0]
    run2 = df[df["id"] == run2_id].iloc[0]

    # -------------------------------------------
    # Display comparison cards
    # -------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Run Details")

    dist1, unit = convert_distance_for_display(run1["distance"])
    dist2, _ = convert_distance_for_display(run2["distance"])

    pace1 = run1["avg_pace"] or "—"
    pace2 = run2["avg_pace"] or "—"

    elev1 = run1["elevation_gain"] or "—"
    elev2 = run2["elevation_gain"] or "—"

    hr1 = run1["avg_hr"] or "—"
    hr2 = run2["avg_hr"] or "—"

    cad1 = run1["cadence"] or "—"
    cad2 = run2["cadence"] or "—"

    eff1 = run1.get("effort", None)
    eff2 = run2.get("effort", None)

    # Show summary rows
    st.markdown(_compare_card("Distance", f"{dist1:.2f} {unit}", f"{dist2:.2f} {unit}"))
    st.markdown(_compare_card("Average Pace", pace1, pace2))
    st.markdown(_compare_card("Avg HR", hr1, hr2))
    st.markdown(_compare_card("Cadence", cad1, cad2))
    st.markdown(_compare_card("Elevation Gain", f"{elev1} ft", f"{elev2} ft"))
    st.markdown(_compare_card("Effort", eff1, eff2))

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------
    # Pace / HR / Cadence trend mini-charts
    # -------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Trend Comparison")

    metrics_small = metrics[metrics["id"].isin([run1_id, run2_id])].copy()

    # Convert pace to numeric
    metrics_small["pace_num"] = metrics_small["avg_pace"].apply(pace_to_float)

    # --- Pace chart ---
    pace_df = metrics_small.dropna(subset=["pace_num"])
    if not pace_df.empty:
        pace_chart = (
            alt.Chart(pace_df)
            .mark_bar()
            .encode(
                x=alt.X("id:O", title="Run ID"),
                y=alt.Y("pace_num:Q", scale=alt.Scale(reverse=True), title="Pace (min/mi)"),
                color="id:N",
                tooltip=["date", "avg_pace", "distance", "run_type"],
            )
        )
        st.altair_chart(pace_chart, use_container_width=True)

    # --- HR chart ---
    hr_df = metrics_small.dropna(subset=["avg_hr"])
    if not hr_df.empty:
        hr_chart = (
            alt.Chart(hr_df)
            .mark_bar()
            .encode(
                x=alt.X("id:O", title="Run ID"),
                y=alt.Y("avg_hr:Q", title="Average HR"),
                color="id:N",
                tooltip=["date", "avg_hr", "distance"],
            )
        )
        st.altair_chart(hr_chart, use_container_width=True)

    # --- Cadence chart ---
    cad_df = metrics_small.dropna(subset=["cadence"])
    if not cad_df.empty:
        cad_chart = (
            alt.Chart(cad_df)
            .mark_bar()
            .encode(
                x=alt.X("id:O", title="Run ID"),
                y=alt.Y("cadence:Q", title="Cadence (spm)"),
                color="id:N",
                tooltip=["date", "cadence", "distance"],
            )
        )
        st.altair_chart(cad_chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
