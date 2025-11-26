import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta, datetime

from db import fetch_runs, fetch_shoes
from utils import convert_distance_for_display
from metrics import (
    prepare_metrics_df,
    compute_efficiency_score,
    calculate_prs,
    compute_streaks,
    add_hr_zones,
    compute_daily_load,
    compute_fitness_fatigue,
    generate_race_prediction_series,
)


def render():
    st.title("📊 Dashboard")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs to view your dashboard.")
        return

    # ----------------------------------------
    # PREP METRICS
    # ----------------------------------------
    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    metrics = add_hr_zones(metrics, hr_max=st.session_state.get("hr_max", 190))

    daily = compute_daily_load(metrics)
    load_df = compute_fitness_fatigue(daily)
    prs = calculate_prs(metrics)

    # ----------------------------------------
    # SUMMARY METRICS
    # ----------------------------------------
    total_miles = metrics["distance"].sum(skipna=True)
    last7 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7_miles = last7["distance"].sum(skipna=True)
    avg_hr = metrics["avg_hr"].mean(skipna=True)
    
    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)
    current_streak, longest_streak = compute_streaks(metrics)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    col3.metric("Avg HR", f"{avg_hr:.1f}" if not pd.isna(avg_hr) else "—")
    col4.metric("Current Streak", f"{current_streak} days")
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # WEEKLY DISTANCE CHART
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weekly Distance")

    dfw = metrics.copy()
    dfw["week"] = dfw["date_dt"].dt.isocalendar().week
    weekly = dfw.groupby("week")["distance"].sum().reset_index()

    if not weekly.empty:
        weekly["display"], _ = zip(*weekly["distance"].apply(convert_distance_for_display))
        chart = (
            alt.Chart(weekly)
            .mark_bar()
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("display:Q", title=f"Distance ({unit})"),
                tooltip=["week", "display"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data for weekly chart.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # PACE TREND
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pace Trend (min/mi)")

    p = metrics.dropna(subset=["pace_min_per_mile"])
    if not p.empty:
        chart = (
            alt.Chart(p)
            .mark_line(point=True)
            .encode(
                x="date_dt:T",
                y=alt.Y("pace_min_per_mile:Q", scale=alt.Scale(reverse=True)),
                tooltip=["date", "distance", "avg_pace"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Log distance + duration to view pace trend.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # VO2 + Efficiency Trend
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("VO₂ Max & Efficiency Over Time")

    eff = metrics.dropna(subset=["efficiency_score"])
    vo2 = metrics.dropna(subset=["vo2max"])

    if not eff.empty or not vo2.empty:
        eff_plot = eff[["date_dt", "efficiency_score"]].copy()
        eff_plot["metric"] = "Efficiency"
        eff_plot = eff_plot.rename(columns={"efficiency_score": "value"})

        vo2_plot = vo2[["date_dt", "vo2max"]].copy()
        vo2_plot["metric"] = "VO₂ Max"
        vo2_plot = vo2_plot.rename(columns={"vo2max": "value"})

        combo = pd.concat([eff_plot, vo2_plot], ignore_index=True)

        chart = (
            alt.Chart(combo)
            .mark_line(point=True)
            .encode(
                x="date_dt:T",
                y="value:Q",
                color="metric:N",
                tooltip=["date_dt", "metric", "value"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough HR or eff data.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # HR + HR ZONES
    # ----------------------------------------
    colA, colB = st.columns(2)

    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Average HR Over Time")

        hr_df = metrics.dropna(subset=["avg_hr"])
        if not hr_df.empty:
            chart = (
                alt.Chart(hr_df)
                .mark_line(point=True)
                .encode(x="date_dt:T", y="avg_hr:Q")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No HR data.")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("HR Zone Distribution")

        z = metrics.dropna(subset=["hr_zone"])
        zone_df = z.groupby("hr_zone")["duration_minutes"].sum().reset_index()

        if not zone_df.empty:
            chart = (
                alt.Chart(zone_df)
                .mark_arc()
                .encode(
                    theta="duration_minutes:Q",
                    color="hr_zone:N",
                    tooltip=["hr_zone", "duration_minutes"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No HR zone data.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # CADENCE
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Cadence Trend (spm)")

    cad = metrics.dropna(subset=["cadence"])
    if not cad.empty:
        chart = (
            alt.Chart(cad)
            .mark_line(point=True)
            .encode(x="date_dt:T", y="cadence:Q")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No cadence data recorded.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # TRAINING LOAD (CTL / ATL / TSB)
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Load: CTL, ATL, TSB")

    if not load_df.empty:
        melted = load_df.melt(id_vars="date_dt", value_vars=["CTL", "ATL", "TSB"])
        chart = (
            alt.Chart(melted)
            .mark_line()
            .encode(x="date_dt:T", y="value:Q", color="variable:N")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough load data yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # RACE PREDICTION TREND
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏁 Race Prediction Trend")

    pred_df = generate_race_prediction_series(metrics)
    if pred_df.empty:
        st.info("Not enough data for race prediction.")
    else:
        melt = pred_df.melt(id_vars="date", var_name="race", value_name="minutes")

        chart = (
            alt.Chart(melt)
            .mark_line(point=True)
            .encode(
                x="date:T",
                y="minutes:Q",
                color="race:N",
                tooltip=["date", "race", "minutes"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # HABITS (Sleep + Stress)
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Habits & Recovery (Last 21 Days)")

    recent = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=21)]

    if recent.empty:
        st.info("Not enough recent data.")
    else:
        avg_sleep = recent["sleep_hours"].mean(skipna=True)
        avg_stress = recent["stress"].mean(skipna=True)

        colH1, colH2 = st.columns(2)
        colH1.metric("Avg Sleep", f"{avg_sleep:.1f} hrs" if not pd.isna(avg_sleep) else "—")
        colH2.metric("Avg Stress", f"{avg_stress:.1f}/5" if not pd.isna(avg_stress) else "—")

        sleep_df = recent.dropna(subset=["sleep_hours"])
        if not sleep_df.empty:
            chart = alt.Chart(sleep_df).mark_line(point=True).encode(
                x="date_dt:T", y="sleep_hours:Q"
            )
            st.altair_chart(chart, use_container_width=True)

        stress_df = recent.dropna(subset=["stress"])
        if not stress_df.empty:
            chart = alt.Chart(stress_df).mark_line(point=True).encode(
                x="date_dt:T", y="stress:Q"
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # INJURY RISK SNAPSHOT
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Injury Risk Snapshot")

    last14 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=14)]
    if last14.empty:
        st.info("Not enough recent data.")
    else:
        last7 = last14[last14["date_dt"] >= datetime.now() - timedelta(days=7)]
        prev7 = last14[
            (last14["date_dt"] < datetime.now() - timedelta(days=7))
            & (last14["date_dt"] >= datetime.now() - timedelta(days=14))
        ]

        last7_mi = last7["distance"].sum(skipna=True)
        prev7_mi = prev7["distance"].sum(skipna=True)
        load_ratio = (last7_mi / prev7_mi) if prev7_mi > 0 else 1.0

        avg_effort = last7["effort"].mean(skipna=True)
        avg_sleep7 = last7["sleep_hours"].mean(skipna=True)
        avg_stress7 = last7["stress"].mean(skipna=True)

        colR1, colR2, colR3 = st.columns(3)
        colR1.metric("Last 7d Miles", f"{last7_mi:.1f}")
        colR2.metric("Load vs Prev Week", f"{load_ratio:.2f}x" if prev7_mi > 0 else "—")
        colR3.metric("Avg Effort", f"{avg_effort:.1f}" if not pd.isna(avg_effort) else "—")

        # Basic scoring model
        risk_score = 0
        if load_ratio > 1.3: risk_score += 2
        if avg_effort and avg_effort > 7: risk_score += 2
        if avg_sleep7 and avg_sleep7 < 6: risk_score += 2
        if avg_stress7 and avg_stress7 > 3: risk_score += 1

        risk_label = (
            "High" if risk_score >= 5 else
            "Moderate" if risk_score >= 3 else
            "Low"
        )

        st.write(f"**Estimated Injury Risk:** {risk_label} (score {risk_score}/7)")

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # SHOE MILEAGE
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👟 Shoe Mileage")

    shoes_df = fetch_shoes(include_retired=True)
    if shoes_df.empty:
        st.info("No shoes tracked yet.")
    else:
        for _, shoe in shoes_df.iterrows():
            shoe_runs = df[df["shoe_id"] == shoe["id"]]
            mileage = shoe_runs["distance"].sum() if not shoe_runs.empty else 0.0
            status = "Active" if shoe["retired"] == 0 else "Retired"

            warn = ""
            if mileage > 350:
                warn = "⚠️ Nearing end of life"
            elif mileage > 300:
                warn = "⚠️ Getting worn"

            st.markdown(
                f"""
                <div class='card'>
                    <strong>{shoe['name']} ({shoe['brand']})</strong> — {status}<br>
                    Total Miles: {mileage:.1f}<br>
                    {warn}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------
    # PERSONAL RECORDS
    # ----------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Personal Records")

    if not prs:
        st.info("Log more runs to generate PRs.")
    else:
        for key, val in prs.items():
            label = key.replace("_", " ").title()
            if "fastest" in key:
                st.write(f"⚡ **{label}:** {val:.2f} min")
            else:
                st.write(f"🔥 **{label}:** {val:.2f} mi")

    st.markdown("</div>", unsafe_allow_html=True)
