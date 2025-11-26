# ===============================================
# SEGMENT 1 — IMPORTS, CONFIG, STYLES, DATABASE
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import ast
from datetime import datetime, timedelta
import altair as alt
from dateutil import parser

# ---------------------------
# OpenAI client
# ---------------------------
from openai import OpenAI
client = OpenAI()


# ======================================
# Streamlit Page Configuration
# ======================================
st.set_page_config(
    page_title="Run Tracker",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================
# Global CSS Styling — Strava-like dark UI
# ======================================
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0F172A !important;
        color: white !important;
    }
    .main {
        background-color: #0F172A;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    .card {
        background: #1E293B;
        padding: 18px 22px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .pr-banner {
        background: linear-gradient(90deg, #16A34A, #4ADE80);
        padding: 10px;
        border-radius: 8px;
        color: black;
        font-weight: 600;
        text-align: center;
        margin: 10px 0;
    }
    .pr-mini {
        color: #4ADE80;
        font-weight: 600;
        margin-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ======================================
# DATABASE INITIALIZATION
# ======================================

def init_db():
    conn = sqlite3.connect("runs.db")
    cur = conn.cursor()

    # Main run table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            run_type TEXT,
            distance REAL,
            duration TEXT,
            avg_pace TEXT,
            avg_hr REAL,
            max_hr REAL,
            elevation_gain REAL,
            cadence REAL,
            effort_rating REAL,
            terrain TEXT,
            weather TEXT,
            notes TEXT,
            splits TEXT,
            sleep REAL,
            stress REAL,
            vo2max REAL,
            hrv REAL
        );
        """
    )

    conn.commit()
    conn.close()


init_db()



# ======================================
# DATABASE HELPER FUNCTIONS
# ======================================

def insert_run(data):
    conn = sqlite3.connect("runs.db")
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO runs (
            date, run_type, distance, duration, avg_pace, avg_hr, max_hr, elevation_gain,
            cadence, effort_rating, terrain, weather, notes, splits, sleep, stress,
            vo2max, hrv
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["date"],
            data["run_type"],
            data["distance"],
            data["duration"],
            data["avg_pace"],
            data["avg_hr"],
            data["max_hr"],
            data["elevation_gain"],
            data["cadence"],
            data["effort_rating"],
            data["terrain"],
            data["weather"],
            data["notes"],
            data["splits"],
            data["sleep"],
            data["stress"],
            data["vo2max"],
            data["hrv"],
        ),
    )

    conn.commit()
    conn.close()



def fetch_runs():
    conn = sqlite3.connect("runs.db")
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY date ASC", conn)
    conn.close()
    return df



# ======================================
# UTILITY FUNCTIONS
# ======================================

def safe_json_load(val):
    """Safely loads JSON or returns original string."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except:
        try:
            return ast.literal_eval(val)
        except:
            return val


def convert_pace_to_float(pace_str):
    """Converts '9:30' -> 9.5 minutes per mile."""
    if not pace_str or isinstance(pace_str, float):
        return None
    try:
        minutes, seconds = pace_str.split(":")
        return float(minutes) + float(seconds) / 60
    except:
        return None


def duration_to_minutes(duration_str):
    """Converts '1:02:30' → minutes decimal."""
    if not duration_str:
        return None
    try:
        parts = duration_str.split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 60 + m + s / 60
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m + s / 60
    except:
        return None
    return None


# ======================================
# METRICS PREPARATION
# ======================================

def prepare_metrics_df(df):
    """Ensures consistent data types and computed metrics."""
    if df.empty:
        return df

    df = df.copy()

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    # Pace numeric
    df["pace_min_per_mile"] = df["avg_pace"].apply(convert_pace_to_float)

    # Duration minutes
    df["duration_min"] = df["duration"].apply(duration_to_minutes)

    # Efficiency score (simple)
    df["efficiency_score"] = None
    if "avg_hr" in df:
        df["efficiency_score"] = df.apply(
            lambda r: (r["distance"] / (r["avg_hr"] / 150)) if (r["avg_hr"] and r["avg_hr"] > 0) else None,
            axis=1,
        )

    # Clean splits
    df["splits"] = df["splits"].apply(safe_json_load)

    return df
# ===============================================
# SEGMENT 2 — HOME, FEED, LOG RUN, GARMIN IMPORT, AI UTILITIES
# ===============================================


# ------------------------------------------------------
# OpenAI AI Helper
# ------------------------------------------------------
def call_ai(prompt):
    """Wrapper for OpenAI completion."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.7,
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ AI Error: {e}"


# ------------------------------------------------------
# PR Detection
# ------------------------------------------------------
def calculate_prs(df):
    if df.empty:
        return {}

    prs = {}

    # Longest distance
    prs["longest_distance"] = df["distance"].max()

    # Highest weekly mileage
    df_copy = df.copy()
    df_copy["week"] = df_copy["date_dt"].dt.isocalendar().week
    weekly = df_copy.groupby("week")["distance"].sum()
    prs["highest_weekly_mileage"] = weekly.max() if not weekly.empty else 0

    # Highest monthly mileage
    df_copy["month"] = df_copy["date_dt"].dt.month
    monthly = df_copy.groupby("month")["distance"].sum()
    prs["highest_monthly_mileage"] = monthly.max() if not monthly.empty else 0

    # Fastest paces (min/mi → fastest = lowest)
    if "pace_min_per_mile" in df:
        prs["fastest_pace"] = df["pace_min_per_mile"].min(skipna=True)

    return prs



# ===============================================
# HOME PAGE
# ===============================================
def render_home_page():
    st.title("🏃‍♂️ Run Tracker")
    st.markdown(
        """
        Welcome to your personal running dashboard!  
        Track your runs, explore insights, analyze trends, and build AI-powered training plans.
        """
    )

    df = fetch_runs()
    if df.empty:
        st.info("Log your first run to get started!")
        return

    df = prepare_metrics_df(df)

    last_run = df.iloc[-1]
    st.subheader("Last Run Summary")
    st.markdown(
        f"""
        **{last_run['date']} — {last_run['run_type']}**  
        **Distance:** {last_run['distance']} mi  
        **Pace:** {last_run['avg_pace']}  
        **HR:** {last_run['avg_hr']}  
        """
    )



# ===============================================
# FEED PAGE (Recent Run List)
# ===============================================
def render_feed_page():
    st.title("📣 Feed")

    df = fetch_runs()
    if df.empty:
        st.info("No runs logged yet.")
        return

    df = prepare_metrics_df(df)

    st.markdown("### Recent Runs")

    for _, row in df.sort_values("date_dt", ascending=False).iterrows():
        st.markdown(
            f"""
            <div class='card'>
                <h4>{row['date']} — {row['run_type']}</h4>
                <strong>Distance:</strong> {row['distance']} mi<br>
                <strong>Pace:</strong> {row['avg_pace']}<br>
                <strong>Avg HR:</strong> {row['avg_hr']}<br>
            </div>
            """,
            unsafe_allow_html=True,
        )



# ===============================================
# LOG A RUN
# ===============================================
def render_log_run_page():
    st.title("📝 Log a Run")

    df = fetch_runs()
    df = prepare_metrics_df(df)

    with st.form("run_form"):
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date")
            run_type = st.selectbox("Run Type", ["Easy", "Tempo", "Intervals", "LongRun", "Recovery", "Race", "Other"])
            distance = st.number_input("Distance (mi)", min_value=0.0, step=0.01)
            duration = st.text_input("Duration (HH:MM:SS)")
            avg_pace = st.text_input("Avg Pace (MM:SS)")
            elevation = st.number_input("Elevation Gain (ft)", min_value=0.0, step=1.0)
        with col2:
            avg_hr = st.number_input("Avg HR", min_value=0, step=1)
            max_hr = st.number_input("Max HR", min_value=0, step=1)
            cadence = st.number_input("Cadence (spm)", min_value=0, step=1)
            effort = st.slider("Effort Rating (1-10)", 1, 10, 5)
            sleep = st.number_input("Sleep (hrs)", min_value=0.0, step=0.1)
            stress = st.slider("Stress (1–5)", 1, 5, 2)

        terrain = st.text_input("Terrain")
        weather = st.text_input("Weather")
        notes = st.text_area("Notes")
        splits = st.text_area("Splits (optional)")

        vo2max = st.number_input("VO2 Max", min_value=0.0, step=0.1)
        hrv = st.number_input("HRV", min_value=0.0, step=0.1)

        submitted = st.form_submit_button("Submit Run")

    if submitted:
        data = {
            "date": str(date),
            "run_type": run_type,
            "distance": distance,
            "duration": duration,
            "avg_pace": avg_pace,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "elevation_gain": elevation,
            "cadence": cadence,
            "effort_rating": effort,
            "terrain": terrain,
            "weather": weather,
            "notes": notes,
            "splits": splits,
            "sleep": sleep,
            "stress": stress,
            "vo2max": vo2max,
            "hrv": hrv,
        }

        insert_run(data)
        st.success("Run logged successfully!")



# ===============================================
# GARMIN IMPORT (CSV ONLY)
# ===============================================
def render_garmin_import_page():
    st.title("📥 Garmin Import")

    uploaded = st.file_uploader("Upload Garmin CSV Export", type=["csv"])

    if not uploaded:
        return

    df = pd.read_csv(uploaded)

    st.write("Preview:", df.head())

    if st.button("Import Runs"):
        imported = 0
        for _, row in df.iterrows():
            try:
                date = row.get("date") or row.get("Date")
                date = pd.to_datetime(date).date()

                data = {
                    "date": str(date),
                    "run_type": row.get("type", "Run"),
                    "distance": float(row.get("distance", 0)),
                    "duration": row.get("duration", ""),
                    "avg_pace": row.get("pace", ""),
                    "avg_hr": row.get("avg_hr", None),
                    "max_hr": row.get("max_hr", None),
                    "elevation_gain": row.get("elev", 0),
                    "cadence": row.get("cadence", None),
                    "effort_rating": row.get("effort", 5),
                    "terrain": "",
                    "weather": "",
                    "notes": "",
                    "splits": "",
                    "sleep": None,
                    "stress": None,
                    "vo2max": row.get("vo2max", None),
                    "hrv": row.get("hrv", None),
                }

                insert_run(data)
                imported += 1
            except:
                pass

        st.success(f"Imported {imported} runs successfully.")
# ===============================================
# SEGMENT 3 — DASHBOARD & ANALYTICS
# ===============================================


# ------------------------------------------------------
# Weekly mileage helper
# ------------------------------------------------------
def get_weekly_mileage(df):
    df_copy = df.copy()
    df_copy["week"] = df_copy["date_dt"].dt.isocalendar().week
    weekly = df_copy.groupby("week")["distance"].sum()
    return weekly


# ------------------------------------------------------
# Dashboard Page
# ------------------------------------------------------
def render_dashboard_page():
    st.title("📊 Dashboard")

    df = fetch_runs()
    if df.empty:
        st.info("Log your first run to see dashboard insights.")
        return

    df = prepare_metrics_df(df)

    # ----------------------------
    # High-level metrics
    # ----------------------------
    total_miles = df["distance"].sum()
    last7 = df[df["date_dt"] >= datetime.today() - timedelta(days=7)]["distance"].sum()
    last30 = df[df["date_dt"] >= datetime.today() - timedelta(days=30)]["distance"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Miles", f"{total_miles:.1f}")
    c2.metric("Last 7 Days", f"{last7:.1f}")
    c3.metric("Last 30 Days", f"{last30:.1f}")

    st.markdown("---")

    # ----------------------------
    # Weekly mileage chart
    # ----------------------------
    st.subheader("📅 Weekly Mileage")

    weekly = get_weekly_mileage(df)

    if not weekly.empty:
        st.bar_chart(weekly)
    else:
        st.info("Not enough data for weekly chart.")

    st.markdown("---")

    # ----------------------------
    # Pace Trend
    # ----------------------------
    st.subheader("🏃 Pace Trend (min/mile)")

    pace_df = df.dropna(subset=["pace_min_per_mile"]).sort_values("date_dt")
    if not pace_df.empty:
        st.line_chart(
            pace_df.set_index("date_dt")["pace_min_per_mile"],
            height=250
        )
    else:
        st.info("No pace data available.")

    st.markdown("---")

    # ----------------------------
    # Heart Rate Trend
    # ----------------------------
    st.subheader("❤️ Avg Heart Rate Trend")

    hr_df = df.dropna(subset=["avg_hr"]).sort_values("date_dt")
    if not hr_df.empty:
        st.line_chart(
            hr_df.set_index("date_dt")["avg_hr"],
            height=250
        )
    else:
        st.info("No HR data available.")

    st.markdown("---")

    # ----------------------------
    # Efficiency Trend
    # ----------------------------
    st.subheader("⚡ Efficiency Score Trend")

    eff_df = df.dropna(subset=["efficiency_score"]).sort_values("date_dt")
    if not eff_df.empty:
        st.line_chart(
            eff_df.set_index("date_dt")["efficiency_score"],
            height=250
        )
    else:
        st.info("Not enough efficiency data yet.")

    st.markdown("---")

    # ----------------------------
    # Run Type Distribution
    # ----------------------------
    st.subheader("🏷 Run Type Distribution")

    type_counts = df["run_type"].value_counts()

    if not type_counts.empty:
        st.bar_chart(type_counts)
    else:
        st.info("No run types found.")

    st.markdown("---")

    # ----------------------------
    # Training Load Progression
    # ----------------------------
    st.subheader("📈 Training Load Progression")

    df["load"] = df.apply(calculate_daily_load, axis=1)
    load_df = df[["date_dt", "load"]].sort_values("date_dt")

    if not load_df.empty:
        st.line_chart(load_df.set_index("date_dt")["load"], height=250)
    else:
        st.info("No load data available.")

    st.markdown("---")

    # ----------------------------
    # PRs
    # ----------------------------
    st.subheader("🏆 Personal Records")

    prs = calculate_prs(df)
    if prs:
        st.markdown("<div class='pr-banner'>PR Achievements</div>", unsafe_allow_html=True)

        if prs.get("longest_distance"):
            st.write(f"**Longest Run:** {prs['longest_distance']:.1f} mi")

        if prs.get("fastest_pace"):
            st.write(f"**Fastest Pace:** {prs['fastest_pace']:.2f} min/mi")

        if prs.get("highest_weekly_mileage"):
            st.write(f"**Highest Weekly Mileage:** {prs['highest_weekly_mileage']:.1f} mi")

        if prs.get("highest_monthly_mileage"):
            st.write(f"**Highest Monthly Mileage:** {prs['highest_monthly_mileage']:.1f} mi")

    else:
        st.info("No PRs recorded yet.")
# ===============================================
# SEGMENT 4 — CALENDAR UTILITY FUNCTIONS
# ===============================================

# ------------------------------------------------------
# Training Load Calculation
# ------------------------------------------------------
def calculate_daily_load(row):
    """
    Training load model:
    - distance
    - effort (RPE)
    - HR intensity modifier
    """
    distance = row.get("distance", 0) or 0
    effort = row.get("effort_rating", 5) or 5
    avg_hr = row.get("avg_hr", None)

    # Base load
    load = distance * effort

    # HR scaling
    if avg_hr and avg_hr > 0:
        load *= (avg_hr / 140)

    return load


# ------------------------------------------------------
# Training Load Color Gradient
# ------------------------------------------------------
def get_training_load_color(load, max_load):
    """
    Creates a blue gradient:
    - low load = light blue
    - high load = deep saturated blue
    """
    if max_load == 0:
        return "rgba(30, 41, 59, 0.4)"

    intensity = load / max_load
    intensity = min(max(intensity, 0.05), 1)

    # Base blue (Tailwind blue-900)
    r, g, b = 30, 58, 138
    alpha = 0.25 + 0.6 * intensity

    return f"rgba({r}, {g}, {b}, {alpha})"


# ------------------------------------------------------
# Injury Risk Modeling
# ------------------------------------------------------
def calculate_injury_risk_day(df_day):
    """
    Uses multiple fatigue + spike indicators to determine daily risk:
    - high effort
    - high HR
    - low sleep
    - high stress
    - elevation spikes
    - long distance spikes
    """
    if df_day.empty:
        return "low"

    row = df_day.iloc[-1]
    risk = 0

    effort = row.get("effort_rating", 5) or 5
    if effort >= 8:
        risk += 1.5

    sleep = row.get("sleep", 7)
    if sleep and sleep < 6:
        risk += 1.2

    stress = row.get("stress", 2)
    if stress and stress >= 4:
        risk += 1.0

    avg_hr = row.get("avg_hr", None)
    if avg_hr and avg_hr > 160:
        risk += 1.0

    dist = row.get("distance", 0)
    if dist > 8:
        risk += 0.5

    elev = row.get("elevation_gain", 0)
    if elev > 500:
        risk += 0.5

    # thresholds
    if risk >= 3:
        return "high"
    elif risk >= 1.5:
        return "moderate"
    return "low"


# ------------------------------------------------------
# Border Colors for Injury Risk
# ------------------------------------------------------
def get_injury_border_color(risk):
    if risk == "high":
        return "3px solid #EF4444"   # red-500
    if risk == "moderate":
        return "3px solid #FACC15"   # yellow-400
    return "1px solid rgba(255,255,255,0.08)"  # low risk
# ===============================================
# SEGMENT 5 — STRAVA STYLE MONTH VIEW
# ===============================================

def render_month_view(df):
    st.markdown("### 📅 Monthly Training Calendar")

    # -------- Month selector --------
    today = datetime.today()
    col1, col2 = st.columns([1, 1])
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=today.year, key="cal_year")
    with col2:
        month = st.selectbox("Month", list(range(1, 12 + 1)), index=today.month - 1, key="cal_month")

    # Determine month range
    month_start = datetime(year, month, 1)
    next_month = month_start + pd.DateOffset(months=1)

    df_month = df[(df["date_dt"] >= month_start) & (df["date_dt"] < next_month)]

    # Compute load
    if not df_month.empty:
        df_month["load"] = df_month.apply(calculate_daily_load, axis=1)
        max_load = df_month["load"].max()
    else:
        df_month["load"] = 0
        max_load = 1

    # Group data by day
    mileage_by_day = df_month.groupby(df_month["date_dt"].dt.date)["distance"].sum().to_dict()
    load_by_day = df_month.groupby(df_month["date_dt"].dt.date)["load"].sum().to_dict()
    types_by_day = df_month.groupby(df_month["date_dt"].dt.date)["run_type"].apply(list).to_dict()

    # Injury risk per day
    injury_by_day = {}
    for d in df_month["date_dt"].dt.date.unique():
        risk = calculate_injury_risk_day(df_month[df_month["date_dt"].dt.date == d])
        injury_by_day[d] = risk

    # ------------ Run type badge colors ------------
    badge_colors = {
        "Easy": "#22C55E",
        "LongRun": "#3B82F6",
        "Tempo": "#F97316",
        "Intervals": "#EC4899",
        "Recovery": "#64748B",
        "Race": "#A855F7",
    }

    # ------------ Calendar layout ------------
    import calendar
    cal = calendar.Calendar(firstweekday=0)
    weeks = cal.monthdatescalendar(year, month)

    # Header row
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    header = st.columns(7)
    for i, d in enumerate(dow):
        header[i].markdown(f"<div style='text-align:center; font-weight:600;'>{d}</div>", unsafe_allow_html=True)

    # Calendar grid
    for week in weeks:
        row = st.columns(7)
        for idx, date_obj in enumerate(week):
            inside_month = (date_obj.month == month)
            date_key = date_obj

            mileage = mileage_by_day.get(date_key, 0)
            load_val = load_by_day.get(date_key, 0)
            run_types = types_by_day.get(date_key, [])

            # Color styling
            bg = get_training_load_color(load_val, max_load)
            border = get_injury_border_color(injury_by_day.get(date_key, "low"))
            opacity = "1.0" if inside_month else "0.28"
            text_color = "white" if inside_month else "#9CA3AF"

            # Run type badges
            badges_html = ""
            for rt in run_types:
                c = badge_colors.get(rt, "#6B7280")
                badges_html += (
                    f"<span style='background:{c}; padding:2px 6px; color:white; "
                    f"border-radius:6px; font-size:0.7rem; margin-right:4px;'>{rt}</span>"
                )

            # Click handler
            if row[idx].button(f"{date_obj}", key=f"day_{date_obj}"):
                st.session_state["calendar_selected_day"] = str(date_obj)

            # Cell content
            row[idx].markdown(
                f"""
                <div style="
                    background:{bg};
                    border:{border};
                    opacity:{opacity};
                    padding:10px;
                    border-radius:10px;
                    min-height:90px;
                ">
                    <div style="font-weight:600; color:{text_color}; font-size:1rem;">
                        {date_obj.day}
                    </div>
                    <div style="color:{text_color}; font-size:0.85rem;">
                        {mileage:.1f} mi
                    </div>
                    <div>{badges_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Weekly mileage chart
    st.markdown("### 🧮 Weekly Mileage Summary")
    weekly = (
        df_month.groupby(df_month["date_dt"].dt.isocalendar().week)["distance"].sum()
        if not df_month.empty else pd.Series()
    )

    if not weekly.empty:
        st.bar_chart(weekly)
    else:
        st.info("No mileage recorded this month.")
# ===============================================
# SEGMENT 6 — STRAVA STYLE WEEK VIEW
# ===============================================

def calculate_week_consistency(df_week):
    """
    Consistency score (0–100):
    - Mileage distribution (not 1 big day + 6 zeros)
    - Intensity balance (effort)
    - Mileage trend vs average
    """
    if df_week.empty:
        return 0

    weekly_total = df_week["distance"].sum()

    # Distribution penalty — high std dev = lower score
    daily_dist = df_week.groupby(df_week["date_dt"].dt.date)["distance"].sum()
    dist_std = daily_dist.std() if len(daily_dist) > 1 else 0
    distribution_score = max(0, 40 - (dist_std * 4))  # 0–40

    # Effort score
    if "effort_rating" in df_week:
        avg_eff = df_week["effort_rating"].mean()
        effort_score = max(0, 30 - abs(avg_eff - 5) * 6)  # 0–30
    else:
        effort_score = 15

    # Mileage score
    avg_daily = df_week["distance"].mean()
    mileage_ratio = min(weekly_total / (avg_daily * 7 + 1e-9), 1.3)
    mileage_score = min(30 * mileage_ratio, 30)  # 0–30

    return int(distribution_score + effort_score + mileage_score)


def render_week_view(df):
    st.markdown("### 🗓 Weekly View")

    today = datetime.today()
    current_year, current_week, _ = today.isocalendar()

    col1, col2 = st.columns([1, 1])
    with col1:
        week = st.number_input("ISO Week", min_value=1, max_value=53, value=current_week, key="week_view_week")
    with col2:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=current_year, key="week_view_year")

    # Convert to date range
    week_start = datetime.fromisocalendar(year, int(week), 1)
    week_end = week_start + timedelta(days=7)

    df_week = df[(df["date_dt"] >= week_start) & (df["date_dt"] < week_end)].copy()

    # Load computation
    df_week["load"] = df_week.apply(calculate_daily_load, axis=1)
    max_load = df_week["load"].max() if not df_week.empty else 1

    # Aggregate by day
    day_dist = df_week.groupby(df_week["date_dt"].dt.date)["distance"].sum().to_dict()
    day_types = df_week.groupby(df_week["date_dt"].dt.date)["run_type"].apply(list).to_dict()
    day_loads = df_week.groupby(df_week["date_dt"].dt.date)["load"].sum().to_dict()

    # Injury risk
    injury_by_day = {}
    for d in df_week["date_dt"].dt.date.unique():
        injury_by_day[d] = calculate_injury_risk_day(df_week[df_week["date_dt"].dt.date == d])

    # Days of the week
    days = [(week_start + timedelta(days=i)).date() for i in range(7)]
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    header = st.columns(7)
    for i, name in enumerate(day_names):
        header[i].markdown(f"<div style='text-align:center; font-weight:600;'>{name}</div>", unsafe_allow_html=True)

    row = st.columns(7)

    # Badge colors
    badge_colors = {
        "Easy": "#22C55E",
        "LongRun": "#3B82F6",
        "Tempo": "#F97316",
        "Intervals": "#EC4899",
        "Recovery": "#64748B",
        "Race": "#A855F7",
    }

    # Render day cells
    for i, date_obj in enumerate(days):
        mileage = day_dist.get(date_obj, 0)
        load_val = day_loads.get(date_obj, 0)
        run_types = day_types.get(date_obj, [])

        bg = get_training_load_color(load_val, max_load)
        border = get_injury_border_color(injury_by_day.get(date_obj, "low"))

        # Run type badges
        badges = ""
        for rt in run_types:
            color = badge_colors.get(rt, "#6B7280")
            badges += f"<span style='background:{color}; color:white; padding:2px 6px; border-radius:6px; font-size:0.75rem; margin-right:4px;'>{rt}</span>"

        # Click handler
        if row[i].button(str(date_obj), key=f"week_click_{date_obj}"):
            st.session_state["calendar_selected_day"] = str(date_obj)

        # Render cell
        row[i].markdown(
            f"""
            <div style="
                background:{bg};
                border:{border};
                padding:12px;
                border-radius:12px;
                min-height:110px;
            ">
                <div style="font-weight:600; margin-bottom:6px;">{date_obj.day}</div>
                <div style="font-size:0.9rem;">{mileage:.1f} mi</div>
                <div style="margin-top:6px;">{badges}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------
    # Weekly Summary
    # --------------------------
    total_week = df_week["distance"].sum()
    consistency = calculate_week_consistency(df_week)

    st.markdown("### 📊 Weekly Summary")
    c1, c2 = st.columns(2)
    c1.metric("Total Mileage", f"{total_week:.1f} mi")
    c2.metric("Consistency Score", f"{consistency}/100")

    # Load progression
    if not df_week.empty:
        sorted_df = df_week.sort_values("date_dt")
        st.line_chart(sorted_df.set_index("date_dt")["load"], height=200)
# ===============================================
# SEGMENT 7 — DAY MODAL PANEL + AI ANALYSIS
# ===============================================

def render_day_modal(df):
    """Displays detailed run data + AI insights for selected day."""
    if "calendar_selected_day" not in st.session_state:
        return

    selected_date = datetime.fromisoformat(st.session_state["calendar_selected_day"]).date()
    df_day = df[df["date_dt"].dt.date == selected_date]

    st.markdown(
        f"""
        <div style="
            background:#0F172A;
            padding:18px;
            border-radius:12px;
            border:1px solid #334155;
            margin-top:20px;
        ">
            <h2 style='margin:0;'>📅 {selected_date.strftime('%A, %B %d, %Y')}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_day.empty:
        st.info("No runs logged on this day.")
        return

    # Run type badge colors
    badge_colors = {
        "Easy": "#22C55E",
        "LongRun": "#3B82F6",
        "Tempo": "#F97316",
        "Intervals": "#EC4899",
        "Recovery": "#64748B",
        "Race": "#A855F7",
        "Other": "#6B7280"
    }

    # Render each run on this day
    for _, row in df_day.iterrows():
        rt = row.get("run_type", "Run")
        color = badge_colors.get(rt, "#6B7280")

        distance = row.get("distance", 0)
        duration = row.get("duration", "")
        pace = row.get("avg_pace", "")
        avg_hr = row.get("avg_hr", None)
        max_hr = row.get("max_hr", None)
        elev = row.get("elevation_gain", 0)
        cadence = row.get("cadence", None)
        effort = row.get("effort_rating", None)
        terrain = row.get("terrain", "")
        weather = row.get("weather", "")
        notes = row.get("notes", "")
        splits = row.get("splits", "")

        st.markdown(
            f"""
            <div style="
                background:#1E293B;
                padding:16px;
                border-radius:12px;
                border:1px solid #334155;
                margin-top:20px;
            ">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <span style="
                        background:{color};
                        padding:6px 12px;
                        color:white;
                        border-radius:8px;
                        font-size:0.9rem;
                        font-weight:600;
                    ">{rt}</span>

                    <span style="font-size:1.2rem; font-weight:600; color:white;">
                        {distance:.2f} mi
                    </span>
                </div>

                <div style="margin-top:10px; color:#E2E8F0; font-size:0.9rem;">
                    <strong>Duration:</strong> {duration}<br>
                    <strong>Pace:</strong> {pace}<br>
                    <strong>Avg HR:</strong> {avg_hr if avg_hr else '—'}<br>
                    <strong>Max HR:</strong> {max_hr if max_hr else '—'}<br>
                    <strong>Elevation:</strong> {elev} ft<br>
                    <strong>Cadence:</strong> {cadence if cadence else '—'} spm<br>
                    <strong>Effort:</strong> {effort if effort else '—'}/10<br>
                    <strong>Terrain:</strong> {terrain}<br>
                    <strong>Weather:</strong> {weather}<br>
                </div>

                <div style="margin-top:12px; color:#CBD5E1; font-size:0.88rem;">
                    <strong>Notes:</strong><br>
                    {notes if notes else '<em>No notes.</em>'}
                </div>

                <div style="margin-top:12px; color:#CBD5E1; font-size:0.88rem;">
                    <strong>Splits:</strong>
                    <pre style="
                        white-space:pre-wrap;
                        background:#0F172A;
                        padding:10px;
                        border-radius:6px;
                        font-size:0.85rem;
                    ">{splits if splits else 'N/A'}</pre>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =======================================================================
    # AI BUTTONS (Analyze Day + Improvements)
    # =======================================================================
    st.markdown("### 🧠 AI Insights")

    col1, col2 = st.columns(2)

    # Prepare data for prompting
    day_records = df_day.to_dict("records")

    with col1:
        if st.button("Analyze This Day"):
            prompt = f"""
You are an elite running coach. Analyze this training day in depth:

DATE: {selected_date}

RUNS:
{day_records}

Provide:
- What went well
- What looked challenging
- HR & pace relationship analysis
- Effort appropriateness
- Technique factors
- Any signs of fatigue or injury risk
- Whether this run supports long-term goals
- Smart advice for the next 2–3 days
            """
            st.write(call_ai(prompt))

    with col2:
        if st.button("How Can I Improve?"):
            prompt = f"""
You are a professional running coach.

Based on these runs from {selected_date}, give specific improvement advice:

RUN DATA:
{day_records}

Provide:
- 3 technique adjustments
- 3 pacing adjustments
- 3 HR training cues
- 3 efficiency boosters
- 2 recovery recommendations
- 3 targeted workouts to address weaknesses
            """
            st.write(call_ai(prompt))
# ===============================================
# SEGMENT 8 — MONTH AI ANALYSIS
# ===============================================

def render_month_analysis(df):
    """AI-powered breakdown of the entire selected month."""

    today = datetime.today()
    year = st.session_state.get("cal_year", today.year)
    month = st.session_state.get("cal_month", today.month)

    month_start = datetime(year, month, 1)
    next_month = month_start + pd.DateOffset(months=1)

    df_month = df[(df["date_dt"] >= month_start) & (df["date_dt"] < next_month)]

    st.markdown(f"### 📊 Monthly Insights — {month_start.strftime('%B %Y')}")

    if df_month.empty:
        st.info("No runs logged for this month.")
        return

    # Prepare metrics
    df_month = prepare_metrics_df(df_month)
    df_month["load"] = df_month.apply(calculate_daily_load, axis=1)

    total_miles = df_month["distance"].sum()
    avg_pace = df_month["pace_min_per_mile"].mean()
    avg_hr = df_month["avg_hr"].mean()
    avg_eff = df_month["efficiency_score"].mean()
    total_load = df_month["load"].sum()

    # Long runs (8+ miles)
    long_runs = df_month[df_month["distance"] >= 8]["distance"].tolist()

    # PRs
    prs = calculate_prs(df_month)

    # ----------------------------
    # Summary Cards
    # ----------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Miles", f"{total_miles:.1f}")
    c2.metric("Avg Pace", f"{avg_pace:.2f} min/mi" if avg_pace else "—")
    c3.metric("Avg HR", f"{avg_hr:.1f}" if avg_hr else "—")
    c4.metric("Total Load", f"{total_load:.1f}")

    if long_runs:
        st.success(f"🏔️ Long Runs: {', '.join([f'{x:.1f} mi' for x in long_runs])}")

    # ----------------------------
    # PR Section
    # ----------------------------
    st.markdown("### 🏆 PRs This Month")

    if prs:
        for k, v in prs.items():
            label = k.replace("_", " ").title()
            if "fastest" in k:
                st.write(f"⚡ **{label}:** {v:.2f} min/mi")
            else:
                st.write(f"🔥 **{label}:** {v:.1f} mi")
    else:
        st.info("No PRs recorded this month.")

    # ----------------------------
    # AI Monthly Analysis
    # ----------------------------
    st.markdown("---")
    st.markdown("### 🧠 AI Coach — Monthly Evaluation")

    if st.button("Analyze This Month"):
        records = df_month.to_dict("records")

        prompt = f"""
You are a world-class running coach.

Analyze the following month of training:

MONTH: {month_start.strftime('%B %Y')}
DATA: {records}

Provide a detailed coaching report with the following sections:

1. **Overall Training Summary**
   - Mileage volume
   - Weekly structure
   - Intensity balance
   - Long-run progression

2. **Training Load & Fatigue**
   - Total load
   - Load ramp-up / ramp-down
   - Fatigue indicators
   - Whether athlete is trending toward overtraining

3. **Performance Trends**
   - Pace progression
   - HR trends (improving vs deteriorating)
   - Efficiency score analysis
   - Cadence consistency

4. **Strengths & Weaknesses**
   - Aerobic base
   - Threshold conditioning
   - Speed development
   - Muscular endurance

5. **Injury Risk Assessment**
   - Load spikes
   - HR drift warnings
   - Low sleep/high stress flags
   - Shin-splint risk or form breakdown signals

6. **PRs This Month**
   {prs}

7. **Actionable Improvements**
   - 5 training adjustments
   - 3 pacing recommendations
   - 3 HR zone cues
   - 3 recovery strategies
   - 3 technique optimizations

8. **Next-Month Training Plan**
   - Weekly mileage targets
   - Long-run goals
   - Recommended workouts (tempo, intervals, easy ratios)
   - Specific steps to progress safely

The goal: Provide clear, actionable, elite-level guidance.
"""

        st.write(call_ai(prompt))
# ===============================================
# SEGMENT 9 — CALENDAR WRAPPER PAGE
# ===============================================

def render_calendar_page():
    st.title("📅 Training Calendar")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs to populate your calendar.")
        return

    df = prepare_metrics_df(df)

    # -------------------------
    # Calendar View Toggle
    # -------------------------
    view = st.radio(
        "Calendar View",
        ["Month View", "Week View"],
        horizontal=True,
        key="calendar_view_mode",
    )

    # -------------------------
    # Render Selected View
    # -------------------------
    if view == "Month View":
        render_month_view(df)
    else:
        render_week_view(df)

    st.markdown("---")

    # -------------------------
    # DAY MODAL (when user clicks a date)
    # -------------------------
    if "calendar_selected_day" in st.session_state:
        st.markdown("## 📖 Day Details")
        render_day_modal(df)
        st.markdown("---")

    # -------------------------
    # MONTHLY AI COACH ANALYSIS
    # -------------------------
    st.markdown("## 🤖 Monthly AI Analysis")
    render_month_analysis(df)
# ===============================================
# SEGMENT 10 — NAVIGATION + MAIN APPLICATION ROUTER
# ===============================================

def main():
    st.sidebar.title("🏃 Run Tracker Menu")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Home",
            "Feed",
            "Log a Run",
            "Dashboard",
            "Calendar",
            "Garmin Import",
        ],
        index=0,
    )

    # Routing
    if page == "Home":
        render_home_page()

    elif page == "Feed":
        render_feed_page()

    elif page == "Log a Run":
        render_log_run_page()

    elif page == "Dashboard":
        render_dashboard_page()

    elif page == "Calendar":
        render_calendar_page()

    elif page == "Garmin Import":
        render_garmin_import_page()


# Run the app
if __name__ == "__main__":
    main()
