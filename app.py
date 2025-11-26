import sqlite3
from datetime import datetime, timedelta, date

import altair as alt
import pandas as pd
import streamlit as st

# OpenAI client (for AI coach)
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

DB_PATH = "run_log.db"


# =========================
# SESSION & THEME
# =========================

def init_session_state():
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    if "units" not in st.session_state:
        st.session_state["units"] = "mi"
    if "ai_verbosity" not in st.session_state:
        st.session_state["ai_verbosity"] = "normal"
    if "ai_focus" not in st.session_state:
        st.session_state["ai_focus"] = "balanced"
    if "race_goal" not in st.session_state:
        st.session_state["race_goal"] = "Pittsburgh Half – Sub 1:40"
    if "race_date_str" not in st.session_state:
        st.session_state["race_date_str"] = "2026-05-03"
    if "hr_max" not in st.session_state:
        st.session_state["hr_max"] = 190
    if "weekly_goal_mi" not in st.session_state:
        st.session_state["weekly_goal_mi"] = 25.0
    if "compact_mode" not in st.session_state:
        st.session_state["compact_mode"] = False
    if "edit_run_id" not in st.session_state:
        st.session_state["edit_run_id"] = None


def inject_css():
    theme = st.session_state.get("theme", "dark")
    compact = st.session_state.get("compact_mode", False)

    if theme == "light":
        bg = "#F7F7FA"
        card_bg = "#FFFFFF"
        text = "#111827"
        border = "rgba(0,0,0,0.06)"
    else:
        bg = "#05060A"
        card_bg = "#111827"
        text = "#F9FAFB"
        border = "rgba(255,255,255,0.08)"

    card_padding = "0.8rem 1.0rem" if compact else "1.1rem 1.4rem"
    card_margin = "0.6rem" if compact else "1.0rem"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: {bg};
            color: {text};
        }}

        .block-container {{
            padding-top: 1.0rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }}

        .card {{
            background: {card_bg};
            padding: {card_padding};
            border-radius: 14px;
            margin-bottom: {card_margin};
            border: 1px solid {border};
            box-shadow: 0 12px 28px rgba(0,0,0,0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 18px 40px rgba(0,0,0,0.32);
            border-color: rgba(59,130,246,0.6);
        }}

        .feed-card {{
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }}

        .feed-header-line {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.35rem;
        }}

        .feed-main-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            align-items: baseline;
        }}

        .big-distance {{
            font-size: 1.6rem;
            font-weight: 600;
        }}

        .muted {{
            font-size: 0.85rem;
            opacity: 0.7;
        }}

        .tag {{
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
            color: white;
        }}
        .tag-Easy {{ background: #22C55E; }}
        .tag-LongRun {{ background: #3B82F6; }}
        .tag-Tempo {{ background: #F97316; }}
        .tag-Intervals {{ background: #EC4899; }}
        .tag-Recovery {{ background: #64748B; }}
        .tag-Race {{ background: #A855F7; }}
        .tag-Other {{ background: #6B7280; }}

        .pr-banner {{
            background: linear-gradient(90deg, #ffb300, #ffdd66);
            padding: 12px 16px;
            margin: 12px 0;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            color: #111827;
            text-align: center;
        }}

        .pr-mini {{
            background: #ffe29a;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 8px;
            display: inline-block;
        }}

        .pr-badge {{
            background: #ffcc00;
            padding: 4px 10px;
            border-radius: 10px;
            display: inline-block;
            margin-top: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #111827;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# DB HELPERS
# =========================

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            run_type TEXT,
            distance REAL,
            duration_minutes REAL,
            avg_pace TEXT,
            splits TEXT,
            avg_hr INTEGER,
            max_hr INTEGER,
            hr_by_segment TEXT,
            cadence INTEGER,
            elevation_gain INTEGER,
            effort INTEGER,
            terrain TEXT,
            weather TEXT,
            how_felt TEXT,
            pain TEXT,
            sleep_hours REAL,
            stress INTEGER,
            nutrition_notes TEXT,
            vo2max REAL,
            hrv INTEGER,
            shoe_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS shoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT,
            start_date TEXT,
            retired INTEGER DEFAULT 0
        )
        """
    )

    cur.execute("PRAGMA table_info(runs)")
    cols = [row[1] for row in cur.fetchall()]
    if "shoe_id" not in cols:
        cur.execute("ALTER TABLE runs ADD COLUMN shoe_id INTEGER")

    conn.commit()
    conn.close()


def init_db_with_migration():
    init_db()


# =========================
# CORE UTILS
# =========================

def duration_to_minutes(time_str: str | None):
    if not time_str:
        return None
    parts = time_str.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:
        m, s = parts
        return m + s / 60.0
    if len(parts) == 3:
        h, m, s = parts
        return h * 60.0 + m + s / 60.0
    return None


def minutes_to_hms(minutes: float | None) -> str:
    if minutes is None:
        return ""
    total_seconds = int(round(minutes * 60))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def pace_to_float(pace_str: str | None):
    if not pace_str or ":" not in pace_str:
        return None
    try:
        m, s = pace_str.split(":")
        return int(m) + int(s) / 60.0
    except Exception:
        return None


def prepare_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    m = df.copy()
    m["date_dt"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date_dt"])
    m["date"] = m["date_dt"].dt.date
    m["week_start"] = m["date_dt"] - pd.to_timedelta(m["date_dt"].dt.weekday, unit="D")
    m["week_start"] = m["week_start"].dt.date

    m["training_load"] = m["effort"].fillna(0) * m["duration_minutes"].fillna(0)

    m["pace_min_per_mile"] = None
    mask = (
        m["distance"].notna()
        & (m["distance"] > 0)
        & m["duration_minutes"].notna()
    )
    m.loc[mask, "pace_min_per_mile"] = (
        m.loc[mask, "duration_minutes"] / m.loc[mask, "distance"]
    )

    hr_max = float(st.session_state.get("hr_max", 190))
    m["rss"] = None
    mask_rss = m["duration_minutes"].notna() & m["avg_hr"].notna()
    m.loc[mask_rss, "rss"] = (
        m.loc[mask_rss, "duration_minutes"]
        * (m.loc[mask_rss, "avg_hr"] / hr_max) ** 2
    )

    return m


def compute_daily_load(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    daily = (
        metrics.groupby("date_dt", as_index=False)
        .agg(
            distance=("distance", "sum"),
            training_load=("training_load", "sum"),
            rss=("rss", "sum"),
        )
    )
    return daily


def compute_fitness_fatigue(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    d = daily.sort_values("date_dt").copy()
    d["CTL"] = d["training_load"].rolling(window=42, min_periods=1).mean()
    d["ATL"] = d["training_load"].rolling(window=7, min_periods=1).mean()
    d["TSB"] = d["CTL"] - d["ATL"]
    return d


def compute_efficiency_score(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()
    m["efficiency_score"] = None
    mask = (
        m["distance"].notna()
        & (m["distance"] > 0)
        & m["duration_minutes"].notna()
        & m["avg_hr"].notna()
    )
    m.loc[mask, "efficiency_score"] = (
        m.loc[mask, "distance"]
        / (m.loc[mask, "duration_minutes"] / 60.0)
        / m.loc[mask, "avg_hr"]
        * 1000.0
    )
    return m


def add_hr_zones(metrics: pd.DataFrame, hr_max=190) -> pd.DataFrame:
    if metrics.empty or "avg_hr" not in metrics.columns:
        return metrics

    def zone(hr):
        if pd.isna(hr):
            return None
        pct = hr / hr_max
        if pct < 0.6:
            return "Z1"
        if pct < 0.7:
            return "Z2"
        if pct < 0.8:
            return "Z3"
        if pct < 0.9:
            return "Z4"
        return "Z5"

    m = metrics.copy()
    m["hr_zone"] = m["avg_hr"].apply(zone)
    return m


def compute_streaks(metrics: pd.DataFrame):
    if metrics.empty:
        return 0, 0
    dates = sorted(set(metrics["date_dt"].dt.date))
    if not dates:
        return 0, 0
    current = 1
    longest = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return current, longest


def convert_distance_for_display(distance_mi: float | None):
    units = st.session_state.get("units", "mi")
    if distance_mi is None:
        return None, units
    if units == "km":
        return distance_mi * 1.60934, "km"
    return distance_mi, "mi"


# =========================
# PR / RACE / ZONES HELPERS
# =========================

def calculate_prs(df: pd.DataFrame):
    prs = {}
    if df.empty:
        return prs

    df = df.copy()
    df["pace_num"] = df["avg_pace"].apply(pace_to_float)

    prs["longest_distance"] = df["distance"].max()

    pace_df = df[df["pace_num"].notna()]
    if not pace_df.empty:
        prs["fastest_pace"] = pace_df["pace_num"].min()

    mile_runs = df[df["distance"] >= 1.0]
    if not mile_runs.empty:
        prs["fastest_mile"] = (
            mile_runs["duration_minutes"] / mile_runs["distance"]
        ).min()

    fivek_runs = df[df["distance"] >= 3.11]
    if not fivek_runs.empty:
        prs["fastest_5k"] = (
            fivek_runs["duration_minutes"] / fivek_runs["distance"] * 3.11
        ).min()

    tenk_runs = df[df["distance"] >= 6.22]
    if not tenk_runs.empty:
        prs["fastest_10k"] = (
            tenk_runs["duration_minutes"] / tenk_runs["distance"] * 6.22
        ).min()

    half_runs = df[df["distance"] >= 13.1]
    if not half_runs.empty:
        prs["fastest_half"] = (
            half_runs["duration_minutes"] / half_runs["distance"] * 13.1
        ).min()

    df["date_dt"] = pd.to_datetime(df["date"])
    df["week"] = df["date_dt"].dt.isocalendar().week
    df["year"] = df["date_dt"].dt.year

    weekly = df.groupby(["year", "week"])["distance"].sum()
    prs["highest_weekly_mileage"] = weekly.max()

    df["month"] = df["date_dt"].dt.month
    monthly = df.groupby(["year", "month"])["distance"].sum()
    prs["highest_monthly_mileage"] = monthly.max()

    return prs


def detect_pr_improvements(prs_before, prs_after):
    if prs_after is None:
        return []
    labels = {
        "longest_distance": "Longest Run",
        "fastest_pace": "Fastest Overall Pace",
        "fastest_mile": "Fastest Mile",
        "fastest_5k": "Fastest 5K",
        "fastest_10k": "Fastest 10K",
        "fastest_half": "Fastest Half Marathon",
        "highest_weekly_mileage": "Highest Weekly Mileage",
        "highest_monthly_mileage": "Highest Monthly Mileage",
    }
    improvements = []
    for key, new_val in prs_after.items():
        if new_val is None or pd.isna(new_val):
            continue
        old_val = prs_before.get(key) if prs_before else None
        if old_val is None or pd.isna(old_val):
            improvements.append(labels.get(key, key))
            continue
        if "fastest" in key:
            if new_val < old_val - 1e-6:
                improvements.append(labels.get(key, key))
        else:
            if new_val > old_val + 1e-6:
                improvements.append(labels.get(key, key))
    return improvements


def predict_race_times(df: pd.DataFrame):
    if df.empty:
        return None
    df = df.copy()
    df["pace_num"] = df["avg_pace"].apply(pace_to_float)
    best_pace = df["pace_num"].min()
    vo2 = df["vo2max"].dropna().mean() if "vo2max" in df.columns else None
    vo2_factor = (50 / vo2) if vo2 else 1.0
    daily = compute_daily_load(df)
    load = compute_fitness_fatigue(daily)
    tsb = load["TSB"].iloc[-1] if not load.empty else 0
    fatigue_factor = 1 - (tsb / 200.0)
    effective_pace = best_pace * vo2_factor * fatigue_factor
    return {
        "5K": effective_pace * 3.11,
        "10K": effective_pace * 6.22,
        "Half": effective_pace * 13.1,
        "Marathon": effective_pace * 26.2,
    }


def generate_race_prediction_series(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame([])
    out = []
    df = df.copy().sort_values("date_dt")
    for _, row in df.iterrows():
        sub = df[df["date_dt"] <= row["date_dt"]]
        pred = predict_race_times(sub)
        if pred:
            out.append(
                {
                    "date": row["date_dt"],
                    "5K": pred["5K"],
                    "10K": pred["10K"],
                    "Half": pred["Half"],
                    "Marathon": pred["Marathon"],
                }
            )
    return pd.DataFrame(out)


def estimate_threshold_pace(prs):
    if not prs:
        return None
    fivek = prs.get("fastest_5k")
    mile = prs.get("fastest_mile")
    if fivek:
        return fivek / 3.11 * 0.98
    if mile:
        return mile * 1.15
    return None


def estimate_easy_pace(best_pace):
    if best_pace is None:
        return None, None
    return best_pace + 1.0, best_pace + 2.0


def estimate_tempo_pace(threshold_pace):
    if threshold_pace is None:
        return None
    return threshold_pace * 0.97, threshold_pace * 1.02


def estimate_interval_pace(best_pace):
    if best_pace is None:
        return None
    return best_pace * 0.90, best_pace * 0.95


def estimate_rep_pace(best_pace):
    if best_pace is None:
        return None
    return best_pace * 0.80, best_pace * 0.90


def calculate_hr_zones(hr_max):
    return {
        "Zone 1 (Recovery)": (0.50 * hr_max, 0.60 * hr_max),
        "Zone 2 (Easy / Aerobic)": (0.60 * hr_max, 0.70 * hr_max),
        "Zone 3 (Tempo)": (0.70 * hr_max, 0.80 * hr_max),
        "Zone 4 (Threshold)": (0.80 * hr_max, 0.90 * hr_max),
        "Zone 5 (VO2 Max)": (0.90 * hr_max, 1.00 * hr_max),
    }


# =========================
# AI HELPERS
# =========================

def ensure_openai_client():
    global client
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            client = None
    return client


def call_ai(prompt: str):
    c = ensure_openai_client()
    if c is None:
        return "OpenAI client not available. Make sure OPENAI_API_KEY is set."
    verbosity = st.session_state.get("ai_verbosity", "normal")
    focus = st.session_state.get("ai_focus", "balanced")
    style = f"Verbosity: {verbosity}. Focus: {focus}."
    try:
        resp = c.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional running coach and data analyst.",
                },
                {
                    "role": "user",
                    "content": style + "\n\n" + prompt,
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling AI: {e}"


# =========================
# DB CRUD
# =========================

def insert_run(data: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO runs
        (date, run_type, distance, duration_minutes, avg_pace, splits,
         avg_hr, max_hr, hr_by_segment, cadence, elevation_gain, effort,
         terrain, weather, how_felt, pain, sleep_hours, stress,
         nutrition_notes, vo2max, hrv, shoe_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["date"],
            data["run_type"],
            data["distance"],
            data["duration_minutes"],
            data["avg_pace"],
            data["splits"],
            data["avg_hr"],
            data["max_hr"],
            data["hr_by_segment"],
            data["cadence"],
            data["elevation_gain"],
            data["effort"],
            data["terrain"],
            data["weather"],
            data["how_felt"],
            data["pain"],
            data["sleep_hours"],
            data["stress"],
            data["nutrition_notes"],
            data["vo2max"],
            data["hrv"],
            data["shoe_id"],
        ),
    )
    conn.commit()
    conn.close()


def update_run(run_id: int, data: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE runs SET
            date = ?,
            run_type = ?,
            distance = ?,
            duration_minutes = ?,
            avg_pace = ?,
            splits = ?,
            avg_hr = ?,
            max_hr = ?,
            hr_by_segment = ?,
            cadence = ?,
            elevation_gain = ?,
            effort = ?,
            terrain = ?,
            weather = ?,
            how_felt = ?,
            pain = ?,
            sleep_hours = ?,
            stress = ?,
            nutrition_notes = ?,
            vo2max = ?,
            hrv = ?,
            shoe_id = ?
        WHERE id = ?
        """,
        (
            data["date"],
            data["run_type"],
            data["distance"],
            data["duration_minutes"],
            data["avg_pace"],
            data["splits"],
            data["avg_hr"],
            data["max_hr"],
            data["hr_by_segment"],
            data["cadence"],
            data["elevation_gain"],
            data["effort"],
            data["terrain"],
            data["weather"],
            data["how_felt"],
            data["pain"],
            data["sleep_hours"],
            data["stress"],
            data["nutrition_notes"],
            data["vo2max"],
            data["hrv"],
            data["shoe_id"],
            run_id,
        ),
    )
    conn.commit()
    conn.close()


def delete_run(run_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


def fetch_runs():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY date ASC", conn)
    conn.close()
    return df


def insert_shoe(name, brand, start_date):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO shoes (name, brand, start_date, retired)
        VALUES (?, ?, ?, 0)
        """,
        (name, brand, start_date),
    )
    conn.commit()
    conn.close()


def fetch_shoes(include_retired=False):
    conn = get_conn()
    if include_retired:
        df = pd.read_sql_query("SELECT * FROM shoes ORDER BY id DESC", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM shoes WHERE retired = 0 ORDER BY id DESC", conn)
    conn.close()
    return df


def retire_shoe(shoe_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE shoes SET retired = 1 WHERE id = ?", (shoe_id,))
    conn.commit()
    conn.close()


# =========================
# GARMIN IMPORT
# =========================

def build_run_from_garmin_df(df: pd.DataFrame):
    if df.empty:
        return None, "CSV appears to be empty."

    row = df.iloc[0]
    cols = list(df.columns)

    def get_val(*names):
        for n in names:
            if n in cols and pd.notna(row[n]):
                return row[n]
        return None

    date_val = get_val("Start Time", "Start", "Date", "Timestamp")
    date_iso = datetime.today().date().isoformat()
    if date_val is not None:
        dt = pd.to_datetime(str(date_val), errors="coerce")
        if pd.notna(dt):
            date_iso = dt.date().isoformat()

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
                    duration_minutes = num / 60.0 if num > 200 else num
                except Exception:
                    pass
            break

    avg_pace = None
    for c in cols:
        if "pace" in c.lower():
            avg_pace = str(row[c])
            break

    avg_hr = get_val("Average Heart Rate", "Avg HR", "Average HR")
    try:
        avg_hr = int(float(avg_hr)) if avg_hr is not None else None
    except Exception:
        avg_hr = None

    max_hr = get_val("Maximum Heart Rate", "Max HR")
    try:
        max_hr = int(float(max_hr)) if max_hr is not None else None
    except Exception:
        max_hr = None

    cadence = get_val("Average Run Cadence", "Avg Run Cadence", "Run Cadence")
    try:
        cadence = int(float(cadence)) if cadence is not None else None
    except Exception:
        cadence = None

    elevation_gain = get_val("Elevation Gain", "Total Ascent", "Ascent")
    try:
        elevation_gain = int(float(elevation_gain)) if elevation_gain is not None else None
    except Exception:
        elevation_gain = None

    vo2 = get_val("VO2 Max", "VO2Max")
    try:
        vo2 = float(vo2) if vo2 is not None else None
    except Exception:
        vo2 = None

    hrv = get_val("HRV", "HRV (ms)")
    try:
        hrv = int(hrv) if hrv is not None else None
    except Exception:
        hrv = None

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
# =========================
# PAGES: HOME & FEED
# =========================

def render_home_page():
    st.title("🏠 Home")

    df = fetch_runs()
    if df.empty:
        st.info("You have no runs logged yet. Start by logging a run or importing from Garmin.")
        return

    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    prs = calculate_prs(metrics)

    total_miles = metrics["distance"].sum(skipna=True)
    last7 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7_miles = last7["distance"].sum(skipna=True)
    eff_avg = metrics["efficiency_score"].mean(skipna=True)
    avg_hr = metrics["avg_hr"].mean(skipna=True)

    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)
    current_streak, longest_streak = compute_streaks(metrics)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    col3.metric("Avg HR", f"{avg_hr:.1f}" if not pd.isna(avg_hr) else "—")
    col4.metric("Current Streak", f"{current_streak} days")

    st.markdown("<h4>Recent Runs</h4>", unsafe_allow_html=True)

    recent = df.tail(5).iloc[::-1]
    for _, row in recent.iterrows():
        rt = row["run_type"] or "Other"
        distance, dunit = convert_distance_for_display(row["distance"])
        pace = row["avg_pace"] or "—"
        hr = row["avg_hr"] or "—"
        elev = row["elevation_gain"] or "—"

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

        if st.button(f"Edit Run {row['id']}", key=f"edit_home_{row['id']}"):
            st.session_state["page"] = "Edit Run"
            st.session_state["edit_run_id"] = int(row["id"])
            st.rerun()


def render_feed_page():
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

        if st.button(f"Edit Run {row['id']}", key=f"edit_feed_{row['id']}"):
            st.session_state["page"] = "Edit Run"
            st.session_state["edit_run_id"] = int(row["id"])
            st.rerun()


# =========================
# LOG / EDIT / IMPORT PAGES
# =========================

def render_log_run_page():
    st.title("📝 Log a Run")

    shoes_df = fetch_shoes()
    shoe_options = ["None"] + [f"{row['name']} ({row['brand']})" for _, row in shoes_df.iterrows()]

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

        if submitted:
            df_before = fetch_runs()
            metrics_before = prepare_metrics_df(df_before) if not df_before.empty else df_before
            prs_before = calculate_prs(metrics_before) if not df_before.empty else {}

            minutes = duration_to_minutes(duration_str)
            if minutes is None:
                st.error("Invalid duration format. Use MM:SS or HH:MM:SS.")
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

            insert_run(data)
            st.success("Run saved! ✅")

            df_after = fetch_runs()
            metrics_after = prepare_metrics_df(df_after)
            prs_after = calculate_prs(metrics_after)
            improved = detect_pr_improvements(prs_before, prs_after)
            if improved:
                st.success("🎉 New PRs: " + " | ".join(improved))


def render_edit_run_page(run_id: int):
    st.title("✏️ Edit Run")

    df = fetch_runs()
    row = df[df["id"] == run_id]
    if row.empty:
        st.error("Run not found.")
        return

    row = row.iloc[0]

    shoes_df = fetch_shoes()
    shoe_options = ["None"] + [f"{r['name']} ({r['brand']})" for _, r in shoes_df.iterrows()]

    current_shoe = "None"
    if row["shoe_id"] and not pd.isna(row["shoe_id"]) and not shoes_df.empty:
        if row["shoe_id"] in shoes_df["id"].values:
            srow = shoes_df[shoes_df["id"] == row["shoe_id"]].iloc[0]
            current_shoe = f"{srow['name']} ({srow['brand']})"

    with st.form("edit_run"):
        date_val = st.date_input("Date", datetime.fromisoformat(row["date"]))
        rt_list = ["Easy", "LongRun", "Tempo", "Intervals", "Recovery", "Race", "Other"]
        try:
            idx_rt = rt_list.index(row["run_type"])
        except Exception:
            idx_rt = 0
        run_type = st.selectbox("Run Type", rt_list, index=idx_rt)

        distance = float(row["distance"]) if not pd.isna(row["distance"]) else 0.0
        distance = st.number_input("Distance (mi)", min_value=0.0, step=0.01, value=distance)

        duration_minutes = row["duration_minutes"] if not pd.isna(row["duration_minutes"]) else None
        duration_str = minutes_to_hms(duration_minutes)
        duration_str = st.text_input("Duration (MM:SS or HH:MM:SS)", value=duration_str)

        avg_pace = st.text_input("Average Pace (MM:SS)", value=row["avg_pace"] or "")
        splits = st.text_area("Splits", value=row["splits"] or "")

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

        effort = st.slider("Effort (1–10)", 1, 10, effort_val)
        terrain = st.text_input("Terrain", value=row["terrain"] or "")
        weather = st.text_input("Weather", value=row["weather"] or "")
        how_felt = st.text_input("How I Felt", value=row["how_felt"] or "")
        pain = st.text_input("Any Pain or Tightness", value=row["pain"] or "")

        sleep = st.number_input("Sleep (hrs)", min_value=0.0, step=0.1, value=sleep_val)
        stress = st.slider("Stress (1–5)", 1, 5, stress_val)
        nutrition = st.text_area("Nutrition / Hydration Notes", value=row["nutrition_notes"] or "")

        vo2 = st.number_input("VO2 Max", min_value=0.0, step=0.1, value=vo2_val)
        hrv = st.number_input("HRV", min_value=0, max_value=300, value=hrv_val)

        shoe_choice = st.selectbox("Shoe", shoe_options, index=shoe_options.index(current_shoe))

        save_btn = st.form_submit_button("Save Changes")
        delete_btn = st.form_submit_button("Delete Run")

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

        if delete_btn:
            delete_run(run_id)
            st.warning("Run deleted.")
            st.session_state["page"] = "Feed"
            st.session_state["edit_run_id"] = None
            st.rerun()


def render_garmin_import_page():
    st.title("📥 Garmin CSV Import")

    uploaded = st.file_uploader("Upload Garmin CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
        data, msg = build_run_from_garmin_df(df)
        st.info(msg)

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


# =========================
# DASHBOARD
# =========================

def render_dashboard_page():
    st.title("📊 Dashboard")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs to view your dashboard.")
        return

    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    metrics = add_hr_zones(metrics, hr_max=st.session_state.get("hr_max", 190))
    daily = compute_daily_load(metrics)
    load_df = compute_fitness_fatigue(daily)
    prs = calculate_prs(metrics)

    total_miles = metrics["distance"].sum(skipna=True)
    last7 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7_miles = last7["distance"].sum(skipna=True)
    avg_eff = metrics["efficiency_score"].mean(skipna=True)
    avg_hr = metrics["avg_hr"].mean(skipna=True)
    current_streak, longest_streak = compute_streaks(metrics)

    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    col3.metric("Avg HR", f"{avg_hr:.1f}" if not pd.isna(avg_hr) else "—")
    col4.metric("Current Streak", f"{current_streak} days")
    st.markdown("</div>", unsafe_allow_html=True)

    if prs:
        if metrics["distance"].max() == prs.get("longest_distance"):
            st.markdown("<div class='pr-banner'>🏆 New Longest Run!</div>", unsafe_allow_html=True)

        df_copy = metrics.copy()
        df_copy["week"] = df_copy["date_dt"].dt.isocalendar().week
        weekly = df_copy.groupby("week")["distance"].sum()
        if weekly.max() == prs.get("highest_weekly_mileage"):
            st.markdown("<div class='pr-banner'>🔥 Highest Weekly Mileage Ever!</div>", unsafe_allow_html=True)

        df_copy["month"] = df_copy["date_dt"].dt.month
        monthly = df_copy.groupby("month")["distance"].sum()
        if monthly.max() == prs.get("highest_monthly_mileage"):
            st.markdown("<div class='pr-banner'>📈 Highest Monthly Mileage Ever!</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weekly Distance")
    dfw = metrics.copy()
    dfw["week"] = dfw["date_dt"].dt.isocalendar().week
    weekly = dfw.groupby("week")["distance"].sum().reset_index()
    if not weekly.empty:
        weekly["display_dist"], _ = zip(*weekly["distance"].apply(convert_distance_for_display))
        chart = (
            alt.Chart(weekly)
            .mark_bar()
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("display_dist:Q", title=f"Distance ({unit})"),
                tooltip=["week", "display_dist"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        if prs.get("highest_weekly_mileage"):
            st.markdown(
                f"<span class='pr-mini'>🔥 PR Week: {prs['highest_weekly_mileage']:.1f} mi</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Not enough data for weekly chart.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pace Trend (min/mi)")
    p = metrics.dropna(subset=["pace_min_per_mile"])
    if not p.empty:
        chart = (
            alt.Chart(p)
            .mark_line(point=True)
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("pace_min_per_mile:Q", title="Pace (min/mi)", scale=alt.Scale(reverse=True)),
                tooltip=["date", "distance", "avg_pace"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        fastest_pace = prs.get("fastest_pace")
        if fastest_pace:
            st.markdown(
                f"<span class='pr-mini'>⚡ Fastest Pace PR: {fastest_pace:.2f} min/mi</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Log distance and duration to see pace trends.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("VO2 Max & Efficiency Over Time")
    eff = metrics.dropna(subset=["efficiency_score"])
    vo2 = metrics.dropna(subset=["vo2max"])

    if not eff.empty or not vo2.empty:
        eff_plot = eff[["date_dt", "efficiency_score"]].copy()
        eff_plot["metric"] = "Efficiency"
        eff_plot = eff_plot.rename(columns={"efficiency_score": "value"})

        vo2_plot = vo2[["date_dt", "vo2max"]].copy()
        vo2_plot["metric"] = "VO2 Max"
        vo2_plot = vo2_plot.rename(columns={"vo2max": "value"})

        combo = pd.concat([eff_plot, vo2_plot], ignore_index=True)
        chart = (
            alt.Chart(combo)
            .mark_line(point=True)
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("value:Q", title="Value"),
                color="metric:N",
                tooltip=["date_dt", "metric", "value"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Log VO2 max and HR-based runs to see these curves.")
    st.markdown("</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Average HR Over Time")
        hr_df = metrics.dropna(subset=["avg_hr"])
        if not hr_df.empty:
            chart = (
                alt.Chart(hr_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date_dt:T", title="Date"),
                    y=alt.Y("avg_hr:Q", title="Avg HR (bpm)"),
                    tooltip=["date_dt", "avg_hr"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No HR data logged.")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("HR Zone Distribution")
        z = metrics.dropna(subset=["hr_zone"])
        zone_df = z.groupby("hr_zone", as_index=False)["duration_minutes"].sum()
        if not zone_df.empty:
            chart = (
                alt.Chart(zone_df)
                .mark_arc()
                .encode(
                    theta=alt.Theta("duration_minutes:Q", title="Minutes"),
                    color=alt.Color("hr_zone:N", title="HR Zone"),
                    tooltip=["hr_zone", "duration_minutes"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Need HR data to show zones.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Cadence Trend (spm)")
    cad = metrics.dropna(subset=["cadence"])
    if not cad.empty:
        chart = (
            alt.Chart(cad)
            .mark_line(point=True)
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("cadence:Q", title="Cadence (spm)"),
                tooltip=["date_dt", "cadence"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Log cadence to view this trend.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Load: Fitness (CTL), Fatigue (ATL), Form (TSB)")
    if not load_df.empty:
        melted = load_df.melt(
            id_vars="date_dt",
            value_vars=["CTL", "ATL", "TSB"],
            var_name="metric",
            value_name="value",
        )
        chart = (
            alt.Chart(melted)
            .mark_line()
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("value:Q", title="Load"),
                color="metric:N",
                tooltip=["date_dt", "metric", "value"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough training load data yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏁 Race Prediction Trend")
    pred_df = generate_race_prediction_series(metrics)
    if pred_df.empty:
        st.info("Not enough data to generate race predictions.")
    else:
        melt = pred_df.melt(id_vars="date", var_name="race", value_name="minutes")
        chart = (
            alt.Chart(melt)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("minutes:Q", title="Predicted Finish Time (min)"),
                color="race:N",
                tooltip=["date", "race", "minutes"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        last = pred_df.iloc[-1]
        colA2, colB2, colC2, colD2 = st.columns(4)
        colA2.metric("5K", f"{last['5K']:.1f} min")
        colB2.metric("10K", f"{last['10K']:.1f} min")
        colC2.metric("Half", f"{last['Half']:.1f} min")
        colD2.metric("Marathon", f"{last['Marathon']:.1f} min")
    st.markdown("</div>", unsafe_allow_html=True)

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
            chart = (
                alt.Chart(sleep_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date_dt:T", title="Date"),
                    y=alt.Y("sleep_hours:Q", title="Sleep (hrs)"),
                    tooltip=["date_dt", "sleep_hours"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

        stress_df = recent.dropna(subset=["stress"])
        if not stress_df.empty:
            chart = (
                alt.Chart(stress_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date_dt:T", title="Date"),
                    y=alt.Y("stress:Q", title="Stress (1–5)"),
                    tooltip=["date_dt", "stress"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Injury Risk Snapshot")
    last14 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=14)]
    if last14.empty:
        st.info("Not enough recent data for injury risk.")
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

        risk_score = 0
        if load_ratio > 1.3:
            risk_score += 2
        if avg_effort and avg_effort > 7:
            risk_score += 2
        if avg_sleep7 and avg_sleep7 < 6:
            risk_score += 2
        if avg_stress7 and avg_stress7 > 3:
            risk_score += 1

        if risk_score >= 5:
            risk_label = "High"
        elif risk_score >= 3:
            risk_label = "Moderate"
        else:
            risk_label = "Low"

        st.write(f"**Estimated Injury Risk:** {risk_label} (score {risk_score}/7)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👟 Shoe Mileage")
    shoes_df = fetch_shoes(include_retired=True)
    runs_df = fetch_runs()
    if shoes_df.empty:
        st.info("No shoes tracked yet.")
    else:
        for _, s in shoes_df.iterrows():
            shoe_runs = runs_df[runs_df["shoe_id"] == s["id"]]
            mileage = shoe_runs["distance"].sum() if not shoe_runs.empty else 0.0
            status = "Active" if s["retired"] == 0 else "Retired"
            warn = ""
            if mileage > 350:
                warn = "⚠️ Nearing end of life"
            elif mileage > 300:
                warn = "⚠️ Getting worn"
            st.markdown(
                f"""
                <div class='card'>
                    <strong>{s['name']} ({s['brand']})</strong> — {status}<br>
                    Total Miles: {mileage:.1f}<br>
                    {warn}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

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
# =========================
# AI COACH PAGE (with advanced schedule controls)
# =========================

def render_ai_coach_page():
    st.title("🤖 AI Coach")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs or import Garmin data to use the AI Coach.")
        return

    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    recent = df.tail(30)
    latest = df.iloc[-1].to_dict()
    race_goal = st.session_state.get("race_goal", "")
    race_date = st.session_state.get("race_date_str", "")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Daily & Weekly",
            "Workout Generator",
            "7-Day Plan",
            "Race Simulator",
            "Injury Risk AI",
            "PR Milestones",
            "Training Block",
        ]
    )

    # --- Tab 1: Daily & Weekly ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'><h3>Last Run Analysis</h3></div>", unsafe_allow_html=True)
            if st.button("Analyze Last Run"):
                prompt = (
                    "Analyze my most recent run in detail. Include pacing, HR, efficiency, fatigue, "
                    "injury risk, and 3–5 concrete actions.\n\n"
                    f"Run: {latest}"
                )
                st.write(call_ai(prompt))

        with col2:
            st.markdown("<div class='card'><h3>Weekly Summary</h3></div>", unsafe_allow_html=True)
            last7_df = df[pd.to_datetime(df["date"]) >= datetime.today() - timedelta(days=7)]
            week_metrics = prepare_metrics_df(last7_df) if not last7_df.empty else last7_df
            week_prs = calculate_prs(week_metrics) if not last7_df.empty else {}
            if week_prs:
                st.write("**PRs This Week:**")
                for k, v in week_prs.items():
                    label = k.replace("_", " ").title()
                    if "fastest" in k:
                        st.write(f"⚡ {label}: {v:.2f} min")
                    else:
                        st.write(f"🔥 {label}: {v:.2f} mi")

            if st.button("Summarize Last 7 Days"):
                prompt = (
                    "Provide a detailed weekly training summary including mileage, pace trends, HR trends, "
                    "efficiency changes, fatigue level, and suggested training load next week.\n\n"
                    f"Weekly PRs: {week_prs}\n\n"
                    f"Weekly runs: {last7_df.to_dict('records')}"
                )
                st.write(call_ai(prompt))

    # --- Tab 2: Workout Generator ---
    with tab2:
        st.markdown("<div class='card'><h3>Generate Tomorrow’s Workout</h3></div>", unsafe_allow_html=True)
        if st.button("Create Tomorrow's Workout"):
            prompt = (
                "Based on my last 2–3 weeks of runs and my goal race, generate a single workout for tomorrow "
                "including warm-up, main set, paces or HR zones, cooldown, and rationale.\n\n"
                f"Race goal: {race_goal} on {race_date}\n\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- Tab 3: 7-Day Plan (with advanced schedule controls) ---
    with tab3:
        st.markdown("<div class='card'><h3>Next 7-Day Plan</h3></div>", unsafe_allow_html=True)

        st.subheader("Training Schedule Preferences")

        days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        training_days_per_week = st.slider(
            "Days per week you want to run",
            min_value=2,
            max_value=7,
            value=5,
            key="7d_days_per_week",
        )

        default_training_days = ["Mon", "Tue", "Thu", "Sat", "Sun"]
        training_days = st.multiselect(
            "Which days do you want to train?",
            options=days_of_week,
            default=default_training_days[:training_days_per_week],
            key="7d_training_days",
        )

        hard_days = st.multiselect(
            "Preferred HARD workout days (tempo/intervals)",
            options=days_of_week,
            default=["Tue", "Thu"],
            key="7d_hard_days",
        )

        rest_days = st.multiselect(
            "Preferred rest days",
            options=days_of_week,
            default=["Fri"],
            key="7d_rest_days",
        )

        long_run_day = st.selectbox(
            "Primary long run day",
            options=days_of_week,
            index=6,  # Sunday
            key="7d_long_run_day",
        )

        secondary_options = ["None"] + days_of_week
        secondary_long_run = st.selectbox(
            "Optional secondary long run day (for back-to-back endurance)",
            options=secondary_options,
            index=0,
            key="7d_secondary_long_run",
        )

        allow_back_to_back = st.checkbox(
            "Allow back-to-back hard days",
            value=False,
            key="7d_allow_back_to_back",
        )

        allow_doubles = st.checkbox(
            "Allow double days (AM/PM)?",
            value=False,
            key="7d_allow_doubles",
        )

        if len(training_days) != training_days_per_week:
            st.warning(
                f"You selected {training_days_per_week} days per week but chose "
                f"{len(training_days)} training days. The AI will still try to honor both, "
                "but you may want to align them."
            )

        if st.button("Generate 7-Day Plan"):
            prefs_text = f"""
Training schedule preferences:
- Days per week: {training_days_per_week}
- Training days: {", ".join(training_days) if training_days else "None specified"}
- Key workout days: {", ".join(hard_days) if hard_days else "None specified"}
- Rest days: {", ".join(rest_days) if rest_days else "None specified"}
- Long run day: {long_run_day}
- Secondary long run: {secondary_long_run}
- Allow back-to-back hard days: {"Yes" if allow_back_to_back else "No"}
- Allow doubles: {"Yes" if allow_doubles else "No"}
"""

            prompt = (
                "Using my recent training and my goal race, build a structured 7-day plan including run types, "
                "distances, suggested paces, HR zones, and rest days. Respect my training schedule preferences.\n\n"
                f"Race goal: {race_goal} on {race_date}\n\n"
                f"{prefs_text}\n\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- Tab 4: Race Simulator ---
    with tab4:
        st.markdown("<div class='card'><h3>Race Day Simulation</h3></div>", unsafe_allow_html=True)
        if st.button("Simulate Race Performance"):
            prompt = (
                "Simulate my race performance for my goal event. Provide predicted finish time range, pacing plan, "
                "HR zones per segment, fueling, and mental strategies.\n\n"
                f"Race goal: {race_goal} on {race_date}\n\n"
                f"All training: {df.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- Tab 5: Injury Risk AI ---
    with tab5:
        st.markdown("<div class='card'><h3>AI Injury Risk</h3></div>", unsafe_allow_html=True)
        if st.button("Evaluate Injury Risk"):
            prompt = (
                "Evaluate my injury risk with emphasis on shin splints. Analyze training load spikes, pace spikes, "
                "elevation, sleep, stress, and pain notes. Give a risk rating and 4–6 adjustments.\n\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- Tab 6: PR Milestones ---
    with tab6:
        st.markdown("<div class='card'><h3>🏆 PR Milestone Analysis</h3></div>", unsafe_allow_html=True)
        prs_all = calculate_prs(metrics)
        st.subheader("Current PRs")
        for key, val in prs_all.items():
            label = key.replace("_", " ").title()
            if "fastest" in key:
                st.write(f"⚡ **{label}:** {val:.2f} min")
            else:
                st.write(f"🔥 **{label}:** {val:.2f} mi")

        if st.button("Analyze PR Progress"):
            trend_data = metrics.sort_values("date_dt").to_dict("records")
            prompt = f"""
You are an elite running coach. Analyze the athlete’s PR progression based on these run records.

1. Summary of PR Progress
2. Reasons for Improvement (efficiency, cadence, HR drift, load)
3. Weak Spots (which PR stalled)
4. Next Most Likely PR & Timeline
5. 3–5 Targeted Workouts to Break That PR
6. 6–8 Week Projection

Training history:
{trend_data}

Current PRs:
{prs_all}
"""
            st.write(call_ai(prompt))

    # --- Tab 7: Training Block (with advanced schedule controls) ---
    with tab7:
        st.markdown("<div class='card'><h3>📆 Training Block Generator</h3></div>", unsafe_allow_html=True)

        block_length_weeks = st.slider(
            "Block Length (weeks)",
            min_value=4,
            max_value=20,
            value=12,
            key="block_length_weeks",
        )

        st.subheader("Training Schedule Preferences for the Block")

        days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        block_days_per_week = st.slider(
            "Days per week you want to run (for this block)",
            min_value=2,
            max_value=7,
            value=5,
            key="block_days_per_week",
        )

        default_block_days = ["Mon", "Tue", "Thu", "Sat", "Sun"]
        block_training_days = st.multiselect(
            "Which days do you want to train in this block?",
            options=days_of_week,
            default=default_block_days[:block_days_per_week],
            key="block_training_days",
        )

        block_hard_days = st.multiselect(
            "Preferred HARD workout days (tempo/intervals) in this block",
            options=days_of_week,
            default=["Tue", "Thu"],
            key="block_hard_days",
        )

        block_rest_days = st.multiselect(
            "Preferred rest days in this block",
            options=days_of_week,
            default=["Fri"],
            key="block_rest_days",
        )

        block_long_run_day = st.selectbox(
            "Primary long run day (block)",
            options=days_of_week,
            index=6,  # Sunday
            key="block_long_run_day",
        )

        block_secondary_long_run = st.selectbox(
            "Optional secondary long run day (block)",
            options=["None"] + days_of_week,
            index=0,
            key="block_secondary_long_run",
        )

        block_allow_back_to_back = st.checkbox(
            "Allow back-to-back hard days in this block",
            value=False,
            key="block_allow_back_to_back",
        )

        block_allow_doubles = st.checkbox(
            "Allow double days (AM/PM) in this block",
            value=False,
            key="block_allow_doubles",
        )

        if len(block_training_days) != block_days_per_week:
            st.warning(
                f"You selected {block_days_per_week} days per week but chose "
                f"{len(block_training_days)} training days. The AI will still try to honor both, "
                "but you may want to align them."
            )

        if st.button("Generate Training Block"):
            prefs_text = f"""
Training schedule preferences for the block:
- Days per week: {block_days_per_week}
- Training days: {", ".join(block_training_days) if block_training_days else "None specified"}
- Key workout days: {", ".join(block_hard_days) if block_hard_days else "None specified"}
- Rest days: {", ".join(block_rest_days) if block_rest_days else "None specified"}
- Long run day: {block_long_run_day}
- Secondary long run: {block_secondary_long_run}
- Allow back-to-back hard days: {"Yes" if block_allow_back_to_back else "No"}
- Allow doubles: {"Yes" if block_allow_doubles else "No"}
"""

            prompt = f"""
You are designing a {block_length_weeks}-week half marathon training block.

Phases:
- Base
- Build
- Peak
- Taper

Use my existing training history and PRs to structure this block. Include each week's:
- total mileage
- key workouts
- easy runs
- long runs
- cutback weeks
- recovery guidance

Respect my training schedule preferences.

Race goal: {race_goal} on {race_date}

{prefs_text}

Training history:
{df.to_dict('records')}

PRs:
{calculate_prs(metrics)}
"""
            st.write(call_ai(prompt))


# =========================
# COMPARE RUNS
# =========================

def render_compare_runs_page():
    st.title("📊 Compare Two Runs")

    df = fetch_runs()
    if df.empty or len(df) < 2:
        st.info("Log at least two runs to compare them.")
        return

    df_sorted = df.sort_values("date", ascending=False)
    labels = [f"{r['date']} — {r['run_type']} — {r['distance']} mi" for _, r in df_sorted.iterrows()]
    mapping = {label: row for label, (_, row) in zip(labels, df_sorted.iterrows())}

    col1, col2 = st.columns(2)
    with col1:
        r1_label = st.selectbox("Run 1", labels)
    with col2:
        r2_label = st.selectbox("Run 2", labels)

    if r1_label == r2_label:
        st.warning("Select two different runs.")
        return

    run1 = mapping[r1_label]
    run2 = mapping[r2_label]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Side-by-Side Metrics")

    metrics_list = [
        ("Distance (mi)", "distance"),
        ("Duration (min)", "duration_minutes"),
        ("Avg HR", "avg_hr"),
        ("Max HR", "max_hr"),
        ("Cadence", "cadence"),
        ("Elev Gain (ft)", "elevation_gain"),
        ("Effort", "effort"),
        ("VO2 Max", "vo2max"),
        ("HRV", "hrv"),
    ]

    colA, colB, colC = st.columns(3)
    colA.write("**Metric**")
    colB.write("**Run 1**")
    colC.write("**Run 2**")

    for label, key in metrics_list:
        colA.write(label)
        colB.write(run1[key] if key in run1 else "—")
        colC.write(run2[key] if key in run2 else "—")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Comparison")
    if st.button("Generate AI Comparison"):
        prompt = f"""
Compare these two runs in detail. Cover pacing, HR response, efficiency, elevation,
recovery, and training impact. Give a diagnosis of what improved or regressed and
advice on what to focus on next.

Run 1:
{dict(run1)}

Run 2:
{dict(run2)}
"""
        st.write(call_ai(prompt))
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# PACE ZONES PAGE
# =========================

def render_pace_zones_page():
    st.title("📏 Pace & HR Zones")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs to generate pace zones.")
        return

    metrics = prepare_metrics_df(df)
    prs = calculate_prs(metrics)
    best_pace = prs.get("fastest_pace")

    threshold_pace = estimate_threshold_pace(prs)
    easy_min, easy_max = estimate_easy_pace(best_pace) if best_pace else (None, None)
    tempo = estimate_tempo_pace(threshold_pace) if threshold_pace else None
    interval = estimate_interval_pace(best_pace) if best_pace else None
    rep = estimate_rep_pace(best_pace) if best_pace else None

    hr_max = st.session_state.get("hr_max", 190)
    hr_zones = calculate_hr_zones(hr_max)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pace Zones (min/mi)")

    pace_data = []

    def add_zone(name, pr):
        if pr and pr[0] and pr[1]:
            pace_data.append(
                {
                    "Zone": name,
                    "Min Pace (min/mi)": f"{pr[0]:.2f}",
                    "Max Pace (min/mi)": f"{pr[1]:.2f}",
                }
            )

    if easy_min and easy_max:
        add_zone("Recovery", (easy_min + 0.4, easy_max + 0.6))
        add_zone("Easy", (easy_min, easy_max))
        add_zone("Aerobic Endurance", (easy_min - 0.3, easy_min + 0.3))
    if tempo:
        add_zone("Tempo", tempo)
    if threshold_pace:
        add_zone("Threshold", (threshold_pace * 0.98, threshold_pace * 1.02))
    if interval:
        add_zone("Interval / VO2", interval)
    if rep:
        add_zone("Repetition", rep)

    if pace_data:
        st.table(pd.DataFrame(pace_data))
    else:
        st.info("Need pace data to calculate zones.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("HR Zones")

    hr_data = []
    for zone, (low, high) in hr_zones.items():
        hr_data.append(
            {"Zone": zone, "Low (bpm)": int(low), "High (bpm)": int(high)}
        )
    st.table(pd.DataFrame(hr_data))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Zone Workout Generator")

    zone_choice = st.selectbox(
        "Select a Zone",
        [
            "Recovery",
            "Easy",
            "Aerobic Endurance",
            "Tempo",
            "Threshold",
            "Interval / VO2",
            "Repetition",
        ],
    )
    if st.button("Generate Zone Workout"):
        prompt = f"""
Create a structured running workout for the **{zone_choice}** zone.
Include:
- warm-up
- main set
- cooldown
- pacing instructions
- HR zone guidance
- purpose of the workout
- where it fits within a half marathon training block
"""
        st.write(call_ai(prompt))

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# SETTINGS PAGE
# =========================

def render_settings_page():
    st.title("⚙ Settings")

    st.markdown("<div class='card'><h3>Appearance</h3></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        theme = st.radio(
            "Theme",
            ["dark", "light"],
            index=0 if st.session_state["theme"] == "dark" else 1,
        )
        st.session_state["theme"] = theme
    with col2:
        units = st.radio(
            "Units",
            ["mi", "km"],
            index=0 if st.session_state["units"] == "mi" else 1,
        )
        st.session_state["units"] = units
    with col3:
        compact = st.checkbox(
            "Compact mode",
            value=st.session_state["compact_mode"],
        )
        st.session_state["compact_mode"] = compact

    st.markdown("<div class='card'><h3>AI Coaching</h3></div>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        verbosity = st.selectbox(
            "AI Verbosity",
            ["short", "normal", "detailed"],
            index=["short", "normal", "detailed"].index(st.session_state["ai_verbosity"]),
        )
        st.session_state["ai_verbosity"] = verbosity
    with col5:
        focus = st.selectbox(
            "AI Focus",
            ["balanced", "performance", "injury prevention"],
            index=["balanced", "performance", "injury prevention"].index(
                st.session_state["ai_focus"]
            ),
        )
        st.session_state["ai_focus"] = focus

    st.markdown("<div class='card'><h3>Race Goal & HR</h3></div>", unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        goal = st.text_input("Race goal description", value=st.session_state["race_goal"])
        st.session_state["race_goal"] = goal
        race_date_input = st.date_input(
            "Race date", value=datetime.fromisoformat(st.session_state["race_date_str"]).date()
        )
        st.session_state["race_date_str"] = race_date_input.isoformat()
    with col7:
        hr_max = st.number_input(
            "HR Max (bpm)",
            min_value=140,
            max_value=220,
            value=st.session_state.get("hr_max", 190),
        )
        st.session_state["hr_max"] = hr_max
        weekly_goal = st.number_input(
            "Weekly mileage goal (mi)",
            min_value=0.0,
            max_value=200.0,
            value=float(st.session_state.get("weekly_goal_mi", 25.0)),
        )
        st.session_state["weekly_goal_mi"] = weekly_goal

    st.markdown("<div class='card'><h3>Shoes</h3></div>", unsafe_allow_html=True)
    st.write("Track your running shoes and mileage.")

    with st.form("add_shoe"):
        colA, colB = st.columns(2)
        with colA:
            shoe_name = st.text_input("Shoe Name (e.g., Clifton 10)")
        with colB:
            shoe_brand = st.text_input("Brand (e.g., Hoka)")
        start_date = st.date_input("Start Date", datetime.today().date())
        add_btn = st.form_submit_button("Add Shoe")
        if add_btn:
            insert_shoe(shoe_name, shoe_brand, start_date.isoformat())
            st.success("Shoe added!")

    shoes_df = fetch_shoes(include_retired=True)
    runs_df = fetch_runs()
    if shoes_df.empty:
        st.info("No shoes added yet.")
    else:
        for _, s in shoes_df.iterrows():
            shoe_runs = runs_df[runs_df["shoe_id"] == s["id"]]
            mileage = shoe_runs["distance"].sum() if not shoe_runs.empty else 0.0
            status = "Retired" if s["retired"] else "Active"
            st.markdown(
                f"""
                <div class='card'>
                    <strong>{s['name']} ({s['brand']})</strong> — {status}<br>
                    Start: {s['start_date']}<br>
                    Miles: {mileage:.1f}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not s["retired"]:
                if st.button(f"Retire {s['name']}", key=f"retire_{s['id']}"):
                    retire_shoe(s["id"])
                    st.warning(f"{s['name']} retired.")

    st.markdown("<div class='card'><h3>Data</h3></div>", unsafe_allow_html=True)
    col8, col9 = st.columns(2)
    with col8:
        if st.button("Export Run Log to CSV"):
            df = fetch_runs()
            if df.empty:
                st.info("No runs to export.")
            else:
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="run_log_export.csv",
                    mime="text/csv",
                )
    with col9:
        if st.button("⚠ Delete ALL Data"):
            conn = get_conn()
            conn.execute("DELETE FROM runs")
            conn.commit()
            conn.close()
            st.warning("All run data deleted.")


# =========================
# MAIN
# =========================

def main():
    st.set_page_config(page_title="Run Tracker & AI Coach", layout="wide")
    init_session_state()
    init_db_with_migration()
    inject_css()

    st.sidebar.header("Navigation")

    st.sidebar.radio(
        "Theme",
        ["dark", "light"],
        index=0 if st.session_state["theme"] == "dark" else 1,
        key="sidebar_theme",
    )
    st.session_state["theme"] = st.session_state["sidebar_theme"]

    st.sidebar.radio(
        "Units",
        ["mi", "km"],
        index=0 if st.session_state["units"] == "mi" else 1,
        key="sidebar_units",
    )
    st.session_state["units"] = st.session_state["sidebar_units"]

    st.sidebar.checkbox(
        "Compact mode",
        value=st.session_state["compact_mode"],
        key="sidebar_compact",
    )
    st.session_state["compact_mode"] = st.session_state["sidebar_compact"]

    st.sidebar.text_input(
        "Race Goal (sidebar)",
        value=st.session_state["race_goal"],
        key="sidebar_race_goal",
    )
    st.session_state["race_goal"] = st.session_state["sidebar_race_goal"]

    page = st.sidebar.radio(
        "Page",
        [
            "Home",
            "Feed",
            "Log a Run",
            "Dashboard",
            "Garmin Import",
            "AI Coach",
            "Compare Runs",
            "Pace Zones",
            "Settings",
        ],
        index=[
            "Home",
            "Feed",
            "Log a Run",
            "Dashboard",
            "Garmin Import",
            "AI Coach",
            "Compare Runs",
            "Pace Zones",
            "Settings",
        ].index(st.session_state.get("page", "Home")),
    )
    st.session_state["page"] = page

    if page == "Home":
        render_home_page()
    elif page == "Feed":
        render_feed_page()
    elif page == "Log a Run":
        render_log_run_page()
    elif page == "Dashboard":
        render_dashboard_page()
    elif page == "Garmin Import":
        render_garmin_import_page()
    elif page == "AI Coach":
        render_ai_coach_page()
    elif page == "Compare Runs":
        render_compare_runs_page()
    elif page == "Pace Zones":
        render_pace_zones_page()
    elif page == "Settings":
        render_settings_page()
    elif page == "Edit Run":
        rid = st.session_state.get("edit_run_id")
        if rid is None:
            st.error("No run selected to edit.")
        else:
            render_edit_run_page(int(rid))


if __name__ == "__main__":
    main()
