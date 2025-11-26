import sqlite3
from datetime import datetime, timedelta, date

import altair as alt
import pandas as pd
import streamlit as st

# ---------- OpenAI client ----------
try:
    from openai import OpenAI
    client = OpenAI()
except ImportError:
    client = None

DB_PATH = "run_log.db"


# ---------- Session defaults ----------

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
    if "hr_max" not in st.session_state:
        st.session_state["hr_max"] = 190
    if "weekly_goal_mi" not in st.session_state:
        st.session_state["weekly_goal_mi"] = 25.0
    if "compact_mode" not in st.session_state:
        st.session_state["compact_mode"] = False
    if "race_date_str" not in st.session_state:
        st.session_state["race_date_str"] = "2026-05-03"


# ---------- GLOBAL CSS ----------

def inject_css():
    theme = st.session_state.get("theme", "dark")
    compact = st.session_state.get("compact_mode", False)

    if theme == "light":
        bg = "#F7F7FA"
        bg2 = "#FFFFFF"
        text = "#16161D"
        card_bg = "#FFFFFF"
        border = "rgba(0,0,0,0.06)"
    else:
        bg = "#05060A"
        bg2 = "#0D1117"
        text = "#F5F7FF"
        card_bg = "#111827"
        border = "rgba(255,255,255,0.08)"

    card_padding = "0.8rem 1.0rem" if compact else "1.1rem 1.3rem"
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
        padding-top: 1.2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }}

    h1, h2, h3, h4 {{
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }}

    .card {{
        background-color: {card_bg};
        padding: {card_padding};
        border-radius: 14px;
        margin-bottom: {card_margin};
        border: 1px solid {border};
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }}
    .card:hover {{
        transform: translateY(-1px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.32);
        border-color: rgba(37,99,235,0.5);
    }}

    .feed-card {{
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }}

    .feed-header-line {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        row-gap: 0.2rem;
    }}

    .feed-main-metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 1.0rem;
        margin-top: 0.3rem;
        align-items: baseline;
    }}

    .big-distance {{
        font-size: 1.6rem;
        font-weight: 600;
    }}

    .muted {{
        opacity: 0.7;
        font-size: 0.85rem;
    }}

    .tag {{
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        color: white;
    }}

    .tag-Easy {{ background-color: #22C55E; }}
    .tag-LongRun {{ background-color: #3B82F6; }}
    .tag-Tempo {{ background-color: #F97316; }}
    .tag-Intervals {{ background-color: #EC4899; }}
    .tag-Recovery {{ background-color: #64748B; }}
    .tag-Race {{ background-color: #A855F7; }}
    .tag-Other {{ background-color: #6B7280; }}

    .pr-banner {{
        background: linear-gradient(90deg, #ffb300, #ffdd66);
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        color: black;
        text-align: center;
    }}

    .pr-mini {{
        background: #ffe29a;
        padding: 6px 10px;
        border-radius: 8px;
        display: inline-block;
        margin-left: 8px;
        font-size: 14px;
        font-weight: bold;
    }}

    .pr-badge {{
        background: #ffcc00;
        padding: 6px 12px;
        margin-top: 6px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }}

    </style>
    """,
        unsafe_allow_html=True,
    )


# ---------- DB helpers ----------

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

    conn.commit()
    conn.close()


def migrate_db():
    conn = get_conn()
    cur = conn.cursor()

    # Shoes table
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

    # ensure shoe_id column exists
    cur.execute("PRAGMA table_info(runs)")
    cols = [row[1] for row in cur.fetchall()]
    if "shoe_id" not in cols:
        cur.execute("ALTER TABLE runs ADD COLUMN shoe_id INTEGER")

    conn.commit()
    conn.close()


def init_db_with_migration():
    init_db()
    migrate_db()


# ---------- Core utilities & analytics ----------

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
        return m + s / 60
    if len(parts) == 3:
        h, m, s = parts
        return h * 60 + m + s / 60
    return None


def pace_to_float(pace_str: str | None):
    if not pace_str or ":" not in pace_str:
        return None
    try:
        m, s = pace_str.split(":")
        return int(m) + int(s) / 60
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
    mask = (m["distance"].notna()) & (m["distance"] > 0) & m["duration_minutes"].notna()
    m.loc[mask, "pace_min_per_mile"] = m.loc[mask, "duration_minutes"] / m.loc[mask, "distance"]

    hr_max = float(st.session_state.get("hr_max", 190))
    m["rss"] = None
    mask_rss = m["duration_minutes"].notna() & m["avg_hr"].notna()
    m.loc[mask_rss, "rss"] = (
        m.loc[mask_rss, "duration_minutes"] * (m.loc[mask_rss, "avg_hr"] / hr_max) ** 2
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
    d = daily.copy().sort_values("date_dt")
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
        if pct < 0.6: return "Z1"
        if pct < 0.7: return "Z2"
        if pct < 0.8: return "Z3"
        if pct < 0.9: return "Z4"
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
    longest = 1
    current = 1
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


# ---------- PR / Race Prediction / Zones helpers ----------

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
        prs["fastest_mile"] = (mile_runs["duration_minutes"] / mile_runs["distance"]).min()

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
    vo2 = df["vo2max"].dropna().mean() if "vo2max" in df else None
    vo2_factor = (50 / vo2) if vo2 else 1.0
    daily = compute_daily_load(df)
    load = compute_fitness_fatigue(daily)
    tsb = load["TSB"].iloc[-1] if not load.empty else 0
    fatigue_factor = 1 - (tsb / 200)
    effective_pace = best_pace * vo2_factor * fatigue_factor
    predictions = {
        "5K": effective_pace * 3.11,
        "10K": effective_pace * 6.22,
        "Half": effective_pace * 13.1,
        "Marathon": effective_pace * 26.2,
    }
    return predictions


def generate_race_prediction_series(df: pd.DataFrame):
    out = []
    if df.empty:
        return pd.DataFrame(out)
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


# Pace zones

def estimate_threshold_pace(prs):
    if not prs:
        return None
    mile = prs.get("fastest_mile")
    fivek = prs.get("fastest_5k")
    if fivek:
        return fivek / 3.11 * 0.98
    if mile:
        return mile * 1.15
    return None


def estimate_easy_pace(best_pace):
    return best_pace + 1.0, best_pace + 2.0


def estimate_tempo_pace(threshold_pace):
    if threshold_pace:
        return threshold_pace * 0.97, threshold_pace * 1.02
    return None


def estimate_interval_pace(best_pace):
    return best_pace * 0.90, best_pace * 0.95


def estimate_rep_pace(best_pace):
    return best_pace * 0.80, best_pace * 0.90


def calculate_hr_zones(hr_max):
    return {
        "Zone 1 (Recovery)": (0.50 * hr_max, 0.60 * hr_max),
        "Zone 2 (Easy / Aerobic)": (0.60 * hr_max, 0.70 * hr_max),
        "Zone 3 (Tempo)": (0.70 * hr_max, 0.80 * hr_max),
        "Zone 4 (Threshold)": (0.80 * hr_max, 0.90 * hr_max),
        "Zone 5 (VO2 Max)": (0.90 * hr_max, 1.00 * hr_max),
    }
# ---------- AI helpers ----------

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
        return "OpenAI client not available. Install 'openai' and set OPENAI_API_KEY."
    verbosity = st.session_state.get("ai_verbosity", "normal")
    focus = st.session_state.get("ai_focus", "balanced")
    style = f"Verbosity: {verbosity}. Focus: {focus}."
    try:
        resp = c.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a professional running coach and data analyst."},
                {"role": "user", "content": style + "\n\n" + prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling AI: {e}"


# ---------- DB operations for runs & shoes ----------

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


def update_run(run_id, data):
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


def delete_run(run_id):
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


def retire_shoe(shoe_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE shoes SET retired = 1 WHERE id = ?", (shoe_id,))
    conn.commit()
    conn.close()


# ---------- Garmin CSV parser ----------

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
                if "km" in c.lower():
                    val *= 0.621371
                distance = val
            except Exception:
                pass
            break

    duration_minutes = None
    for c in cols:
        if any(x in c.lower() for x in ("elapsed", "duration", "time")):
            raw = row[c]
            s = str(raw)
            if ":" in s:
                duration_minutes = duration_to_minutes(s)
                break
            try:
                num = float(s)
                duration_minutes = num / 60 if num > 200 else num
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

    msg = "Imported successfully."
    if warnings:
        msg += " " + " ".join(warnings)

    return data, msg


# ---------- Pages: Home & Feed ----------

def render_home_page():
    st.title("🏠 Home")

    st.markdown(
        """
        <div class='card'>
        <h3>Welcome to Your Run Tracker + AI Coach</h3>
        <p>Log runs, analyze your training, and get AI coaching towards your race goals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = fetch_runs()
    if df.empty:
        st.info("You have no logged runs yet. Log one or import from Garmin!")
        return

    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    total_miles = metrics["distance"].sum(skipna=True)
    last_7 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7_miles = last_7["distance"].sum(skipna=True)
    eff_avg = metrics["efficiency_score"].mean(skipna=True)
    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)

    current_streak, longest_streak = compute_streaks(metrics)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    col3.metric("Avg Efficiency", f"{eff_avg:.1f}" if not pd.isna(eff_avg) else "—")
    col4.metric("Current Streak", f"{current_streak} days")

    st.markdown("<h4>Recent Runs</h4>", unsafe_allow_html=True)

    df_recent = df.tail(5).iloc[::-1]
    prs = calculate_prs(metrics)
    for _, row in df_recent.iterrows():
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

        badge_text = " | ".join(badges)

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
                    <span class='muted'>HR: {hr} bpm</span>
                    <span class='muted'>Elev: {elev} ft</span>
                </div>
                <span class='muted'>Felt: {row['how_felt'] or '—'}</span><br>
                <span class='muted'>Effort: {row['effort'] or '—'} / 10</span>
                {"<div class='pr-badge'>" + badge_text + "</div>" if badge_text else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"Edit Run {row['id']}", key=f"edit_home_{row['id']}"):
            st.session_state["page"] = "Edit Run"
            st.session_state["edit_run_id"] = row["id"]
            st.experimental_rerun()


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

        badge_text = " | ".join(badges)

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
                    <span class='muted'>HR: {hr} bpm</span>
                    <span class='muted'>Elev: {elev} ft</span>
                </div>
                <span class='muted'>Effort: {row['effort'] or '—'} / 10</span><br>
                <span class='muted'>Felt: {row['how_felt'] or '—'}</span>
                {"<div class='pr-badge'>" + badge_text + "</div>" if badge_text else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"Edit Run {row['id']}", key=f"edit_feed_{row['id']}"):
            st.session_state["page"] = "Edit Run"
            st.session_state["edit_run_id"] = row["id"]
            st.experimental_rerun()
# ---------- Dashboard Page ----------

def render_dashboard_page():
    st.title("📊 Dashboard")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs to see your dashboard analytics.")
        return

    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)

    prs = calculate_prs(metrics)

    # ----- Summary Metrics (top row) -----
    total_miles = metrics["distance"].sum(skipna=True)
    last_7_days = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7 = last_7_days["distance"].sum(skipna=True)
    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7)

    avg_eff = metrics["efficiency_score"].mean(skipna=True)
    avg_hr = metrics["avg_hr"].mean(skipna=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    col3.metric("Avg HR", f"{avg_hr:.1f}" if avg_hr else "—")
    col4.metric("Efficiency", f"{avg_eff:.1f}" if avg_eff else "—")

    # ----- PR Banners -----
    if prs:
        if metrics["distance"].max() == prs["longest_distance"]:
            st.markdown("<div class='pr-banner'>🏆 New Longest Run!</div>", unsafe_allow_html=True)

        df_copy = metrics.copy()
        df_copy["week"] = df_copy["date_dt"].dt.isocalendar().week
        weekly = df_copy.groupby("week")["distance"].sum()

        if weekly.max() == prs["highest_weekly_mileage"]:
            st.markdown("<div class='pr-banner'>🔥 Highest Weekly Mileage Ever!</div>", unsafe_allow_html=True)

        df_copy["month"] = df_copy["date_dt"].dt.month
        monthly = df_copy.groupby("month")["distance"].sum()

        if monthly.max() == prs["highest_monthly_mileage"]:
            st.markdown("<div class='pr-banner'>📈 Highest Monthly Mileage Ever!</div>", unsafe_allow_html=True)

    # ---------- Weekly Mileage Chart ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weekly Mileage")

    dfw = metrics.copy()
    dfw["week"] = dfw["date_dt"].dt.isocalendar().week
    weekly = dfw.groupby("week")["distance"].sum().reset_index()

    if not weekly.empty:
        chart = alt.Chart(weekly).mark_bar().encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("distance:Q", title=f"Distance ({unit})")
        )
        st.altair_chart(chart, use_container_width=True)

        if prs.get("highest_weekly_mileage"):
            st.markdown(
                f"<span class='pr-mini'>🔥 PR Week: {prs['highest_weekly_mileage']:.1f} mi</span>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Pace Trend ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pace Trend (min/mi)")

    pace_df = metrics.dropna(subset=["pace_min_per_mile"])

    if not pace_df.empty:
        chart = alt.Chart(pace_df).mark_line(point=True).encode(
            x="date_dt:T",
            y="pace_min_per_mile:Q",
        )
        st.altair_chart(chart, use_container_width=True)

        if prs.get("fastest_pace"):
            st.markdown(
                f"<span class='pr-mini'>⚡ Pace PR: {prs['fastest_pace']:.2f}</span>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Efficiency Trend ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Efficiency Score Trend")

    eff_df = metrics.dropna(subset=["efficiency_score"])
    if not eff_df.empty:
        chart = alt.Chart(eff_df).mark_line(point=True).encode(
            x="date_dt:T",
            y="efficiency_score:Q",
        )
        st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- VO2 Max Trend ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("VO2 Max")

    vo2_df = metrics.dropna(subset=["vo2max"])
    if not vo2_df.empty:
        chart = alt.Chart(vo2_df).mark_line(point=True).encode(
            x="date_dt:T",
            y="vo2max:Q",
        )
        st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Injury Early Signals ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚕️ Injury Early Signals")

    signals = []
    if avg_eff and avg_eff < metrics["efficiency_score"].quantile(0.25):
        signals.append("⚠️ Efficiency drop — potential overreaching.")

    if avg_hr and avg_hr > metrics["avg_hr"].quantile(0.75):
        signals.append("⚠️ Elevated HR — possible fatigue.")

    if not signals:
        st.success("No significant injury signals detected.")
    else:
        for s in signals:
            st.warning(s)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Race Prediction ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏁 Race Prediction Trend")

    pred = generate_race_prediction_series(metrics)
    if not pred.empty:
        melted = pred.melt(id_vars="date", var_name="race", value_name="minutes")
        chart = alt.Chart(melted).mark_line(point=True).encode(
            x="date:T",
            y="minutes:Q",
            color="race:N",
            tooltip=["date", "race", "minutes"],
        )
        st.altair_chart(chart, use_container_width=True)

        last = pred.iloc[-1]
        colA, colB, colC, colD = st.columns(4)
        colA.metric("5K", f"{last['5K']:.1f} min")
        colB.metric("10K", f"{last['10K']:.1f} min")
        colC.metric("Half", f"{last['Half']:.1f} min")
        colD.metric("Marathon", f"{last['Marathon']:.1f} min")
    else:
        st.info("Not enough data for race prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Shoe Mileage ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👟 Shoe Mileage")

    shoes = fetch_shoes(include_retired=True)
    if shoes.empty:
        st.info("No shoes added yet.")
    else:
        for _, s in shoes.iterrows():
            shoe_runs = metrics[metrics["shoe_id"] == s["id"]]
            miles = shoe_runs["distance"].sum() if not shoe_runs.empty else 0

            st.markdown(
                f"""
                <div>
                    <strong>{s['name']} ({s['brand']})</strong> — {"Retired" if s['retired'] else "Active"}<br>
                    Miles: {miles:.1f}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
# ---------- AI Coach Page (includes PR Milestones & Training Block Generator) ----------

def render_ai_coach_page():
    st.title("🤖 AI Coach")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs or import Garmin data for AI analysis.")
        return

    metrics = prepare_metrics_df(df)
    recent = df.tail(30)
    latest = df.iloc[-1].to_dict()
    race_goal = st.session_state.get("race_goal", "")
    race_date = st.session_state.get("race_date_str", "")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Daily & Weekly",
            "Workout Generator",
            "7-Day Plan",
            "Race Sim",
            "Injury Risk AI",
            "PR Milestones",
            "Training Block",
        ]
    )

    # --- TAB 1: Daily & Weekly Analysis ---
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'><h3>Last Run Analysis</h3></div>", unsafe_allow_html=True)
            if st.button("Analyze Last Run"):
                prompt = (
                    "Analyze my most recent run. Include pacing, HR, efficiency, "
                    "effort, fatigue estimate, and 3 actionable recommendations.\n\n"
                    f"Run data: {latest}"
                )
                st.write(call_ai(prompt))

        with col2:
            st.markdown("<div class='card'><h3>Weekly Summary</h3></div>", unsafe_allow_html=True)
            if st.button("Summarize Last 7 Days"):
                last7 = df[pd.to_datetime(df["date"]) >= datetime.today() - timedelta(days=7)]
                week_metrics = prepare_metrics_df(last7) if not last7.empty else last7
                week_prs = calculate_prs(week_metrics) if not last7.empty else {}

                if week_prs:
                    st.markdown("<strong>PR Achievements This Week:</strong>", unsafe_allow_html=True)
                    for key, val in week_prs.items():
                        label = key.replace("_", " ").title()
                        if "fastest" in key:
                            st.write(f"⚡ {label}: {val:.2f} min")
                        else:
                            st.write(f"🔥 {label}: {val:.2f} mi")

                prompt = (
                    "Provide a detailed weekly training summary including mileage, pace trends, HR trends, "
                    "efficiency changes, fatigue level, suggested training load next week, and recovery advice.\n\n"
                    f"Weekly data: {last7.to_dict('records')}\n\n"
                    f"Weekly PRs: {week_prs}"
                )
                st.write(call_ai(prompt))

    # --- TAB 2: Workout Generator ---
    with tab2:
        st.markdown("<div class='card'><h3>Generate Tomorrow’s Workout</h3></div>", unsafe_allow_html=True)
        if st.button("Create Tomorrow's Workout"):
            prompt = (
                "Based on my last 2–3 weeks of training, fitness level, and goal race, "
                "generate a complete training session for tomorrow including warm-up, "
                "main set, paces or HR zones, cooldown, and purpose.\n\n"
                f"Race goal: {race_goal}\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- TAB 3: 7-Day Planner ---
    with tab3:
        st.markdown("<div class='card'><h3>7-Day Training Plan</h3></div>", unsafe_allow_html=True)
        if st.button("Generate Next 7-Day Plan"):
            prompt = (
                "Using my past 2–4 weeks of training, build a structured 7-day plan including "
                "run types, distances, paces, HR zones, goals for each day, and one key workout.\n\n"
                f"Race: {race_goal} on {race_date}\n"
                f"Training history: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- TAB 4: Race Simulator ---
    with tab4:
        st.markdown("<div class='card'><h3>Race Day Simulation</h3></div>", unsafe_allow_html=True)
        if st.button("Simulate Race Performance"):
            prompt = (
                "Simulate my race day performance. Provide: estimated finish time range, pacing strategy, "
                "HR zones, fueling schedule, mental checkpoints, and final mile strategy.\n\n"
                f"Goal: {race_goal}\n"
                f"Race date: {race_date}\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- TAB 5: Injury Risk AI (J) ---
    with tab5:
        st.markdown("<div class='card'><h3>AI Injury Risk Analysis</h3></div>", unsafe_allow_html=True)
        if st.button("Evaluate Injury Risk"):
            prompt = (
                "Evaluate my injury risk with focus on shin splints. Analyze training load changes, "
                "pace/hard effort spikes, elevation, fatigue, sleep, and pain notes. "
                "Give a risk rating and 4–6 recommended adjustments.\n\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- TAB 6: PR Milestone Analysis ---
    with tab6:
        st.markdown("<div class='card'><h3>🏆 PR Milestone Analysis</h3></div>", unsafe_allow_html=True)

        df_all = fetch_runs()
        metrics_all = prepare_metrics_df(df_all)
        prs_all = calculate_prs(metrics_all)

        st.subheader("Your Current Personal Records")
        for key, val in prs_all.items():
            label = key.replace("_", " ").title()
            if "fastest" in key:
                st.write(f"⚡ **{label}:** {val:.2f} min")
            else:
                st.write(f"🔥 **{label}:** {val:.2f} mi")

        st.markdown("---")

        if st.button("Analyze PR Progress"):
            df_sorted = metrics_all.sort_values("date_dt")
            trend_data = df_sorted.to_dict("records")

            prompt = f"""
You are an elite running coach. Analyze the athlete’s PR progression based on these run records.
Break your answer into 6 sections:

1. Summary of PR Progress
2. Reasons for Improvement
3. Identify Weak Spots
4. Predict the Next Likely PR
5. Workout Plan to Break That PR
6. Long-Term Projection

Training history:
{trend_data}

Current PRs:
{prs_all}
"""
            analysis = call_ai(prompt)
            st.write(analysis)

    # --- TAB 7: Training Block Generator (I) ---
    with tab7:
        st.markdown("<div class='card'><h3>📆 Training Block Generator</h3></div>", unsafe_allow_html=True)
        block_length = st.selectbox("Block Length (weeks)", [8, 10, 12, 16], index=2)

        if st.button("Generate Training Block"):
            prompt = f"""
You are a professional running coach. Create a {block_length}-week half-marathon training block
for me leading into my goal race.

Details:
- Goal race & time: {race_goal}, date {race_date}
- Training history: {recent.to_dict('records')}
- Current PRs: {calculate_prs(metrics)}
- Weekly mileage goal (user setting): {st.session_state.get("weekly_goal_mi", 25.0)}
- Constraints: focus on gradual progression, 1–2 key workouts/week, avoid overloading shins.

Output a week-by-week plan with:
- Weekly mileage target
- 4–5 runs per week with type (easy, long, tempo, intervals, recovery)
- Specific workouts for key days
- Notes on recovery weeks and deloads
- Taper plan in the final 1–2 weeks
"""
            st.write(call_ai(prompt))


# ---------- Compare Runs Page ----------

def render_compare_runs_page():
    st.title("📊 Compare Two Runs")

    df = fetch_runs()
    if df.empty or len(df) < 2:
        st.info("Log at least two runs to compare them.")
        return

    df_sorted = df.sort_values("date", ascending=False)
    run_labels = [f"{r['date']} — {r['run_type']} — {r['distance']} mi" for _, r in df_sorted.iterrows()]
    id_to_run = {label: r for label, (_, r) in zip(run_labels, df_sorted.iterrows())}

    col1, col2 = st.columns(2)
    with col1:
        r1_label = st.selectbox("Run 1", run_labels, key="cmp1")
        run1 = id_to_run[r1_label]
    with col2:
        r2_label = st.selectbox("Run 2", run_labels, key="cmp2")
        run2 = id_to_run[r2_label]

    if r1_label == r2_label:
        st.warning("Please select two different runs.")
        return

    st.markdown("### 🔍 Side-by-Side Comparison")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    colA.markdown("#### Metric")
    colB.markdown("#### Run 1")
    colC.markdown("#### Run 2")

    metrics_to_compare = [
        ("Distance (mi)", "distance"),
        ("Duration (min)", "duration_minutes"),
        ("Avg HR", "avg_hr"),
        ("Max HR", "max_hr"),
        ("Cadence", "cadence"),
        ("Elevation Gain", "elevation_gain"),
        ("Effort", "effort"),
        ("VO2 Max", "vo2max"),
        ("HRV", "hrv"),
    ]

    for label, key in metrics_to_compare:
        val1 = run1[key]
        val2 = run2[key]
        colA.write(label)
        colB.write(val1 if val1 is not None else "—")
        colC.write(val2 if val2 is not None else "—")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📈 Performance Comparison")
    df_compare = pd.DataFrame([
        {"run": "Run 1", "distance": run1["distance"], "hr": run1["avg_hr"], "elev": run1["elevation_gain"]},
        {"run": "Run 2", "distance": run2["distance"], "hr": run2["avg_hr"], "elev": run2["elevation_gain"]},
    ])

    chart = (
        alt.Chart(df_compare)
        .mark_bar()
        .encode(
            x=alt.X("run:N", title="Run"),
            y=alt.Y("distance:Q", title="Distance (mi)"),
            color="run:N",
            tooltip=["run", "distance", "hr", "elev"]
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### 🤖 AI Comparison Summary")
    if st.button("Generate AI Comparison"):
        prompt = (
            "Compare these two runs in detail. Cover pacing, heart rate response, "
            "efficiency, elevation impact, recovery, and give a diagnosis of what improved "
            "or regressed.\n\n"
            f"Run 1: {run1}\n\n"
            f"Run 2: {run2}"
        )
        st.write(call_ai(prompt))


# ---------- Pace Zones Page ----------

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
    tempo = estimate_tempo_pace(threshold_pace)
    interval = estimate_interval_pace(best_pace) if best_pace else None
    rep = estimate_rep_pace(best_pace) if best_pace else None

    hr_max = st.session_state.get("hr_max", 190)
    hr_zones = calculate_hr_zones(hr_max)

    # ---------- Pace Zones Table ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pace Zones (min/mile)")

    pace_data = []

    def add_zone(name, pace_range):
        if not pace_range or not pace_range[0] or not pace_range[1]:
            return
        pace_data.append({
            "Zone": name,
            "Min Pace (min/mi)": f"{pace_range[0]:.2f}",
            "Max Pace (min/mi)": f"{pace_range[1]:.2f}",
        })

    if easy_min and easy_max:
        add_zone("Recovery", (easy_min + 0.5, easy_max + 0.5))
        add_zone("Easy", (easy_min, easy_max))
        add_zone("Aerobic Endurance", (easy_min - 0.3, easy_min + 0.3))

    if tempo:
        add_zone("Tempo", tempo)

    if threshold_pace:
        add_zone("Threshold", (threshold_pace * 0.98, threshold_pace * 1.02))

    if interval:
        add_zone("Interval", interval)

    if rep:
        add_zone("Repetition", rep)

    if pace_data:
        st.table(pd.DataFrame(pace_data))
    else:
        st.info("Need at least a few runs with pace data to calculate zones.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- HR Zones ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("HR Zones")

    hr_data = []
    for zone, (low, high) in hr_zones.items():
        hr_data.append({
            "Zone": zone,
            "Low (bpm)": int(low),
            "High (bpm)": int(high),
        })

    st.table(pd.DataFrame(hr_data))
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Zone Usage Examples ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("How to Use These Zones")

    st.write("""
- **Recovery** — Day after hard workouts; very easy run  
- **Easy (Zone 2)** — Most of your mileage and long runs  
- **Aerobic** — Steady runs; aerobic development  
- **Tempo** — Strong but controlled efforts; marathon strength  
- **Threshold** — Close to 10K / half pace; lactate threshold work  
- **Interval** — 3–5 minute repeats near VO2 max  
- **Repetition** — Very fast, short reps for leg speed and form  
    """)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AI Zone Workout Generator ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Zone Workout Generator")

    zone_choice = st.selectbox(
        "Select a Zone",
        ["Recovery", "Easy", "Aerobic Endurance", "Tempo", "Threshold", "Interval", "Repetition"]
    )

    if st.button("Generate Zone Workout"):
        prompt = f"""
Create a structured running workout for the **{zone_choice}** zone.
Include:
- warm-up
- main set
- cooldown
- pacing instructions (min/mile)
- HR zone instructions
- purpose of the workout
- where in the week to place it
"""
        st.write(call_ai(prompt))

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Settings Page ----------

def render_settings_page():
    st.title("⚙ Settings")

    # --- Appearance ---
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

    # --- AI Coaching Settings ---
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

    # --- Race Goal & HR ---
    st.markdown("<div class='card'><h3>Race Goal & HR</h3></div>", unsafe_allow_html=True)
    col6, col7 = st.columns(2)

    with col6:
        goal = st.text_input(
            "Race goal description",
            value=st.session_state["race_goal"],
        )
        st.session_state["race_goal"] = goal

        race_date_input = st.date_input(
            "Race date",
            value=datetime.fromisoformat(st.session_state["race_date_str"]).date(),
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

    # --- Data Export / Delete ---
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
        if st.button("⚠ Delete ALL data"):
            conn = get_conn()
            conn.execute("DELETE FROM runs")
            conn.commit()
            conn.close()
            st.warning("All run data deleted.")

    # --- Shoe Management ---
    st.markdown("<div class='card'><h3>Shoes</h3></div>", unsafe_allow_html=True)
    st.write("Track your running shoes and mileage.")

    with st.form("add_shoe"):
        colA, colB = st.columns(2)
        with colA:
            shoe_name = st.text_input("Shoe Name (e.g., Clifton 10)")
        with colB:
            shoe_brand = st.text_input("Brand (e.g., Hoka)")

        start_date = st.date_input("Start Date", datetime.today().date())
        add_shoe_btn = st.form_submit_button("Add Shoe")

        if add_shoe_btn:
            insert_shoe(shoe_name, shoe_brand, start_date.isoformat())
            st.success("Shoe added!")

    st.subheader("Active Shoes")
    shoes_df = fetch_shoes()
    runs_df = fetch_runs()

    if shoes_df.empty:
        st.info("No shoes added yet.")
    else:
        for _, s in shoes_df.iterrows():
            shoe_runs = runs_df[runs_df["shoe_id"] == s["id"]]
            mileage = shoe_runs["distance"].sum() if not shoe_runs.empty else 0
            miles_left = 400 - mileage
            warn = ""
            if mileage > 350:
                warn = "⚠️ Nearing end of life"
            elif mileage > 300:
                warn = "⚠️ Getting worn"

            st.markdown(
                f"""
                <div class='card'>
                    <strong>{s['name']} ({s['brand']})</strong><br>
                    Start: {s['start_date']}<br>
                    Miles: {mileage:.1f}<br>
                    {warn}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(f"Retire {s['name']}", key=f"retire_{s['id']}"):
                retire_shoe(s["id"])
                st.warning(f"{s['name']} retired.")


# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="Run Tracker & AI Coach", layout="wide")
    init_session_state()
    init_db_with_migration()
    inject_css()

    # Sidebar quick settings & navigation
    st.sidebar.header("Quick Settings")

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
        "Race Goal",
        value=st.session_state["race_goal"],
        key="sidebar_race_goal",
    )
    if st.session_state.get("sidebar_race_goal"):
        st.session_state["race_goal"] = st.session_state["sidebar_race_goal"]

    st.sidebar.markdown("---")
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

    # Route
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


if __name__ == "__main__":
    main()
