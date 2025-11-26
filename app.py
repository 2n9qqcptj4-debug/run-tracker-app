###############################################################
# SEGMENT 1 / 12
# IMPORTS • SESSION STATE • CSS • ATHLETE PROFILE SYSTEM
###############################################################

import sqlite3
from datetime import datetime, timedelta, date

import altair as alt
import pandas as pd
import streamlit as st

# OpenAI client
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

DB_PATH = "run_log.db"


# ============================================================
# SESSION STATE INITIALIZATION (with Athlete Profile fields)
# ============================================================

def init_session_state():

    defaults = {
        "page": "Home",
        "theme": "dark",
        "units": "mi",
        "ai_verbosity": "normal",
        "ai_focus": "balanced",
        "weekly_goal_mi": 25.0,
        "compact_mode": False,
        "edit_run_id": None,

        # Race goal (general)
        "race_goal": "Pittsburgh Half – Sub 1:40",
        "race_date_str": "2026-05-03",

        # Athlete Profile System
        "ath_age": 26,
        "ath_height_in": 69,
        "ath_weight_lb": 167,
        "ath_vo2max": 52,
        "ath_rest_hr": 49,
        "ath_max_hr": 190,
        "ath_experience": "9 months of consistent training",
        "ath_limitations": "Shin splints",
        "ath_preferences": "5 days/week, long runs on Sunday",
        "ath_terrain": "Pavement with rolling hills",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# THEME / CSS
# ============================================================

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
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 18px 40px rgba(0,0,0,0.32);
            border-color: rgba(59,130,246,0.6);
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

        </style>
        """,
        unsafe_allow_html=True,
    )
###############################################################
# SEGMENT 2 / 12
# DATABASE HELPERS & MIGRATIONS
###############################################################

def get_conn():
    """Get SQLite DB connection with row dict support."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create DB tables if they do not exist, including migrations."""
    conn = get_conn()
    cur = conn.cursor()

    # -------------------------
    # RUNS TABLE
    # -------------------------
    cur.execute("""
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
    """)

    # -------------------------
    # SHOES TABLE
    # -------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT,
            start_date TEXT,
            retired INTEGER DEFAULT 0
        )
    """)

    # --- Migration: Ensure shoe_id exists ---
    cur.execute("PRAGMA table_info(runs)")
    cols = [row[1] for row in cur.fetchall()]
    if "shoe_id" not in cols:
        cur.execute("ALTER TABLE runs ADD COLUMN shoe_id INTEGER")

    conn.commit()
    conn.close()


def init_db_with_migration():
    """Call migration-ready DB initialization."""
    init_db()


# ============================================================
# DB CRUD HELPERS
# ============================================================

def insert_run(data: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs
        (date, run_type, distance, duration_minutes, avg_pace, splits,
         avg_hr, max_hr, hr_by_segment, cadence, elevation_gain, effort,
         terrain, weather, how_felt, pain, sleep_hours, stress,
         nutrition_notes, vo2max, hrv, shoe_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["date"], data["run_type"], data["distance"], data["duration_minutes"],
        data["avg_pace"], data["splits"], data["avg_hr"], data["max_hr"],
        data["hr_by_segment"], data["cadence"], data["elevation_gain"],
        data["effort"], data["terrain"], data["weather"], data["how_felt"],
        data["pain"], data["sleep_hours"], data["stress"], data["nutrition_notes"],
        data["vo2max"], data["hrv"], data["shoe_id"]
    ))
    conn.commit()
    conn.close()


def update_run(run_id: int, data: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE runs SET
            date = ?, run_type = ?, distance = ?, duration_minutes = ?, avg_pace = ?,
            splits = ?, avg_hr = ?, max_hr = ?, hr_by_segment = ?, cadence = ?,
            elevation_gain = ?, effort = ?, terrain = ?, weather = ?, how_felt = ?,
            pain = ?, sleep_hours = ?, stress = ?, nutrition_notes = ?, vo2max = ?,
            hrv = ?, shoe_id = ?
        WHERE id = ?
    """, (
        data["date"], data["run_type"], data["distance"], data["duration_minutes"],
        data["avg_pace"], data["splits"], data["avg_hr"], data["max_hr"],
        data["hr_by_segment"], data["cadence"], data["elevation_gain"],
        data["effort"], data["terrain"], data["weather"], data["how_felt"],
        data["pain"], data["sleep_hours"], data["stress"], data["nutrition_notes"],
        data["vo2max"], data["hrv"], data["shoe_id"], run_id
    ))
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


# -------------------------
# SHOES CRUD
# -------------------------

def insert_shoe(name, brand, start_date):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO shoes (name, brand, start_date, retired)
        VALUES (?, ?, ?, 0)
    """, (name, brand, start_date))
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
###############################################################
# SEGMENT 3 / 12
# CORE UTILITY FUNCTIONS
###############################################################

# -------------------------
# TIME & PACE HELPERS
# -------------------------

def duration_to_minutes(time_str: str | None):
    """Convert 'MM:SS' or 'HH:MM:SS' into float minutes."""
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
        return h * 60 + m + s / 60.0
    return None


def minutes_to_hms(minutes: float | None) -> str:
    """Convert float minutes → HH:MM:SS string."""
    if minutes is None:
        return ""
    total_seconds = int(round(minutes * 60))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def pace_to_float(pace_str: str | None):
    """Convert 'MM:SS' → float minutes."""
    if not pace_str or ":" not in pace_str:
        return None
    try:
        m, s = pace_str.split(":")
        return int(m) + int(s) / 60.0
    except Exception:
        return None


# -------------------------
# METRIC PROCESSING
# -------------------------

def prepare_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields: date, week start, pace min/mi, RSS, load."""
    if df.empty:
        return df

    m = df.copy()

    m["date_dt"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date_dt"])
    m["date"] = m["date_dt"].dt.date

    # Week start (Monday)
    m["week_start"] = m["date_dt"] - pd.to_timedelta(m["date_dt"].dt.weekday, unit="D")
    m["week_start"] = m["week_start"].dt.date

    # Load
    m["training_load"] = m["effort"].fillna(0) * m["duration_minutes"].fillna(0)

    # Pace (numeric)
    mask = (
        m["distance"].notna()
        & (m["distance"] > 0)
        & m["duration_minutes"].notna()
    )
    m["pace_min_per_mile"] = None
    m.loc[mask, "pace_min_per_mile"] = (
        m.loc[mask, "duration_minutes"] / m.loc[mask, "distance"]
    )

    # RSS (relative stress score)
    hr_max = float(st.session_state.get("hr_max", 190))
    mask_rss = m["duration_minutes"].notna() & m["avg_hr"].notna()
    m["rss"] = None
    m.loc[mask_rss, "rss"] = (
        m.loc[mask_rss, "duration_minutes"]
        * (m.loc[mask_rss, "avg_hr"] / hr_max) ** 2
    )

    return m


def compute_daily_load(metrics: pd.DataFrame) -> pd.DataFrame:
    """Group by day → distance, load, rss totals."""
    if metrics.empty:
        return metrics
    return (
        metrics.groupby("date_dt", as_index=False)
        .agg(
            distance=("distance", "sum"),
            training_load=("training_load", "sum"),
            rss=("rss", "sum"),
        )
    )


def compute_fitness_fatigue(daily: pd.DataFrame) -> pd.DataFrame:
    """CTL (42-day avg), ATL (7-day avg), TSB."""
    if daily.empty:
        return daily

    d = daily.sort_values("date_dt").copy()
    d["CTL"] = d["training_load"].rolling(window=42, min_periods=1).mean()
    d["ATL"] = d["training_load"].rolling(window=7, min_periods=1).mean()
    d["TSB"] = d["CTL"] - d["ATL"]
    return d


def compute_efficiency_score(metrics: pd.DataFrame) -> pd.DataFrame:
    """Efficiency = speed / HR."""
    if metrics.empty:
        return metrics

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
    """Convert avg HR → Z1–Z5."""
    if metrics.empty:
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


# -------------------------
# STREAKS
# -------------------------

def compute_streaks(metrics: pd.DataFrame):
    """Return (current_streak, longest_streak)."""
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


# -------------------------
# UNIT CONVERSION
# -------------------------

def convert_distance_for_display(distance_mi: float | None):
    """Convert miles → km if required."""
    units = st.session_state.get("units", "mi")
    if distance_mi is None:
        return None, units
    if units == "km":
        return distance_mi * 1.60934, "km"
    return distance_mi, "mi"


# -------------------------
# PR CALCULATIONS
# -------------------------

def calculate_prs(df: pd.DataFrame):
    """Return dictionary of PRs across events, longest, weekly, monthly."""
    prs = {}
    if df.empty:
        return prs

    df = df.copy()
    df["pace_num"] = df["avg_pace"].apply(pace_to_float)

    prs["longest_distance"] = df["distance"].max()

    pace_df = df[df["pace_num"].notna()]
    if not pace_df.empty:
        prs["fastest_pace"] = pace_df["pace_num"].min()

    # 1 mile
    mile = df[df["distance"] >= 1.0]
    if not mile.empty:
        prs["fastest_mile"] = (mile["duration_minutes"] / mile["distance"]).min()

    # 5K
    fivek = df[df["distance"] >= 3.11]
    if not fivek.empty:
        prs["fastest_5k"] = (
            fivek["duration_minutes"] / fivek["distance"] * 3.11
        ).min()

    # 10K
    tenk = df[df["distance"] >= 6.22]
    if not tenk.empty:
        prs["fastest_10k"] = (
            tenk["duration_minutes"] / tenk["distance"] * 6.22
        ).min()

    # Half
    half = df[df["distance"] >= 13.1]
    if not half.empty:
        prs["fastest_half"] = (
            half["duration_minutes"] / half["distance"] * 13.1
        ).min()

    # Weekly mileage
    df["date_dt"] = pd.to_datetime(df["date"])
    df["week"] = df["date_dt"].dt.isocalendar().week
    df["year"] = df["date_dt"].dt.year
    weekly = df.groupby(["year", "week"])["distance"].sum()
    prs["highest_weekly_mileage"] = weekly.max()

    # Monthly mileage
    df["month"] = df["date_dt"].dt.month
    monthly = df.groupby(["year", "month"])["distance"].sum()
    prs["highest_monthly_mileage"] = monthly.max()

    return prs


def detect_pr_improvements(old_prs, new_prs):
    """Return list of PR categories improved."""
    if new_prs is None:
        return []

    labels = {
        "longest_distance": "Longest Run",
        "fastest_pace": "Fastest Pace",
        "fastest_mile": "Fastest Mile",
        "fastest_5k": "Fastest 5K",
        "fastest_10k": "Fastest 10K",
        "fastest_half": "Fastest Half Marathon",
        "highest_weekly_mileage": "Highest Weekly Mileage",
        "highest_monthly_mileage": "Highest Monthly Mileage",
    }

    improvements = []

    for key, new_val in new_prs.items():
        if new_val is None or pd.isna(new_val):
            continue

        old_val = old_prs.get(key) if old_prs else None

        if old_val is None or pd.isna(old_val):
            improvements.append(labels.get(key, key))
            continue

        # For pace: lower = better
        if "fastest" in key:
            if new_val < old_val - 1e-6:
                improvements.append(labels.get(key, key))
        else:
            # For mileage/distance: higher = better
            if new_val > old_val + 1e-6:
                improvements.append(labels.get(key, key))

    return improvements


# -------------------------
# RACE PREDICTIONS
# -------------------------

def predict_race_times(df: pd.DataFrame):
    """Return predicted finish times based on best pace + fatigue."""
    if df.empty:
        return None

    df = df.copy()
    df["pace_num"] = df["avg_pace"].apply(pace_to_float)
    best_pace = df["pace_num"].min()

    # VO2 factor
    vo2 = df["vo2max"].dropna().mean() if "vo2max" in df.columns else None
    vo2_factor = (50 / vo2) if vo2 else 1.0

    # Fatigue factor
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
    """Generate prediction over time for charts."""
    if df.empty:
        return pd.DataFrame([])

    out = []
    df = df.copy().sort_values("date_dt")

    for _, row in df.iterrows():
        sub = df[df["date_dt"] <= row["date_dt"]]
        pred = predict_race_times(sub)
        if pred:
            out.append({
                "date": row["date_dt"],
                "5K": pred["5K"],
                "10K": pred["10K"],
                "Half": pred["Half"],
                "Marathon": pred["Marathon"],
            })

    return pd.DataFrame(out)


# -------------------------
# PACE ZONE ESTIMATES
# -------------------------

def estimate_threshold_pace(prs):
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
    return best_pace + 1, best_pace + 2


def estimate_tempo_pace(threshold):
    if threshold is None:
        return None
    return threshold * 0.97, threshold * 1.02


def estimate_interval_pace(best_pace):
    if best_pace is None:
        return None
    return best_pace * 0.90, best_pace * 0.95


def estimate_rep_pace(best_pace):
    if best_pace is None:
        return None
    return best_pace * 0.80, best_pace * 0.90


# -------------------------
# HR ZONES
# -------------------------

def calculate_hr_zones(hr_max):
    return {
        "Zone 1 (Recovery)": (0.50 * hr_max, 0.60 * hr_max),
        "Zone 2 (Easy / Aerobic)": (0.60 * hr_max, 0.70 * hr_max),
        "Zone 3 (Tempo)": (0.70 * hr_max, 0.80 * hr_max),
        "Zone 4 (Threshold)": (0.80 * hr_max, 0.90 * hr_max),
        "Zone 5 (VO2 Max)": (0.90 * hr_max, 1.00 * hr_max),
    }
###############################################################
# SEGMENT 4 / 12
# AI HELPERS (OpenAI Integration + Athlete Context Engine)
###############################################################

# -------------------------
# Ensure OpenAI client
# -------------------------

def ensure_openai_client():
    """Initialize the OpenAI client safely."""
    global client
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            client = None
    return client


# -------------------------
# Fetch Athlete Profile to embed into AI prompts
# -------------------------

def get_athlete_profile_for_ai():
    """Return athlete profile as a formatted string for AI context."""
    profile = fetch_athlete_profile()
    if not profile:
        return "No athlete profile on file."

    return f"""
ATHLETE PROFILE
---------------
Name: {profile['name']}
Age: {profile['age']}
Sex: {profile['sex']}
Experience Level: {profile['experience_level']}
Primary Goal: {profile['primary_goal']}

Weekly Mileage Target: {profile['weekly_mileage_target']}
Typical Easy Pace: {profile['typical_easy_pace']}
Typical Long Run Pace: {profile['typical_long_run_pace']}

Sleep (typical): {profile['sleep_baseline']}
Stress Level (baseline 1–5): {profile['stress_baseline']}
History of Injury: {profile['injury_history']}

Notes:
{profile['notes']}
"""


# -------------------------
# Build AI prompt context
# -------------------------

def build_ai_context(user_prompt: str):
    """Insert athlete profile, settings, verbosity, and focus."""
    verbosity = st.session_state.get("ai_verbosity", "normal")
    focus = st.session_state.get("ai_focus", "balanced")

    profile = get_athlete_profile_for_ai()

    return f"""
You are a professional running coach and human performance expert.

AI SETTINGS
-----------
Verbosity: {verbosity}
Focus: {focus}

{profile}

USER REQUEST
------------
{user_prompt}
"""


# -------------------------
# AI call wrapper
# -------------------------

def call_ai(prompt: str):
    """
    Main AI interface.
    - Automatically loads athlete profile
    - Adds verbosity + focus settings
    - Handles client errors gracefully
    """

    c = ensure_openai_client()
    if c is None:
        return "⚠️ OpenAI client unavailable — missing or invalid OPENAI_API_KEY."

    # Build the full enriched prompt
    full_prompt = build_ai_context(prompt)

    try:
        resp = c.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.45,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world-class running coach, sports scientist, "
                        "and data analyst. Provide clear, practical, and accurate guidance."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Error calling AI: {e}"
###############################################################
# SEGMENT 5 / 12
# ATHLETE PROFILE — DATABASE + CRUD
###############################################################

# ============================================================
# CREATE ATHLETE PROFILE TABLE
# ============================================================

def init_athlete_profile_table():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS athlete_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT,
            age INTEGER,
            sex TEXT,
            height_in REAL,
            weight_lb REAL,

            experience_level TEXT,
            primary_goal TEXT,
            weekly_mileage_target REAL,

            typical_easy_pace TEXT,
            typical_long_run_pace TEXT,

            sleep_baseline TEXT,
            stress_baseline TEXT,

            injury_history TEXT,
            notes TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()


# ============================================================
# INSERT / UPDATE PROFILE (UPSERT)
# ============================================================

def save_athlete_profile(data: dict):
    """
    If a profile exists → update it.
    If no profile exists → insert a new one.
    """

    conn = get_conn()
    cur = conn.cursor()

    # Check if any profile exists
    cur.execute("SELECT id FROM athlete_profile LIMIT 1")
    row = cur.fetchone()

    if row:
        # UPDATE
        profile_id = row["id"]
        placeholders = ", ".join([f"{k} = ?" for k in data.keys()])
        values = list(data.values())
        values.append(profile_id)

        cur.execute(
            f"""
            UPDATE athlete_profile
            SET {placeholders}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            values,
        )
    else:
        # INSERT
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = list(data.values())

        cur.execute(
            f"""
            INSERT INTO athlete_profile ({cols})
            VALUES ({placeholders})
            """,
            values,
        )

    conn.commit()
    conn.close()


# ============================================================
# FETCH PROFILE (DICT)
# ============================================================

def fetch_athlete_profile():
    """Return full athlete profile as a dict or None."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM athlete_profile LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {col: row[col] for col in row.keys()}


# ============================================================
# PROVIDE SAFE DEFAULT PROFILE FOR NEW USERS
# ============================================================

def get_profile_or_default():
    """Return an athlete profile with safe defaults if not yet created."""
    profile = fetch_athlete_profile()
    if profile:
        return profile

    # Default empty profile
    return {
        "name": "",
        "age": None,
        "sex": "",
        "height_in": None,
        "weight_lb": None,
        "experience_level": "",
        "primary_goal": "",
        "weekly_mileage_target": None,
        "typical_easy_pace": "",
        "typical_long_run_pace": "",
        "sleep_baseline": "",
        "stress_baseline": "",
        "injury_history": "",
        "notes": "",
    }


# ============================================================
# API FOR AI CONTEXT (STRING FORMATTED)
# ============================================================

def format_profile_for_ai(profile: dict):
    if not profile:
        return "No athlete profile is saved."

    return f"""
Name: {profile.get("name", "")}
Age: {profile.get("age", "")}
Sex: {profile.get("sex", "")}
Height (in): {profile.get("height_in", "")}
Weight (lb): {profile.get("weight_lb", "")}

Experience Level: {profile.get("experience_level", "")}
Primary Goal: {profile.get("primary_goal", "")}

Weekly Mileage Target: {profile.get("weekly_mileage_target", "")}
Typical Easy Pace: {profile.get("typical_easy_pace", "")}
Typical Long Run Pace: {profile.get("typical_long_run_pace", "")}

Avg Sleep: {profile.get("sleep_baseline", "")}
Stress Baseline: {profile.get("stress_baseline", "")}

Injury History: {profile.get("injury_history", "")}

Notes:
{profile.get("notes", "")}
"""


# ============================================================
# INIT PROFILE TABLE ON BOOT
# ============================================================

def init_all_tables():
    """Call once when app boots."""
    init_db_with_migration()
    init_athlete_profile_table()
###############################################################
# SEGMENT 6 / 12
# ATHLETE PROFILE PAGE (UI + AI SUMMARY)
###############################################################

def render_athlete_profile_page():
    st.title("🏃 Athlete Profile")

    st.markdown("<div class='card'><h3>Athlete Information</h3></div>", unsafe_allow_html=True)

    # Load current profile or defaults
    profile = get_profile_or_default()

    # -----------------------------------------
    # FORM BLOCK
    # -----------------------------------------
    with st.form("athlete_profile_form"):
        st.subheader("Basic Information")
        name = st.text_input("Name", value=profile.get("name", ""))
        age = st.number_input("Age", min_value=5, max_value=100, value=profile.get("age") or 25)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=["Male","Female","Other"].index(profile.get("sex") or "Male"))

        st.subheader("Physical Metrics")
        height_in = st.number_input("Height (inches)", min_value=36, max_value=90, value=profile.get("height_in") or 69)
        weight_lb = st.number_input("Weight (lbs)", min_value=50, max_value=400, value=profile.get("weight_lb") or 165)

        st.subheader("Training Background")
        experience_level = st.text_input("Experience Level", value=profile.get("experience_level") or "")
        primary_goal = st.text_input("Primary Goal", value=profile.get("primary_goal") or "")
        weekly_mileage_target = st.number_input(
            "Weekly Mileage Target", min_value=0.0, max_value=200.0,
            value=profile.get("weekly_mileage_target") or 25.0
        )

        st.subheader("Pacing Metrics")
        typical_easy_pace = st.text_input("Typical Easy Pace (MM:SS)", value=profile.get("typical_easy_pace") or "")
        typical_lr_pace = st.text_input("Typical Long Run Pace (MM:SS)", value=profile.get("typical_long_run_pace") or "")

        st.subheader("Lifestyle Metrics")
        sleep_base = st.text_input("Typical Sleep (hours)", value=profile.get("sleep_baseline") or "")
        stress_base = st.selectbox(
            "Baseline Stress (1–5)",
            [1, 2, 3, 4, 5],
            index=(profile.get("stress_baseline") or 3) - 1
        )

        st.subheader("Injury History")
        injury_history = st.text_area("Injury History", value=profile.get("injury_history") or "")

        st.subheader("Notes")
        notes = st.text_area("Additional Notes", value=profile.get("notes") or "")

        submitted = st.form_submit_button("Save Profile", use_container_width=True)

        if submitted:
            save_athlete_profile({
                "name": name,
                "age": age,
                "sex": sex,
                "height_in": height_in,
                "weight_lb": weight_lb,
                "experience_level": experience_level,
                "primary_goal": primary_goal,
                "weekly_mileage_target": weekly_mileage_target,
                "typical_easy_pace": typical_easy_pace,
                "typical_long_run_pace": typical_lr_pace,
                "sleep_baseline": sleep_base,
                "stress_baseline": stress_base,
                "injury_history": injury_history,
                "notes": notes,
            })
            st.success("Profile updated successfully!")

    st.markdown("---")

    # -----------------------------------------
    # AI SUMMARY SECTION
    # -----------------------------------------
    st.subheader("AI Athlete Summary")

    if st.button("Generate AI Summary", use_container_width=True):
        df = fetch_runs()
        profile_for_ai = format_profile_for_ai(fetch_athlete_profile())

        last_20 = df.tail(20).to_dict("records") if not df.empty else []

        prompt = f"""
Analyze the following athlete and provide:

- Training identity
- Strengths & weaknesses
- Recommended training structure
- Ideal intensity balance
- Race distance suitability
- Major risks
- Personalized improvement roadmap (next 4–8 weeks)

ATHLETE PROFILE:
{profile_for_ai}

RECENT RUNS (last 20):
{last_20}
"""

        summary = call_ai(prompt)
        st.write(summary)

    st.markdown("---")

    # -----------------------------------------
    # QUICK SNAPSHOT CARD
    # -----------------------------------------
    st.subheader("Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric("Age", profile.get("age", "—"))
    col2.metric("Weekly Mileage Target", f"{profile.get('weekly_mileage_target', '—')} mi")
    col3.metric("Experience", profile.get("experience_level", "—"))
###############################################################
# SEGMENT 7 / 12
# LOG A RUN PAGE (UI + CRUD)
###############################################################

def render_log_run_page():
    st.title("🏃 Log a Run")

    st.markdown("<div class='card'><h3>Enter Run Details</h3></div>", unsafe_allow_html=True)

    editing = st.session_state.get("edit_run_id", None)
    run_to_edit = None

    if editing:
        df = fetch_runs()
        run_to_edit = df[df["id"] == editing]
        if not run_to_edit.empty:
            run_to_edit = run_to_edit.iloc[0].to_dict()
        else:
            st.session_state["edit_run_id"] = None
            editing = None

    # ---------------------------------------------------------
    # RUN FORM
    # ---------------------------------------------------------
    with st.form("run_form"):
        date_value = st.date_input(
            "Date",
            value=datetime.now().date() if not editing else datetime.strptime(run_to_edit["date"], "%Y-%m-%d").date(),
        )

        run_types = ["Easy", "Long Run", "Intervals", "Tempo", "Recovery", "Race", "Other"]
        run_type = st.selectbox("Run Type", run_types, index=run_types.index(run_to_edit["run_type"]) if editing else 0)

        distance = st.number_input(
            "Distance (miles)", min_value=0.0, max_value=200.0,
            value=float(run_to_edit["distance"]) if editing else 0.0,
            step=0.01
        )

        duration_str = st.text_input(
            "Duration (HH:MM:SS or MM:SS)",
            value=minutes_to_hms(run_to_edit["duration_minutes"]) if editing else "",
        )

        avg_pace = st.text_input(
            "Average Pace (MM:SS)",
            value=run_to_edit["avg_pace"] if editing else "",
        )

        splits = st.text_area("Splits", value=run_to_edit["splits"] if editing else "")

        avg_hr = st.number_input("Average HR", min_value=0, max_value=250,
                                 value=int(run_to_edit["avg_hr"]) if editing and run_to_edit["avg_hr"] else 0)

        max_hr = st.number_input("Max HR", min_value=0, max_value=250,
                                 value=int(run_to_edit["max_hr"]) if editing and run_to_edit["max_hr"] else 0)

        hr_by_segment = st.text_area("HR by Segment", value=run_to_edit["hr_by_segment"] if editing else "")

        cadence = st.number_input("Cadence (spm)", min_value=0, max_value=300,
                                  value=int(run_to_edit["cadence"]) if editing and run_to_edit["cadence"] else 0)

        elevation = st.number_input("Elevation Gain (ft)", min_value=0, max_value=10000,
                                    value=int(run_to_edit["elevation_gain"]) if editing else 0)

        effort = st.slider("Effort (1–10)", 1, 10, value=int(run_to_edit["effort"]) if editing else 5)

        terrain = st.text_input("Terrain", value=run_to_edit["terrain"] if editing else "")
        weather = st.text_input("Weather", value=run_to_edit["weather"] if editing else "")

        how_felt = st.text_area("How I Felt", value=run_to_edit["how_felt"] if editing else "")
        pain = st.text_area("Any Pain or Tightness", value=run_to_edit["pain"] if editing else "")

        sleep_hours = st.number_input("Sleep (hours)", min_value=0.0, max_value=24.0,
                                      value=float(run_to_edit["sleep_hours"]) if editing and run_to_edit["sleep_hours"] else 0.0)

        stress = st.slider("Stress (1–5)", 1, 5, value=int(run_to_edit["stress"]) if editing else 3)
        nutrition = st.text_area("Nutrition / Hydration Notes", value=run_to_edit["nutrition_notes"] if editing else "")

        vo2 = st.number_input("VO2 Max", min_value=0.0, max_value=100.0,
                              value=float(run_to_edit["vo2max"]) if editing and run_to_edit["vo2max"] else 0.0)

        hrv = st.number_input("HRV", min_value=0, max_value=300,
                              value=int(run_to_edit["hrv"]) if editing and run_to_edit["hrv"] else 0)

        # Shoe selection
        shoes_df = fetch_shoes()
        shoe_options = ["None"] + [f"{row['id']} - {row['name']}" for _, row in shoes_df.iterrows()]

        selected_shoe = st.selectbox(
            "Shoe Used",
            shoe_options,
            index=0 if not editing or not run_to_edit["shoe_id"]
            else shoe_options.index(f"{run_to_edit['shoe_id']} - {shoes_df[shoes_df['id']==run_to_edit['shoe_id']]['name'].iloc[0]}")
        )

        if selected_shoe == "None":
            shoe_id = None
        else:
            shoe_id = int(selected_shoe.split(" - ")[0])

        submitted = st.form_submit_button("Save Run", use_container_width=True)

        # ---------------------------------------------------------
        # ON SAVE
        # ---------------------------------------------------------
        if submitted:
            minutes = duration_to_minutes(duration_str)

            data = {
                "date": date_value.strftime("%Y-%m-%d"),
                "run_type": run_type,
                "distance": distance,
                "duration_minutes": minutes,
                "avg_pace": avg_pace,
                "splits": splits,
                "avg_hr": avg_hr,
                "max_hr": max_hr,
                "hr_by_segment": hr_by_segment,
                "cadence": cadence,
                "elevation_gain": elevation,
                "effort": effort,
                "terrain": terrain,
                "weather": weather,
                "how_felt": how_felt,
                "pain": pain,
                "sleep_hours": sleep_hours,
                "stress": stress,
                "nutrition_notes": nutrition,
                "vo2max": vo2,
                "hrv": hrv,
                "shoe_id": shoe_id,
            }

            if editing:
                update_run(editing, data)
                st.success("Run updated!")
                st.session_state["edit_run_id"] = None
            else:
                insert_run(data)
                st.success("Run logged!")

    st.markdown("---")

    # ---------------------------------------------------------
    # LIST OF RECENT RUNS
    # ---------------------------------------------------------
    st.subheader("Recent Runs")

    df = fetch_runs()

    if df.empty:
        st.info("No runs logged yet.")
        return

    for _, row in df.iloc[::-1].head(10).iterrows():
        run_id = row["id"]
        with st.expander(f"{row['date']} — {row['run_type']} — {row['distance']} mi"):
            st.write(f"**Duration:** {minutes_to_hms(row['duration_minutes'])}")
            st.write(f"**Pace:** {row['avg_pace']}")
            st.write(f"**Effort:** {row['effort']}")
            st.write(f"**Terrain:** {row['terrain']}")
            st.write(f"**Weather:** {row['weather']}")
            st.write(f"**How Felt:** {row['how_felt']}")
            st.write("---")

            c1, c2 = st.columns(2)
            if c1.button(f"✏️ Edit Run {run_id}", key=f"edit_{run_id}"):
                st.session_state["edit_run_id"] = run_id
                st.rerun()
            if c2.button(f"🗑️ Delete Run {run_id}", key=f"delete_{run_id}"):
                delete_run(run_id)
                st.success("Run deleted.")
                st.rerun()
###############################################################
# SEGMENT 8 / 12
# FEED PAGE (STRAVA-STYLE ACTIVITY FEED)
###############################################################

def render_feed_page():
    st.title("📣 Feed")

    df = fetch_runs()
    if df.empty:
        st.info("No runs logged yet.")
        return

    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    prs = calculate_prs(prepare_metrics_df(df))

    # Helper to tag PRs
    def check_pr(row):
        pr_events = []

        if prs.get("fastest_mile") and row["distance"] >= 1:
            pace = row["duration_minutes"] / row["distance"]
            if abs(pace - prs["fastest_mile"]) < 0.001:
                pr_events.append("🏅 Mile PR")

        if prs.get("fastest_5k") and row["distance"] >= 3.11:
            eff = row["duration_minutes"] / row["distance"] * 3.11
            if abs(eff - prs["fastest_5k"]) < 0.001:
                pr_events.append("🏅 5K PR")

        if prs.get("fastest_10k") and row["distance"] >= 6.22:
            eff = row["duration_minutes"] / row["distance"] * 6.22
            if abs(eff - prs["fastest_10k"]) < 0.001:
                pr_events.append("🏅 10K PR")

        if prs.get("fastest_half") and row["distance"] >= 13.1:
            eff = row["duration_minutes"] / row["distance"] * 13.1
            if abs(eff - prs["fastest_half"]) < 0.001:
                pr_events.append("🏅 HM PR")

        return pr_events

    # --------------------------------------------------------
    # FEED CARDS
    # --------------------------------------------------------
    for _, row in df.iterrows():
        run_date = row["date"]
        run_type = row["run_type"]
        distance = row["distance"]
        duration = minutes_to_hms(row["duration_minutes"])
        pace = row["avg_pace"]
        effort = row["effort"]
        terrain = row["terrain"]
        weather = row["weather"]
        shoe_id = row.get("shoe_id")

        shoes_df = fetch_shoes(include_retired=True)
        shoe_name = None
        if shoe_id and shoe_id in list(shoes_df["id"]):
            shoe_name = shoes_df[shoes_df["id"] == shoe_id]["name"].iloc[0]

        # PR Tag
        pr_tags = check_pr(row)

        # Run type tag color
        type_tag_color = {
            "Easy": "tag-Easy",
            "Long Run": "tag-LongRun",
            "Intervals": "tag-Intervals",
            "Tempo": "tag-Tempo",
            "Recovery": "tag-Recovery",
            "Race": "tag-Race",
            "Other": "tag-Other",
        }.get(run_type, "tag-Other")

        # Card
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            # Header Row
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {distance} miles — <span class='{type_tag_color}'>{run_type}</span>", unsafe_allow_html=True)
                st.markdown(f"**{run_date}**")
            with col2:
                if pr_tags:
                    for tag in pr_tags:
                        st.markdown(f"<div class='tag tag-Race'>{tag}</div>", unsafe_allow_html=True)

            # Main Metrics
            colA, colB, colC = st.columns(3)
            colA.metric("Duration", duration)
            colB.metric("Pace", pace)
            colC.metric("Effort", effort)

            # HR row
            colD, colE = st.columns(2)
            colD.write(f"**Avg HR:** {row['avg_hr']}")
            colE.write(f"**Max HR:** {row['max_hr']}")

            # Shoe
            if shoe_name:
                st.write(f"**Shoes:** {shoe_name}")

            # Terrain + weather
            st.write(f"**Terrain:** {terrain}")
            st.write(f"**Weather:** {weather}")

            # Notes
            if row["how_felt"]:
                st.write(f"**How I Felt:** {row['how_felt']}")
            if row["pain"]:
                st.write(f"**Pain / Tightness:** {row['pain']}")

            # Splits
            if row["splits"]:
                with st.expander("Splits"):
                    try:
                        split_dict = eval(row["splits"])
                        if isinstance(split_dict, dict):
                            for mile, split in split_dict.items():
                                st.write(f"Mile {mile}: {split}")
                        else:
                            st.write(row["splits"])
                    except:
                        st.write(row["splits"])

            # Spacer
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

###############################################################
# SEGMENT 9 / 12
# DASHBOARD PAGE (METRICS, CHARTS, TRENDS)
###############################################################

def render_dashboard_page():
    st.title("📊 Dashboard")

    df = fetch_runs()
    if df.empty:
        st.info("Log a few runs to unlock your dashboard.")
        return

    # Prepare metrics
    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    metrics = add_hr_zones(metrics, hr_max=st.session_state.get("hr_max", 190))

    daily = compute_daily_load(metrics)
    load_df = compute_fitness_fatigue(daily)
    prs = calculate_prs(metrics)

    # -----------------------------------------------------------
    # TOP SUMMARY METRICS
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    total_miles = metrics["distance"].sum(skipna=True)
    last7 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7_miles = last7["distance"].sum(skipna=True)

    avg_hr = metrics["avg_hr"].mean(skipna=True)
    current_streak, longest_streak = compute_streaks(metrics)

    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Distance", f"{total_display:.1f} {unit}")
    c2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    c3.metric("Avg HR", f"{avg_hr:.1f}" if not pd.isna(avg_hr) else "—")
    c4.metric("Current Streak", f"{current_streak} days")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # WEEKLY DISTANCE CHART
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Weekly Mileage Trend")

    dfw = metrics.copy()
    dfw["week"] = dfw["date_dt"].dt.isocalendar().week
    weekly = dfw.groupby("week")["distance"].sum().reset_index()

    if not weekly.empty:
        weekly["display_dist"], _ = zip(*weekly["distance"].apply(convert_distance_for_display))

        chart = (
            alt.Chart(weekly)
            .mark_bar()
            .encode(
                x=alt.X("week:O", title="Week #"),
                y=alt.Y("display_dist:Q", title=f"Mileage ({unit})"),
                tooltip=["week", "display_dist"]
            )
        )
        st.altair_chart(chart, use_container_width=True)

        if prs.get("highest_weekly_mileage"):
            st.markdown(
                f"<div class='pr-mini'>🔥 Highest Week: {prs['highest_weekly_mileage']:.1f} mi</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Not enough data for weekly chart.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # PACE TREND
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⏱ Pace Trend (min/mi)")

    p = metrics.dropna(subset=["pace_min_per_mile"])
    if not p.empty:
        chart = (
            alt.Chart(p)
            .mark_line(point=True)
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("pace_min_per_mile:Q", title="Pace (min/mi)", scale=alt.Scale(reverse=True)),
                tooltip=["date", "distance", "avg_pace"]
            )
        )
        st.altair_chart(chart, use_container_width=True)

        if prs.get("fastest_pace"):
            st.markdown(
                f"<div class='pr-mini'>⚡ Fastest Pace PR: {prs['fastest_pace']:.2f} min/mi</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Not enough pace data yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # VO2 / EFFICIENCY CHART
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧬 VO2 Max & Efficiency Trend")

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
                y=alt.Y("value:Q", title="Metric Value"),
                color="metric:N",
                tooltip=["date_dt", "metric", "value"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Log runs with HR/VO2 data to see this.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # HR TREND
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("❤️ Average HR Trend")

    hr_df = metrics.dropna(subset=["avg_hr"])
    if not hr_df.empty:
        chart = (
            alt.Chart(hr_df)
            .mark_line(point=True)
            .encode(
                x="date_dt:T",
                y="avg_hr:Q",
                tooltip=["date_dt", "avg_hr"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No HR data logged.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # HR ZONES PIE
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💓 HR Zone Distribution")

    zone_df = metrics.dropna(subset=["hr_zone"]).groupby("hr_zone")["duration_minutes"].sum().reset_index()

    if not zone_df.empty:
        chart = (
            alt.Chart(zone_df)
            .mark_arc()
            .encode(
                theta="duration_minutes:Q",
                color="hr_zone:N",
                tooltip=["hr_zone", "duration_minutes"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need heart rate data to show HR zones.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # CADENCE TREND
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🦵 Cadence Trend")

    cad = metrics.dropna(subset=["cadence"])
    if not cad.empty:
        chart = (
            alt.Chart(cad)
            .mark_line(point=True)
            .encode(
                x="date_dt:T",
                y="cadence:Q",
                tooltip=["date_dt", "cadence"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No cadence data yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # TRAINING LOAD (CTL / ATL / TSB)
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Training Load: CTL / ATL / TSB")

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
                x="date_dt:T",
                y="value:Q",
                color="metric:N",
                tooltip=["date_dt", "metric", "value"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough training load data yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # PR PANEL
    # -----------------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏅 Personal Records")

    if not prs:
        st.info("No PRs detected yet — go run!")
    else:
        for k, v in prs.items():
            label = k.replace("_", " ").title()
            if "fastest" in k:
                st.write(f"⚡ **{label}:** {v:.2f} min")
            else:
                st.write(f"🔥 **{label}:** {v:.2f} mi")

    st.markdown("</div>", unsafe_allow_html=True)
###############################################################
# SEGMENT 10 / 12
# COMPARE RUNS + PACE ZONES + ATHLETE PROFILE CORE
###############################################################

###############################################################
# HOME PAGE
###############################################################

def render_home_page():
    st.title("🏃 Run Tracker & AI Coach")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Welcome Back!")

    st.write("""
This app helps you track your training, analyze your runs, visualize progress,
and get AI-powered coaching based on your personal profile and run history.
""")

    st.write("Use the sidebar to log runs, visualize data, and generate training plans.")

    st.markdown("</div>", unsafe_allow_html=True)
###############################################################
# GARMIN IMPORT PAGE (placeholder)
###############################################################

def render_garmin_import_page():
    st.title("⌚ Garmin Import")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.write("""
This placeholder allows manual imports.  
Full Garmin automatic sync requires a partner API key, which Garmin only grants to companies.

But we can still import:

- GPX files  
- TCX files  
- FIT files (with an extra parser)
""")

    uploaded = st.file_uploader("Upload GPX/TCX/FIT file")

    if uploaded:
        st.success("File imported (placeholder). We'll build full parsing later.")

    st.markdown("</div>", unsafe_allow_html=True)
###############################################################
# AI COACH PAGE
###############################################################

def render_ai_coach_page():
    st.title("🤖 AI Coach")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.write("Ask your AI coach anything about training, pacing, race prep, injury, etc.")

    prompt = st.text_area("Your question:")

    if st.button("Ask AI"):
        if not prompt.strip():
            st.warning("Please enter a question.")
        else:
            response = call_ai(prompt)
            st.write(response)

    st.markdown("</div>", unsafe_allow_html=True)
###############################################################
# EDIT RUN PAGE (redirects into Log Run UI with item loaded)
###############################################################

def render_edit_run_page(run_id: int):
    st.title("✏️ Edit Run")

    df = fetch_runs()
    run = df[df["id"] == run_id]
    if run.empty:
        st.error("Run not found.")
        return

    # Store the ID so the Log a Run page loads editing mode
    st.session_state["edit_run_id"] = run_id

    st.info("Redirecting you to the Log a Run page to edit this run…")
    st.session_state["page"] = "Log a Run"
    st.rerun()


# =========================
# COMPARE RUNS PAGE
# =========================

def render_compare_runs_page():
    st.title("📊 Compare Two Runs")

    df = fetch_runs()
    if df.empty or len(df) < 2:
        st.info("Log at least two runs to compare them.")
        return

    df_sorted = df.sort_values("date", ascending=False)

    # Build label → row mapping
    labels = [
        f"{r['date']} — {r['run_type']} — {r['distance']} mi"
        for _, r in df_sorted.iterrows()
    ]
    mapping = {label: row for label, (_, row) in zip(labels, df_sorted.iterrows())}

    col1, col2 = st.columns(2)
    with col1:
        r1_label = st.selectbox("Run 1", labels)
    with col2:
        r2_label = st.selectbox("Run 2", labels)

    if r1_label == r2_label:
        st.warning("Please choose two different runs.")
        return

    run1 = mapping[r1_label]
    run2 = mapping[r2_label]

    # ---- Metrics table ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Side-by-Side Metrics")

    metrics_list = [
        ("Distance (mi)", "distance"),
        ("Duration (min)", "duration_minutes"),
        ("Avg HR", "avg_hr"),
        ("Max HR", "max_hr"),
        ("Cadence", "cadence"),
        ("Elevation Gain (ft)", "elevation_gain"),
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

    # ---- AI comparison ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 AI Comparison")

    if st.button("Generate AI Comparison"):
        prompt = f"""
Compare these two runs in detail. Include:
- pacing analysis
- heart rate response
- efficiency differences
- elevation effects
- stress + recovery
- training impact
- what improved or regressed
- what to focus on next

Run 1:
{dict(run1)}

Run 2:
{dict(run2)}
"""
        st.write(call_ai(prompt))

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# PACE & HR ZONES PAGE
# =========================

def render_pace_zones_page():
    st.title("📏 Pace & HR Zones")

    df = fetch_runs()
    if df.empty:
        st.info("Log runs to generate pace zones.")
        return

    metrics = prepare_metrics_df(df)
    prs = calculate_prs(metrics)

    best_pace = prs.get("fastest_pace")
    threshold_pace = estimate_threshold_pace(prs)

    easy_min, easy_max = estimate_easy_pace(best_pace) if best_pace else (None, None)
    tempo = estimate_tempo_pace(threshold_pace) if threshold_pace else None
    interval = estimate_interval_pace(best_pace) if best_pace else None
    rep = estimate_rep_pace(best_pace) if best_pace else None

    # HR zones
    hr_max = st.session_state.get("hr_max", 190)
    hr_zones = calculate_hr_zones(hr_max)

    # ---- Pace Zone Table ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏃 Pace Zones (min/mi)")

    pace_data = []

    def add_zone(name, pr):
        if pr and pr[0] and pr[1]:
            pace_data.append({
                "Zone": name,
                "Min Pace": f"{pr[0]:.2f}",
                "Max Pace": f"{pr[1]:.2f}",
            })

    if easy_min and easy_max:
        add_zone("Recovery", (easy_min + 0.4, easy_max + 0.6))
        add_zone("Easy", (easy_min, easy_max))
        add_zone("Aerobic Endurance", (easy_min - 0.3, easy_min + 0.3))

    if tempo:
        add_zone("Tempo", tempo)
    if threshold_pace:
        add_zone("Threshold", (threshold_pace * 0.98, threshold_pace * 1.02))
    if interval:
        add_zone("Interval (VO2)", interval)
    if rep:
        add_zone("Repetition", rep)

    if pace_data:
        st.table(pd.DataFrame(pace_data))
    else:
        st.info("Not enough pace data yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- HR Zones ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("❤️ Heart Rate Zones")

    hr_data = []
    for zone, (low, high) in hr_zones.items():
        hr_data.append({
            "Zone": zone,
            "Low (bpm)": int(low),
            "High (bpm)": int(high),
        })

    st.table(pd.DataFrame(hr_data))
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- AI Zone Workout ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 Zone Workout Generator")

    zone_choice = st.selectbox(
        "Select a zone",
        [
            "Recovery", "Easy", "Aerobic Endurance",
            "Tempo", "Threshold", "Interval (VO2)", "Repetition",
        ],
    )

    if st.button("Generate Workout"):
        prompt = f"""
Create a structured running workout targeting the **{zone_choice}** zone.
Include warm-up, main set, cooldown, paces, HR zones, purpose, and where it fits
in half-marathon or marathon training.
"""
        st.write(call_ai(prompt))

    st.markdown("</div>", unsafe_allow_html=True)



# ============================================================
# ATHLETE PROFILE CORE (loaded & added to training logic)
# ============================================================

def load_athlete_profile():
    """
    Returns a dictionary of the athlete’s saved profile.
    Used by the AI Coach, Training Block Generator, and future modules.
    """
    return {
        "age": st.session_state.get("athlete_age", 26),
        "gender": st.session_state.get("athlete_gender", "Male"),
        "height": st.session_state.get("athlete_height", "5'9\""),
        "weight_lbs": st.session_state.get("athlete_weight_lbs", 167),
        "experience_years": st.session_state.get("athlete_experience", 1),
        "primary_goal": st.session_state.get("race_goal", "Half Marathon"),
        "race_date": st.session_state.get("race_date_str", "2026-05-03"),
        "preferred_days": st.session_state.get("athlete_preferred_days", ["Mon", "Tue", "Thu", "Sat", "Sun"]),
        "avoid_days": st.session_state.get("athlete_avoid_days", ["Fri"]),
        "injury_history": st.session_state.get("athlete_injury_history", "Shin splints"),
        "hr_max": st.session_state.get("hr_max", 190),
        "weekly_mileage_goal": st.session_state.get("weekly_goal_mi", 25),
    }
###############################################################
# SEGMENT 11 / 12
# SETTINGS PAGE — including full ATHLETE PROFILE UI + STORAGE
###############################################################

def render_settings_page():
    st.title("⚙ Settings")

    # =========================
    # Appearance
    # =========================
    st.markdown("<div class='card'><h3>🎨 Appearance</h3></div>", unsafe_allow_html=True)

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
            "Compact Mode",
            value=st.session_state["compact_mode"],
        )
        st.session_state["compact_mode"] = compact


    # =========================
    # AI Settings
    # =========================
    st.markdown("<div class='card'><h3>🤖 AI Coaching Settings</h3></div>", unsafe_allow_html=True)

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



    # =========================
    # Athlete Profile
    # =========================
    st.markdown("<div class='card'><h3>🏃 Athlete Profile</h3></div>", unsafe_allow_html=True)

    # Load existing values or defaults
    age = st.number_input("Age", min_value=10, max_value=90, value=st.session_state.get("athlete_age", 26))
    gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary", "Other"], index=["Male","Female","Non-Binary","Other"].index(st.session_state.get("athlete_gender","Male")))
    height = st.text_input("Height (e.g. 5'9\")", value=st.session_state.get("athlete_height","5'9\""))
    weight_lbs = st.number_input("Weight (lbs)", min_value=70, max_value=400, value=st.session_state.get("athlete_weight_lbs",167))
    experience_years = st.number_input("Years Running", min_value=0, max_value=40, value=st.session_state.get("athlete_experience",1))

    injuries = st.text_area(
        "Injury History",
        value=st.session_state.get("athlete_injury_history","Shin splints"),
        placeholder="List any injuries, sensitivity areas, medical notes…"
    )

    preferred_days = st.multiselect(
        "Preferred Training Days",
        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        default=st.session_state.get("athlete_preferred_days",["Mon","Tue","Thu","Sat","Sun"]),
    )

    avoid_days = st.multiselect(
        "Days to Avoid",
        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        default=st.session_state.get("athlete_avoid_days",["Fri"]),
    )


    # Save values
    st.session_state["athlete_age"] = age
    st.session_state["athlete_gender"] = gender
    st.session_state["athlete_height"] = height
    st.session_state["athlete_weight_lbs"] = weight_lbs
    st.session_state["athlete_experience"] = experience_years
    st.session_state["athlete_injury_history"] = injuries
    st.session_state["athlete_preferred_days"] = preferred_days
    st.session_state["athlete_avoid_days"] = avoid_days


    # =========================
    # Race Goal & HR
    # =========================
    st.markdown("<div class='card'><h3>🎯 Race Goal & Heart Rate</h3></div>", unsafe_allow_html=True)

    col6, col7 = st.columns(2)
    with col6:
        goal = st.text_input("Race Goal Description", value=st.session_state["race_goal"])
        st.session_state["race_goal"] = goal

        rd = st.date_input(
            "Race Date",
            value=datetime.fromisoformat(st.session_state["race_date_str"]).date()
        )
        st.session_state["race_date_str"] = rd.isoformat()

    with col7:
        hr_max = st.number_input(
            "HR Max (bpm)",
            min_value=140,
            max_value=220,
            value=st.session_state.get("hr_max", 190)
        )
        st.session_state["hr_max"] = hr_max

        weekly_goal = st.number_input(
            "Weekly Mileage Goal (mi)",
            min_value=5.0,
            max_value=200.0,
            value=float(st.session_state.get("weekly_goal_mi", 25.0))
        )
        st.session_state["weekly_goal_mi"] = weekly_goal



    # =========================
    # Shoe Management
    # =========================
    st.markdown("<div class='card'><h3>👟 Shoes</h3></div>", unsafe_allow_html=True)
    st.write("Track shoe mileage and retire shoes when worn out.")

    with st.form("add_shoe"):
        colA, colB = st.columns(2)
        with colA:
            shoe_name = st.text_input("Shoe Name")
        with colB:
            shoe_brand = st.text_input("Brand")

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
                    <strong>{s['name']} ({s['brand']})</strong><br>
                    Status: {status}<br>
                    Started: {s['start_date']}<br>
                    Total Miles: {mileage:.1f}
                </div>
                """,
                unsafe_allow_html=True
            )

            if not s["retired"]:
                if st.button(f"Retire {s['name']}", key=f"retire_{s['id']}"):
                    retire_shoe(s["id"])
                    st.warning(f"{s['name']} retired.")


    # =========================
    # Data Export / Delete
    # =========================
    st.markdown("<div class='card'><h3>📁 Data Controls</h3></div>", unsafe_allow_html=True)

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
        if st.button("⚠ Delete ALL Run Data"):
            conn = get_conn()
            conn.execute("DELETE FROM runs")
            conn.commit()
            conn.close()
            st.warning("All run data deleted.")
###############################################################
# SEGMENT 12 / 12
# MAIN APPLICATION ROUTER + INITIALIZATION
###############################################################

def main():
    # Set page config (must run before anything else)
    st.set_page_config(
        page_title="Run Tracker & AI Coach",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize state, DB, CSS
    init_session_state()
    init_db_with_migration()
    inject_css()

    # =========================
    # SIDEBAR NAVIGATION
    # =========================
    st.sidebar.header("Navigation")

    # Appearance Controls
    st.sidebar.subheader("Appearance")
    st.session_state["theme"] = st.sidebar.radio(
        "Theme",
        ["dark", "light"],
        index=0 if st.session_state["theme"] == "dark" else 1,
        key="sidebar_theme_select",
    )

    st.session_state["units"] = st.sidebar.radio(
        "Units",
        ["mi", "km"],
        index=0 if st.session_state["units"] == "mi" else 1,
        key="sidebar_units_select",
    )

    st.session_state["compact_mode"] = st.sidebar.checkbox(
        "Compact Mode",
        value=st.session_state["compact_mode"],
        key="sidebar_compact_select",
    )

    # Race Goal Quick Edit
    st.sidebar.subheader("Race Goal")
    st.session_state["race_goal"] = st.sidebar.text_input(
        "Race Goal",
        value=st.session_state["race_goal"],
        key="sidebar_race_goal_edit",
    )

    # Page Router
    st.sidebar.subheader("Pages")

    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Feed",
            "Log a Run",
            "Dashboard",
            "Garmin Import",
            "AI Coach",
            "Compare Runs",
            "Pace Zones",
            "Athlete Profile",
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
            "Athlete Profile",
            "Settings",
        ].index(st.session_state.get("page", "Home")),
        key="sidebar_page_picker",
    )

    st.session_state["page"] = page

    # =========================
    # ROUTE TO PAGE
    # =========================
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

    elif page == "Athlete Profile":
        render_athlete_profile_page()

    elif page == "Settings":
        render_settings_page()

    elif page == "Edit Run":
        rid = st.session_state.get("edit_run_id")
        if rid is None:
            st.error("No run selected to edit.")
        else:
            render_edit_run_page(int(rid))


# Run the app
if __name__ == "__main__":
    main()
