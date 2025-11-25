import os
import sqlite3
from datetime import datetime, timedelta, date

import pandas as pd
import streamlit as st
import altair as alt

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
        st.session_state["theme"] = "dark"  # 'dark' or 'light'
    if "units" not in st.session_state:
        st.session_state["units"] = "mi"    # 'mi' or 'km'
    if "ai_verbosity" not in st.session_state:
        st.session_state["ai_verbosity"] = "normal"  # 'short','normal','detailed'
    if "ai_focus" not in st.session_state:
        st.session_state["ai_focus"] = "balanced"    # 'performance','injury','balanced'
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

    html, body, [class*="css"]  {{
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

    /* NAVBAR */
    .top-nav {{
        position: sticky;
        top: 0;
        z-index: 100;
        background: linear-gradient(90deg, {bg2}, {bg});
        padding: 0.5rem 1rem 0.4rem 1rem;
        border-bottom: 1px solid {border};
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
    }}

    .top-nav-inner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 1200px;
        margin: 0 auto;
    }}

    .nav-left {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}

    .nav-title {{
        font-weight: 600;
        font-size: 1.1rem;
    }}

    .nav-links {{
        display: flex;
        gap: 0.3rem;
        margin-left: 1.5rem;
    }}

    .nav-right {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
    }}

    /* Cards */
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

    .stButton>button {{
        border-radius: 999px !important;
        padding: 0.35rem 1.0rem !important;
        font-weight: 600 !important;
    }}

    input, textarea, select {{
        border-radius: 8px !important;
    }}

    .stDataFrame {{
        border-radius: 10px;
        overflow: hidden;
    }}

    section[data-testid="stSidebar"] {{
        padding-top: 2rem;
    }}

    .section-divider {{
        text-align: center;
        opacity: 0.6;
        margin: 0.8rem 0 0.4rem 0;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()
# ---------- Utilities & analytics ----------

def duration_to_minutes(time_str):
    """Convert MM:SS or HH:MM:SS into total minutes."""
    if not time_str:
        return None
    parts = time_str.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None

    if len(parts) == 2:      # MM:SS
        m, s = parts
        return m + s / 60.0
    if len(parts) == 3:      # HH:MM:SS
        h, m, s = parts
        return h * 60 + m + s / 60.0

    return None


def prepare_metrics_df(df):
    """Normalize and compute derived metrics."""
    if df.empty:
        return df

    m = df.copy()

    # Date normalization
    m["date_dt"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date_dt"])
    m["date"] = m["date_dt"].dt.date

    # Start of week (Monday)
    m["week_start"] = m["date_dt"] - pd.to_timedelta(m["date_dt"].dt.weekday, unit="D")
    m["week_start"] = m["week_start"].dt.date

    # Training load = effort × duration
    m["training_load"] = m["effort"].fillna(0) * m["duration_minutes"].fillna(0)

    # Pace (min/mi)
    m["pace_min_per_mile"] = None
    mask = (
        m["distance"].notna()
        & (m["distance"] > 0)
        & m["duration_minutes"].notna()
    )
    m.loc[mask, "pace_min_per_mile"] = (
        m.loc[mask, "duration_minutes"] / m.loc[mask, "distance"]
    )

    # Running Stress Score (RSS)
    hr_max = float(st.session_state.get("hr_max", 190))
    m["rss"] = None
    mask_rss = m["duration_minutes"].notna() & m["avg_hr"].notna()
    m.loc[mask_rss, "rss"] = (
        m.loc[mask_rss, "duration_minutes"]
        * (m.loc[mask_rss, "avg_hr"] / hr_max) ** 2
    )

    return m


def compute_daily_load(metrics):
    """Summarize distance, training load, RSS per day."""
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


def compute_fitness_fatigue(daily):
    """Compute CTL/ATL/TSB (fitness, fatigue, form)."""
    if daily.empty:
        return daily

    d = daily.copy().sort_values("date_dt")
    d["CTL"] = d["training_load"].rolling(window=42, min_periods=1).mean()
    d["ATL"] = d["training_load"].rolling(window=7, min_periods=1).mean()
    d["TSB"] = d["CTL"] - d["ATL"]
    return d


def compute_efficiency_score(metrics):
    """Simple efficiency metric combining distance, duration, and HR."""
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


def add_hr_zones(metrics, hr_max=190):
    """Assign each run to an HR zone based on avg HR."""
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


def compute_streaks(metrics):
    """Compute current and longest run streaks."""
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


def convert_distance_for_display(distance_mi):
    """Convert between miles and km based on user setting."""
    units = st.session_state.get("units", "mi")
    if distance_mi is None:
        return None, units
    if units == "km":
        return distance_mi * 1.60934, "km"
    return distance_mi, "mi"
    

# ---------- AI helpers ----------

def ensure_openai_client():
    """Ensure OpenAI client is initialized."""
    global client
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            client = None
    return client


def call_ai(prompt):
    """Unified helper to call OpenAI with the app's settings."""
    c = ensure_openai_client()
    if c is None:
        return "OpenAI client not installed. Install 'openai' and set OPENAI_API_KEY."

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


# ---------- Garmin CSV parser ----------

def build_run_from_garmin_df(df):
    """Attempt to parse Garmin CSV fields and convert into a standard run entry."""
    if df.empty:
        return None, "CSV appears to be empty."

    row = df.iloc[0]
    cols = list(df.columns)

    def get_val(*names):
        for n in names:
            if n in cols and pd.notna(row[n]):
                return row[n]
        return None

    # Date parsing
    date_val = get_val("Start Time", "Start", "Date", "Timestamp")
    date_iso = datetime.today().date().isoformat()

    if date_val is not None:
        dt = pd.to_datetime(str(date_val), errors="coerce")
        if pd.notna(dt):
            date_iso = dt.date().isoformat()

    # Distance extraction
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

    # Duration extraction
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

    # Pace
    avg_pace = None
    for c in cols:
        if "pace" in c.lower():
            avg_pace = str(row[c])
            break

    # HR
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

    # Cadence
    cadence = get_val("Average Run Cadence", "Avg Run Cadence", "Run Cadence")
    try:
        cadence = int(float(cadence)) if cadence is not None else None
    except Exception:
        cadence = None

    # Elevation
    elevation_gain = get_val("Elevation Gain", "Total Ascent", "Ascent")
    try:
        elevation_gain = int(float(elevation_gain)) if elevation_gain is not None else None
    except Exception:
        elevation_gain = None

    # VO2
    vo2 = get_val("VO2 Max", "VO2Max")
    try:
        vo2 = float(vo2) if vo2 is not None else None
    except Exception:
        vo2 = None

    # HRV
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
# ---------- Navbar ----------

def render_navbar():
    st.markdown(
        """
        <div class="top-nav">
            <div class="top-nav-inner">
                <div class="nav-left">
                    <span class="nav-title">🏃 Run Tracker + AI Coach</span>
                    <div class="nav-links">
                        <a href="#" onclick="window.location.search='?page=Home'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">Home</button>
                        </a>
                        <a href="#" onclick="window.location.search='?page=Feed'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">Feed</button>
                        </a>
                        <a href="#" onclick="window.location.search='?page=Log%20a%20Run'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">Log Run</button>
                        </a>
                        <a href="#" onclick="window.location.search='?page=Dashboard'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">Dashboard</button>
                        </a>
                        <a href="#" onclick="window.location.search='?page=Garmin%20Import'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">Garmin Import</button>
                        </a>
                        <a href="#" onclick="window.location.search='?page=AI%20Coach'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">AI Coach</button>
                        </a>
                        <a href="#" onclick="window.location.search='?page=Compare%20Runs'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">Compare</button>
                        </a>
                        <a href="#" onclick="window.location.search='?page=Settings'" style="text-decoration:none;">
                            <button class="css-1q8dd3e edgvbvh10">Settings</button>
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Home Page ----------

def render_home_page():
    st.title("🏠 Home")

    st.markdown(
        """
        <div class='card'>
        <h3>Welcome to Your Run Tracker + AI Coach</h3>
        <p>Analyze your runs, visualize training trends, and get personalized coaching.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-divider'>Overview</div>", unsafe_allow_html=True)

    df = fetch_runs()
    if df.empty:
        st.info("You have no logged runs yet. Log one or import from Garmin!")
        return

    metrics = prepare_metrics_df(df)
    total_miles = metrics["distance"].sum(skipna=True)
    last_7 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7_miles = last_7["distance"].sum(skipna=True)

    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)

    col1, col2 = st.columns(2)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")

    st.markdown("<div class='section-divider'>Recent Activity</div>", unsafe_allow_html=True)

    for _, row in df.tail(5).iloc[::-1].iterrows():
        icon = "🏃"
        rt = row["run_type"] or "Other"

        distance, dunit = convert_distance_for_display(row["distance"])
        st.markdown(
            f"""
            <div class='card feed-card'>
                <div class='feed-header-line'>
                    <span><strong>{row['date']}</strong></span>
                    <span class='tag tag-{rt}'>{rt}</span>
                </div>
                <div class='feed-main-metrics'>
                    <span class='big-distance'>{distance:.2f} {dunit}</span>
                    <span class='muted'>Avg Pace: {row['avg_pace'] or '—'}</span>
                    <span class='muted'>HR: {row['avg_hr'] or '—'} bpm</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Feed Page ----------

def render_feed_page():
    st.title("📜 Training Feed")

    df = fetch_runs()
    if df.empty:
        st.info("No runs yet — log one or import from Garmin.")
        return

    df = df.sort_values("date", ascending=False)

    for _, row in df.iterrows():
        rt = row["run_type"] or "Other"
        distance, dunit = convert_distance_for_display(row["distance"])

        st.markdown(
            f"""
            <div class='card feed-card'>
                <div class='feed-header-line'>
                    <span><strong>{row['date']}</strong></span>
                    <span class='tag tag-{rt}'>{rt}</span>
                </div>

                <div class='feed-main-metrics'>
                    <span class='big-distance'>{distance:.2f} {dunit}</span>
                    <span class='muted'>Pace: {row['avg_pace'] or '—'}</span>
                    <span class='muted'>HR: {row['avg_hr'] or '—'} bpm</span>
                    <span class='muted'>Elev: {row['elevation_gain'] or '—'} ft</span>
                </div>

                <span class='muted'>Effort: {row['effort'] or '—'} / 10</span>
                <span class='muted'>Felt: {row['how_felt'] or '—'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Log Run Page ----------

def insert_run(data):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO runs
        (date, run_type, distance, duration_minutes, avg_pace, splits,
         avg_hr, max_hr, hr_by_segment, cadence, elevation_gain, effort,
         terrain, weather, how_felt, pain, sleep_hours, stress,
         nutrition_notes, vo2max, hrv)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ),
    )
    conn.commit()
    conn.close()


def fetch_runs():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY date ASC", conn)
    conn.close()
    return df


def render_log_run_page():
    st.title("📝 Log a Run")

    with st.form("log_run"):
        date_val = st.date_input("Date", datetime.today())
        run_type = st.selectbox("Run Type", ["Easy", "LongRun", "Tempo", "Intervals", "Recovery", "Race", "Other"])
        distance = st.number_input("Distance (mi)", min_value=0.0, step=0.01)
        duration = st.text_input("Duration (MM:SS or HH:MM:SS)")
        avg_pace = st.text_input("Avg Pace (optional)")
        splits = st.text_area("Splits (optional)")

        avg_hr = st.number_input("Avg HR", min_value=0, max_value=240, value=0)
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
        nutrition = st.text_area("Nutrition/Hydration Notes")

        vo2 = st.number_input("VO2 Max", min_value=0.0, step=0.1)
        hrv = st.number_input("HRV", min_value=0, max_value=200)

        submitted = st.form_submit_button("Save Run")

        if submitted:
            minutes = duration_to_minutes(duration)
            if minutes is None:
                st.error("Invalid duration format.")
                return

            data = {
                "date": date_val.isoformat(),
                "run_type": run_type,
                "distance": distance,
                "duration_minutes": minutes,
                "avg_pace": avg_pace,
                "splits": splits,
                "avg_hr": avg_hr,
                "max_hr": max_hr,
                "hr_by_segment": "",
                "cadence": cadence,
                "elevation_gain": elevation_gain,
                "effort": effort,
                "terrain": terrain,
                "weather": weather,
                "how_felt": how_felt,
                "pain": pain,
                "sleep_hours": sleep,
                "stress": stress,
                "nutrition_notes": nutrition,
                "vo2max": vo2,
                "hrv": hrv,
            }

            insert_run(data)
            st.success("Run saved!")


# ---------- Garmin Import Page ----------

def render_garmin_import_page():
    st.title("📥 Garmin CSV Import")

    uploaded = st.file_uploader("Upload Garmin CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            data, msg = build_run_from_garmin_df(df)

            st.info(msg)

            if st.button("Import Run"):
                insert_run(data)
                st.success("Garmin run imported!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
# ---------- Dashboard Page ----------

def render_dashboard_page():
    st.title("📊 Dashboard")

    df = fetch_runs()
    if df.empty:
        st.info("Log or import some runs to view your dashboard.")
        return

    # Prepare metrics
    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    metrics = add_hr_zones(metrics, hr_max=st.session_state.get("hr_max", 190))

    daily = compute_daily_load(metrics)
    load_df = compute_fitness_fatigue(daily)
    current_streak, longest_streak = compute_streaks(metrics)

    # --- TOP SUMMARY METRICS ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    total_miles = metrics["distance"].sum(skipna=True)
    last7 = metrics[metrics["date_dt"] >= datetime.now() - timedelta(days=7)]
    last7_miles = last7["distance"].sum(skipna=True)
    eff_avg = metrics["efficiency_score"].mean(skipna=True)

    total_display, unit = convert_distance_for_display(total_miles)
    last7_display, _ = convert_distance_for_display(last7_miles)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{total_display:.1f} {unit}")
    col2.metric("Last 7 Days", f"{last7_display:.1f} {unit}")
    col3.metric("Avg Efficiency", f"{eff_avg:.1f}" if not pd.isna(eff_avg) else "—")
    col4.metric("Current Streak", f"{current_streak} days")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Weekly Distance Chart
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weekly Distance")

    weekly = metrics.groupby("week_start", as_index=False)["distance"].sum()
    if not weekly.empty:
        weekly["display_dist"], _ = zip(*weekly["distance"].apply(convert_distance_for_display))

        chart = (
            alt.Chart(weekly)
            .mark_bar()
            .encode(
                x=alt.X("week_start:T", title="Week"),
                y=alt.Y("display_dist:Q", title=f"Distance ({unit})"),
                tooltip=["week_start", "display_dist"]
            )
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data for weekly chart.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Pace Trend (min/mi)
    # -----------------------------
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
                tooltip=["date", "distance", "avg_pace"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Log distance+duration to view pace trends.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # HR Trend & HR Zone Distribution
    # -----------------------------
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
                    tooltip=["date", "avg_hr"]
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
                    tooltip=["hr_zone", "duration_minutes"]
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Need HR data to show zones.")

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Cadence Trend
    # -----------------------------
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
                tooltip=["date", "cadence"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Log cadence to view this trend.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Efficiency Trend
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Efficiency Trend")

    eff = metrics.dropna(subset=["efficiency_score"])
    if not eff.empty:
        chart = (
            alt.Chart(eff)
            .mark_line(point=True)
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("efficiency_score:Q", title="Efficiency Score"),
                tooltip=["date", "efficiency_score"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need HR + distance + duration to compute efficiency.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # VO2max Trend
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("VO2 Max Trend")

    vo2 = metrics.dropna(subset=["vo2max"])
    if not vo2.empty:
        chart = (
            alt.Chart(vo2)
            .mark_line(point=True)
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("vo2max:Q", title="VO2 Max"),
                tooltip=["date", "vo2max"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Log VO2max to view trend.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Training Load (CTL / ATL / TSB)
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Load – Fitness (CTL), Fatigue (ATL), Form (TSB)")

    if not load_df.empty:
        melted = load_df.melt(
            id_vars="date_dt",
            value_vars=["CTL", "ATL", "TSB"],
            var_name="metric",
            value_name="value"
        )

        chart = (
            alt.Chart(melted)
            .mark_line()
            .encode(
                x=alt.X("date_dt:T", title="Date"),
                y=alt.Y("value:Q", title="Load"),
                color="metric:N",
                tooltip=["date_dt", "metric", "value"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Training load data unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Monthly Heatmap
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Monthly Distance Heatmap")

    cal = daily.copy()
    if not cal.empty:
        cal["date"] = cal["date_dt"].dt.date
        cal["dow"] = cal["date_dt"].dt.weekday
        cal["week"] = cal["date_dt"].dt.isocalendar().week

        chart = (
            alt.Chart(cal)
            .mark_rect()
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("dow:O", title="Day of Week"),
                color=alt.Color("distance:Q", title=f"Distance ({unit})"),
                tooltip=["date", "distance"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No daily data for heatmap.")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Training Type Breakdown
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Type Distribution")

    type_df = metrics.groupby("run_type", as_index=False)["distance"].sum()
    if not type_df.empty:
        chart = (
            alt.Chart(type_df)
            .mark_arc()
            .encode(
                theta="distance:Q",
                color="run_type:N",
                tooltip=["run_type", "distance"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need run types logged.")

    st.markdown("</div>", unsafe_allow_html=True)
# ---------- AI Coach Page ----------

def render_ai_coach_page():
    st.title("🤖 AI Coach")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs or import Garmin data for AI analysis.")
        return

    recent = df.tail(30)
    latest = df.iloc[-1].to_dict()
    race_goal = st.session_state.get("race_goal", "")
    race_date = st.session_state.get("race_date_str", "")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Daily & Weekly Analysis", "Workout Generator", "7-Day Planner",
         "Race Simulator", "Injury Risk"]
    )

    # --- TAB 1: Basic Run / Weekly Analysis ---
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
                prompt = (
                    "Provide a detailed weekly training summary including mileage, pace trends, HR trends, "
                    "efficiency changes, fatigue level, suggested training load next week, and recovery advice.\n\n"
                    f"Weekly data: {last7.to_dict('records')}"
                )
                st.write(call_ai(prompt))

    # --- TAB 2: Workout Generator for Tomorrow ---
    with tab2:
        st.markdown("<div class='card'><h3>Generate Tomorrow’s Workout</h3></div>", unsafe_allow_html=True)
        if st.button("Create Tomorrow's Workout"):
            prompt = (
                "Based on my last 2–3 runs, fitness level, and goal race, "
                "generate a complete training session for tomorrow including warm-up, "
                "main set, paces or HR zones, cooldown, and purpose.\n\n"
                f"Race goal: {race_goal}\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))

    # --- TAB 3: AI-Generated 7-Day Training Plan ---
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

    # --- TAB 5: Injury Risk Assessment ---
    with tab5:
        st.markdown("<div class='card'><h3>Injury Risk Check</h3></div>", unsafe_allow_html=True)
        if st.button("Evaluate Injury Risk"):
            prompt = (
                "Evaluate my injury risk with focus on shin splints. Analyze training load changes, "
                "pace/hard effort spikes, elevation, fatigue, sleep, and pain notes. "
                "Give a risk rating and 4–6 recommended adjustments.\n\n"
                f"Recent runs: {recent.to_dict('records')}"
            )
            st.write(call_ai(prompt))



# ---------- Compare Runs Page ----------

def render_compare_runs_page():
    st.title("📊 Compare Two Runs")

    df = fetch_runs()
    if df.empty or len(df) < 2:
        st.info("Log at least two runs to compare them.")
        return

    run_ids = df["id"].tolist()
    run_labels = [f"{r['date']} — {r['run_type']} — {r['distance']} mi" for _, r in df.iterrows()]
    id_to_run = {label: r for label, (_, r) in zip(run_labels, df.iterrows())}

    col1, col2 = st.columns(2)

    with col1:
        r1_label = st.selectbox("Run 1", run_labels)
        run1 = id_to_run[r1_label]

    with col2:
        r2_label = st.selectbox("Run 2", run_labels)
        run2 = id_to_run[r2_label]

    # Prevent comparing same run
    if r1_label == r2_label:
        st.warning("Please select two different runs.")
        return

    # Compute differences
    def diff(a, b):
        if a is None or b is None:
            return "—"
        return round(b - a, 2)

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

    # Mini Pace Comparison Chart
    st.markdown("### 📈 Performance Comparison")

    df_compare = pd.DataFrame([
        {"run": r1_label, "distance": run1["distance"], "hr": run1["avg_hr"], "elev": run1["elevation_gain"]},
        {"run": r2_label, "distance": run2["distance"], "hr": run2["avg_hr"], "elev": run2["elevation_gain"]},
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

    # AI Summary of Comparison
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

    # --- Data Management ---
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


# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="Run Tracker & AI Coach", layout="wide")
    init_session_state()
    init_db()

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

    # ✅ Race goal in sidebar uses a DIFFERENT key than settings page
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
            "Settings",
        ].index(st.session_state.get("page", "Home")),
    )
    st.session_state["page"] = page

    inject_css()
    # You *can* call render_navbar() if you want the top bar visible, or skip it
    # render_navbar()

    # Route to the selected page
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
    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
