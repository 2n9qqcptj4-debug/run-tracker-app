from datetime import timedelta
import pandas as pd
import streamlit as st

# =========================
# BASIC TIME / PACE UTILS
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


# =========================
# METRICS PREP
# =========================

def prepare_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    m = df.copy()

    # date handling
    m["date_dt"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date_dt"])
    m["date"] = m["date_dt"].dt.date
    m["week_start"] = m["date_dt"] - pd.to_timedelta(m["date_dt"].dt.weekday, unit="D")
    m["week_start"] = m["week_start"].dt.date

    # training load
    m["training_load"] = m["effort"].fillna(0) * m["duration_minutes"].fillna(0)

    # pace in min/mi
    m["pace_min_per_mile"] = None
    mask = (
        m["distance"].notna()
        & (m["distance"] > 0)
        & m["duration_minutes"].notna()
    )
    m.loc[mask, "pace_min_per_mile"] = (
        m.loc[mask, "duration_minutes"] / m.loc[mask, "distance"]
    )

    # RSS (heart-rate based load)
    hr_max = float(st.session_state.get("hr_max", 190))
    m["rss"] = None
    mask_rss = m["duration_minutes"].notna() & m["avg_hr"].notna()
    m.loc[mask_rss, "rss"] = (
        m.loc[mask_rss, "duration_minutes"]
        * (m.loc[mask_rss, "avg_hr"] / hr_max) ** 2
    )

    return m


# =========================
# LOAD / FITNESS / FATIGUE
# =========================

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


# =========================
# PR / MILEAGE HELPERS
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

    def best_time(min_dist):
        r = df[df["distance"] >= min_dist]
        if r.empty:
            return None
        return (r["duration_minutes"] / r["distance"] * min_dist).min()

    prs["fastest_mile"] = best_time(1.0)
    prs["fastest_5k"] = best_time(3.11)
    prs["fastest_10k"] = best_time(6.22)
    prs["fastest_half"] = best_time(13.1)

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


# =========================
# STREAKS
# =========================

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


# =========================
# HR ZONES
# =========================

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


def calculate_hr_zones(hr_max):
    return {
        "Zone 1 (Recovery)": (0.50 * hr_max, 0.60 * hr_max),
        "Zone 2 (Easy / Aerobic)": (0.60 * hr_max, 0.70 * hr_max),
        "Zone 3 (Tempo)": (0.70 * hr_max, 0.80 * hr_max),
        "Zone 4 (Threshold)": (0.80 * hr_max, 0.90 * hr_max),
        "Zone 5 (VO2 Max)": (0.90 * hr_max, 1.00 * hr_max),
    }


# =========================
# RACE PREDICTIONS
# =========================

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


# =========================
# PACE ZONE HELPERS
# =========================

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
