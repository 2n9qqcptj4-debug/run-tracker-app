import pandas as pd
from datetime import timedelta
import streamlit as st

# ------------------------------------------------------
# Basic time + pace utilities
# ------------------------------------------------------

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


# ------------------------------------------------------
# Prepare metrics dataframe: the core cleaner
# ------------------------------------------------------

def prepare_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    m = df.copy()

    # Convert string date → datetime
    m["date_dt"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date_dt"])
    m["date"] = m["date_dt"].dt.date

    # Week start (Monday)
    m["week_start"] = m["date_dt"] - pd.to_timedelta(m["date_dt"].dt.weekday, unit="D")
    m["week_start"] = m["week_start"].dt.date

    # Training load used for CTL/ATL
    m["training_load"] = m["effort"].fillna(0) * m["duration_minutes"].fillna(0)

    # Pace
    m["pace_min_per_mile"] = None
    mask = (
        m["distance"].notna()
        & (m["distance"] > 0)
        & m["duration_minutes"].notna()
    )
    m.loc[mask, "pace_min_per_mile"] = (
        m.loc[mask, "duration_minutes"] / m.loc[mask, "distance"]
    )

    # Heart-rate based RSS load
    hr_max = float(st.session_state.get("hr_max", 190))
    m["rss"] = None
    mask_rss = m["duration_minutes"].notna() & m["avg_hr"].notna()
    m.loc[mask_rss, "rss"] = (
        m.loc[mask_rss, "duration_minutes"]
        * (m.loc[mask_rss, "avg_hr"] / hr_max) ** 2
    )

    return m


# ------------------------------------------------------
# Aggregation functions
# ------------------------------------------------------

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


def compute_fitness_fatigue(daily: pd.DataFrame):
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


# ------------------------------------------------------
# PR / mileage / pace records
# ------------------------------------------------------

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

    # Standard race distances
    def best_time(dist):
        r = df[df["distance"] >= dist]
        if r.empty:
            return None
        return (r["duration_minutes"] / r["distance"] * dist).min()

    prs["fastest_mile"] = best_time(1.0)
    prs["fastest_5k"] = best_time(3.11)
    prs["fastest_10k"] = best_time(6.22)
    prs["fastest_half"] = best_time(13.1)

    # Weekly mileage PR
    df["week"] = df["date_dt"].dt.isocalendar().week
    df["year"] = df["date_dt"].dt.year
    weekly = df.groupby(["year", "week"])["distance"].sum()
    prs["highest_weekly_mileage"] = weekly.max()

    # Monthly mileage PR
    df["month"] = df["date_dt"].dt.month
    monthly = df.groupby(["year", "month"])["distance"].sum()
    prs["highest_monthly_mileage"] = monthly.max()

    return prs


# ------------------------------------------------------
# Streaks
# ------------------------------------------------------

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


# ------------------------------------------------------
# HR Zone mapping
# ------------------------------------------------------

def add_hr_zones(metrics: pd.DataFrame, hr_max=190):
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


# ------------------------------------------------------
# Race predictions (simple model)
# ------------------------------------------------------

def predict_race_times(df: pd.DataFrame):
    if df.empty:
        return None

    df = df.copy()
    df["pace_num"] = df["avg_pace"].apply(pace_to_float)
    best_pace = df["pace_num"].min()

    # Adjust based on VO2 and fatigue
    vo2 = df["vo2max"].dropna().mean() if "vo2max" in df.columns else None
    vo2_factor = (50 / vo2) if vo2 else 1.0

    from .metrics import compute_daily_load, compute_fitness_fatigue  # safe here
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
