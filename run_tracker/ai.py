import textwrap
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import streamlit as st
from openai import OpenAI

from metrics import calculate_prs, compute_efficiency_score, prepare_metrics_df


# Create OpenAI client (expects OPENAI_API_KEY in env or Streamlit secrets)
def get_client():
    # If you use st.secrets["OPENAI_API_KEY"], uncomment next line & adjust
    # return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return OpenAI()


def _summarize_recent_runs(metrics: pd.DataFrame, max_runs: int = 10) -> str:
    if metrics.empty:
        return "No runs logged yet."

    df = metrics.sort_values("date_dt", ascending=False).head(max_runs)
    lines = []
    for _, r in df.iterrows():
        date = r["date_dt"].strftime("%Y-%m-%d")
        dist = r.get("distance", None)
        rt = r.get("run_type", "") or ""
        pace = r.get("avg_pace", "") or ""
        effort = r.get("effort", None)
        hr = r.get("avg_hr", None)
        line = f"- {date}: {dist:.2f} mi {rt}"
        if pace:
            line += f" @ {pace}"
        if hr and not pd.isna(hr):
            line += f", avg HR {int(hr)}"
        if effort and not pd.isna(effort):
            line += f", effort {int(effort)}/10"
        lines.append(line)
    return "\n".join(lines)


def build_base_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute key stats used in multiple AI prompts."""
    if df.empty:
        return {
            "summary_text": "No training data yet.",
            "weekly_avg": 0.0,
            "recent_mileage": 0.0,
            "prs": {},
        }

    metrics = prepare_metrics_df(df)
    metrics = compute_efficiency_score(metrics)
    prs = calculate_prs(metrics)

    # last 4 weeks average
    now = datetime.now()
    last28 = metrics[metrics["date_dt"] >= (now - pd.Timedelta(days=28))]
    weekly_avg = 0.0
    if not last28.empty:
        total = last28["distance"].sum(skipna=True)
        weekly_avg = total / 4.0

    last7 = metrics[metrics["date_dt"] >= (now - pd.Timedelta(days=7))]
    recent_mileage = last7["distance"].sum(skipna=True) if not last7.empty else 0.0

    summary_text = _summarize_recent_runs(metrics, max_runs=10)

    return {
        "metrics": metrics,
        "prs": prs,
        "summary_text": summary_text,
        "weekly_avg": weekly_avg,
        "recent_mileage": recent_mileage,
    }


def _verbosity_instruction() -> str:
    verb = st.session_state.get("ai_verbosity", "normal")
    if verb == "concise":
        return "Be concise and bullet-point focused. Limit yourself to short, direct advice."
    if verb == "deep":
        return "Give a very detailed, step-by-step response with explanations for your reasoning."
    return "Give a balanced level of detail—clear and structured but not overly long."


def _focus_instruction() -> str:
    focus = st.session_state.get("ai_focus", "balanced")
    if focus == "race":
        return "Prioritize race-day performance and sharpening workouts leading into the goal event."
    if focus == "injury":
        return "Prioritize injury prevention, conservative mileage, and lower-impact structure."
    if focus == "base":
        return "Prioritize aerobic base-building, easy mileage, and gradual long-run progression."
    return "Keep the advice balanced across performance, health, and long-term development."


def call_ai_coach(user_prompt: str, system_role: str = "You are an expert running coach.") -> str:
    """Call OpenAI Chat Completions with the given prompt and return text."""
    client = get_client()
    model = st.session_state.get("ai_model", "gpt-4o-mini")

    verbosity = _verbosity_instruction()
    focus = _focus_instruction()

    system_message = textwrap.dedent(
        f"""
        {system_role}

        Always make your advice specific to the user's paces, mileage, and constraints.
        {verbosity}
        {focus}
        """
    ).strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()
