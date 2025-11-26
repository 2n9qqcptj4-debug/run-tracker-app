import streamlit as st
import pandas as pd
from datetime import datetime

from db import fetch_runs
from ai import build_base_context, call_ai_coach


def _settings_bar():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        verb_label = col1.selectbox(
            "AI Detail Level",
            ["Balanced", "Concise", "Deep"],
            index=0
            if st.session_state.get("ai_verbosity", "normal") == "normal"
            else 1
            if st.session_state.get("ai_verbosity") == "concise"
            else 2,
        )
        st.session_state["ai_verbosity"] = {
            "Balanced": "normal",
            "Concise": "concise",
            "Deep": "deep",
        }[verb_label]

    with col2:
        focus_label = col2.selectbox(
            "Focus",
            ["Balanced", "Race Performance", "Injury Prevention", "Base Building"],
            index=0,
        )
        st.session_state["ai_focus"] = {
            "Balanced": "balanced",
            "Race Performance": "race",
            "Injury Prevention": "injury",
            "Base Building": "base",
        }[focus_label]

    with col3:
        model_label = col3.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o"],
            index=0,
        )
        st.session_state["ai_model"] = model_label

    st.markdown("</div>", unsafe_allow_html=True)


def render():
    st.title("🤖 AI Running Coach")

    df = fetch_runs()
    if df.empty:
        st.info("Log some runs first so the AI coach has context.")
        return

    # Shared settings & context
    _settings_bar()
    context = build_base_context(df)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.expander("📋 Training Summary Used by the AI", expanded=False):
        st.write(f"**Approx. Weekly Mileage (last 4 weeks):** {context['weekly_avg']:.1f} mi")
        st.write(f"**Last 7 Days Mileage:** {context['recent_mileage']:.1f} mi")
        st.write("**Recent Runs:**")
        st.text(context["summary_text"])
    st.markdown("</div>", unsafe_allow_html=True)

    tab_week, tab_workouts, tab_race, tab_block = st.tabs(
        ["📅 Weekly Plan", "🏋️ Workouts", "🏁 Race Strategy", "📆 Training Block"]
    )

    # ------------------------------
    # WEEKLY PLAN TAB
    # ------------------------------
    with tab_week:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Generate a 7-Day Training Week")

        race_goal = st.text_input(
            "Race Goal (distance + target, e.g. 'Pittsburgh Half Marathon – sub 1:40')",
            value=st.session_state.get("race_goal", "Pittsburgh Half – Sub 1:40"),
        )
        race_date_str = st.text_input(
            "Race Date (YYYY-MM-DD)",
            value=st.session_state.get("race_date_str", "2026-05-03"),
        )

        days_per_week = st.slider("Days per week to run", 3, 7, 5)
        long_run_day = st.selectbox(
            "Preferred Long Run Day", ["Saturday", "Sunday"], index=1
        )
        weak_spots = st.text_area(
            "Anything you want to focus on? (e.g. 'tempo pace', 'downhill running', 'shin splints')",
            "",
        )
        other_notes = st.text_area(
            "Injuries, constraints, or preferences the coach should know about:",
            "",
        )

        if st.button("Generate Weekly Plan"):
            try:
                race_date = datetime.fromisoformat(race_date_str).date()
            except Exception:
                race_date = None

            base_prompt = f"""
            You are designing a 7-day running week for an intermediate runner.

            Race goal: {race_goal}
            Race date: {race_date} (if None, treat as 4–6 months away)
            Usual weekly mileage: ~{context['weekly_avg']:.1f} miles
            Recent mileage (last 7 days): {context['recent_mileage']:.1f} miles
            Days per week to run: {days_per_week}
            Preferred long run day: {long_run_day}

            Runner notes / weak spots:
            {weak_spots}

            Health / constraints:
            {other_notes}

            Recent training history:
            {context['summary_text']}

            Please return:
            - A 7-day schedule with each day labeled (e.g. 'Mon', 'Tue')
            - For each day: run type (easy, tempo, long run, etc.), distance, suggested pace or effort, and any notes
            - A short explanation (2–4 bullets) of the overall structure and what the week is trying to achieve
            """

            with st.spinner("Asking your AI coach for a weekly plan..."):
                reply = call_ai_coach(base_prompt)
            st.markdown(reply)
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------
    # WORKOUT TAB
    # ------------------------------
    with tab_workouts:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Get Specific Workout Ideas")

        goal_distance = st.selectbox(
            "Goal Distance",
            ["5K", "10K", "Half Marathon", "Marathon", "General Fitness"],
            index=2,
        )
        session_type = st.selectbox(
            "Type of Workout",
            ["Easy", "Tempo", "Intervals", "Hill Repeats", "Long Run", "Speed / Reps"],
        )
        available_time = st.text_input(
            "Time Available (e.g. '45 minutes', '90 minutes')", "60 minutes"
        )
        workout_notes = st.text_area(
            "Any specific constraints? (e.g. treadmill only, bad weather, shin discomfort)",
            "",
        )

        if st.button("Suggest Workouts"):
            base_prompt = f"""
            I am a runner training primarily for {goal_distance}.
            I have about {available_time} available.

            I would like a {session_type} workout.

            Constraints / notes:
            {workout_notes}

            My recent training history:
            {context['summary_text']}

            Please give:
            - 2–3 specific workout options
            - For each: warm-up, main set, cool-down, suggested paces or effort, and brief rationale
            """

            with st.spinner("Getting workout ideas from your AI coach..."):
                reply = call_ai_coach(base_prompt)
            st.markdown(reply)

        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------
    # RACE STRATEGY TAB
    # ------------------------------
    with tab_race:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Race Strategy & Pacing")

        race_type = st.selectbox(
            "Race Distance",
            ["5K", "10K", "Half Marathon", "Marathon"],
            index=2,
        )
        target_time = st.text_input(
            "Target Time (e.g. '1:40:00')", "1:40:00" if race_type == "Half Marathon" else ""
        )
        course_desc = st.text_area(
            "Course & conditions (e.g. 'hilly first half, flat finish, likely cool weather')",
            "",
        )
        race_mindset = st.text_area(
            "What do you struggle with on race day? (e.g. 'going out too fast', 'panic mid-race')",
            "",
        )

        if st.button("Generate Race Plan"):
            base_prompt = f"""
            I am training for a {race_type}.
            Target finish time: {target_time or 'Not sure'}

            Course description:
            {course_desc}

            Race-day challenges:
            {race_mindset}

            My recent training:
            {context['summary_text']}

            Please provide:
            - Suggested pacing plan (e.g. per mile or per segment)
            - How I should feel/think at different parts of the race
            - Fuel / hydration suggestions (general guidance, not medical)
            - 3–5 mental cues or strategies for handling tough patches
            """

            with st.spinner("Building your race strategy..."):
                reply = call_ai_coach(base_prompt)
            st.markdown(reply)

        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------
    # TRAINING BLOCK TAB
    # ------------------------------
    with tab_block:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Multi-Week Training Block Builder")

        block_weeks = st.slider("Length of block (weeks)", 4, 20, 12)
        focus_block = st.selectbox(
            "Main Focus of this Block",
            ["Base Building", "Speed Development", "Race-Specific", "Injury Rehab / Return to Run"],
        )
        block_notes = st.text_area(
            "Important context (injury history, previous peak mileage, time availability, etc.)",
            "",
        )

        if st.button("Generate Training Block Outline"):
            base_prompt = f"""
            I want to build a {block_weeks}-week training block focused on {focus_block}.

            My approximate current weekly mileage: {context['weekly_avg']:.1f} miles.
            My recent mileage (last 7 days): {context['recent_mileage']:.1f} miles.

            Context:
            {block_notes}

            Recent training:
            {context['summary_text']}

            Please provide:
            - A week-by-week mileage progression (approximate miles each week)
            - 1–2 key sessions per week and their purpose
            - How cutback/deload weeks are structured
            - A brief overview of how this block prepares me for future racing or health
            """

            with st.spinner("Designing your training block..."):
                reply = call_ai_coach(base_prompt)
            st.markdown(reply)

        st.markdown("</div>", unsafe_allow_html=True)
