import streamlit as st

from styles import inject_css
from db import init_db
from utils import init_session_state
from run_tracker.pages import (
    home,
    feed,
    log_run,
    edit_run,
    dashboard,
    garmin_import,
    ai_coach,
    compare_runs,
    pace_zones,
)


def main():
    st.set_page_config(page_title="Run Tracker & AI Coach", layout="wide")

    init_session_state()
    init_db()
    inject_css()

    st.sidebar.header("Navigation")

    page = st.sidebar.selectbox(
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
        ],
    )

    if page == "Home":
        home.render()
    elif page == "Feed":
        feed.render()
    elif page == "Log a Run":
        log_run.render()
    elif page == "Dashboard":
        dashboard.render()
    elif page == "Garmin Import":
        garmin_import.render()
    elif page == "AI Coach":
        ai_coach.render()
    elif page == "Compare Runs":
        compare_runs.render()
    elif page == "Pace Zones":
        pace_zones.render()
