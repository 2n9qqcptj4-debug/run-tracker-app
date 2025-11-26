import streamlit as st

# =========================
# SESSION & GLOBAL SETTINGS
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


# =========================
# UNIT CONVERSION
# =========================

def convert_distance_for_display(distance_mi: float | None):
    """
    Takes a distance stored in miles and returns (value, unit) according
    to the user's chosen unit ("mi" or "km").
    """
    units = st.session_state.get("units", "mi")
    if distance_mi is None:
        return None, units
    if units == "km":
        return distance_mi * 1.60934, "km"
    return distance_mi, "mi"
