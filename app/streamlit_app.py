"""Dubai AI Travel Planner — Streamlit home page.

Run with ``streamlit run app/streamlit_app.py`` (``LLM_PROVIDER=mock`` works offline).
"""

from __future__ import annotations

import streamlit as st
from ui_shared import (
    current_session_id,
    deployment_notices,
    rejected_link_notice,
    services,
    sidebar,
)

st.set_page_config(page_title="Dubai AI Travel Planner", page_icon="🌴", layout="wide")
svc = services()
sidebar(svc)

st.title("🌴 Dubai AI Travel Planner")
rejected_link_notice("session")
deployment_notices(svc)
st.write(
    "Describe your trip in plain English. Specialist agents research attractions, hotels, "
    "restaurants, transport and the weather in parallel; a budget controller allocates "
    "your money; an itinerary architect schedules the days; a reviewer checks every plan "
    "against deterministic rules (budget, heat, opening hours, dietary needs) before you "
    "see it. The same agents run under **LangGraph** or **CrewAI**: pick the mode in the "
    "sidebar.")
st.page_link("pages/1_Planner.py", label="Plan a trip", icon="🗺️")
st.page_link("pages/2_Sessions.py", label="Sessions & versions", icon="🗂️")
st.page_link("pages/3_Execution.py", label="Agent execution (graph, timeline, compare)",
             icon="🔭")
st.page_link("pages/4_Evaluation.py", label="Evaluation: LangGraph vs CrewAI", icon="📊")
st.markdown(
    "**Try:** *2 adults, 10-12 Feb 2027, AED 6000, history and food* — then refine it: "
    "*reduce the budget by 20%* or *replace day 2 with indoor activities*.")
st.caption("All prices are estimates from curated public data, not live prices."
           + ("" if svc.isolated else
              " Sessions are saved locally (SQLite) and survive restarts."))
if current_session_id():
    st.caption("Your open session is in the Planner.")
