"""Shared Streamlit pieces: process-wide services, the sidebar and UI state rules.

UI state rules (CLAUDE.md): ``st.session_state`` holds only the current ``session_id``
and UI flags (the selected mode, which version is shown…). Everything else is read
from SQLite through ``SessionService``. Planner runs live in ``RunManager`` (one per
process), never in the page script, so reruns cannot cancel or duplicate them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitAPIException

from dubai_planner import bootstrap
from dubai_planner.config import Settings, get_settings
from dubai_planner.data.seed import SeedCatalog
from dubai_planner.models import PlannerMode
from dubai_planner.services.runs import PlannerHub, RunManager
from dubai_planner.services.session import SessionService
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox

MODE_LABELS = {
    PlannerMode.LANGGRAPH: "LangGraph (graph, parallel research)",
    PlannerMode.CREWAI_SEQUENTIAL: "CrewAI — sequential crew",
    PlannerMode.CREWAI_HIERARCHICAL: "CrewAI — hierarchical (manager delegates)",
}
SESSION_KEY, MODE_KEY = "session_id", "mode"


@dataclass(frozen=True)
class Services:
    settings: Settings
    sessions: SessionService
    hub: PlannerHub
    runs: RunManager
    catalog: SeedCatalog


@st.cache_resource(show_spinner=False)
def _services(db_path: str, checkpoint_path: str, provider: str) -> Services:
    """One set per process (and per DB, so tests stay isolated). Keys are never shown."""
    settings = get_settings()
    repo = Repository(settings=settings)
    sessions = SessionService(repo)
    tools = Toolbox.default()
    hub = PlannerHub(repo, settings=settings, checkpoint_path=checkpoint_path, tools=tools)
    return Services(settings=settings, sessions=sessions,
                    hub=hub, runs=RunManager(hub, sessions, today=date.today),
                    catalog=tools.catalog)


def _secrets_to_env() -> None:
    """Expose Streamlit secrets (Community Cloud) as env vars for ``config.Settings``.

    Only top-level string/number/bool values are copied, never logged, and an existing
    environment variable always wins. Locally there is usually no ``secrets.toml``; the
    `.env` file is used instead.
    """
    try:
        items = list(st.secrets.items())
    except Exception:  # no secrets file: normal for local runs
        return
    for name, value in items:
        if isinstance(value, str | int | float | bool) and name.upper() not in os.environ:
            os.environ[name.upper()] = str(value)


def services() -> Services:
    _secrets_to_env()  # before the first Settings() is built
    settings = bootstrap()  # logging + telemetry opt-outs before any framework import
    return _services(str(settings.db_path), str(settings.checkpoint_db_path),
                     settings.llm_provider.value)


def current_mode() -> PlannerMode:
    return PlannerMode(st.session_state.get(MODE_KEY, PlannerMode.LANGGRAPH.value))


def current_session_id() -> str | None:
    value = st.session_state.get(SESSION_KEY)
    return str(value) if value else None


def select_session(session_id: str | None) -> None:
    st.session_state[SESSION_KEY] = session_id
    if session_id:
        st.query_params["session"] = session_id
    elif "session" in st.query_params:
        del st.query_params["session"]


def _restore_from_url(svc: Services) -> None:
    """A browser refresh keeps the open session (``?session=<id>`` in the URL)."""
    if SESSION_KEY in st.session_state:
        return
    wanted = st.query_params.get("session")
    st.session_state[SESSION_KEY] = wanted if wanted and svc.sessions.get(wanted) else None


def sidebar(svc: Services) -> None:
    """Mode selector, provider display, session picker and "new session"."""
    _restore_from_url(svc)
    with st.sidebar:
        st.header("Dubai Planner")
        modes = list(MODE_LABELS)
        st.selectbox("Orchestration mode", modes, key=MODE_KEY,
                     format_func=lambda m: MODE_LABELS[PlannerMode(m)],
                     help="Which framework plans your next message. Both use the same "
                          "agents, rules and data.")
        s = svc.settings
        model = s.model_for(None)
        st.caption(f"LLM: **{s.llm_provider.value}** · `{model}`")
        if s.llm_provider.value == "mock":
            st.caption("Offline mock model: plans are deterministic demos, not AI output.")
        st.divider()
        sessions = svc.sessions.list_sessions()
        ids = [x.id for x in sessions]
        labels = {x.id: f"{x.title} · {x.mode.value}" for x in sessions}
        current = current_session_id()
        if st.button("➕ New session", use_container_width=True):
            select_session(svc.sessions.create(current_mode()).id)
            st.rerun()
        if ids:
            index = ids.index(current) if current in ids else None
            chosen = st.selectbox("Session", ids, index=index, format_func=labels.__getitem__,
                                  placeholder="Choose a saved session")
            if chosen and chosen != current:
                select_session(chosen)
                st.rerun()
        else:
            st.caption("No saved sessions yet.")


def money(value: Any) -> str:
    return f"AED {float(value):,.2f}"


def nav_link(page: str, label: str, icon: str) -> None:
    """``st.page_link`` that degrades to a hint when the page is run on its own (tests,
    ``streamlit run app/pages/…``), where sibling pages cannot be resolved."""
    try:
        st.page_link(page, label=label, icon=icon)
    except StreamlitAPIException:
        st.caption(f"{icon} {label}: open it from the sidebar.")
