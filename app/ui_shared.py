"""Shared Streamlit pieces: process-wide services, the sidebar and UI state rules.

UI state rules (CLAUDE.md): ``st.session_state`` holds only the current ``session_id``,
UI flags (the selected mode, which version is shown…) and, with
``SESSION_SCOPE=visitor``, the tab's random visitor token. Everything else is read
from SQLite through ``SessionService``. Planner runs live in ``RunManager`` (one per
process), never in the page script, so reruns cannot cancel or duplicate them.

Visitor isolation: :func:`services` returns services *scoped to the current visitor*;
they enforce ownership themselves (``dubai_planner.services.ownership``), so a page
cannot show or change another visitor's session even by passing its id. The visitor
token stays in the server-side ``st.session_state``: it is never put in the URL, shown,
logged or stored (SQLite keeps only its SHA-256).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitAPIException

from dubai_planner import bootstrap
from dubai_planner.config import LLMProvider, SessionScope, Settings, get_settings
from dubai_planner.data.seed import SeedCatalog
from dubai_planner.models import PlannerMode
from dubai_planner.services.ownership import Owner, new_visitor_token
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
#: Server-side only (never a widget key, never in the URL): the tab's visitor token.
VISITOR_KEY = "_visitor_token"
REJECTED_KEY = "_rejected_link"


@dataclass(frozen=True)
class Services:
    settings: Settings
    sessions: SessionService
    hub: PlannerHub
    runs: RunManager
    catalog: SeedCatalog

    @property
    def isolated(self) -> bool:
        """True when the services are scoped to one visitor (``SESSION_SCOPE=visitor``)."""
        return not self.sessions.owner.is_local


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


def process_services() -> Services:
    """The process-wide, *unscoped* services (scripts and tests; pages use ``services``)."""
    _secrets_to_env()  # before the first Settings() is built
    settings = bootstrap()  # logging + telemetry opt-outs before any framework import
    return _services(str(settings.db_path), str(settings.checkpoint_db_path),
                     settings.llm_provider.value)


def services() -> Services:
    """Services for the current visitor: they only see and change this visitor's data."""
    base = process_services()
    owner = visitor_owner(base.settings)
    if owner.is_local:
        return base
    sessions = base.sessions.scoped(owner)
    return replace(base, sessions=sessions, runs=base.runs.scoped(sessions))


def _authenticated_identity() -> tuple[str, str] | None:
    """(issuer, subject) of an ``st.login`` user, if the app is configured for it."""
    try:
        user = st.user
        if not user.is_logged_in:
            return None
        issuer, subject = user.get("iss"), user.get("sub")
    except Exception:  # auth not configured / not available in this runtime
        return None
    return (str(issuer), str(subject)) if issuer and subject else None


def visitor_owner(settings: Settings) -> Owner:
    """Who is calling: local (``shared`` scope), a logged-in user, or this tab's visitor.

    The visitor token is created on the server with ``secrets.token_urlsafe`` the first
    time a tab runs a page, and lives only in that tab's server-side session state.
    """
    if settings.session_scope is SessionScope.SHARED:
        return Owner.LOCAL
    identity = _authenticated_identity()
    if identity is not None:
        return Owner.from_identity(*identity)
    token = st.session_state.get(VISITOR_KEY)
    try:
        return Owner.from_token(token)
    except ValueError:
        token = new_visitor_token()
        st.session_state[VISITOR_KEY] = token
        return Owner.from_token(token)


def current_mode() -> PlannerMode:
    return PlannerMode(st.session_state.get(MODE_KEY, PlannerMode.LANGGRAPH.value))


def current_session_id() -> str | None:
    value = st.session_state.get(SESSION_KEY)
    return str(value) if value else None


def select_session(session_id: str | None) -> None:
    """Open a session. Only in ``shared`` scope is its id also put in the URL (so a
    refresh reopens it); a visitor's refresh starts a new visitor anyway."""
    st.session_state[SESSION_KEY] = session_id
    if session_id and get_settings().session_scope is SessionScope.SHARED:
        st.query_params["session"] = session_id
    elif "session" in st.query_params:
        del st.query_params["session"]


def _restore_from_url(svc: Services) -> None:
    """Keep the open session valid; ``?session=<id>`` opens a session this caller owns.

    A session id that is not the caller's (a link from another visitor, an id from a
    previous tab, a deleted session) is refused and removed from the URL.
    """
    if SESSION_KEY in st.session_state:
        current = st.session_state[SESSION_KEY]
        if current and not svc.sessions.owns(str(current)):
            st.session_state[SESSION_KEY] = None
        return
    wanted = st.query_params.get("session")
    if wanted and svc.sessions.get(wanted) is not None:
        st.session_state[SESSION_KEY] = wanted
        return
    st.session_state[SESSION_KEY] = None
    if wanted:
        st.session_state[REJECTED_KEY] = "session"
        del st.query_params["session"]


def rejected_link_notice(kind: str) -> None:
    """Explain (once) why a ``?session=`` / ``?run=`` link did not open anything."""
    if st.session_state.pop(REJECTED_KEY, None) == kind:
        st.warning(f"That {kind} link is not available here. Sessions and runs are "
                   "private to the browser tab that created them.", icon="🔒")


def deployment_notices(svc: Services) -> None:
    """Mock-mode and storage notices for the main area (home and planner pages)."""
    if svc.settings.llm_provider is LLMProvider.MOCK:
        st.info("**Demo mode: deterministic mock model, not real AI inference.** "
                "Rule-based stand-ins replace the language model so the agent "
                "orchestration can be shown without an API key; the same request always "
                "gives the same plan.", icon="🧪")
    if svc.isolated:
        st.caption("🔒 Your sessions are private to this browser tab: other visitors "
                   "cannot see them, and they are not reopened after a refresh or in a "
                   "new tab. They are kept in a temporary SQLite database; on Streamlit "
                   "Community Cloud it is erased whenever the app restarts, sleeps or is "
                   "redeployed.")


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
        if svc.isolated:
            st.caption("🔒 Sessions are private to this browser tab and temporary.")
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
