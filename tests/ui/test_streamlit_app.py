"""Phase 5 UI tests: the real Streamlit pages, driven by ``AppTest`` (offline, mock LLM).

Each test gets its own SQLite files (``conftest._isolated_storage``); the app's cached
services are keyed by those paths, so tests never share planners or data. Planner runs
happen in the app's background ``RunManager`` thread, exactly as in the browser.

``SESSION_SCOPE=visitor`` (the default) gives every ``AppTest`` its own visitor, like a
separate browser tab. Pages that the *same* tab opens get its visitor token with
``open_page(..., visitor=at)`` (switching pages keeps ``st.session_state`` in a browser).
Assertions on stored data use the unscoped ``process_services()``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dubai_planner.config import get_settings
from dubai_planner.models import PlanStatus

APP = Path(__file__).resolve().parents[2] / "app"
HOME, PLANNER, SESSIONS = (str(APP / "streamlit_app.py"), str(APP / "pages" / "1_Planner.py"),
                           str(APP / "pages" / "2_Sessions.py"))
WINTER = "2 adults, 10-12 Feb 2027, AED 6000, history and food"
TIMEOUT = 120


@pytest.fixture(autouse=True)
def _offline_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HTTP_OFFLINE", "true")
    monkeypatch.syspath_prepend(str(APP))
    get_settings.cache_clear()
    yield
    services = _services_or_none()
    if services is not None:  # unscoped: every visitor's sessions
        for session in services.sessions.list_sessions():
            handle = services.runs.active(session.id)
            if handle is not None:
                handle.wait(TIMEOUT)
        services.hub.close()
    st.cache_resource.clear()


def _services_or_none() -> Any:
    if "ui_shared" not in sys.modules:
        return None
    return sys.modules["ui_shared"].process_services()


def services() -> Any:
    """The app's process-wide services, *not* scoped to a visitor (test assertions)."""
    import ui_shared

    return ui_shared.process_services()


VISITOR_KEY = "_visitor_token"


def token_of(tab: AppTest | str) -> str:
    return tab if isinstance(tab, str) else str(tab.session_state[VISITOR_KEY])


def open_page(path: str, session_id: str | None = None, mode: str | None = None, *,
              visitor: AppTest | str | None = None) -> AppTest:
    """Run a page; ``visitor`` = the tab (or token) it belongs to (default: a new tab)."""
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    if visitor is not None:
        at.session_state[VISITOR_KEY] = token_of(visitor)
    if session_id is not None:
        at.session_state["session_id"] = session_id
    if mode is not None:
        at.session_state["mode"] = mode
    return at.run()


def send(at: AppTest, text: str) -> Any:
    """Type into the chat box, wait for the background run, then render the result."""
    at.chat_input[0].set_value(text).run()
    session_id = at.session_state["session_id"]
    handle = services().runs.last(session_id)
    assert handle is not None, "no run was started"
    assert handle.wait(TIMEOUT), "the planner run did not finish"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return handle


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.success] + [e.value for e in at.warning]
    parts += [e.value for e in at.error] + [e.value for e in at.info]
    return "\n".join(str(p) for p in parts)


# ------------------------------------------------------------------ AC5.1


def test_home_page_renders_with_sidebar() -> None:
    at = open_page(HOME)
    assert not at.exception
    assert "Dubai AI Travel Planner" in at.title[0].value
    labels = at.sidebar.selectbox(key="mode").options
    assert len(labels) == 3
    assert any("mock" in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize("mode", ["langgraph", "crewai_sequential", "crewai_hierarchical"])
def test_ac51_plans_a_trip_from_the_ui(mode: str) -> None:
    at = open_page(PLANNER, mode=mode)
    assert not at.exception and not at.chat_input[0].disabled
    handle = send(at, WINTER)
    assert handle.result is not None and handle.result.status is PlanStatus.APPROVED
    assert handle.mode.value == mode
    assert any("Approved" in s.value for s in at.success)
    tabs = [t.label for t in at.tabs]
    assert tabs[:2] == ["Overview", "Day 1 · Wed 10 Feb"] and "Map" in tabs
    assert [m.avatar for m in at.chat_message] == ["user", "assistant"]
    record = services().sessions.latest(at.session_state["session_id"])
    assert record is not None and record.mode.value == mode and record.approved


def test_mode_selector_drives_the_next_run() -> None:
    at = open_page(PLANNER)
    at.sidebar.selectbox(key="mode").set_value("crewai_sequential").run()
    handle = send(at, WINTER)
    assert handle.mode.value == "crewai_sequential"


# ------------------------------------------------------------------ AC5.2


def test_ac52_restart_shows_the_chat_and_latest_itinerary(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Local single-user mode (``SESSION_SCOPE=shared``): a refresh reopens the session
    from ``?session=<id>`` and the picker lists every session. The visitor-scope
    equivalent is ``test_visitor_session_survives_an_app_restart_in_the_same_tab``."""
    monkeypatch.setenv("SESSION_SCOPE", "shared")
    get_settings.cache_clear()
    at = open_page(PLANNER)
    send(at, WINTER)
    session_id = at.session_state["session_id"]
    services().hub.close()
    st.cache_resource.clear()  # app restart: new services, planners and caches
    fresh = AppTest.from_file(PLANNER, default_timeout=TIMEOUT)
    fresh.query_params["session"] = session_id  # reopened from the URL…
    fresh.run()
    assert fresh.session_state["session_id"] == session_id
    assert [m.avatar for m in fresh.chat_message] == ["user", "assistant"]
    assert "Overview" in [t.label for t in fresh.tabs]
    picker = AppTest.from_file(PLANNER, default_timeout=TIMEOUT).run()  # …or the picker
    assert picker.session_state["session_id"] is None
    picker.sidebar.selectbox[1].set_value(session_id).run()
    assert [m.avatar for m in picker.chat_message] == ["user", "assistant"]


# ------------------------------------------------------------------ AC5.3 / AC5.4


def test_ac53_ac54_refinements_and_version_diff_in_the_ui() -> None:
    at = open_page(PLANNER)
    send(at, WINTER)
    session_id = at.session_state["session_id"]
    cut = send(at, "reduce the budget by 20%")
    assert cut.refinement.kind.value == "request_delta"
    v2 = services().sessions.latest(session_id)
    assert v2.version == 2 and v2.request.total_budget_aed() == Decimal("4800")
    assert v2.itinerary.cost.total <= Decimal("6000") * Decimal("0.8") * Decimal("1.05")
    edit = send(at, "Replace day 2 with indoor activities")
    assert edit.refinement.kind.value == "local_edit"
    v3 = services().sessions.latest(session_id)
    assert v3.version == 3
    assert v3.itinerary.days[0] == v2.itinerary.days[0]
    assert v3.itinerary.days[2] == v2.itinerary.days[2]
    assert v3.itinerary.days[1] != v2.itinerary.days[1]
    sessions = open_page(SESSIONS, session_id=session_id, visitor=at)
    assert not sessions.exception
    versions = sessions.dataframe[1].value
    assert list(versions["Version"]) == [1, 2, 3]
    metric = sessions.metric[0]
    assert "v2 → v3" in metric.label and "AED" in metric.delta
    page = texts(sessions)
    assert "Unchanged days: 1, 3" in page and "**Day 2**" in page
    assert ":green[+ " in page and ":red[− " in page
    sessions.selectbox(key=f"from-{session_id}").set_value(1).run()
    assert "v1 → v3" in sessions.metric[0].label


# ------------------------------------------------------------------ AC5.5


def test_ac55_provider_error_is_friendly_and_the_app_stays_usable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "   ")  # set but blank: counts as missing
    get_settings.cache_clear()
    at = open_page(PLANNER)
    handle = send(at, WINTER)
    assert handle.user_error is not None
    [error] = at.error
    assert "not configured" in error.value and "GROQ_API_KEY" in error.value
    assert "Traceback" not in texts(at) and not at.exception
    assert not at.chat_input[0].disabled  # usable: the next message can be sent


def test_api_key_never_appears_in_the_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    canary = "gsk_CANARY_ui_never_shows_this_123456"
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", canary)
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    get_settings.cache_clear()
    at = open_page(HOME)
    assert canary not in texts(at)
    assert all(canary not in str(c.value) for c in at.sidebar.caption)


def test_clarification_question_is_shown() -> None:
    at = open_page(PLANNER)
    handle = send(at, "plan a Dubai trip")
    assert handle.result is not None
    assert handle.result.status is PlanStatus.NEEDS_CLARIFICATION
    assert any("Question" in w.value for w in at.warning)
    answer = send(at, "5 days in Feb, AED 8000")
    assert answer.refinement.kind.value == "clarification_answer"
    assert answer.result is not None and answer.result.status is PlanStatus.APPROVED


# ------------------------------------------------------------------ AC5.6


def test_ac56_every_price_is_an_estimate_with_its_source() -> None:
    at = open_page(PLANNER)
    send(at, WINTER)
    priced = [m for m in at.markdown if "AED" in m.value and "(estimate)" in m.value]
    assert priced, "no prices shown"
    for m in at.markdown:
        if "≈ AED" in m.value:
            assert "estimate" in m.value
    with_help = [m for m in at.markdown if m.help and "Source:" in m.help]
    assert with_help and all("as of" in m.help for m in with_help
                             if "estimated" in m.help)
    assert any("estimates from curated public data" in i.value for i in at.info)
    assert any("OpenStreetMap" in c.value for c in at.caption)  # attribution kept


# ------------------------------------------------------------------ rerun safety


def test_reruns_never_start_a_duplicate_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from dubai_planner.services.runs import RunInProgress, RunManager

    gate = threading.Event()
    original = RunManager._run

    def slow(self: RunManager, handle: Any) -> None:
        gate.wait(30)
        original(self, handle)

    monkeypatch.setattr(RunManager, "_run", slow)
    at = open_page(PLANNER)
    at.chat_input[0].set_value(WINTER).run()
    session_id = at.session_state["session_id"]
    runs = services().runs
    first = runs.active(session_id)
    assert first is not None
    for _ in range(3):  # page reruns while the run is active start nothing new
        at.run()
    assert at.chat_input[0].disabled
    assert runs.active(session_id) is first
    with pytest.raises(RunInProgress):
        runs.start(session_id, first.mode, "another request")
    gate.set()
    assert first.wait(TIMEOUT)
    at.run()
    assert runs.last(session_id) is first and not at.chat_input[0].disabled
    assert len(services().sessions.versions(session_id)) == 1


def test_sessions_page_rename_and_delete() -> None:
    at = open_page(PLANNER)
    send(at, WINTER)
    session_id = at.session_state["session_id"]
    page = open_page(SESSIONS, session_id=session_id, visitor=at)
    page.text_input(key=f"rename-{session_id}").set_value("Winter trip").run()
    page.button(key=f"save-{session_id}").click().run()
    assert services().sessions.get(session_id).title == "Winter trip"
    page.button(key=f"del-{session_id}").click().run()
    assert services().sessions.get(session_id) is None
    assert services().sessions.versions(session_id) == []
    assert page.session_state["session_id"] is None


Clicker = Callable[[AppTest], AppTest]


# ================================================================== Phase 6: Execution page

EXECUTION = str(APP / "pages" / "3_Execution.py")


def graph_spec_text(at: AppTest) -> str:
    charts = at.get("graphviz_chart")
    assert charts, "no graph rendered"
    return "\n".join(c.proto.spec for c in charts)


def all_text(at: AppTest) -> str:
    parts = [texts(at)] + [str(m.value) for m in at.metric] + [str(j.value) for j in at.json]
    parts += [str(c.value) for c in at.code] + [str(d.value) for d in at.dataframe]
    return "\n".join(parts)


def test_execution_page_without_runs_explains_what_to_do() -> None:
    at = open_page(EXECUTION)
    assert not at.exception
    assert any("No runs yet" in i.value for i in at.info)


def test_ac61_ac64_execution_page_graph_timeline_and_inspector() -> None:
    at = open_page(PLANNER)
    send(at, WINTER)
    session_id = at.session_state["session_id"]
    page = open_page(EXECUTION, session_id=session_id, visitor=at)
    assert not page.exception, [e.value for e in page.exception]
    dot = graph_spec_text(page)
    assert "parallel research (one super-step)" in dot and "● done" in dot
    assert "◐ running" not in dot and "✖ failed" not in dot
    assert len(page.get("plotly_chart")) >= 1  # the Gantt timeline (+ comparison charts)
    metrics = {m.label: m.value for m in page.metric}
    assert metrics["Outcome"] == "approved" and metrics["Research"].startswith("parallel ×")
    assert "Mock provider" in texts(page)
    # Inspector: pick the compose step's end event; its output re-validates.
    run_id = page.session_state["exec_run"]
    trace = ob_trace(run_id)
    compose = next(i for i, e in enumerate(trace.events) if e.agent == "compose_itinerary"
                   and e.event.value == "end" and "output_schema" in e.payload)
    page.selectbox(key=f"insp-{run_id}-event").set_value(compose).run()
    assert any(s.value == "✓ validates as `Itinerary`" for s in page.success)
    page.radio(key=f"insp-{run_id}-filter").set_value("LLM calls").run()
    assert not page.exception and "Prompt" in texts(page)


def ob_trace(run_id: str) -> Any:
    from dubai_planner.services import observability as ob

    trace = ob.load_trace(services().sessions.repo, run_id)
    assert trace is not None
    return trace


def test_ac63_live_graph_highlights_the_running_step() -> None:
    import threading

    from dubai_planner.agents import composer
    from dubai_planner.agents.suite import AgentSuite
    from dubai_planner.models import PlannerMode

    entered, release = threading.Event(), threading.Event()

    def blocking(data: Any, llm: Any, tools: Any) -> Any:
        entered.set()
        release.wait(60)
        return composer.run(data, llm, tools)

    from dubai_planner.services.ownership import Owner

    home = open_page(HOME)
    svc = services()
    svc.hub.planner_kwargs["agents"] = AgentSuite(composer=blocking)  # before any planner
    tab = Owner.from_token(token_of(home))
    session_id = svc.sessions.scoped(tab).create(PlannerMode.LANGGRAPH).id
    handle = svc.runs.start(session_id, PlannerMode.LANGGRAPH, WINTER)
    try:
        assert entered.wait(60)
        page = open_page(EXECUTION, session_id=session_id, visitor=home)
        assert not page.exception, [e.value for e in page.exception]
        dot = graph_spec_text(page)
        compose_line = next(line for line in dot.splitlines()
                            if line.strip().startswith('"compose_itinerary" [label='))
        assert "◐ running" in compose_line and "penwidth=3" in compose_line
        assert "now running: **compose_itinerary**" in texts(page)
        assert any("refreshes every second" in c.value for c in page.caption)
        page.run()  # a later poll: still the same state, no duplicate run started
        assert svc.runs.active(session_id) is handle
    finally:
        release.set()
    assert handle.wait(TIMEOUT)
    page.run()
    assert not page.exception
    dot = graph_spec_text(page)
    assert "◐ running" not in dot and r'"finalize" [label="finalize\n● done"' in dot


def test_ac65_execution_history_survives_an_app_restart() -> None:
    at = open_page(PLANNER)
    send(at, WINTER)
    session_id = at.session_state["session_id"]
    services().hub.close()
    st.cache_resource.clear()  # app restart: nothing in memory, SQLite only
    page = open_page(EXECUTION, session_id=session_id, visitor=at)
    assert not page.exception
    assert "● done" in graph_spec_text(page) and page.get("plotly_chart")
    assert {m.label: m.value for m in page.metric}["Outcome"] == "approved"
    planner = open_page(PLANNER, session_id=session_id, visitor=at)  # review detail kept
    assert any("from the saved trace" in e.label for e in planner.expander)


def test_execution_page_compares_langgraph_and_crewai_on_the_same_request() -> None:
    lg = open_page(PLANNER, mode="langgraph")
    send(lg, WINTER)
    cr = open_page(PLANNER, mode="crewai_sequential", visitor=lg)  # same tab, new session
    send(cr, WINTER)
    page = open_page(EXECUTION, session_id=cr.session_state["session_id"], visitor=lg)
    assert not page.exception
    assert any("Same message, request, seed data" in s.value for s in page.success)
    table = next(d.value for d in page.dataframe if "B − A" in d.value.columns)
    metrics = set(table["metric"])
    assert {"Wall-clock time (s)", "LLM calls", "Total tokens", "Research execution",
            "Review iterations (refinements)"} <= metrics
    research = table.set_index("metric").loc["Research execution"]
    assert research["A"].startswith("parallel") and research["B"].startswith("parallel")


def test_execution_page_never_shows_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    canary = "gsk_CANARY_execution_page_never_shows_this_77"
    monkeypatch.setenv("GROQ_API_KEY", canary)  # configured, but the mock provider runs
    get_settings.cache_clear()
    at = open_page(PLANNER)
    send(at, WINTER)
    page = open_page(EXECUTION, session_id=at.session_state["session_id"], visitor=at)
    assert canary not in all_text(page) and canary not in graph_spec_text(page)
    [download] = page.get("download_button")
    assert canary not in str(download.proto)


# ------------------------------------------------------------------ Phase 7


def test_evaluation_page_shows_published_results_with_mock_label() -> None:
    """The dashboard renders the committed benchmark (eval/results/) with no eval DB."""
    at = open_page(str(APP / "pages" / "4_Evaluation.py"))
    assert not at.exception, [e.value for e in at.exception]
    assert any("LangGraph vs CrewAI" in t.value for t in at.title)
    assert at.selectbox and at.selectbox[0].options, "no evaluation results listed"
    options = at.selectbox[0].options
    mock = next(o for o in options if "mock" in o.lower())
    at.selectbox[0].set_value(mock).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Mock provider" in w.value for w in at.warning)
    assert at.dataframe, "metric table missing"
    real = next((o for o in options if "groq" in o.lower()), None)
    if real is not None:  # a real-provider result is labelled as such, without the banner
        at.selectbox[0].set_value(real).run()
        assert not at.exception, [e.value for e in at.exception]
        assert not any("Mock provider" in w.value for w in at.warning)
        assert any("Real provider" in i.value for i in at.info)
    labels = [t.label for t in at.tabs]
    assert labels == ["Summary", "Distributions", "Per scenario", "Fault injection & judge"]


# ================================================================== visitor isolation


def sidebar_texts(at: AppTest) -> str:
    return "\n".join(str(c.value) for c in at.sidebar.caption)


def test_two_visitors_never_see_each_others_sessions_or_runs() -> None:
    alice = open_page(PLANNER)
    send(alice, WINTER)
    a_sid = alice.session_state["session_id"]
    bob = open_page(PLANNER)  # another browser tab: another visitor
    assert token_of(bob) != token_of(alice)
    assert bob.session_state["session_id"] is None
    assert "No saved sessions yet." in sidebar_texts(bob)
    assert any("No sessions yet" in i.value for i in open_page(SESSIONS, visitor=bob).info)
    assert any("No runs yet" in i.value for i in open_page(EXECUTION, visitor=bob).info)
    send(bob, "1 adult, 3-5 Mar 2027, AED 3000, beaches")
    b_sid = bob.session_state["session_id"]
    assert b_sid != a_sid
    for tab, own in ((alice, a_sid), (bob, b_sid)):
        page = open_page(SESSIONS, visitor=tab)
        picker = page.selectbox(key="sessions_pick")
        assert picker.value == own and len(picker.options) == 1
        execution = open_page(EXECUTION, visitor=tab)
        assert execution.radio(key="exec_scope").value == "All my sessions"
        run_box = execution.selectbox(key="exec_run")
        own_runs = services().sessions.run_records(own)
        assert len(run_box.options) == len(own_runs) == 1
        assert run_box.value == own_runs[0].id
    assert {s.id for s in services().sessions.list_sessions()} == {a_sid, b_sid}


def test_direct_links_to_another_visitors_session_or_run_are_rejected() -> None:
    alice = open_page(PLANNER)
    send(alice, WINTER)
    a_sid = alice.session_state["session_id"]
    [a_run] = [r.id for r in services().sessions.run_records(a_sid)]
    bob = AppTest.from_file(PLANNER, default_timeout=TIMEOUT)
    bob.query_params["session"] = a_sid  # a copied link
    bob.run()
    assert not bob.exception and bob.session_state["session_id"] is None
    assert not bob.chat_message and "Overview" not in [t.label for t in bob.tabs]
    assert any("not available here" in w.value for w in bob.warning)
    assert "session" not in bob.query_params
    tampered = open_page(PLANNER, session_id=a_sid)  # a forged UI state is reset as well
    assert tampered.session_state["session_id"] is None and not tampered.chat_message
    page = AppTest.from_file(EXECUTION, default_timeout=TIMEOUT)
    page.query_params["run"] = a_run
    page.run()
    assert not page.exception
    assert any("not available here" in w.value for w in page.warning)
    assert any("No runs yet" in i.value for i in page.info)
    assert WINTER not in all_text(page)
    assert services().sessions.get(a_sid) is not None  # nothing was changed


def test_the_visitor_token_stays_on_the_server() -> None:
    at = open_page(PLANNER)
    send(at, WINTER)
    token = token_of(at)
    assert len(token) >= 43
    assert all(token not in str(v) for v in at.query_params.values())
    assert "session" not in at.query_params  # visitor scope: no ids in the URL either
    assert token not in all_text(at) and token not in sidebar_texts(at)
    assert token.encode() not in Path(get_settings().db_path).read_bytes()


def test_visitor_session_survives_an_app_restart_in_the_same_tab() -> None:
    """Ownership is stored in SQLite: after the services restart, the same tab (same
    server-side token) still sees its session; a new tab or a refresh does not."""
    at = open_page(PLANNER)
    send(at, WINTER)
    session_id = at.session_state["session_id"]
    services().hub.close()
    st.cache_resource.clear()
    at.run()
    assert not at.exception
    assert [m.avatar for m in at.chat_message] == ["user", "assistant"]
    assert "Overview" in [t.label for t in at.tabs]
    picker = at.sidebar.selectbox[1]
    assert picker.value == session_id and len(picker.options) == 1
    refreshed = AppTest.from_file(PLANNER, default_timeout=TIMEOUT)
    refreshed.query_params["session"] = session_id
    refreshed.run()
    assert refreshed.session_state["session_id"] is None and not refreshed.chat_message


def test_mock_demo_and_temporary_storage_notices() -> None:
    for path in (HOME, PLANNER):
        at = open_page(path)
        assert any("deterministic mock model, not real AI inference" in i.value
                   for i in at.info), path
        assert any("temporary SQLite database" in c.value
                   and "Streamlit Community Cloud" in c.value for c in at.caption), path
    assert "survive restarts" not in texts(open_page(HOME))


def test_logged_in_users_keep_their_sessions_across_tabs(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``st.login`` configured, the owner comes from the identity (iss, sub), so a
    new tab of the same user sees the session and another user does not."""
    import ui_shared

    def login(subject: str) -> None:
        monkeypatch.setattr(ui_shared, "_authenticated_identity",
                            lambda: ("https://idp.example", subject))

    login("user-1")
    at = open_page(PLANNER)
    send(at, WINTER)
    session_id = at.session_state["session_id"]
    picker = open_page(SESSIONS).selectbox(key="sessions_pick")  # a new tab, same user
    assert picker.value == session_id and len(picker.options) == 1
    login("user-2")
    assert any("No sessions yet" in i.value for i in open_page(SESSIONS).info)
