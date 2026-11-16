"""Phase 5 UI tests: the real Streamlit pages, driven by ``AppTest`` (offline, mock LLM).

Each test gets its own SQLite files (``conftest._isolated_storage``); the app's cached
services are keyed by those paths, so tests never share planners or data. Planner runs
happen in the app's background ``RunManager`` thread, exactly as in the browser.
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
    if services is not None:
        for session in services.sessions.list_sessions():
            handle = services.runs.active(session.id)
            if handle is not None:
                handle.wait(TIMEOUT)
        services.hub.close()
    st.cache_resource.clear()


def _services_or_none() -> Any:
    if "ui_shared" not in sys.modules:
        return None
    return sys.modules["ui_shared"].services()


def services() -> Any:
    import ui_shared

    return ui_shared.services()


def open_page(path: str, session_id: str | None = None, mode: str | None = None) -> AppTest:
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
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


def test_ac52_restart_shows_the_chat_and_latest_itinerary() -> None:
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
    sessions = open_page(SESSIONS, session_id=session_id)
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
    page = open_page(SESSIONS, session_id=session_id)
    page.text_input(key=f"rename-{session_id}").set_value("Winter trip").run()
    page.button(key=f"save-{session_id}").click().run()
    assert services().sessions.get(session_id).title == "Winter trip"
    page.button(key=f"del-{session_id}").click().run()
    assert services().sessions.get(session_id) is None
    assert services().sessions.versions(session_id) == []
    assert page.session_state["session_id"] is None


Clicker = Callable[[AppTest], AppTest]
