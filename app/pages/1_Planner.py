"""Planner page: conversation, live progress, itinerary, review outcome.

A message starts a run in ``RunManager`` (a background thread). While it runs, a
fragment polls its progress once a second; when it ends, the page reruns once to show
the result. Nothing here plans anything itself: classification, planning and
persistence happen in the services.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st
from ui_itinerary import render_itinerary
from ui_shared import Services, current_mode, current_session_id, select_session, services, sidebar

from dubai_planner.models import ChatRole, PlanStatus
from dubai_planner.services import presentation as view
from dubai_planner.services.runs import RunHandle, RunInProgress

NODE_LABELS = {
    "parse_request": "Understanding your request", "ask_clarification": "Asking a question",
    "research_attractions": "Researching attractions", "research_accommodation":
    "Finding hotels", "research_dining": "Choosing restaurants", "research_transport":
    "Planning transport", "research_weather_events": "Checking weather & events",
    "merge_research": "Combining research", "allocate_budget": "Allocating the budget",
    "compose_itinerary": "Composing the itinerary", "validate": "Checking the rules",
    "review": "Reviewing the plan", "review_itinerary": "Reviewing the plan",
    "refine_itinerary": "Revising the plan", "finalize": "Saving the plan",
    "__end__": "Done",
}

st.set_page_config(page_title="Planner · Dubai AI Travel Planner", page_icon="🗺️",
                   layout="wide")
svc = services()
sidebar(svc)


def progress_lines(handle: RunHandle) -> list[str]:
    lines = []
    for u in handle.snapshot():
        label = NODE_LABELS.get(u.node, u.node.replace("_", " ").capitalize())
        loop = f" (revision {u.iteration})" if u.iteration else ""
        lines.append(f"✓ {label}{loop}")
    return lines


def live_progress(svc: Services, session_id: str) -> None:
    """Polls the active run; reruns the page once when it finishes (no loop)."""
    handle = svc.runs.active(session_id)
    if handle is None:
        st.rerun()  # the run has just finished: show its result
        return
    elapsed = (datetime.now(UTC) - handle.started_at).total_seconds()
    with st.status(f"Planning… {elapsed:.0f} s ({handle.mode.value})", expanded=True):
        st.caption(handle.refinement.explanation)
        for line in progress_lines(handle):
            st.write(line)
        st.caption("You can switch pages; the plan keeps running and is saved.")


def outcome(handle: RunHandle) -> None:
    """The finished run: error, question, or review outcome with all attempts."""
    if handle.user_error:
        st.error(handle.user_error, icon="🚫")
        st.caption("Nothing was lost; send the request again when the problem is fixed.")
        return
    result = handle.result
    if result is None:
        return
    if result.status is PlanStatus.NEEDS_CLARIFICATION:
        st.warning(f"**Question:** {result.clarification_question}", icon="❓")
        return
    if result.status is PlanStatus.CANCELLED:
        st.warning("The run was interrupted before it finished.")
        return
    kind = st.success if result.approved else st.warning
    kind(f"{view.status_text(result)} after {result.iterations} revision(s) · "
         f"{handle.refinement.explanation}")
    with st.expander(f"Review: {len(result.attempts)} attempt(s), "
                     f"{len(result.blocking_issues)} unresolved issue(s)",
                     expanded=not result.approved):
        if result.review is not None:
            st.write(f"Reviewer score: **{result.review.score:.2f}** · {result.review.summary}")
            for issue in result.review.issues:
                where = f"day {issue.day_index + 1}: " if issue.day_index is not None else ""
                icon = {"error": "🔴", "warning": "🟠"}.get(issue.severity.value, "🔵")
                fix = f" — *{issue.suggested_fix}*" if issue.suggested_fix else ""
                st.markdown(f"{icon} {where}{issue.message}{fix}")
        if result.attempts:
            st.dataframe(pd.DataFrame(view.attempt_rows(result.attempts)), hide_index=True,
                         use_container_width=True)
        if result.assumptions:
            st.caption("Assumptions: " + "; ".join(result.assumptions))


session_id = current_session_id()
session = svc.sessions.get(session_id) if session_id else None
st.title("🗺️ Plan your Dubai trip")
if session is not None:
    st.caption(f"Session **{session.title}** · started in {session.mode.value} · next "
               f"message uses **{current_mode().value}**")

if session_id is not None:
    for message in svc.sessions.history(session_id):
        role = "user" if message.role is ChatRole.USER else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

active = svc.runs.active(session_id) if session_id else None
if active is not None and session_id is not None:
    st.fragment(run_every=1.0)(live_progress)(svc, session_id)
elif session_id is not None:
    last = svc.runs.last(session_id)
    if last is not None:
        outcome(last)

if session_id is not None and active is None:
    latest = svc.sessions.latest(session_id)
    if latest is not None:
        render_itinerary(latest, svc, key=f"planner-{latest.version}")
elif session_id is None:
    st.info("Type your trip below to start a new session, e.g. *2 adults, 10-12 Feb 2027, "
            "AED 6000, history and food*.")

prompt = st.chat_input("Describe your trip, answer a question, or ask for a change…",
                       disabled=active is not None, max_chars=4000)
if prompt:
    if session_id is None:
        session_id = svc.sessions.create(current_mode()).id
        select_session(session_id)
    try:
        svc.runs.start(session_id, current_mode(), prompt)
    except RunInProgress as exc:
        st.warning(str(exc))
    except ValueError as exc:
        st.warning(str(exc))
    else:
        st.rerun()
