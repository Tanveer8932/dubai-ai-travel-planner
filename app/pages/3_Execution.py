"""Execution page: how the agents ran — graph, timeline, inspector, state, comparison.

Everything is read from SQLite (``runs`` + ``trace_events``), so finished runs can be
viewed after a restart. While the selected run is executing (in this app's
``RunManager``), a fragment re-reads its trace once a second: the recorder writes at
every node/task transition, so the graph highlights each step as it starts and ends.
When the run finishes, the page reruns once (no loop). Nothing here starts a run.
"""

from __future__ import annotations

import streamlit as st
from ui_execution import (
    export_trace,
    render_comparison,
    render_graph,
    render_inspector,
    render_state,
    render_summary,
    render_timeline,
    run_label,
)
from ui_shared import Services, current_session_id, nav_link, services, sidebar

from dubai_planner.models import RunRecord
from dubai_planner.services import observability as ob

RUN_KEY, SCOPE_KEY = "exec_run", "exec_scope"

st.set_page_config(page_title="Execution · Dubai AI Travel Planner", page_icon="🔭",
                   layout="wide")
svc = services()
sidebar(svc)
st.title("🔭 Agent execution")


def is_active(svc: Services, run: RunRecord) -> bool:
    """Running *in this process* now (a ``running`` row left by a killed app is not)."""
    if run.session_id is None:
        return False
    handle = svc.runs.active(run.session_id)
    return handle is not None and run.status.value == "running"


def load(svc: Services, run: RunRecord) -> ob.RunTrace | None:
    return ob.load_trace(svc.sessions.repo, run.id, active=is_active(svc, run))


def live_view(svc: Services, run_id: str) -> None:
    """Polled while the run executes; reruns the whole page once when it ends."""
    run = svc.sessions.repo.get_run(run_id)
    trace = load(svc, run) if run is not None else None
    if trace is None or not trace.live:
        st.rerun()
        return
    render_summary(trace)
    st.subheader("Live graph")
    render_graph(svc, trace)
    st.subheader("Timeline so far")
    render_timeline(trace, key=f"live-tl-{len(trace.events)}", live=True)


session_id = current_session_id()
scopes = ["This session", "All sessions"]
linked = st.query_params.get("run")  # ?run=<id>: open that run (links, screenshots)
if linked and RUN_KEY not in st.session_state:
    st.session_state[RUN_KEY] = linked
    st.session_state.setdefault(SCOPE_KEY, scopes[1])
scope = st.radio("Runs from", scopes, horizontal=True, key=SCOPE_KEY,
                 index=0 if session_id else 1)
runs = ob.list_run_traces(svc.sessions.repo,
                          session_id if scope == scopes[0] and session_id else None)
if not runs:
    st.info("No runs yet. Plan a trip on the Planner page, then come back here to see how "
            "the agents executed.")
    nav_link("pages/1_Planner.py", "Go to the Planner", "🗺️")
    st.stop()

by_id = {r.id: r for r in runs}
active_ids = [r.id for r in runs if is_active(svc, r)]
if st.session_state.get(RUN_KEY) not in by_id:
    st.session_state[RUN_KEY] = active_ids[0] if active_ids else runs[0].id
run_id = st.selectbox("Run", list(by_id), key=RUN_KEY,
                      format_func=lambda i: run_label(by_id[i])
                      + ("  ◐ live" if i in active_ids else ""))
run = by_id[run_id]
trace = load(svc, run)
if trace is None:  # pragma: no cover - deleted between listing and loading
    st.warning("This run no longer exists.")
    st.stop()

if trace.live:
    st.caption("◐ This run is executing: the view refreshes every second.")
    st.fragment(run_every=1.0)(live_view)(svc, run.id)
    st.stop()

if trace.message:
    st.caption(f"Request: *{trace.message[:300]}*")
render_summary(trace)
if not trace.events:
    st.info(ob.NO_TRACE + " Runs made before trace persistence (Phase 6) only have their "
            "run record.")

TABS = ["Graph", "Timeline", "Inspector", "State & review", "Compare runs"]
view = st.query_params.get("view")  # ?view=Timeline opens that tab
graph_tab, timeline_tab, inspect_tab, state_tab, compare_tab = st.tabs(
    TABS, default=view if view in TABS else None)
with graph_tab:
    render_graph(svc, trace)
with timeline_tab:
    render_timeline(trace, key=f"tl-{run.id}")
with inspect_tab:
    render_inspector(trace, key=f"insp-{run.id}")
with state_tab:
    render_state(trace)
    st.download_button("Download trace (JSON)", export_trace(trace),
                       file_name=f"trace-{run.id[:12]}.json", mime="application/json")
with compare_tab:
    every = ob.list_run_traces(svc.sessions.repo)
    labels = {r.id: run_label(r) for r in every}
    others = [r for r in every if r.id != run.id]
    messages = ob.messages_by_run(svc.sessions.repo, every)
    same_msg = [r for r in others if r.mode is not run.mode and trace.message
                and messages.get(r.id) == trace.message]
    default_b = (same_msg or [r for r in others if r.mode is not run.mode] or others or [run])[0]
    st.caption("A is the run selected above. Pick B: ideally the same request in another "
               "mode (plan the same message in a new session with the other mode).")
    b_id = st.selectbox("Run B", list(labels), index=list(labels).index(default_b.id),
                        format_func=labels.__getitem__, key=f"cmp-b-{run.id}")
    other = ob.load_trace(svc.sessions.repo, b_id, active=False)
    if other is not None:
        render_comparison(svc, trace, other)
