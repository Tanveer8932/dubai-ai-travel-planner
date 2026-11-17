"""Execution-view widgets: orchestration graph, Gantt timeline, inspector, comparison.

Rendering only. Every number comes from ``services.observability`` (which reads the
``runs`` row and ``trace_events`` from SQLite); nothing is computed from LLM text and
nothing is estimated. Graph descriptions come from the frameworks themselves:
LangGraph's compiled ``get_graph()`` and CrewAI's ``CrewFactory.describe()``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui_shared import Services

from dubai_planner.models import PlannerMode
from dubai_planner.orchestration.langgraph.graph import describe_graph
from dubai_planner.services import observability as ob

# Categorical slots (validated reference palette) in fixed order: one per review iteration.
ITERATION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
TASK_WINDOW = "#c3c2b7"
STATUS_COLORS = {ob.StepStatus.RUNNING: "#fab219", ob.StepStatus.FAILED: "#d03b3b",
                 ob.StepStatus.STOPPED: "#8a8985", ob.StepStatus.WAITING: "#ec835a"}
MARKER_STYLE = {
    "llm_call": ("circle", "#52514e", 7, "LLM call"),
    "guardrail_rejected": ("x", "#d03b3b", 12, "Guardrail rejected → retry"),
    "guardrail_passed": ("line-ns-open", "#0ca30c", 10, "Guardrail passed"),
    "delegation": ("triangle-right", "#6250d6", 10, "Delegation"),
}


# --------------------------------------------------------------------------- graph


@st.cache_data(show_spinner=False)
def _langgraph_description(_svc_key: str, _svc: Services) -> dict[str, Any]:
    return describe_graph(_svc.hub.planner(PlannerMode.LANGGRAPH).graph)


@st.cache_data(show_spinner=False)
def _crew_description(mode: str, _svc_key: str, _svc: Services) -> dict[str, Any]:
    planner = _svc.hub.planner(PlannerMode(mode))
    return dict(planner.factory.describe())


def graph_spec(svc: Services, mode: PlannerMode) -> ob.GraphSpec:
    """The orchestration picture for ``mode``, read from the framework itself."""
    key = str(svc.settings.db_path)
    if mode is PlannerMode.LANGGRAPH:
        return ob.langgraph_spec(_langgraph_description(key, svc))
    return ob.crewai_spec(_crew_description(mode.value, key, svc))


def render_graph(svc: Services, trace: ob.RunTrace) -> dict[str, ob.StepStatus]:
    spec = graph_spec(svc, trace.mode)
    status = ob.step_status(ob.step_spans(trace), spec.steps)
    st.graphviz_chart(ob.to_dot(spec, status), width="stretch")
    running = [s for s, v in status.items() if v is ob.StepStatus.RUNNING]
    legend = " · ".join(f"{ob.STATUS_ICON[s]} {s.value}" for s in ob.StepStatus)
    st.caption(f"**{spec.title}** · {legend}"
               + (f" · now running: **{', '.join(running)}**" if running else ""))
    with st.expander("Step status table / Mermaid source"):
        st.dataframe(pd.DataFrame([{"step": k, "status": f"{ob.STATUS_ICON[v]} {v.value}"}
                                   for k, v in status.items()]),
                     hide_index=True, width="stretch")
        st.code(ob.to_mermaid(spec, status), language="text")
    return status


# --------------------------------------------------------------------------- timeline


def _bar(rows: list[ob.TimelineRow], name: str, color: str, *, width: float,
         opacity: float = 1.0, legend: bool = True) -> go.Bar:
    return go.Bar(
        orientation="h", y=[r.lane for r in rows], x=[max(r.duration_s, 0.002) for r in rows],
        base=[r.start_s for r in rows], name=name, marker={"color": color, "line": {"width": 0}},
        opacity=opacity, width=width, showlegend=legend,
        customdata=[[r.name, r.iteration, r.start_s, r.end_s, round(r.duration_s, 3),
                     r.status.value, r.tokens, r.detail] for r in rows],
        hovertemplate=("<b>%{customdata[0]}</b> · %{y}<br>%{customdata[2]:.2f}s → "
                       "%{customdata[3]:.2f}s (%{customdata[4]:.3f}s)<br>iteration "
                       "%{customdata[1]} · %{customdata[5]} · %{customdata[7]}<br>"
                       "LLM tokens: %{customdata[6]}<extra></extra>"))


def timeline_figure(trace: ob.RunTrace, *, now: datetime | None = None,
                    height: int | None = None) -> go.Figure:
    """Gantt chart: one lane per node (LangGraph) or role (CrewAI), bars per step span,
    one shaded band per review iteration, markers for LLM calls / guardrails / delegation."""
    rows = ob.timeline_rows(trace, now=now)
    fig = go.Figure()
    lanes: list[str] = []
    for r in rows:
        if r.lane not in lanes:
            lanes.append(r.lane)
    windows = [r for r in rows if r.kind == "crew_task"]
    if windows:
        fig.add_trace(_bar(windows, "CrewAI task window", TASK_WINDOW, width=0.8, opacity=0.6))
    work = [r for r in rows if r.kind != "crew_task"]
    for it in sorted({r.iteration for r in work}):
        done = [r for r in work if r.iteration == it and r.status is ob.StepStatus.DONE]
        if done:
            label = "first draft" if it == 0 else f"revision {it}"
            fig.add_trace(_bar(done, f"steps · {label}",
                               ITERATION_COLORS[it % len(ITERATION_COLORS)],
                               width=0.45 if windows else 0.6))
    for status, color in STATUS_COLORS.items():
        sel = [r for r in rows if r.status is status]
        if sel:
            fig.add_trace(_bar(sel, status.value, color, width=0.45 if windows else 0.6))
    for kind, (symbol, color, size, label) in MARKER_STYLE.items():
        pts = [m for m in ob.markers(trace) if m.kind == kind and m.lane in lanes]
        if pts:
            fig.add_trace(go.Scatter(
                x=[m.at_s for m in pts], y=[m.lane for m in pts], mode="markers", name=label,
                marker={"symbol": symbol, "color": color, "size": size,
                        "line": {"width": 2, "color": color}},
                text=[m.label for m in pts],
                hovertemplate="%{text}<br>at %{x:.2f}s<extra></extra>"))
    for n, band in enumerate(ob.iteration_bands(rows)):
        color = ITERATION_COLORS[band.iteration % len(ITERATION_COLORS)]
        fig.add_vrect(x0=band.start_s, x1=max(band.end_s, band.start_s + 0.001),
                      fillcolor=color, opacity=0.08, line_width=0, layer="below",
                      annotation_text=f"iteration {band.iteration}: {band.label}",
                      annotation_position="top left", annotation_font_size=10,
                      annotation_yshift=-14 * (n % 2))  # narrow bands: no overlap
    fig.update_layout(
        barmode="overlay", height=height or max(260, 34 * len(lanes) + 120),
        margin={"l": 10, "r": 10, "t": 30, "b": 40}, bargap=0.25,
        legend={"orientation": "h", "y": -0.18, "x": 0},
        xaxis={"title": "seconds since the run started", "zeroline": False,
               "showgrid": True, "gridwidth": 1},
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(lanes)),
               "automargin": True},
        hoverlabel={"align": "left"})
    return fig


def render_timeline(trace: ob.RunTrace, *, key: str, live: bool = False) -> None:
    if not trace.events:
        st.info(ob.NO_TRACE)
        return
    now = datetime.now(UTC) if live else None
    st.plotly_chart(timeline_figure(trace, now=now), width="stretch", key=key)
    rows = ob.timeline_rows(trace, now=now)
    par = ob.parallelism(ob.step_spans(trace))
    if par is not None:
        st.caption(f"Research: **{par.label}**: window {par.window_s:.3f} s, sum of step "
                   f"durations {par.sum_s:.3f} s"
                   + (f", speed-up ×{par.speedup}" if par.speedup else "")
                   + " (measured from recorded timestamps).")
    with st.expander("Timeline as a table"):
        st.dataframe(pd.DataFrame([{"lane": r.lane, "step": r.name, "kind": r.kind,
                                    "iteration": r.iteration, "start (s)": r.start_s,
                                    "end (s)": r.end_s, "duration (s)": round(r.duration_s, 3),
                                    "status": r.status.value, "LLM tokens": r.tokens}
                                   for r in rows]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- summary


def fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}{unit}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}{unit}"
    return f"{value}{unit}"


def render_summary(trace: ob.RunTrace) -> ob.RunMetrics:
    m = ob.run_metrics(trace)
    top = st.columns(4)
    top[0].metric("Outcome", m.outcome or m.status)
    top[1].metric("Wall-clock", "running…" if trace.live else fmt(m.duration_s, " s"))
    top[2].metric("LLM calls", fmt(m.llm_calls))
    top[3].metric("Tokens", fmt(m.tokens), help=f"prompt {m.tokens_in:,} · completion "
                  f"{m.tokens_out:,}", delta_color="off")
    bottom = st.columns(4)
    bottom[0].metric("Review iterations", fmt(m.review_iterations))
    bottom[1].metric("Guardrail rejections", fmt(m.guardrail_rejections))
    bottom[2].metric("Research", m.research.short if m.research else "—",
                     help=m.research.label if m.research else "not recorded")
    bottom[3].metric("Time in LLM calls", fmt(m.llm_latency_s, " s"))
    ctx = trace.run.context
    st.caption(f"Run `{trace.run.id[:12]}` · {trace.mode.value} · started "
               f"{trace.run.started_at:%Y-%m-%d %H:%M:%S} UTC · {len(trace.events)} trace "
               f"events · provider **{ctx.get('provider', 'not recorded')}** · model "
               f"`{ctx.get('model', 'not recorded')}` · seed data "
               f"`{ctx.get('data', 'not recorded')}`")
    if ctx.get("provider") == "mock":
        st.caption("Mock provider: these runs prove the orchestration only. Token counts "
                   "are approximations from prompt length and latencies are not real-model "
                   "measurements.")
    if trace.run.error:
        st.error(f"Run error: {trace.run.error}")
    return m


# --------------------------------------------------------------------------- inspector

EVENT_FILTERS = {
    "Steps (nodes / tasks)": lambda e: e.payload.get("kind") in ob.STEP_KINDS,
    "LLM calls": lambda e: e.event.value == "llm_call",
    "Guardrails & delegation": lambda e: e.event.value in ("guardrail", "delegation"),
    "State updates": lambda e: e.payload.get("kind") == "state_update",
    "All events": lambda e: True,
}


def render_inspector(trace: ob.RunTrace, *, key: str) -> None:
    if not trace.events:
        st.info(ob.NO_TRACE)
        return
    kind = st.radio("Show", list(EVENT_FILTERS), horizontal=True, key=f"{key}-filter")
    origin = trace.origin
    chosen = [(i, e) for i, e in enumerate(trace.events) if EVENT_FILTERS[kind](e)]
    if not chosen:
        st.info("No events of this kind in this run.")
        return
    labels = {i: ob.event_label(i, e, origin) for i, e in chosen}
    default = next((n for n, (_, e) in enumerate(chosen) if "output_schema" in e.payload), 0)
    index = st.selectbox("Event", [i for i, _ in chosen], index=default,
                         format_func=labels.__getitem__, key=f"{key}-event")
    view = ob.inspect_event(trace.events[int(index)], origin)
    st.dataframe(pd.DataFrame([{"field": k, "value": fmt(v)} for k, v in view.header.items()]),
                 hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**Inputs (summary)**")
        if view.inputs:
            st.json(view.inputs, expanded=1)
        else:
            st.caption("No inputs recorded for this event.")
    with right:
        st.markdown("**Output**")
        out = view.output
        if out is None:
            st.caption("This event carries no output model.")
        else:
            if out.valid:
                st.success(f"✓ validates as `{out.schema}`")
            elif out.valid is False:
                st.error(f"✗ does not validate as `{out.schema}`: {out.error}")
            else:
                st.warning(f"`{out.schema}`: {out.error}")
            st.json(out.data, expanded=1)
    if view.prompt:
        st.markdown("**Prompt** (opening text of each message)")
        for n, msg in enumerate(view.prompt):
            with st.expander(f"{n + 1}. {msg.get('role')} · {msg.get('chars', 0):,} chars"):
                st.code(str(msg.get("preview", "")), language="markdown")
    if view.response:
        with st.expander("Response (opening text)"):
            st.code(view.response, language="json")
    if view.details:
        with st.expander("Other recorded fields"):
            st.json(view.details, expanded=False)


def render_state(trace: ob.RunTrace) -> None:
    changes = ob.state_changes(trace)
    st.markdown("**Shared state updates** (LangGraph node updates · CrewAI task results)")
    if changes:
        st.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
    else:
        st.caption("No state updates were recorded for this run.")
    attempts = ob.recorded_attempts(trace)
    st.markdown("**Review attempts** (from the saved trace)")
    if not attempts:
        st.caption("No review attempt was recorded (the run ended before review).")
        return
    st.dataframe(pd.DataFrame([attempt_row(a) for a in attempts]), hide_index=True,
                 width="stretch")
    for a in attempts:
        if a["issues"]:
            with st.expander(f"Iteration {a['iteration']}: {len(a['issues'])} issue(s)"):
                for i in a["issues"]:
                    icon = {"error": "🔴", "warning": "🟠"}.get(str(i.get("severity")), "🔵")
                    day = f"day {int(i['day']) + 1}: " if i.get("day") is not None else ""
                    rule = f" `{i['rule']}`" if i.get("rule") else ""
                    st.markdown(f"{icon} {day}{i.get('message')}{rule}")


def attempt_row(a: dict[str, Any]) -> dict[str, Any]:
    at = a.get("attempt") or {}
    return {"iteration": a["iteration"], "approved": at.get("approved"),
            "score": at.get("score"), "total (AED, est.)": at.get("total_aed"),
            "errors": at.get("errors"), "warnings": at.get("warnings"),
            "blocking rules": ", ".join(at.get("blocking_rules") or []),
            "locked days": ", ".join(str(d + 1) for d in at.get("locked_days") or []),
            "issues": len(a["issues"])}


# --------------------------------------------------------------------------- comparison


def run_label(run: Any) -> str:
    return (f"{run.started_at:%m-%d %H:%M:%S} · {run.mode.value} · {run.status.value} · "
            f"{run.id[:8]}")


def comparison_frame(a: ob.RunMetrics, b: ob.RunMetrics) -> pd.DataFrame:
    return pd.DataFrame([{"metric": r.metric, "A": fmt(r.a), "B": fmt(r.b),
                          "B − A": fmt(r.delta), "note": r.note} for r in ob.compare(a, b)])


def render_comparison(svc: Services, a: ob.RunTrace, b: ob.RunTrace) -> None:
    checks = ob.fairness(a, b)
    unfair = [c for c in checks if c.same is False]
    unknown = [c for c in checks if c.same is None]
    if unfair:
        st.warning("Not a like-for-like comparison: "
                   + "; ".join(f"{c.label.lower()}: no" for c in unfair), icon="⚠️")
    elif unknown:
        st.info("Some inputs were not recorded, so fairness cannot be fully confirmed.")
    else:
        st.success("Same message, request, seed data, provider/model and loop cap.")
    st.dataframe(pd.DataFrame([{"check": c.label,
                                "same": {True: "✓ yes", False: "✗ no", None: "? unknown"}[c.same],
                                "A": c.a, "B": c.b} for c in checks]),
                 hide_index=True, width="stretch")
    st.dataframe(comparison_frame(ob.run_metrics(a), ob.run_metrics(b)), hide_index=True,
                 width="stretch")
    st.caption("Measured from each run's recorded timestamps and LLM-call events: one run "
               "per side, no repetition, so differences are not statistically tested "
               "(Phase 7 evaluates repeated scenarios).")
    left, right = st.columns(2)
    for col, trace, name in ((left, a, "A"), (right, b, "B")):
        with col:
            st.markdown(f"**{name}: {trace.mode.value}**")
            if trace.events:
                st.plotly_chart(timeline_figure(trace, height=420), width="stretch",
                                key=f"cmp-{name}-{trace.run.id}")
            else:
                st.info(ob.NO_TRACE)


def export_trace(trace: ob.RunTrace) -> str:
    """The run's record and events as JSON (secrets redacted)."""
    return ob.export_trace(trace)
