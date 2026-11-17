"""Evaluation dashboard: metric table, distributions per mode, per-scenario drill-down.

Reads published results (``eval/results/<name>/``, committed, so a deployed app can
show them) and the local evaluation database (``data/eval.db``, written by
``scripts/run_eval.py``). It never runs an evaluation itself. Mock-provider results
are labelled as orchestration evidence, not model quality.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui_shared import nav_link, services, sidebar

from dubai_planner.evaluation.faults import detection_rate, fix_rate
from dubai_planner.evaluation.metrics import NUMERIC_METRICS, aggregate
from dubai_planner.evaluation.report import MOCK_BANNER, MODE_COLORS, label
from dubai_planner.evaluation.results import EvalDataset, load_all

st.set_page_config(page_title="Evaluation · Dubai AI Travel Planner", page_icon="📊",
                   layout="wide")
svc = services()
sidebar(svc)
st.title("📊 Evaluation: LangGraph vs CrewAI")

datasets = load_all()
if not datasets:
    st.info("No evaluation results yet. Run `python scripts/run_eval.py --provider mock` "
            "(offline) and reload this page.")
    nav_link("streamlit_app.py", "Home", "🏠")
    st.stop()

by_key = {d.key: d for d in datasets}
key = st.selectbox("Results", list(by_key), format_func=lambda k: by_key[k].title)
ds: EvalDataset = by_key[key]
cfg = ds.config
if ds.is_mock:
    st.warning(MOCK_BANNER.removeprefix("> "))
else:
    st.info(f"Real provider: **{cfg.get('provider')}** / `{cfg.get('model')}`. Each cell is "
            "one run; free-tier rate limits add waiting time to latency.")

modes = list(dict.fromkeys(c.mode for c in ds.cells))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Scenarios", len(ds.scenarios))
c2.metric("Modes", len(modes))
c3.metric("Cells", len(ds.cells))
c4.metric("Repetitions", cfg.get("seeds", 1))
st.caption(f"Data fingerprint `{cfg.get('data_fingerprint', '—')}` · HTTP offline "
           f"{cfg.get('http_offline')} · review cap {cfg.get('max_review_iterations')} · "
           f"budget tolerance {cfg.get('budget_tolerance')}")
if not ds.cells:
    st.info("This evaluation has no scored cells yet.")
    st.stop()

tab_sum, tab_dist, tab_scen, tab_faults = st.tabs(
    ["Summary", "Distributions", "Per scenario", "Fault injection & judge"])

with tab_sum:
    agg = aggregate(ds.cells)
    table = pd.DataFrame(
        {label(m): [agg[m][k].show() for k, _ in NUMERIC_METRICS] for m in modes},
        index=[name for _, name in NUMERIC_METRICS])
    st.dataframe(table, width="stretch")
    st.caption("Mean ± sample standard deviation over cells; n = cells with a recorded value. "
               "Shares: 1.00 = every cell. “—” = not recorded.")
    if ds.report_path is not None:
        with st.expander("Generated report (Markdown)"):
            st.markdown(ds.report_path.read_text(encoding="utf-8").split("## Per scenario")[0]
                        .replace("![performance.png](performance.png)", "")
                        .replace("![quality.png](quality.png)", ""))

with tab_dist:
    options = {name: k for k, name in NUMERIC_METRICS
               if k not in ("schema_valid", "budget_compliant", "outcome_ok")}
    metric = st.selectbox("Metric", list(options), index=list(options).index("Latency (s)"))
    field = options[metric]
    fig = go.Figure()
    for m in modes:
        values = [getattr(c, field) for c in ds.cells if c.mode == m
                  and getattr(c, field) is not None]
        fig.add_trace(go.Box(x=values, name=label(m), orientation="h", boxpoints="all",
                             jitter=0.3, pointpos=0, marker_color=MODE_COLORS.get(m),
                             line_color=MODE_COLORS.get(m),
                             hovertemplate=f"{label(m)}: %{{x}}<extra></extra>"))
    fig.update_layout(height=120 + 90 * len(modes), showlegend=False,
                      margin={"l": 10, "r": 10, "t": 30, "b": 10}, title=metric)
    st.plotly_chart(fig, width="stretch")
    st.caption("One dot per scenario × repetition. Box = quartiles; line = median.")

with tab_scen:
    scenario = st.selectbox("Scenario", ds.scenarios)
    cells = [c for c in ds.cells if c.scenario_id == scenario]
    rows = [{"mode": label(c.mode), "seed": c.seed, "status": c.final_status,
             "expected outcome met": c.outcome_ok, "total AED": c.total_aed,
             "budget AED": c.budget_aed, "constraints met":
             f"{c.constraints_satisfied}/{c.constraints_checked}",
             "rule errors": c.violations_error, "rule warnings": c.violations_warning,
             "latency s": c.latency_s, "LLM calls": c.llm_calls, "tokens": c.tokens,
             "review iterations": c.review_iterations} for c in cells]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    first = cells[0]
    st.markdown("**Hard constraints (first repetition of each mode)**")
    for c in {c.mode: c for c in reversed(cells)}.values():
        st.markdown(f"*{label(c.mode)}*: " + (", ".join(
            f"{'✅' if k.satisfied else '❌'} {k.name} ({k.detail})" for k in c.constraints)
            or "—"))
    if len(first.turns) > 1:
        st.markdown("**Conversation routing**")
        for c in {c.mode: c for c in reversed(cells)}.values():
            st.markdown(f"*{label(c.mode)}*")
            for i, t in enumerate(c.turns, 1):
                st.markdown(f"{i}. “{t.message}” → `{t.routing}` → {t.status} "
                            f"(v{t.saved_version or '—'})")
    errors = [c.error for c in cells if c.error]
    if errors:
        st.error("Errors: " + " · ".join(e[:200] for e in errors))

with tab_faults:
    if ds.faults:
        fmodes = list(dict.fromkeys(f.mode for f in ds.faults))
        cols = st.columns(len(fmodes))
        for col, m in zip(cols, fmodes, strict=True):
            det, fix = detection_rate(ds.faults, m), fix_rate(ds.faults, m)
            col.metric(f"{label(m)}: defects detected",
                       "—" if det is None else f"{det:.0%}",
                       help=f"Error defects fixed: {'—' if fix is None else f'{fix:.0%}'}")
        st.dataframe(pd.DataFrame([{
            "defect": f.defect, "rule": f.rule_id, "severity": f.severity,
            "mode": label(f.mode), "detected": f.detected, "fixed": f.fixed,
            "final status": f.final_status, "iterations": f.iterations,
            "detected by": ", ".join(f.detected_by)} for f in ds.faults]),
            width="stretch", hide_index=True)
    else:
        st.info("No fault-injection results in this evaluation.")
    if ds.pairs:
        st.markdown("**Pairwise judge (position-swapped)**")
        st.dataframe(pd.DataFrame([{
            "scenario": p.scenario_id, "A": label(p.a), "B": label(p.b),
            "winner": {"a": label(p.a), "b": label(p.b)}.get(p.winner, "tie"),
            "order-consistent": p.consistent} for p in ds.pairs]),
            width="stretch", hide_index=True)
