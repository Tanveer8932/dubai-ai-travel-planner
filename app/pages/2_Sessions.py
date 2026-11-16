"""Sessions page: list, open, rename, delete; version history and version diff."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import streamlit as st
from ui_itinerary import render_itinerary
from ui_shared import current_session_id, money, select_session, services, sidebar

from dubai_planner.models import VersionDiff
from dubai_planner.services.presentation import COST_LABELS

st.set_page_config(page_title="Sessions · Dubai AI Travel Planner", page_icon="🗂️",
                   layout="wide")
svc = services()
sidebar(svc)
st.title("🗂️ Sessions & versions")


def show_diff(diff: VersionDiff, new_total: Decimal) -> None:
    total = diff.cost_delta.get("total", Decimal("0"))
    st.metric(f"Estimated total, v{diff.old_version} → v{diff.new_version}", money(new_total),
              delta=f"{float(total):+,.2f} AED", delta_color="inverse")
    if diff.budget_before != diff.budget_after:
        st.caption(f"Budget: {money(diff.budget_before or 0)} → {money(diff.budget_after or 0)}")
    rows = [{"Category": label, "Change (AED, estimate)": float(diff.cost_delta[key])}
            for key, label in COST_LABELS.items()]
    rows.append({"Category": "Total", "Change (AED, estimate)": float(total)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    unchanged = ", ".join(str(d + 1) for d in diff.unchanged_days) or "none"
    st.caption(f"Unchanged days: {unchanged}")
    for day in diff.days:
        if not day.changed:
            continue
        with st.container(border=True):
            delta = float(day.cost_after - day.cost_before)
            st.markdown(f"**Day {day.day_index + 1}**"
                        + (f" · {day.date:%a %d %b}" if day.date else "")
                        + f" · day cost {delta:+,.2f} AED (estimate)")
            if day.theme_before != day.theme_after:
                st.caption(f"Theme: {day.theme_before or '-'} → {day.theme_after or '-'}")
            for slot in day.removed:
                st.markdown(f":red[− {slot.label()}]")
            for slot in day.added:
                st.markdown(f":green[+ {slot.label()}]")


sessions = svc.sessions.list_sessions()
if not sessions:
    st.info("No sessions yet. Plan a trip on the Planner page.")
    st.stop()

table = [{"Title": s.title, "Mode": s.mode.value,
          "Versions": len(svc.sessions.versions(s.id)),
          "Updated (UTC)": f"{s.updated_at:%Y-%m-%d %H:%M}"} for s in sessions]
st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

ids = [s.id for s in sessions]
titles = {s.id: s.title for s in sessions}
current = current_session_id()
chosen = st.selectbox("Session", ids, index=ids.index(current) if current in ids else 0,
                      format_func=titles.__getitem__, key="sessions_pick")
cols = st.columns(3)
if cols[0].button("Open in Planner", use_container_width=True):
    select_session(chosen)
    st.switch_page("pages/1_Planner.py")
with cols[1].popover("Rename", use_container_width=True):
    new_title = st.text_input("New title", value=titles[chosen], max_chars=80,
                              key=f"rename-{chosen}")
    if st.button("Save title", key=f"save-{chosen}"):
        try:
            svc.sessions.rename(chosen, new_title)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.rerun()
with cols[2].popover("Delete", use_container_width=True):
    st.write("Delete this session, its conversation and every saved version?")
    busy = svc.runs.active(chosen) is not None
    if busy:
        st.caption("A plan is still being made for this session; wait for it to finish.")
    if st.button("Delete permanently", type="primary", disabled=busy, key=f"del-{chosen}"):
        svc.sessions.delete(chosen)
        svc.hub.forget_thread(chosen)
        svc.runs.forget(chosen)
        if current == chosen:
            select_session(None)
        st.rerun()

st.divider()
versions = svc.sessions.versions(chosen)
st.subheader(f"Versions of “{titles[chosen]}”")
if not versions:
    st.caption("No itinerary has been saved in this session yet.")
    st.stop()
st.dataframe(pd.DataFrame([{
    "Version": v.version, "Saved (UTC)": f"{v.created_at:%Y-%m-%d %H:%M}",
    "Mode": v.mode.value, "Approved": v.approved,
    "Total (AED, estimate)": float(v.total_aed),
    "Budget (AED)": float(v.request.total_budget_aed()) if v.request else None,
} for v in versions]), hide_index=True, use_container_width=True)

numbers = [v.version for v in versions]
if len(versions) >= 2:
    st.subheader("Compare versions")
    left, right = st.columns(2)
    old = left.selectbox("From", numbers, index=len(numbers) - 2, key=f"from-{chosen}")
    new = right.selectbox("To", numbers, index=len(numbers) - 1, key=f"to-{chosen}")
    a, b = svc.sessions.version(chosen, old), svc.sessions.version(chosen, new)
    if a is not None and b is not None and a.version != b.version:
        show_diff(svc.sessions.diff(a, b), b.total_aed)
    else:
        st.caption("Pick two different versions.")

st.subheader("View a version")
shown = st.selectbox("Version", numbers, index=len(numbers) - 1, key=f"view-{chosen}")
record = svc.sessions.version(chosen, shown)
if record is not None:
    render_itinerary(record, svc, key=f"sessions-{chosen}-{shown}")
