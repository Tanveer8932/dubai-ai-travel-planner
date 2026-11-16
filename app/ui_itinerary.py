"""Render a saved itinerary version: overview, day tabs, map, exports.

Every price shown is an estimate and says so; its seed price range, source and as-of
date are in the hover help next to it (AC5.6). Data attribution stays visible.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from ui_shared import Services, money

from dubai_planner.models import ActivityKind, ItineraryRecord, TimeSlot
from dubai_planner.services import presentation as view

KIND_ICONS = {ActivityKind.ATTRACTION: "🏛️", ActivityKind.DINING: "🍽️",
              ActivityKind.HOTEL: "🏨", ActivityKind.EVENT: "🎉", ActivityKind.FREE_TIME: "☕"}
ATTRIBUTION = ("Venue data curated from public sources (© OpenStreetMap contributors, ODbL; "
               "Wikivoyage, CC BY-SA; Visit Dubai). Weather: Open-Meteo. Fares: RTA tariffs.")


def budget_gauge(record: ItineraryRecord) -> go.Figure:
    cost = record.itinerary.cost
    budget = float(cost.budget_aed or cost.total)
    limit = budget * (1 + float(cost.tolerance))
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=float(cost.total),
        number={"prefix": "≈ AED ", "valueformat": ",.0f"},
        delta={"reference": budget, "valueformat": ",.0f", "increasing": {"color": "#c0392b"},
               "decreasing": {"color": "#27ae60"}},
        title={"text": "Estimated total vs budget", "font": {"size": 16}},
        gauge={"axis": {"range": [0, max(limit, float(cost.total)) * 1.1]},
               "bar": {"color": "#1f77b4"},
               "steps": [{"range": [0, budget], "color": "#e8f5e9"},
                         {"range": [budget, limit], "color": "#fff3e0"}],
               "threshold": {"line": {"color": "#c0392b", "width": 3}, "value": limit}}))
    fig.update_layout(height=260, margin={"t": 70, "b": 10, "l": 30, "r": 30})
    return fig


def _slot(slot: TimeSlot, svc: Services) -> None:
    activity = slot.activity
    icon = KIND_ICONS.get(activity.kind, "•")
    setting = "" if activity.kind is ActivityKind.FREE_TIME else (
        " · :green-badge[indoor]" if slot.indoor else " · :orange-badge[outdoor]")
    with st.container(border=True):
        left, right = st.columns([3, 2])
        left.markdown(f"**{slot.start:%H:%M}–{slot.end:%H:%M}** {icon} **{activity.name}**"
                      f"{setting}")
        if activity.area:
            left.caption(activity.area.replace("_", " ").title()
                         + (f" · {activity.description}" if activity.description else ""))
        note = view.venue_price(slot, svc.catalog)
        right.markdown(note.label, help=f"{note.basis}\n\nSource: {note.source}".strip())
        leg = slot.transport_from_prev
        if leg is not None:
            fare = view.transport_price(leg, svc.catalog)
            right.caption(f"🚇 {leg.mode.value.replace('_', ' ')} · {leg.duration_min} min · "
                          f"{fare.label}", help=f"{fare.basis}\n\nSource: {fare.source}")


def render_itinerary(record: ItineraryRecord, svc: Services, *, key: str) -> None:
    it = record.itinerary
    status = "✅ approved" if record.approved else "⚠️ not approved (see the issues)"
    st.subheader(it.title or "Your Dubai itinerary")
    st.caption(f"Version {record.version} · {record.mode.value} · {status} · saved "
               f"{record.created_at:%Y-%m-%d %H:%M} UTC")
    st.info(view.ESTIMATE_NOTE, icon="ℹ️")
    days = [f"Day {i + 1} · {d.date:%a %d %b}" for i, d in enumerate(it.days)]
    tabs = st.tabs(["Overview", *days, "Map", "Export"])
    with tabs[0]:
        if it.summary:
            st.write(it.summary)
        cols = st.columns([1, 1])
        with cols[0]:
            st.plotly_chart(budget_gauge(record), use_container_width=True, key=f"{key}-gauge")
        with cols[1]:
            st.markdown("**Budget breakdown (AED, estimates)**")
            st.dataframe(pd.DataFrame(view.cost_rows(it)), hide_index=True,
                         use_container_width=True)
        if it.hotel is not None:
            hp = view.hotel_price(it.hotel)
            st.markdown(f"🏨 **{it.hotel.name}** · {it.hotel.area.replace('_', ' ').title()} · "
                        f"{hp.basis}", help=f"Source: {hp.source}")
        if record.request is not None:
            req = record.request
            st.caption(f"{req.start_date:%d %b %Y} – {req.end_date:%d %b %Y} · "
                       f"{req.travellers.total} traveller(s) · budget "
                       f"{money(req.total_budget_aed())} · interests: "
                       f"{', '.join(req.interests) or '-'} · dietary: "
                       f"{', '.join(req.dietary) or '-'}")
        for n in it.notes:
            st.caption(f"• {n}")
    for i, day in enumerate(it.days):
        with tabs[i + 1]:
            st.markdown(f"**{day.theme or 'Day plan'}** · day cost ≈ "
                        f"{money(day.day_cost_aed)} (estimate)")
            for slot in day.slots:
                _slot(slot, svc)
    with tabs[len(days) + 1]:
        points = view.map_points(it)
        if points:
            frame = pd.DataFrame(points)
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=float(frame.lat.mean()),
                                                 longitude=float(frame.lon.mean()), zoom=10),
                layers=[pdk.Layer("ScatterplotLayer", data=frame, get_position="[lon, lat]",
                                  get_radius=250, get_fill_color="[200, 30 + day * 50, 160]",
                                  pickable=True)],
                tooltip={"text": "Day {day}: {name}"}), key=f"{key}-map")
            st.caption("Locations are approximate (curated seed data).")
        else:
            st.caption("No locations to show.")
    with tabs[len(days) + 2]:
        st.download_button("Download JSON", view.export_json(record),
                           file_name=f"dubai-itinerary-v{record.version}.json",
                           mime="application/json", key=f"{key}-json")
        st.download_button("Download Markdown", view.export_markdown(record, svc.catalog),
                           file_name=f"dubai-itinerary-v{record.version}.md",
                           mime="text/markdown", key=f"{key}-md")
        st.caption("Exports contain the request and itinerary only (no settings or keys).")
    st.caption(ATTRIBUTION)
