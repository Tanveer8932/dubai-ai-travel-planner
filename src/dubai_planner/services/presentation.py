"""View helpers shared by the Streamlit pages and the exports (no Streamlit import).

Prices are always shown as **estimates** with where they came from (AC5.6): each
slot's price is traced back to its seed record (``PriceRange`` + ``Provenance``), and
transport legs to the curated RTA/taxi tariff. Exports contain the itinerary and the
request only; settings and keys are never part of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from dubai_planner.data.seed import SeedCatalog
from dubai_planner.models import (
    ActivityKind,
    AttemptSummary,
    DiningOption,
    HotelOption,
    Itinerary,
    ItineraryRecord,
    PlanResult,
    PriceRange,
    Provenance,
    TimeSlot,
    TransportLeg,
)

ESTIMATE_NOTE = ("All prices are estimates from curated public data (not live prices); "
                 "confirm before booking.")
COST_LABELS = {"accommodation": "Accommodation", "activities": "Activities", "food": "Food",
               "transport": "Local transport", "contingency": "Contingency"}


@dataclass(frozen=True)
class PriceNote:
    """How one price is shown: the amount, that it is an estimate, and its source."""

    amount_aed: Decimal
    basis: str  # e.g. "≈ AED 149–189 (estimate, as of 2026-11-15) per person"
    source: str  # e.g. "estimated · curated seed (Visit Dubai…) · as of 2026-11-15"

    @property
    def label(self) -> str:
        if self.amount_aed == 0:
            return "Free / included"
        return f"≈ AED {self.amount_aed:,.2f} (estimate)"


def _price_basis(price: PriceRange) -> str:
    unit = price.unit.value.replace("_", " ")
    return f"{price.label()} {unit}".strip()


def _provenance(p: Provenance) -> str:
    return p.label() + (f" · {p.source_url}" if p.source_url else "")


def venue_price(slot: TimeSlot, catalog: SeedCatalog) -> PriceNote:
    """The slot's party price with the seed price range and provenance behind it."""
    ref = slot.activity.ref_id
    venue = catalog.venue(ref) if ref else None
    if venue is None:
        source = ("no charge recorded" if slot.cost_aed == 0
                  else "estimate (source not in the catalogue)")
        return PriceNote(slot.cost_aed, "", source)
    if isinstance(venue, HotelOption):
        price = venue.nightly_rate
    elif isinstance(venue, DiningOption):
        price = venue.avg_spend
    else:
        price = venue.price
    return PriceNote(slot.cost_aed, _price_basis(price), _provenance(price.provenance))


def transport_price(leg: TransportLeg, catalog: SeedCatalog) -> PriceNote:
    tariff = catalog.transport.taxi.provenance if leg.mode.value == "taxi" \
        else catalog.transport.nol_fares.provenance
    basis = f"{leg.mode.value.replace('_', ' ')}, {leg.distance_km:.1f} km, ~{leg.duration_min} min"
    kind = "estimate" if leg.cost_is_estimate else "fare"
    return PriceNote(leg.cost_aed, f"{basis} ({kind})", _provenance(tariff))


def hotel_price(hotel: HotelOption) -> PriceNote:
    return PriceNote(Decimal("0"), _price_basis(hotel.nightly_rate),
                     _provenance(hotel.nightly_rate.provenance))


def cost_rows(itinerary: Itinerary) -> list[dict[str, Any]]:
    """The budget breakdown as table rows (every amount labelled an estimate)."""
    cost = itinerary.cost
    rows = [{"Category": label, "AED (estimate)": float(getattr(cost, key))}
            for key, label in COST_LABELS.items()]
    rows.append({"Category": "Total", "AED (estimate)": float(cost.total)})
    if cost.budget_aed is not None:
        rows.append({"Category": "Budget", "AED (estimate)": float(cost.budget_aed)})
    return rows


def budget_use(itinerary: Itinerary) -> float | None:
    """Total as a share of the budget (1.0 = exactly on budget)."""
    budget = itinerary.cost.budget_aed
    if not budget:
        return None
    return float(itinerary.cost.total / budget)


def map_points(itinerary: Itinerary) -> list[dict[str, Any]]:
    """One point per located slot (for the map), numbered by day."""
    points = []
    for i, day in enumerate(itinerary.days):
        for slot in day.slots:
            loc = slot.activity.location
            if loc is not None:
                points.append({"lat": loc.lat, "lon": loc.lon, "day": i + 1,
                               "name": slot.activity.name, "kind": slot.activity.kind.value})
    if itinerary.hotel is not None:
        loc = itinerary.hotel.location
        points.append({"lat": loc.lat, "lon": loc.lon, "day": 0,
                       "name": itinerary.hotel.name, "kind": "hotel"})
    return points


def attempt_rows(attempts: list[AttemptSummary]) -> list[dict[str, Any]]:
    return [{"Attempt": a.iteration + 1, "Approved": a.approved, "Score": a.score,
             "Total AED (estimate)": float(a.total_aed), "Errors": a.errors,
             "Warnings": a.warnings, "Blocking rules": ", ".join(a.blocking_rules) or "-",
             "Kept days": ", ".join(str(d + 1) for d in a.locked_days) or "-"}
            for a in attempts]


def status_text(result: PlanResult) -> str:
    return {"approved": "Approved", "needs_revision": "Not approved (review cap reached)",
            "needs_clarification": "Needs more information", "failed": "Failed",
            "cancelled": "Cancelled"}[result.status.value]


# --------------------------------------------------------------------------- exports


def export_json(record: ItineraryRecord) -> str:
    """The saved version (request + itinerary) as JSON. No settings, no secrets."""
    data = {
        "session_id": record.session_id, "version": record.version,
        "mode": record.mode.value, "approved": record.approved,
        "created_at": record.created_at.isoformat(),
        "prices_are_estimates": True, "note": ESTIMATE_NOTE,
        "request": record.request.model_dump(mode="json") if record.request else None,
        "itinerary": record.itinerary.model_dump(mode="json"),
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_markdown(record: ItineraryRecord, catalog: SeedCatalog) -> str:
    it, req = record.itinerary, record.request
    lines = [f"# {it.title or 'Dubai itinerary'} (version {record.version})", ""]
    status = "approved" if record.approved else "not approved"
    lines.append(f"*Planned with {record.mode.value}; {status}.*")
    if req is not None:
        lines.append(f"*{req.start_date:%d %b %Y} to {req.end_date:%d %b %Y}, "
                     f"{req.travellers.total} traveller(s), budget AED "
                     f"{req.total_budget_aed():,.2f}.*")
    lines += ["", f"> {ESTIMATE_NOTE}", ""]
    if it.summary:
        lines += [it.summary, ""]
    if it.hotel is not None:
        hp = hotel_price(it.hotel)
        lines += [f"**Hotel:** {it.hotel.name} ({it.hotel.area}) — {hp.basis}; "
                  f"source: {hp.source}", ""]
    for i, day in enumerate(it.days):
        theme = f": {day.theme}" if day.theme else ""
        lines += [f"## Day {i + 1} — {day.date:%a %d %b}{theme}", "",
                  "| Time | Activity | Area | Setting | Price (estimate) | Source |",
                  "|---|---|---|---|---|---|"]
        for slot in day.slots:
            note = venue_price(slot, catalog)
            setting = "indoor" if slot.indoor else "outdoor"
            if slot.activity.kind is ActivityKind.FREE_TIME:
                setting = "-"
            lines.append(f"| {slot.start:%H:%M}–{slot.end:%H:%M} | {slot.activity.name} | "
                         f"{slot.activity.area or '-'} | {setting} | {note.label} | "
                         f"{note.source} |")
        lines += ["", f"Day cost: ≈ AED {day.day_cost_aed:,.2f} (estimate)", ""]
    lines += ["## Budget (AED, estimates)", "", "| Category | AED |", "|---|---|"]
    for row in cost_rows(it):
        lines.append(f"| {row['Category']} | {row['AED (estimate)']:,.2f} |")
    if it.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in it.notes]
    lines += ["", "Venue data: curated from public sources (OpenStreetMap contributors, "
              "ODbL; Wikivoyage, CC BY-SA; Visit Dubai). Weather: Open-Meteo."]
    return "\n".join(lines) + "\n"
