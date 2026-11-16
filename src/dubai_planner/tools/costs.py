"""Deterministic cost arithmetic (hard rule 4): party pricing, hotel cost, cost breakdowns.

All money is ``Decimal`` AED, quantised to fils (2 dp, half-up) at each step. Every
amount is an **estimate** derived from curated seed price ranges (midpoints), and is
labelled as such (``CostItem.kind = estimated``).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from decimal import Decimal

from dubai_planner.data.seed import SeedCatalog
from dubai_planner.models.budget import BudgetCategory, CostBreakdown, CostItem
from dubai_planner.models.common import DataKind, PriceBand, PriceRange, PriceUnit, quantize_aed
from dubai_planner.models.itinerary import ActivityKind, DayPlan
from dubai_planner.models.options import HotelOption
from dubai_planner.models.request import Travellers
from dubai_planner.validation.thresholds import ADULTS_PER_ROOM, TAXI_CAPACITY

ZERO = Decimal("0")


def rooms_needed(travellers: Travellers) -> int:
    """Rooms for the party: two adults per room, children share."""
    return max(1, math.ceil(travellers.adults / ADULTS_PER_ROOM))


def vehicles_needed(travellers: Travellers) -> int:
    return max(1, math.ceil(travellers.total / TAXI_CAPACITY))


def party_cost(price: PriceRange, travellers: Travellers) -> Decimal:
    """Estimated cost of ``price`` for the whole party (midpoint of the range).

    Per-person and per-adult-ticket prices are charged for every traveller: child
    discounts vary by venue and are not in the seed, so this errs on the high side.
    """
    if price.is_free or price.unit is PriceUnit.FREE:
        return ZERO
    mid = price.midpoint_aed
    if price.unit in (PriceUnit.PER_PERSON, PriceUnit.PER_ADULT_TICKET):
        return quantize_aed(mid * travellers.total)
    if price.unit is PriceUnit.PER_ROOM_NIGHT:
        return quantize_aed(mid * rooms_needed(travellers))
    return quantize_aed(mid)  # per trip / per group


def hotel_cost(hotel: HotelOption, nights: int, travellers: Travellers) -> Decimal:
    """Nights × nightly midpoint × rooms."""
    if nights <= 0:
        return ZERO
    return quantize_aed(hotel.nightly_rate.midpoint_aed * nights * rooms_needed(travellers))


def food_daily_rates(catalog: SeedCatalog) -> dict[PriceBand, Decimal]:
    """Estimated food spend per person per day, by price band, derived from seed data.

    Lunch + dinner at the band's median average spend, plus breakfast at the
    budget-band median. Bands without restaurants fall back to the next cheaper band.
    """
    medians: dict[PriceBand, Decimal] = {}
    for band in PriceBand:
        mids = [r.avg_spend.midpoint_aed for r in catalog.restaurants if r.price_band is band]
        if mids:
            medians[band] = quantize_aed(Decimal(statistics.median(mids)))
    if PriceBand.BUDGET not in medians:  # pragma: no cover - seed always has budget venues
        raise ValueError("seed has no budget-band restaurants")
    breakfast = medians[PriceBand.BUDGET]
    rates: dict[PriceBand, Decimal] = {}
    previous = medians[PriceBand.BUDGET]
    for band in PriceBand:
        meal = medians.get(band, previous)
        previous = meal
        rates[band] = quantize_aed(breakfast + 2 * meal)
    return rates


def day_cost(day: DayPlan) -> Decimal:
    """Slot costs plus the cost of each slot's inbound transport leg."""
    legs = sum((s.transport_from_prev.cost_aed for s in day.slots if s.transport_from_prev),
               ZERO)
    return quantize_aed(day.slot_cost_sum + legs)


_SLOT_CATEGORY = {
    ActivityKind.ATTRACTION: BudgetCategory.ACTIVITIES,
    ActivityKind.EVENT: BudgetCategory.ACTIVITIES,
    ActivityKind.DINING: BudgetCategory.FOOD,
    ActivityKind.TRANSIT: BudgetCategory.TRANSPORT,
    ActivityKind.HOTEL: BudgetCategory.ACCOMMODATION,
    ActivityKind.FREE_TIME: BudgetCategory.ACTIVITIES,
}


def cost_items(
    days: Sequence[DayPlan],
    *,
    hotel: HotelOption | None,
    nights: int,
    travellers: Travellers,
    extra_items: Sequence[CostItem] = (),
) -> list[CostItem]:
    """Itemise accommodation, every slot and every transport leg (no contingency)."""
    items: list[CostItem] = []
    if hotel is not None and nights > 0:
        items.append(CostItem(
            category=BudgetCategory.ACCOMMODATION,
            description=f"{hotel.name}: {nights} night(s) × {rooms_needed(travellers)} room(s)",
            amount_aed=hotel_cost(hotel, nights, travellers),
            ref_id=hotel.id,
        ))
    for index, day in enumerate(days):
        for slot in day.slots:
            if slot.cost_aed > 0:
                items.append(CostItem(
                    category=_SLOT_CATEGORY[slot.activity.kind],
                    description=slot.activity.name,
                    amount_aed=slot.cost_aed,
                    ref_id=slot.activity.ref_id,
                    day_index=index,
                ))
            leg = slot.transport_from_prev
            if leg is not None and leg.cost_aed > 0:
                items.append(CostItem(
                    category=BudgetCategory.TRANSPORT,
                    description=f"{leg.mode.value} to {slot.activity.name}",
                    amount_aed=leg.cost_aed,
                    day_index=index,
                ))
    items.extend(extra_items)
    return items


def calculate_costs(
    days: Sequence[DayPlan],
    *,
    hotel: HotelOption | None,
    nights: int,
    travellers: Travellers,
    budget_aed: Decimal | None,
    contingency_pct: Decimal,
    tolerance: Decimal = Decimal("0.05"),
    extra_items: Sequence[CostItem] = (),
) -> CostBreakdown:
    """Consistent, itemised :class:`CostBreakdown` with a contingency line.

    The total equals the category sum and the item sum by construction.
    """
    items = cost_items(days, hotel=hotel, nights=nights, travellers=travellers,
                       extra_items=extra_items)
    subtotal = sum((i.amount_aed for i in items), ZERO)
    contingency = quantize_aed(subtotal * contingency_pct)
    if contingency > 0:
        items.append(CostItem(
            category=BudgetCategory.CONTINGENCY,
            description=f"Contingency {contingency_pct * 100:.0f}% of estimated costs",
            amount_aed=contingency,
            kind=DataKind.ESTIMATED,
        ))
    return CostBreakdown.from_items(items, budget_aed=budget_aed, tolerance=tolerance)
