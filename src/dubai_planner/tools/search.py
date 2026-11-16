"""Seed search tools: filter (hard constraints) and score (soft preferences) venues.

Hard constraints remove candidates: dietary needs (restaurants), known wheelchair
inaccessibility when wheelchair access is required, family suitability when children
travel, seasonal closure, and ``avoid`` terms. Soft preferences only change the
score in ``[0, 1]``: interest match, preferred areas, ``must_include``, price fit, heat.

Results are copies of seed records with ``score`` set, in descending score order
(ties by id, so results are deterministic). Research agents may only pick from these.
"""

from __future__ import annotations

import re
from datetime import timedelta
from decimal import Decimal

from dubai_planner.data.seed import SeedCatalog
from dubai_planner.models.common import PriceBand
from dubai_planner.models.options import (
    AttractionOption,
    DiningOption,
    HotelOption,
    TimeOfDay,
    Venue,
)
from dubai_planner.models.request import AccessibilityNeed, TripRequest
from dubai_planner.tools.costs import food_daily_rates, rooms_needed
from dubai_planner.validation.thresholds import ACCOMMODATION_SHARE, HOT_MONTHS

#: Interest keyword → words to look for in a venue's category, tags, name, description.
INTEREST_SYNONYMS: dict[str, tuple[str, ...]] = {
    "history": ("history", "heritage", "museum", "culture", "souk"),
    "culture": ("culture", "heritage", "emirati", "mosque", "museum", "art", "religious"),
    "food": ("food", "dining", "souk", "market", "spice", "fish market"),
    "shopping": ("shopping", "mall", "souk", "market"),
    "beach": ("beach", "waterfront", "promenade"),
    "adventure": ("adventure", "adrenaline", "desert", "water park", "theme park", "snow"),
    "desert": ("desert", "safari", "dune"),
    "family": ("family", "theme park", "water park", "aquarium", "young children", "wildlife"),
    "kids": ("family", "theme park", "water park", "aquarium", "young children", "wildlife"),
    "art": ("art", "gallery", "contemporary art", "islamic art"),
    "nature": ("nature", "park", "wildlife", "birdlife", "mountains", "flowers"),
    "views": ("views", "iconic", "photo spot", "observation"),
    "architecture": ("architecture", "iconic", "landmark", "mosque"),
    "nightlife": ("evening", "dinner", "cruise", "entertainment"),
    "luxury": ("luxury", "fine dining"),
    "religion": ("mosque", "religious", "modest dress"),
}


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "


def venue_text(venue: Venue) -> str:
    extra = ""
    if isinstance(venue, AttractionOption):
        extra = f" {venue.category.value.replace('_', ' ')}"
    elif isinstance(venue, DiningOption):
        extra = " " + " ".join(c.replace("_", " ") for c in venue.cuisines)
    return _words(f"{venue.name} {venue.description} {' '.join(venue.tags)}{extra}")


def _terms(interest: str) -> tuple[str, ...]:
    key = interest.strip().lower()
    return INTEREST_SYNONYMS.get(key, (key,))


def interest_matches(text: str, interests: list[str]) -> int:
    return sum(1 for i in interests if any(_words(t) in text for t in _terms(i)))


def _mentions(text: str, phrases: list[str]) -> bool:
    return any(_words(p).strip() and _words(p) in text for p in phrases)


def _name_matches(venue: Venue, phrases: list[str]) -> bool:
    name = _words(venue.name)
    return any(_words(p).strip() and _words(p) in name for p in phrases)


def budget_per_person_day(request: TripRequest) -> Decimal:
    return request.total_budget_aed() / request.travellers.total / request.num_days


def trip_months(request: TripRequest) -> set[int]:
    return {(request.start_date + timedelta(days=i)).month for i in range(request.num_days)}


def _needs_wheelchair(request: TripRequest) -> bool:
    return AccessibilityNeed.WHEELCHAIR in request.accessibility


def _passes_common(venue: Venue, request: TripRequest, text: str) -> bool:
    if _needs_wheelchair(request) and venue.wheelchair_accessible is False:
        return False
    if request.travellers.children and not venue.family_friendly:
        return False
    return not (request.avoid and _mentions(text, request.avoid))


def _clamp(score: float) -> float:
    return round(min(1.0, max(0.0, score)), 3)


def resolve_areas(request: TripRequest, catalog: SeedCatalog) -> set[str]:
    """Map free-text preferred areas ("Marina", "old dubai") to seed area ids."""
    wanted = [_words(a) for a in request.preferred_areas]
    ids = set()
    for area in catalog.areas:
        names = _words(f"{area.id.replace('_', ' ')} {area.name}")
        if any(w.strip() and (w in names or names.strip() in w) for w in wanted):
            ids.add(area.id)
    return ids


def search_attractions(
    request: TripRequest, catalog: SeedCatalog, *, limit: int = 25
) -> list[AttractionOption]:
    months = trip_months(request)
    areas = resolve_areas(request, catalog)
    hot = bool(months & HOT_MONTHS)
    per_day = budget_per_person_day(request)
    scored: list[tuple[float, AttractionOption]] = []
    for att in catalog.attractions:
        text = venue_text(att)
        must = _name_matches(att, request.must_include)
        if not must and not _passes_common(att, request, text):
            continue
        if not any(att.open_in_month(m) for m in months):
            continue
        score = 0.3
        matches = interest_matches(text, request.interests)
        score += 0.4 * min(1.0, matches / 2) if request.interests else 0.1
        if att.area in areas:
            score += 0.15
        if "iconic" in att.tags:
            score += 0.1
        if request.travellers.children and att.family_friendly:
            score += 0.05
        if _needs_wheelchair(request) and att.wheelchair_accessible is None:
            score -= 0.1
        if hot:
            if att.indoor:
                score += 0.05
            elif not {TimeOfDay.EVENING, TimeOfDay.NIGHT} & set(att.best_time):
                score -= 0.1
        if att.day_trip and request.num_days < 3:
            score -= 0.2
        price = att.price.midpoint_aed  # per person; activities get ~1/3 of the daily budget
        if price > per_day * Decimal("0.6"):
            score -= 0.3
        elif price > per_day * Decimal("0.35"):
            score -= 0.15
        if must:
            score = 1.0
        scored.append((_clamp(score), att))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [att.model_copy(update={"score": s}) for s, att in scored[:limit]]


def target_nightly_rate(request: TripRequest) -> Decimal:
    """Nightly room rate the budget can carry (accommodation share ÷ nights ÷ rooms)."""
    nights = max(request.nights, 1)
    return request.total_budget_aed() * ACCOMMODATION_SHARE / nights / rooms_needed(
        request.travellers)


def search_hotels(request: TripRequest, catalog: SeedCatalog, *, limit: int = 12
                  ) -> list[HotelOption]:
    if request.nights <= 0:
        return []
    target = target_nightly_rate(request)
    areas = resolve_areas(request, catalog)
    scored: list[tuple[float, HotelOption]] = []
    for hotel in catalog.hotels:
        area = catalog.area(hotel.area)
        if area is None or area.day_trip or not _passes_common(hotel, request, venue_text(hotel)):
            continue
        mid = hotel.nightly_rate.midpoint_aed
        gap = float(abs(mid - target) / target) if target > 0 else 1.0
        score = 0.8 - 0.5 * min(gap, 1.0)
        if mid > target:
            score -= 0.3 * min(float((mid - target) / target), 1.0)
        if hotel.area in areas:
            score += 0.2
        if hotel.nearest_metro:
            score += 0.1
        if request.travellers.children and hotel.family_friendly:
            score += 0.05
        scored.append((_clamp(score), hotel))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [h.model_copy(update={"score": s}) for s, h in scored[:limit]]


_BAND_ORDER = list(PriceBand)


def food_band_for(request: TripRequest, catalog: SeedCatalog) -> PriceBand:
    """Most expensive band whose daily food cost fits ~30% of the per-person daily budget."""
    allowance = budget_per_person_day(request) * Decimal("0.30")
    rates = food_daily_rates(catalog)
    fitting = [b for b in _BAND_ORDER if rates[b] <= allowance]
    return fitting[-1] if fitting else PriceBand.BUDGET


def search_restaurants(
    request: TripRequest, catalog: SeedCatalog, *, limit: int = 25
) -> list[DiningOption]:
    band = food_band_for(request, catalog)
    target_idx = _BAND_ORDER.index(band)
    areas = resolve_areas(request, catalog)
    scored: list[tuple[float, DiningOption]] = []
    for venue in catalog.restaurants:
        text = venue_text(venue)
        if not venue.caters_for(list(request.dietary)):
            continue  # dietary needs are a hard constraint
        if not _passes_common(venue, request, text):
            continue
        score = 0.5
        distance = abs(_BAND_ORDER.index(venue.price_band) - target_idx)
        score -= 0.15 * distance
        if _BAND_ORDER.index(venue.price_band) > target_idx:
            score -= 0.1
        if venue.area in areas:
            score += 0.2
        if interest_matches(text, request.interests):
            score += 0.1
        if venue.fully_vegetarian and "vegetarian" in request.dietary:
            score += 0.05
        if _name_matches(venue, request.must_include):
            score = 1.0
        scored.append((_clamp(score), venue))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [r.model_copy(update={"score": s}) for s, r in scored[:limit]]
