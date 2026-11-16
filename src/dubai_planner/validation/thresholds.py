"""Planning thresholds shared by the agents (to plan) and the rules (to check).

Keeping them in one place means the composer and the validator can never disagree
about, say, what "midday" is. Values follow ``architecture.md §7`` and
``research.md §4``.
"""

from __future__ import annotations

from datetime import time
from decimal import Decimal

from dubai_planner.models.context import HEAT_THRESHOLD_C
from dubai_planner.models.request import Pace

#: ``pace.density``: maximum sightseeing activities (attractions + events) per day.
PACE_MAX_ACTIVITIES: dict[Pace, int] = {Pace.RELAXED: 3, Pace.MODERATE: 5, Pace.PACKED: 7}
#: What the planner aims for per day (below the maximum, leaving slack for meals/travel).
PACE_TARGET_ACTIVITIES: dict[Pace, int] = {Pace.RELAXED: 2, Pace.MODERATE: 3, Pace.PACKED: 4}
#: Local transport legs per day used for transport budgeting.
PACE_LEGS_PER_DAY: dict[Pace, int] = {Pace.RELAXED: 3, Pace.MODERATE: 4, Pace.PACKED: 5}

#: ``heat.midday_outdoor``: outdoor slots overlapping this window are flagged on hot days.
MIDDAY_START = time(11, 0)
MIDDAY_END = time(16, 0)
#: Months in which the heat rule is an error rather than a warning (May–September).
HOT_MONTHS = frozenset({5, 6, 7, 8, 9})
HEAT_LIMIT_C = HEAT_THRESHOLD_C

#: ``geo.travel_time``: maximum plausible single leg, except for day trips.
MAX_LEG_MINUTES = 90

#: Share of the subtotal kept as contingency by the budget agent.
CONTINGENCY_PCT = Decimal("0.10")
#: Share of the budget the hotel search aims to spend on accommodation.
ACCOMMODATION_SHARE = Decimal("0.40")
#: Adults per hotel room (children share).
ADULTS_PER_ROOM = 2
#: Passengers per taxi.
TAXI_CAPACITY = 4
