"""Budget data structures: category allocations (plan) and itemised costs (actuals).

These are pure data structures with exact ``Decimal`` arithmetic helpers. The
allocation *policy* (how much to give each category) is the Budget agent's job
(Phase 2); the models only guarantee consistent, non-negative AED amounts.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from dubai_planner.models.common import AED, DataKind, StrictModel, quantize_aed
from dubai_planner.models.request import BudgetScope

ZERO = Decimal("0")


class BudgetCategory(StrEnum):
    ACCOMMODATION = "accommodation"
    ACTIVITIES = "activities"
    FOOD = "food"
    TRANSPORT = "transport"
    CONTINGENCY = "contingency"


class CategoryAllocation(StrictModel):
    category: BudgetCategory
    amount_aed: AED
    basis: str = Field(default="", description="How the amount was derived")


class BudgetPlan(StrictModel):
    """Per-category allocation of the total trip budget (AED)."""

    budget_aed: AED = Field(description="Total budget for the whole trip, all travellers")
    scope: BudgetScope = BudgetScope.TOTAL
    allocations: list[CategoryAllocation]
    contingency_pct: Decimal = Field(default=Decimal("0.10"), ge=0, le=Decimal("0.5"))
    feasible: bool = True
    infeasibility_reason: str | None = None
    rationale: str = ""
    currency: Literal["AED"] = "AED"

    @model_validator(mode="after")
    def _consistent(self) -> BudgetPlan:
        cats = [a.category for a in self.allocations]
        if len(cats) != len(set(cats)):
            raise ValueError("each budget category may be allocated only once")
        if not self.feasible and not self.infeasibility_reason:
            raise ValueError("an infeasible budget plan must explain why")
        return self

    def allocation(self, category: BudgetCategory) -> Decimal:
        return next((a.amount_aed for a in self.allocations if a.category is category), ZERO)

    @property
    def allocated_total_aed(self) -> Decimal:
        return quantize_aed(sum((a.amount_aed for a in self.allocations), ZERO))

    @property
    def unallocated_aed(self) -> Decimal:
        """Positive = money left; negative = allocations exceed the budget."""
        return quantize_aed(self.budget_aed - self.allocated_total_aed)


class CostItem(StrictModel):
    """One line of the itemised cost breakdown."""

    category: BudgetCategory
    description: str
    amount_aed: AED
    ref_id: str | None = Field(default=None, description="Seed id of the venue, if any")
    day_index: int | None = Field(default=None, ge=0)
    kind: DataKind = DataKind.ESTIMATED


class CostBreakdown(StrictModel):
    """Itemised trip cost. ``total`` should equal the category sum; the
    ``budget.arithmetic`` validator rule (Phase 2) checks that, so a broken breakdown
    can still be represented and reported rather than crash the pipeline."""

    accommodation: AED = ZERO
    activities: AED = ZERO
    food: AED = ZERO
    transport: AED = ZERO
    contingency: AED = ZERO
    total: AED = ZERO
    budget_aed: AED | None = None
    tolerance: Decimal = Field(default=Decimal("0.05"), ge=0, le=1)
    within_budget: bool = True
    items: list[CostItem] = Field(default_factory=list)
    currency: Literal["AED"] = "AED"
    is_estimate: bool = Field(
        default=True, description="True unless every item is verified; shown in the UI"
    )

    @classmethod
    def from_items(
        cls,
        items: list[CostItem],
        *,
        budget_aed: Decimal | None = None,
        tolerance: Decimal = Decimal("0.05"),
    ) -> CostBreakdown:
        """Build a consistent breakdown (category sums, total, within_budget) from items."""
        sums = {cat: ZERO for cat in BudgetCategory}
        for item in items:
            sums[item.category] += item.amount_aed
        total = sum(sums.values(), ZERO)
        within = True if budget_aed is None else total <= budget_aed * (1 + tolerance)
        return cls(
            accommodation=sums[BudgetCategory.ACCOMMODATION],
            activities=sums[BudgetCategory.ACTIVITIES],
            food=sums[BudgetCategory.FOOD],
            transport=sums[BudgetCategory.TRANSPORT],
            contingency=sums[BudgetCategory.CONTINGENCY],
            total=total,
            budget_aed=budget_aed,
            tolerance=tolerance,
            within_budget=within,
            items=items,
            is_estimate=any(i.kind is not DataKind.VERIFIED for i in items) or not items,
        )

    @property
    def category_sum(self) -> Decimal:
        return quantize_aed(
            self.accommodation + self.activities + self.food + self.transport + self.contingency
        )

    @property
    def is_consistent(self) -> bool:
        """Category amounts add up to ``total`` (and items, if any, add up too)."""
        if self.category_sum != self.total:
            return False
        if self.items:
            return quantize_aed(sum((i.amount_aed for i in self.items), ZERO)) == self.total
        return True

    @property
    def overshoot_aed(self) -> Decimal | None:
        """Amount over budget (negative = under). None when no budget is attached."""
        if self.budget_aed is None:
            return None
        return quantize_aed(self.total - self.budget_aed)
