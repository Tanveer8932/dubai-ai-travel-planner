"""Build the planner ``StateGraph`` (architecture §3).

::

    START → parse_request ─┬─(incomplete)→ ask_clarification ──(interrupt/resume)→ parse_request
                           ├─(gave up)───→ finalize
                           └─(complete)──→ 5 research nodes in parallel → merge_research
    merge_research → allocate_budget → compose_itinerary → validate → review
    review ─(approved, or iteration ≥ MAX_REVIEW_ITERATIONS)→ finalize → END
    review ─(otherwise)→ refine_itinerary → compose_itinerary   (targeted re-plan)

Termination is guaranteed by construction: ``refine_itinerary`` is the only back edge
of the automated loop and it increments ``iteration``, which the router compares with
the cap; clarification rounds wait for a human (``interrupt``) and are capped too. A
``recursion_limit`` derived from the same caps (:func:`recursion_limit`) is a second,
independent stop.

``refine`` goes back to ``compose`` rather than ``allocate_budget``: the budget agent is
deterministic for an unchanged request and research, so re-running it would only cost
an LLM call (its rationale) and could not change the allocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from dubai_planner.llm.errors import LLMCallError
from dubai_planner.models import Severity
from dubai_planner.orchestration.langgraph import nodes as n
from dubai_planner.orchestration.langgraph.state import PlannerState

#: Super-steps outside the review loop: parse, research, merge, budget, compose,
#: validate, review, finalize (+ margin).
_BASE_STEPS = 12
_STEPS_PER_REVISION = 4  # refine, compose, validate, review
_STEPS_PER_CLARIFICATION = 2  # ask_clarification, parse_request

_TRANSIENT_MARKERS = ("429", "rate limit", "ratelimit", "rate_limit", "timeout", "timed out",
                      "502", "503", "504", "overloaded", "temporarily unavailable",
                      "connection")


def is_transient(exc: BaseException) -> bool:
    """Provider hiccups worth retrying (rate limits, timeouts, 5xx); never bad output."""
    if not isinstance(exc, LLMCallError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def default_retry_policy(max_attempts: int = 4, initial_interval: float = 8.0) -> RetryPolicy:
    """Bounded node-level retry for transient LLM errors (on top of the SDK's own).

    Waits ≈ 8 s, 16 s, 32 s: together about one minute, the window of per-minute token
    limits on free tiers (e.g. Groq's 8 000 TPM; one full run sends ≈ 11 000 prompt
    tokens, three research calls of them at once).
    """
    return RetryPolicy(max_attempts=max_attempts, initial_interval=initial_interval,
                       backoff_factor=2.0, max_interval=60.0, jitter=True,
                       retry_on=is_transient)


def recursion_limit(max_review_iterations: int, max_clarifications: int) -> int:
    return (_BASE_STEPS + _STEPS_PER_REVISION * max_review_iterations
            + _STEPS_PER_CLARIFICATION * (max_clarifications + 1))


# --------------------------------------------------------------------------- routers


def route_after_parse(state: PlannerState, *, max_clarifications: int) -> list[str] | str:
    if state.get("request") is not None:
        return list(n.RESEARCH_NODES)  # fan-out: same super-step → run in parallel
    if state.get("clarifications", 0) >= max_clarifications:
        return n.FINALIZE  # stop asking; the run ends as needs_clarification
    return n.CLARIFY


def route_after_review(state: PlannerState, *, max_review_iterations: int) -> str:
    """``finalize`` when approved (and no error-severity rule fails) or the cap is hit."""
    review = state.get("review")
    hard_errors = any(v.severity is Severity.ERROR for v in state.get("validation", []))
    if review is not None and review.approved and not hard_errors:
        return n.FINALIZE
    if state.get("iteration", 0) >= max_review_iterations:
        return n.FINALIZE
    return n.REFINE


# --------------------------------------------------------------------------- builder


def build_graph(deps: n.PlannerDeps, *, checkpointer: Any = None,
                retry: RetryPolicy | None = None) -> Any:
    """Compile the planner graph. ``checkpointer`` is required for interrupts/resume."""
    nodes = n.PlannerNodes(deps)
    retry = retry if retry is not None else default_retry_policy()
    g: StateGraph[PlannerState, Any, PlannerState, PlannerState] = StateGraph(PlannerState)

    llm_nodes = {
        n.PARSE: nodes.parse_request,
        n.RESEARCH_ATTRACTIONS: nodes.research_attractions,
        n.RESEARCH_ACCOMMODATION: nodes.research_accommodation,
        n.RESEARCH_DINING: nodes.research_dining,
        n.BUDGET: nodes.allocate_budget,
        n.COMPOSE: nodes.compose_itinerary,
        n.REVIEW: nodes.review,
    }
    for name, fn in llm_nodes.items():
        g.add_node(name, fn, retry_policy=retry)
    g.add_node(n.CLARIFY, nodes.ask_clarification)
    g.add_node(n.RESEARCH_TRANSPORT, nodes.research_transport)
    g.add_node(n.RESEARCH_WEATHER, nodes.research_weather_events)
    g.add_node(n.MERGE, nodes.merge_research)
    g.add_node(n.VALIDATE, nodes.validate)
    g.add_node(n.REFINE, nodes.refine_itinerary)
    g.add_node(n.FINALIZE, nodes.finalize)

    g.add_edge(START, n.PARSE)

    def after_parse(state: PlannerState) -> list[str] | str:
        return route_after_parse(state, max_clarifications=deps.max_clarifications)

    g.add_conditional_edges(n.PARSE, after_parse,
                            [*n.RESEARCH_NODES, n.CLARIFY, n.FINALIZE])
    g.add_edge(n.CLARIFY, n.PARSE)
    g.add_edge(list(n.RESEARCH_NODES), n.MERGE)  # fan-in waits for all five branches
    g.add_edge(n.MERGE, n.BUDGET)
    g.add_edge(n.BUDGET, n.COMPOSE)
    g.add_edge(n.COMPOSE, n.VALIDATE)
    g.add_edge(n.VALIDATE, n.REVIEW)

    def after_review(state: PlannerState) -> str:
        return route_after_review(state, max_review_iterations=deps.max_review_iterations)

    g.add_conditional_edges(n.REVIEW, after_review, [n.REFINE, n.FINALIZE])
    g.add_edge(n.REFINE, n.COMPOSE)
    g.add_edge(n.FINALIZE, END)
    return g.compile(checkpointer=checkpointer, name="dubai_planner")


def draw_mermaid(graph: Any) -> str:
    """Mermaid source of the compiled graph (for the report and the Phase 6 viewer)."""
    return str(graph.get_graph().draw_mermaid())


def export_mermaid(graph: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(draw_mermaid(graph), encoding="utf-8")
    return path
