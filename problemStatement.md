# Problem Statement — Dubai AI Travel Planner (Multi-Agent System)

## 1. Background

Planning a trip to Dubai means making many decisions that depend on each other. These include attractions, neighbourhoods, hotels, dining, getting around (Metro, tram, taxi, RTA buses), weather (summer temperatures often go above 40 °C), cultural calendars (Ramadan, Eid, Dubai Shopping Festival) and a fixed budget. Today travellers piece this together from several websites by hand. Single-prompt LLM chatbots produce plans that look plausible but often:

- go over the stated budget or leave costs out completely;
- schedule outdoor activities at midday in July;
- ignore travel time between places that are far apart (e.g. Hatta → Dubai Marina);
- invent prices, opening hours or venues;
- cannot be refined step by step ("swap day 2 for something cheaper") without losing context.

## 2. Problem

Design and build a **multi-agent system** that produces a **budget-aware, feasible, explainable, day-by-day Dubai itinerary** from a natural-language request. The system must support review, self-correction and conversational refinement across a persisted session.

The same planning problem must be solved with **two orchestration paradigms** so they can be compared:

| Mode | Paradigm | What it demonstrates |
|------|----------|----------------------|
| **LangGraph mode** | Explicit state machine / graph | Typed shared state, parallel fan-out/fan-in, conditional edges, review loops, checkpointing |
| **CrewAI mode** | Role-based crew | Agents with role/goal/backstory, task delegation, sequential/hierarchical collaboration |

## 3. Objectives

1. **Specialised agents.** Split planning into focused agents: requirements, attractions, accommodation, dining, transport, weather/events, budget, itinerary composition, review.
2. **Graph orchestration with parallelism.** Run independent research agents at the same time in LangGraph and merge their outputs deterministically.
3. **Role-based collaboration.** Rebuild the workflow as a CrewAI crew where agents collaborate through task context and delegation.
4. **Budget awareness.** Every itinerary carries an itemised cost breakdown in AED. It must stay within the user's budget (with a configurable tolerance) or explain why it cannot.
5. **Review and refinement loops.** A reviewer agent checks the itinerary against explicit rules and sends it back for revision, up to a bounded number of iterations.
6. **Session persistence and conversation.** Users can come back to a session and refine it ("make day 3 family-friendly", "cut budget by 20%"). Earlier versions are kept.
7. **Execution visualisation.** The UI shows the agent graph, which agents ran, in what order and in parallel, how long they took, and what they produced.
8. **Evaluation.** Benchmark both modes on a fixed scenario set using objective metrics (budget compliance, constraint satisfaction, feasibility, latency, cost/tokens, iterations) plus optional LLM-as-judge quality scores.

## 4. Users and example requests

- *Budget backpacker:* "4 days in Dubai in March, AED 3,000 total excluding flights, I love street food and history."
- *Family:* "Family of 4 with two kids (6, 9), 5 days in December, mid-range budget AED 12,000, no late nights."
- *Luxury couple:* "Honeymoon, 3 nights, AED 25,000, fine dining, desert safari, beach."
- *Constraint-heavy:* "Vegetarian, wheelchair user, July, 3 days, indoor activities preferred."

## 5. Scope

**In scope**
- Dubai (plus optional day trips to Abu Dhabi, Hatta, Sharjah) itinerary planning.
- Natural-language input parsed into a structured `TripRequest`.
- Estimated costs in AED from curated seed data and free public sources.
- Streamlit web UI, SQLite session store, JSON seed data and caches.
- Pluggable LLM providers: local/free first (Ollama, Groq free tier, Gemini free tier), paid optional, plus an offline mock provider.

**Out of scope**
- Real bookings, payments or live hotel/flight inventory (no free API exists; prices are *estimates* and labelled as such).
- Flights and visa processing (visa info is given as static guidance only).
- Multi-user authentication (single local user, many sessions).

## 6. Constraints

- **Academic assignment**: clarity, reproducibility and documented design decisions matter more than production scale.
- **No paid APIs required**: the whole system must run end to end with a free/local LLM and keyless data sources. The deterministic mock LLM must be able to run the full pipeline offline for tests.
- **No hardcoded secrets**: all credentials come from environment variables or a git-ignored `.env`. A `.env.example` documents them.
- **Respect data source terms**: OSM Nominatim/Overpass usage policies (rate limits, User-Agent), attribution for OpenStreetMap / Wikivoyage (CC BY-SA).
- **Windows-friendly** development (VS Code + Claude Code on Windows 11). Should also run on macOS/Linux.

## 7. Success criteria

| # | Criterion | Measure |
|---|-----------|---------|
| S1 | Valid structured output | 100% of runs produce an `Itinerary` that validates against Pydantic schema |
| S2 | Budget compliance | ≥ 90% of benchmark scenarios within budget + 5% tolerance (or with an explicit infeasibility explanation) |
| S3 | Constraint satisfaction | ≥ 85% of hard constraints (days, dietary, accessibility, pace) satisfied |
| S4 | Parallelism | LangGraph research stage wall-clock < 0.6 × sum of its node durations |
| S5 | Review loop | Reviewer catches injected defects (over-budget, heat-exposed midday outdoor slots) in ≥ 80% of seeded tests |
| S6 | Persistence | Sessions survive app restart; refinements produce new itinerary versions linked to the session |
| S7 | Visualisation | Graph structure plus per-run timeline visible in UI for both modes |
| S8 | Evaluation | Comparative report (tables + charts) across ≥ 10 scenarios for both modes |

## 8. Deliverables

- Source code (Python package plus Streamlit app).
- Documentation: this file, `architecture.md`, `research.md`, `implementationPlan.md`, `CLAUDE.md`.
- Seed data (JSON), SQLite schema, evaluation scenarios and generated evaluation report.
- Test suite (pytest) that runs offline with the mock LLM.
