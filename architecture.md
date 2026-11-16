# Architecture — Dubai AI Travel Planner

## 1. Design principles

1. **One domain, two orchestrators.** Agents' *capabilities* (prompts, tools, Pydantic I/O contracts) live in a framework-agnostic core. LangGraph and CrewAI are thin orchestration adapters over the same core, so the evaluation compares orchestration and not different business logic.
2. **Structured everything.** Every agent output is a Pydantic model. LLM text is parsed and validated, and retried on validation failure.
3. **Deterministic where possible.** Budget arithmetic, distance/travel-time estimates, heat rules and schema checks are plain Python, not LLM calls. LLMs pick, rank, write and explain.
4. **Free and offline first.** Keyless public APIs with on-disk JSON caching, curated seed data as fallback, and a mock LLM provider so the full pipeline runs with no network.
5. **Observable by construction.** Every agent step emits a `TraceEvent` to one event bus, which feeds SQLite and the UI visualisation in both modes.

## 2. High-level system diagram

```mermaid
flowchart TB
    subgraph UI["Presentation layer — Streamlit"]
        U1["Chat / Planner page"]
        U2["Itinerary view<br/>day cards, map, cost table"]
        U3["Execution visualiser<br/>graph, timeline, agent I/O"]
        U4["Sessions page<br/>history and versions"]
        U5["Evaluation dashboard"]
    end

    subgraph SVC["Application services"]
        S1["PlannerService<br/>mode = langgraph or crewai"]
        S2["SessionService"]
        S3["RefinementService<br/>diff-aware follow-ups"]
        S4["EvaluationRunner"]
        BUS["TraceEventBus"]
    end

    subgraph ORCH["Orchestration adapters"]
        LG["LangGraph adapter<br/>StateGraph + SqliteSaver"]
        CR["CrewAI adapter<br/>Crew, Agents, Tasks"]
    end

    subgraph CORE["Framework-agnostic agent core"]
        A0["RequirementsAgent"]
        A1["AttractionsAgent"]
        A2["AccommodationAgent"]
        A3["DiningAgent"]
        A4["TransportAgent"]
        A5["WeatherEventsAgent"]
        A6["BudgetAgent"]
        A7["ItineraryComposerAgent"]
        A8["ReviewerAgent"]
        RULES["Deterministic validators<br/>budget, heat, geo, hours, schema"]
    end

    subgraph LLM["LLM provider layer"]
        F["LLMFactory<br/>env-configured"]
        P1["Ollama — local, free"]
        P2["Groq — free tier"]
        P3["Gemini — free tier"]
        P4["OpenAI / Anthropic — optional"]
        P5["MockLLM — offline tests"]
    end

    subgraph DATA["Data layer"]
        T["Tool registry"]
        C["HTTP client + JSON cache"]
        SEED[("Seed JSON<br/>attractions, hotels, dining,<br/>transport fares, events")]
        DB[("SQLite<br/>sessions, messages,<br/>itineraries, traces, evals,<br/>LangGraph checkpoints")]
    end

    subgraph EXT["Free public sources"]
        E1["Open-Meteo<br/>forecast + climate"]
        E2["OSM Nominatim / Overpass"]
        E3["Wikivoyage / Wikipedia API"]
        E4["Frankfurter FX"]
        E5["Dubai Pulse / RTA<br/>open data, static fares"]
    end

    U1 --> S1
    U1 --> S3
    U4 --> S2
    U5 --> S4
    S1 --> LG
    S1 --> CR
    S3 --> S1
    S4 --> S1
    LG --> CORE
    CR --> CORE
    CORE --> F
    F --> P1 & P2 & P3 & P4 & P5
    CORE --> T
    T --> C
    T --> SEED
    C --> EXT
    S2 --> DB
    LG -. checkpoints .-> DB
    LG -- events --> BUS
    CR -- events --> BUS
    BUS --> DB
    BUS --> U3
    S1 --> U2
```

## 3. LangGraph orchestration graph

```mermaid
flowchart TD
    START((START)) --> parse["parse_request<br/>RequirementsAgent"]
    parse --> gate{"request<br/>complete?"}
    gate -- "no: missing dates/budget" --> clarify["ask_clarification<br/>interrupt for human input"]
    clarify --> parse
    gate -- yes --> fan(["fan-out"])

    fan --> attr["research_attractions"]
    fan --> hotel["research_accommodation"]
    fan --> dine["research_dining"]
    fan --> trans["research_transport"]
    fan --> wx["research_weather_events"]

    attr --> join(["fan-in / merge reducer"])
    hotel --> join
    dine --> join
    trans --> join
    wx --> join

    join --> budget["allocate_budget<br/>BudgetAgent + deterministic calc"]
    budget --> compose["compose_itinerary<br/>ItineraryComposerAgent"]
    compose --> validate["validate<br/>deterministic rules"]
    validate --> review["review<br/>ReviewerAgent"]
    review --> decide{"approved OR<br/>iteration ≥ MAX?"}
    decide -- "revise" --> refine["refine_itinerary<br/>targeted fixes from review"]
    refine --> budget
    decide -- "approve" --> finalize["finalize<br/>cost summary, explanation, persist"]
    finalize --> END((END))
```

**State** (`PlannerState`, a `TypedDict` with Pydantic-typed values):

| Key | Type | Reducer |
|-----|------|---------|
| `session_id` | `str` | overwrite |
| `messages` | `list[ChatMessage]` | `add_messages` (append) |
| `request` | `TripRequest \| None` | overwrite |
| `research` | `ResearchBundle` (attractions, hotels, dining, transport, weather, events) | custom merge per key, so parallel branches never collide |
| `budget_plan` | `BudgetPlan` | overwrite |
| `itinerary` | `Itinerary` | overwrite |
| `validation` | `list[RuleViolation]` | overwrite |
| `review` | `ReviewReport` | overwrite |
| `iteration` | `int` | overwrite (incremented by `refine`) |
| `trace` | `list[TraceEvent]` | `operator.add` (append) |

**Parallelism:** the five research nodes come from one fan-out (static edges, or the `Send` API when the number of branches depends on the request, such as one research task per day-trip destination). LangGraph runs them in the same super-step. Their results merge through reducers. Measured speed-up is recorded in the trace.

**Checkpointing:** `SqliteSaver` with `thread_id = session_id`, which enables resume, time travel ("show the version before my last change") and human-in-the-loop clarification through `interrupt`.

## 4. CrewAI collaboration design

```mermaid
flowchart LR
    subgraph Crew["Dubai Planning Crew"]
        direction LR
        M["Trip Manager<br/>manager agent, hierarchical mode"]
        R0["Requirements Analyst"]
        R1["Attractions Specialist"]
        R2["Accommodation Specialist"]
        R3["Dining Specialist"]
        R4["Transport & Logistics Specialist"]
        R5["Climate & Events Analyst"]
        R6["Budget Controller"]
        R7["Itinerary Architect"]
        R8["Quality Reviewer"]
    end
    M -. delegates .-> R0 & R1 & R2 & R3 & R4 & R5 & R6 & R7 & R8
    R0 -->|"TripRequest"| R1 & R2 & R3 & R4 & R5
    R1 & R2 & R3 & R4 & R5 -->|"task context"| R6
    R6 -->|"BudgetPlan"| R7
    R7 -->|"Itinerary"| R8
    R8 -->|"guardrail fail: feedback"| R7
```

- **Agents** have `role`, `goal`, `backstory`, a tool list (wrapping the shared tool registry) and an LLM from `LLMFactory` (LiteLLM model strings, e.g. `ollama/llama3.1`, `groq/llama-3.1-8b-instant`, `gemini/gemini-2.0-flash`).
- **Tasks** declare `expected_output`, `output_pydantic` (same models as LangGraph mode) and `context=[...]` dependencies. Research tasks use `async_execution=True` so they run concurrently.
- **Process**: `Process.sequential` by default (predictable, cheaper). `Process.hierarchical` with a manager agent is a configurable variant evaluated separately.
- **Review loop**: the Itinerary Architect task gets a **task guardrail** that runs deterministic validators plus the Reviewer. A failure returns feedback and CrewAI retries the task (`guardrail_max_retries`). An outer loop in the adapter re-kicks off the compose/review tasks if needed. Both modes share `MAX_REVIEW_ITERATIONS`.
- **Observability**: `step_callback` / `task_callback` plus CrewAI's event listener map to the same `TraceEvent` schema.

## 5. Agent catalogue

| Agent | Input | Output model | Tools | LLM? |
|-------|-------|--------------|-------|------|
| Requirements | user message + session history | `TripRequest` | — | yes (extraction) |
| Attractions | `TripRequest` | `list[AttractionOption]` | seed search, Overpass POIs, Wikivoyage summary | yes (rank/select) |
| Accommodation | `TripRequest` | `list[HotelOption]` | seed hotels by area/price band | yes (rank) |
| Dining | `TripRequest` | `list[DiningOption]` | seed restaurants, Overpass amenity=restaurant | yes (rank) |
| Transport | `TripRequest`, areas | `TransportPlan` | RTA fare table, haversine travel-time estimator | mostly deterministic |
| Weather & Events | dates | `WeatherContext`, `list[EventInfo]` | Open-Meteo forecast/climate, events seed, holiday calendar | light |
| Budget | request + research | `BudgetPlan` (per-category allocation in AED) | currency converter, cost calculator | deterministic + LLM rationale |
| Itinerary Composer | everything above | `Itinerary` (days → time slots) | travel-time estimator | yes |
| Reviewer | `Itinerary`, `TripRequest`, violations | `ReviewReport` (approved, issues[], severity, suggested fixes) | validators | yes |

## 6. Core domain models (Pydantic v2)

```mermaid
classDiagram
    class TripRequest {
        start_date: date
        end_date: date
        travellers: Travellers
        budget_aed: Decimal
        budget_scope: BudgetScope
        interests: list~str~
        pace: Pace
        dietary: list~str~
        accessibility: list~str~
        preferred_areas: list~str~
        must_include: list~str~
        avoid: list~str~
    }
    class Itinerary {
        id: UUID
        session_id: str
        version: int
        mode: Literal
        days: list~DayPlan~
        hotel: HotelOption
        cost: CostBreakdown
        notes: list~str~
    }
    class DayPlan {
        date: date
        theme: str
        slots: list~TimeSlot~
        day_cost_aed: Decimal
    }
    class TimeSlot {
        start: time
        end: time
        activity: Activity
        transport_from_prev: TransportLeg
        cost_aed: Decimal
        indoor: bool
    }
    class CostBreakdown {
        accommodation: Decimal
        activities: Decimal
        food: Decimal
        transport: Decimal
        contingency: Decimal
        total: Decimal
        within_budget: bool
    }
    class ReviewReport {
        approved: bool
        score: float
        issues: list~ReviewIssue~
    }
    class TraceEvent {
        run_id: str
        mode: str
        agent: str
        event: str
        started_at: datetime
        ended_at: datetime
        tokens_in: int
        tokens_out: int
        payload_ref: str
    }
    TripRequest --> Itinerary : planned into
    Itinerary "1" *-- "many" DayPlan
    DayPlan "1" *-- "many" TimeSlot
    Itinerary *-- CostBreakdown
    Itinerary --> ReviewReport : reviewed by
```

**Data provenance (as built, Phase 1).** Every externally sourced fact carries a `Provenance` (`kind`, `source`, `source_url`, `as_of`, `licence`, `note`). `DataKind` is one of:

| Kind | Meaning | Example |
|------|---------|---------|
| `verified` | checked against an authoritative source; requires `source_url` | AED/USD peg 3.6725 |
| `api` | returned by a live public API on `as_of` | Open-Meteo forecast, Nominatim geocode, Frankfurter rates |
| `curated` | compiled by hand from public info, not re-checked live | venue location, category, indoor/outdoor |
| `estimated` | approximation | every seed price (`PriceRange`), event windows, Hijri-computed holidays, last-year weather proxy |

Seed prices are always `estimated` (enforced by `tests/unit/test_seed.py`), and `PriceRange.label()` renders them as `≈ AED x–y (estimate, as of …)`. Money is the `AED` annotated type: `Decimal`, ≥ 0, quantised to 2 dp half-up. Venue activities (`attraction`, `dining`, `hotel`) must carry a `ref_id`, which grounds them at the contract level. `CostBreakdown` may hold an inconsistent total on purpose, so the `budget.arithmetic` rule can *report* it instead of the pipeline crashing.

## 7. Deterministic validators (shared by both modes)

| Rule | Severity | Description |
|------|----------|-------------|
| `budget.total` | error | `cost.total ≤ budget × (1 + BUDGET_TOLERANCE)` |
| `budget.arithmetic` | error | category sums equal total; day costs sum correctly |
| `time.overlap` | error | no overlapping slots within a day |
| `time.opening_hours` | warning | activity within known opening hours (where data exists) |
| `heat.midday_outdoor` | warning→error (May–Sep) | no outdoor slot 11:00–16:00 when forecast/climatology max > 38 °C |
| `geo.travel_time` | warning | travel leg time plausible; ≤ 90 min per leg unless day trip |
| `pace.density` | warning | slots/day match `pace` (relaxed ≤ 3, moderate ≤ 5, packed ≤ 7) |
| `constraints.dietary` | error | dining options honour dietary flags |
| `constraints.accessibility` | warning | wheelchair-flagged activities only when required |
| `culture.ramadan` | info | daytime eating/drinking etiquette notes when dates overlap Ramadan |
| `grounding.known_entities` | warning | every venue resolves to seed data or a cited source (anti-hallucination) |

## 8. Persistence

**SQLite** (`data/app.db`, created by migrations in `storage/schema.sql`):

```mermaid
erDiagram
    SESSIONS ||--o{ MESSAGES : has
    SESSIONS ||--o{ ITINERARIES : has
    SESSIONS ||--o{ RUNS : has
    RUNS ||--o{ TRACE_EVENTS : emits
    ITINERARIES ||--o| ITINERARIES : "parent_version"
    EVAL_RUNS ||--o{ EVAL_RESULTS : contains
    SESSIONS {
        text id PK
        text title
        text mode
        text created_at
        text updated_at
    }
    MESSAGES {
        int id PK
        text session_id FK
        text role
        text content
        text created_at
    }
    ITINERARIES {
        text id PK
        text session_id FK
        int version
        text parent_id FK
        text mode
        text request_json
        text itinerary_json
        real total_aed
        int approved
    }
    RUNS {
        text id PK
        text session_id FK
        text mode
        text status
        real duration_s
        int llm_calls
        int tokens
    }
    TRACE_EVENTS {
        int id PK
        text run_id FK
        text agent
        text event
        text started_at
        text ended_at
        text payload_json
    }
    EVAL_RUNS {
        text id PK
        text created_at
        text config_json
    }
    EVAL_RESULTS {
        int id PK
        text eval_run_id FK
        text scenario_id
        text mode
        text metrics_json
    }
```

LangGraph checkpoints go to a separate `data/checkpoints.db` (owned by `langgraph-checkpoint-sqlite`).

*As built (Phase 1):* `itineraries` also stores `run_id` and `created_at`, with `UNIQUE(session_id, version)`; `messages` has `metadata_json`; `runs` has `started_at`/`ended_at`/`error`; `eval_results` has `seed`. Child rows use `ON DELETE CASCADE`. The schema version lives in `PRAGMA user_version`.

**JSON**: `data/seed/*.json` (curated, versioned, cites sources; as built: `attractions`, `hotels`, `restaurants`, `areas`, `transport`, `events`, `climate`, `holidays`, `fx_rates`, each `{meta, items|data}`), `data/cache/*.json` (HTTP response cache with TTL, git-ignored), `data/exports/*.json|md` (exported itineraries), `eval/scenarios/*.json`.

## 9. Conversational refinement flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit
    participant RS as RefinementService
    participant PS as PlannerService
    participant G as Orchestrator (LG or Crew)
    participant DB as SQLite

    User->>UI: "Make day 2 cheaper and add a desert safari"
    UI->>RS: refine(session_id, message)
    RS->>DB: load latest Itinerary + TripRequest + history
    RS->>RS: classify change (request delta vs local edit)
    alt request delta (dates, budget, travellers)
        RS->>PS: replan(updated TripRequest)
    else local edit (specific days/slots)
        RS->>PS: partial replan(locked days, edit instructions)
    end
    PS->>G: invoke(state, thread_id=session_id)
    G-->>UI: streamed TraceEvents (live visualiser)
    G-->>PS: Itinerary v(n+1) + ReviewReport
    PS->>DB: save version n+1 (parent = n), messages, run
    PS-->>UI: render diff v(n) → v(n+1)
```

## 10. LLM provider configuration

Selected by environment variables only (see `.env.example`):

```
LLM_PROVIDER=ollama|groq|gemini|openai|anthropic|mock
LLM_MODEL=llama3.1:8b
LLM_TEMPERATURE=0.2
LLM_BASE_URL=http://localhost:11434   # Ollama
GROQ_API_KEY= / GEMINI_API_KEY= / OPENAI_API_KEY= / ANTHROPIC_API_KEY=
```

`LLMFactory` returns (a) a LangChain `BaseChatModel` for LangGraph nodes and (b) a CrewAI `LLM` (LiteLLM string) for crews, both from the same settings. Per-agent overrides are possible (e.g. a smaller model for extraction). `MockLLM` returns fixture-based structured outputs so tests and CI are deterministic and free.

## 11. Repository layout

```
dubai-ai-travel-planner/
├── CLAUDE.md, problemStatement.md, architecture.md, research.md, implementationPlan.md
├── README.md                      # quick start (Phase 0)
├── pyproject.toml                 # deps + tool config (Phase 0)
├── .env.example                   # documented env vars, no secrets
├── .gitignore
├── app/                           # Streamlit entry + pages
│   ├── streamlit_app.py
│   └── pages/                     # 1_Planner, 2_Sessions, 3_Execution, 4_Evaluation
├── src/dubai_planner/
│   ├── config.py                  # pydantic-settings
│   ├── models/                    # Pydantic domain models
│   ├── llm/                       # factory, providers, mock, structured-output helpers
│   ├── data/                      # source clients, cache, seed loaders
│   ├── tools/                     # framework-agnostic tool functions
│   ├── agents/                    # agent specs: prompts + I/O contracts + run functions
│   ├── validation/                # deterministic rules
│   ├── orchestration/
│   │   ├── langgraph/             # state, nodes, graph builder
│   │   └── crewai/                # agents.yaml, tasks.yaml, crew builder, guardrails
│   ├── services/                  # planner, session, refinement
│   ├── storage/                   # sqlite repo + schema.sql
│   ├── tracing/                   # TraceEvent bus, adapters for both frameworks
│   └── evaluation/                # metrics, judge, runner, report
├── data/
│   ├── seed/                      # curated JSON (committed)
│   ├── cache/                     # API cache (ignored)
│   └── exports/                   # (ignored)
├── eval/
│   ├── scenarios/                 # benchmark requests (JSON)
│   └── reports/                   # generated reports
├── scripts/                       # seed refresh, run_eval, init_db
└── tests/
    ├── unit/  integration/  fixtures/
```

## 12. Key trade-offs

| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| Shared agent core | yes | separate implementations per framework | fair comparison, less duplication |
| Budget maths | deterministic Python | LLM | LLMs are unreliable at arithmetic |
| Default LLM | Ollama local | hosted paid | free, private, reproducible; slower |
| Hotel/restaurant prices | curated seed bands | scraping booking sites | ToS and legality; no free API |
| Persistence | SQLite + JSON | Postgres/Redis | zero-ops, single file, enough for one user |
| Visualisation | Mermaid from compiled graph + Plotly timeline | LangSmith | free, offline, works for both frameworks |
