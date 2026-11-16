# Research Notes — Dubai AI Travel Planner

> Status: pre-implementation research (Phase 0). Items marked **[verify]** must be checked against current docs or endpoints during the phase that first uses them. APIs, free tiers and framework APIs change often.

## 1. Orchestration frameworks

### 1.1 LangGraph
- **Model:** `StateGraph` over a typed state (TypedDict / Pydantic). Nodes are functions `state -> partial update`. Edges can be static or conditional. The graph is compiled before running.
- **Parallelism:** nodes reachable from the same super-step run concurrently. Fan-out is done with several edges from one node, or dynamically with `Send(node, payload)` from a conditional edge (map-reduce). Concurrent writes to the same key need a **reducer** (`Annotated[list, operator.add]` or a custom merge), or LangGraph raises `InvalidUpdateError`.
- **Loops:** conditional edges back to earlier nodes plus an iteration counter in state. The `recursion_limit` config is a safety net.
- **Persistence:** checkpointers (`MemorySaver`, `SqliteSaver` / `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite`). `thread_id` identifies a conversation. Supports `get_state_history` (time travel) and `interrupt()` + `Command(resume=...)` for human-in-the-loop.
- **Streaming:** `graph.stream(..., stream_mode="updates"|"values"|"debug"|"custom")` gives per-node updates, which feed the live visualiser.
- **Visualisation:** `graph.get_graph().draw_mermaid()` gives Mermaid source; `draw_mermaid_png()` needs network/pyppeteer, so we prefer the Mermaid text rendered in the browser.
- **Fit:** explicit control flow, deterministic routing, reproducible loops. Best for showing orchestration and parallel execution.

### 1.2 CrewAI
- **Model:** `Agent(role, goal, backstory, tools, llm, allow_delegation)`, `Task(description, expected_output, agent, context, output_pydantic, async_execution, guardrail)`, `Crew(agents, tasks, process)`.
- **Processes:** `Process.sequential` (tasks in order, earlier outputs passed as context) and `Process.hierarchical` (a manager LLM/agent plans and delegates). Hierarchical mode costs more LLM calls and is less predictable.
- **Parallelism:** tasks with `async_execution=True` run concurrently. A later task that lists them in `context` waits for them. *Verified (1.15.41):* a crew may end with at most one async task, and an async task may not list an immediately preceding async task in its `context`; async tasks run in threads with `contextvars.copy_context()`. In hierarchical mode one manager executor runs every task, so concurrent tasks are not safe there.
- **Structured output:** `output_pydantic=Model` → `task.output.pydantic`. Local models sometimes produce invalid JSON, so we add a repair/validate-retry wrapper.
- **Review loops:** task **guardrails** (function returning `(ok, result_or_feedback)`) with retry. CrewAI **Flows** (`@start`, `@listen`, `@router`) are an option for an explicit outer loop. *Verified (1.15.41):* `guardrail_max_retries` (`max_retries` is deprecated). A guardrail must be a function or bound method returning `(bool, Any)`; a success value may be a whole `TaskOutput`.
- **LLM:** `crewai.LLM(model="provider/model", base_url=..., api_key=...)` via LiteLLM. Supports `ollama/…`, `groq/…`, `gemini/…`, `openai/…`, `anthropic/…`.
- **Observability:** `step_callback`, `task_callback`, and the event bus (*verified:* `crewai.events`, handlers run on a thread pool, `crewai_event_bus.flush()` waits for them) for `TaskStartedEvent`, `AgentExecutionCompletedEvent`, `ToolUsageFinishedEvent`, etc.
- **Telemetry:** CrewAI sends anonymous telemetry by default. Set `CREWAI_DISABLE_TELEMETRY=true` (and `OTEL_SDK_DISABLED=true`) in `.env.example` for privacy and offline runs. *Phase 4:* also `CREWAI_TRACING_ENABLED=false`, else a first-run trace prompt blocks each kickoff for 20 s.
- **Fit:** fast to describe collaborating "personas"; less explicit control flow. Best for showing role-based collaboration and delegation.

### 1.3 Comparison hypotheses (to test in Phase 7)
| Dimension | Expectation |
|-----------|-------------|
| Latency | LangGraph lower (fewer LLM calls; deterministic routing) |
| LLM calls / tokens | CrewAI higher (ReAct-style agent loops, manager overhead in hierarchical mode) |
| Output validity | LangGraph ≥ CrewAI with small local models |
| Narrative quality | CrewAI comparable or higher (richer role prompts) |
| Developer effort for loops | LangGraph lower (native cycles); CrewAI needs guardrails/Flows |

## 2. LLM providers (free / low-cost first)

| Provider | Cost | Key needed | Notes |
|----------|------|-----------|-------|
| **Ollama** (local) | free | no | Default. `llama3.1:8b`, `qwen2.5:7b` (good JSON/tool use), `mistral`. Needs ~8 GB RAM; slower on CPU. |
| **Groq** | free tier, rate-limited | yes (`GROQ_API_KEY`) | Very fast Llama/Qwen models; good for demos and eval runs. **[verify]** current model IDs and limits. |
| **Google Gemini** | free tier, rate-limited | yes (`GEMINI_API_KEY`) | Flash models; generous context. **[verify]** model IDs. |
| **OpenRouter** | some free models | yes | Optional; OpenAI-compatible. |
| OpenAI / Anthropic | paid | yes | Optional only, never required. |
| **Mock** | free | no | Fixture-backed deterministic responses for tests/CI. |

Implementation notes:
- LangChain integrations: `langchain-ollama`, `langchain-groq`, `langchain-google-genai`, `langchain-openai`, `langchain-anthropic`, imported lazily so only the chosen provider's package is needed.
- Structured output: prefer `llm.with_structured_output(Model)` where the provider supports it. Otherwise use a JSON-mode prompt + `Model.model_validate_json` + one repair retry.
- Rate limiting: a token-bucket wrapper plus exponential backoff for free tiers, especially during evaluation runs.

## 3. Free / public Dubai data sources

| Data | Source | Access | Licence / policy | Use |
|------|--------|--------|------------------|-----|
| Weather forecast (≤16 days) | **Open-Meteo** Forecast API | keyless REST | free for non-commercial; attribution | daily max temp, precipitation, UV for heat rules |
| Climate normals (trips beyond forecast range) | **Open-Meteo** Historical/Climate API | keyless | as above | month-typical temps for future dates |
| POIs (attractions, restaurants, metro stations) | **OpenStreetMap Overpass API** | keyless, fair use | ODbL, attribution required | coordinates, opening_hours, cuisine, wheelchair tags |
| Geocoding | **Nominatim** | keyless | max 1 req/s, custom User-Agent required, cache results | resolve user-named places |
| Descriptions / travel advice | **Wikivoyage** + **Wikipedia** (MediaWiki API) | keyless | CC BY-SA, attribution; User-Agent **must include contact info** (403 otherwise, checked 2026-11-15) | short descriptions, district guides, etiquette (`prop=extracts` works on en.wikivoyage) |
| Currency | **Frankfurter** (ECB rates) | keyless | free | ECB doesn't publish AED. Convert via USD using the fixed peg **1 USD = 3.6725 AED**. Endpoint in use: `api.frankfurter.dev/v1/latest?base=USD` (checked 2026-11-15) |
| Public transport fares | **RTA** published Nol fare tables | static (manually curated into seed JSON) | public info | Metro/tram/bus fare by zones and card class **[verify current fares]** |
| Open government data | **Dubai Pulse / Dubai Data** portal | some datasets downloadable | per-dataset licence **[verify portal URL/status]** | tourism stats, transport stations (optional enrichment) |
| Events / seasons | Curated seed (Dubai Shopping Festival, Dubai Food Festival, Global Village season, NYE) | static | cite official pages | event-aware suggestions |
| Islamic calendar (Ramadan / Eid) | `hijridate` (formerly `hijri-converter`) Python lib | offline | MIT | approximate Ramadan window; flagged as approximate (moon sighting) |
| Public holidays | Nager.Date API | keyless | **Checked 2026-11-15: no UAE coverage** (HTTP 204, `AE` not listed). Holidays come from seed + `hijridate` instead | holiday awareness |

**Not available for free (handled with curated estimates):** live hotel prices and availability, restaurant average spend, attraction ticket prices. The seed JSON stores **price bands** (`budget | mid | premium | luxury`) and typical AED ranges with a `source` and `as_of` date. The UI always labels these as estimates.

### Seed data plan (`data/seed/`)
- `attractions.json` (~60): id, name, area, lat/lon, category, indoor, typical_duration_min, price_aed_min/max, opening_hours, family_friendly, wheelchair, best_time, source, as_of.
- `hotels.json` (~40 across areas: Downtown, Marina, JBR, Deira, Bur Dubai, Palm, Al Barsha, Business Bay): area, star, price band, nightly AED range, nearest metro.
- `restaurants.json` (~80): cuisine, dietary flags (vegetarian, vegan, halal, gluten-free), price band, area, avg spend AED.
- `transport.json`: metro lines/stations (Red, Green), tram, fare rules, taxi base fare and per-km estimate, Careem/Uber estimate band.
- `events.json`: recurring events with month windows.
- `areas.json`: neighbourhood centroids and descriptions (for clustering days geographically).

## 4. Domain knowledge used by rules
- **Heat:** June–September daytime highs are often 40 °C+. Schedule outdoor activities early morning / after sunset; desert safaris run in the afternoon/evening.
- **Weekend:** UAE weekend is Saturday–Sunday (since 2022). Friday prayer around midday affects some opening hours.
- **Ramadan:** reduced daytime dining in public. Iftar experiences; altered opening hours.
- **Dress code / etiquette:** mosques (e.g. Jumeirah Mosque tours), modest dress in traditional areas.
- **Transport:** Metro Red/Green lines, Gold Class carriages; Nol card required. Taxis are metered; Abu Dhabi day trip ~1.5 h each way.
- **Geography:** cluster each day by area to minimise travel (e.g. Old Dubai day: Al Fahidi, Abra, Gold & Spice Souks).

## 5. Evaluation methodology research
- **Objective metrics** (computed by code): schema validity, budget compliance/overshoot %, hard-constraint satisfaction, rule-violation counts by severity, geographic coherence (mean intra-day travel km), grounding rate (share of venues found in seed/OSM), latency, LLM call count, token usage, review iterations, failure rate.
- **LLM-as-judge** (optional, rubric-based, 1–5): relevance to interests, feasibility, explanation quality, local insight. Use a different model than the generator where possible. Report inter-run variance and use pairwise comparison (A/B with position swap) to reduce bias.
- **Experimental design:** ≥ 10 scenarios × 2 modes (+ CrewAI hierarchical variant) × N=3 seeds, same provider/model/temperature. Report mean ± std, plus a short qualitative case study.
- **Fault injection** for the review loop: start from a deliberately broken itinerary (over budget, midday desert safari in July, overlapping slots) and measure detection/fix rate.

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Small local models produce invalid JSON | structured output + validate/repair retry; smaller schemas per agent; `qwen2.5` recommended |
| Free-tier rate limits during eval | caching, backoff, run eval on Ollama or throttle |
| Hallucinated venues/prices | grounding rule; agents select from tool results by ID; composer only references IDs |
| Framework API churn (CrewAI especially) | pin versions in `pyproject.toml`; adapter layer isolates framework code |
| External API downtime | JSON cache + seed fallback; offline mode |
| Python not yet installed on dev machine | Phase 0 installs Python 3.11/3.12 (CrewAI requires < 3.14 **[verify]**) |
| OneDrive-synced project folder locks SQLite / venv files | keep `.venv` and `data/*.db` excluded from sync, or move the repo outside OneDrive |

## 7. References (to cite in the report)
- LangGraph documentation — concepts: graphs, state/reducers, `Send`, persistence, human-in-the-loop.
- CrewAI documentation — agents, tasks, processes, guardrails, flows, event listeners.
- Open-Meteo API docs; OSM Overpass API and Nominatim usage policy; MediaWiki Action API; Frankfurter API.
- RTA Dubai — Nol fares; Visit Dubai (official tourism) for curated attraction facts.
- Wu et al., *AutoGen* (2023) and Park et al., *Generative Agents* (2023) — multi-agent background.
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench* (2023) — judge bias/mitigations.
- Xie et al., *TravelPlanner: A Benchmark for Real-World Planning with Language Agents* (2024) — constraint-based travel plan evaluation (directly relevant to metric design).
