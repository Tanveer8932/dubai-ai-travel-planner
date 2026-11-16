# Implementation Plan — Phases 0–7

Each phase ends with a **demoable increment** and a set of **acceptance criteria (AC)**. A phase counts as done only when every AC passes and `pytest` is green **offline with `LLM_PROVIDER=mock`**. Phases build on each other. Do not start phase N+1 with failing ACs in phase N.

| Phase | Theme | Key requirement(s) covered |
|-------|-------|----------------------------|
| 0 | Environment, scaffolding, docs | foundations, no-secrets policy |
| 1 | Domain models, data layer, persistence | Pydantic, SQLite + JSON, free data sources |
| 2 | LLM providers, tools, specialised agents | configurable LLMs, specialised agents (1), budget (4) |
| 3 | LangGraph orchestration | graph + parallelism (2), review loops (5) |
| 4 | CrewAI orchestration | role-based collaboration (3), review loops (5) |
| 5 | Streamlit UI, sessions, conversational refinement | persistence + refinement (6) |
| 6 | Execution visualisation and observability | visualisation (7) |
| 7 | Evaluation, hardening, final report | evaluation of both modes (8) |

---

## Phase 0 — Environment, scaffolding and documentation

**Tasks**
1. Install Python 3.11 or 3.12 (not currently installed on the dev machine). `winget install Python.Python.3.12` or python.org installer, and turn off the Microsoft Store `python` alias.
2. Optional: install `uv` for fast env management. Otherwise use `python -m venv .venv`.
3. `git init`; add `.gitignore` (venv, `.env`, `data/cache`, `data/exports`, `*.db`, `__pycache__`, `.pytest_cache`, `eval/reports/*` except `.gitkeep`).
4. Create `pyproject.toml` with pinned dependency groups:
   - core: `pydantic>=2`, `pydantic-settings`, `httpx`, `python-dotenv`, `tenacity`
   - langgraph: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-core`
   - crewai: `crewai`
   - providers (optional extras): `langchain-ollama`, `langchain-groq`, `langchain-google-genai`, `langchain-openai`, `langchain-anthropic`
   - ui: `streamlit`, `plotly`, `pandas`, `pydeck` (map)
   - misc: `hijridate`
   - dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, `respx` (httpx mocking)
5. Create the directory skeleton from `architecture.md §11` with `__init__.py` files.
6. `.env.example` with every variable documented and **no real values**.
7. `config.py` using `pydantic-settings` (reads env/.env; secrets typed `SecretStr`).
8. `README.md` quick start; a `Makefile`-equivalent `scripts/tasks.ps1` for Windows (`setup`, `test`, `lint`, `run`, `eval`).
9. Install Ollama and pull a model (`ollama pull qwen2.5:7b` or `llama3.1:8b`), optional but recommended.

**Acceptance criteria**
- [x] AC0.1 `python --version` reports 3.11.x or 3.12.x inside `.venv`.
- [x] AC0.2 `pip install -e ".[dev,langgraph,crewai,ui]"` (or `uv sync`) succeeds on Windows.
- [x] AC0.3 `python -c "import langgraph, crewai, streamlit, pydantic"` succeeds.
- [x] AC0.4 `pytest` runs (a smoke test passes); `ruff check .` passes.
- [x] AC0.5 `Settings()` loads with only `.env.example` values copied, and `LLM_PROVIDER=mock` works.
- [x] AC0.6 `git grep -iE "(api[_-]?key|secret|token)\s*=\s*['\"][A-Za-z0-9]"` finds no hardcoded secrets.
- [x] AC0.7 All five docs exist; Mermaid diagrams render in VS Code preview / GitHub.
  - *Status note (2026-11-15):* existence and Mermaid syntax (closed fences, valid diagram types) are checked by `tests/unit/test_smoke.py`; visual rendering in VS Code/GitHub still needs a one-off manual look.
- *Task 9 (Ollama, optional):* not installed. Not needed for Phase 0–1 (tests use `LLM_PROVIDER=mock`). AC2.3 was met with Groq's free tier instead; AC3.6 accepts Ollama or Groq.

---

## Phase 1 — Domain models, data layer and persistence

**Tasks**
1. `models/`: `TripRequest`, `Travellers`, `AttractionOption`, `HotelOption`, `DiningOption`, `TransportPlan`, `TransportLeg`, `WeatherContext`, `EventInfo`, `BudgetPlan`, `CostBreakdown`, `TimeSlot`, `DayPlan`, `Itinerary`, `RuleViolation`, `ReviewIssue`, `ReviewReport`, `TraceEvent`, `ChatMessage`. Money as `Decimal` with AED. Use validators for date ranges, non-negative costs, and slot ordering.
2. `data/http.py`: `httpx` client with User-Agent from config, timeout, `tenacity` retry, rate limiter (Nominatim 1 req/s), and a **JSON file cache** keyed by request hash with TTL.
3. Source clients: `open_meteo.py` (forecast + climate fallback), `overpass.py`, `nominatim.py`, `wikivoyage.py`, `fx.py` (Frankfurter + USD-peg conversion), `calendar.py` (Ramadan via `hijridate`, holidays with seed fallback).
4. Curate seed JSON (`data/seed/*.json`) per `research.md §3` with `source` + `as_of` on each record. Add `scripts/refresh_seed.py` to enrich coordinates and opening hours from OSM.
5. `storage/schema.sql` + `storage/repository.py` (sessions, messages, itineraries with versioning, runs, trace_events, eval tables). Lightweight migration (`PRAGMA user_version`).
6. `scripts/init_db.py`.

**Acceptance criteria**
- [x] AC1.1 All models round-trip `model_dump_json` → `model_validate_json`. Invalid data (end < start, negative cost, overlapping slots) raises `ValidationError` (unit tests).
  - *Evidence:* `tests/unit/test_models.py` (21 model round-trips, every exported model's JSON schema, 30+ invalid-data cases); `tests/unit/test_framework_compat.py` (LangGraph `JsonPlusSerializer` round-trip, CrewAI `Task(output_pydantic=Itinerary)`).
- [x] AC1.2 Seed files validate against their Pydantic models (≥ 50 attractions, ≥ 30 hotels across ≥ 6 areas, ≥ 60 restaurants with dietary flags).
  - *Evidence:* 61 attractions, 43 hotels across 12 areas, 65 restaurants (all with dietary flags), 27 areas; `tests/unit/test_seed.py`. Every record has `provenance` with source + `as_of`; all prices are `estimated`, none `verified`.
- [x] AC1.3 Each source client has a unit test with mocked HTTP (`respx`) and a cache-hit test. A second identical call makes no network request.
  - *Evidence:* `tests/unit/test_sources.py` (Open-Meteo forecast and archive, Overpass, Nominatim, Wikivoyage, Frankfurter), `tests/unit/test_http.py`. `calendar.py` makes no network calls by design (see note).
- [x] AC1.4 With network disabled, every client falls back to seed/cache without raising.
  - *Evidence:* `tests/integration/test_offline_fallbacks.py` covers `HTTP_OFFLINE=true`, connection errors on every route, and default clients with sockets blocked. The whole suite runs with real sockets blocked (`tests/conftest.py`).
- [x] AC1.5 Repository CRUD tests pass. Saving itinerary v2 links `parent_id` to v1. Data survives process restart (file DB).
  - *Evidence:* `tests/integration/test_repository.py`, including a restart check that reads the DB from a separate Python process.
- [x] AC1.6 `fx.convert(100, "USD", "AED") == 367.25`.
  - *Evidence:* `test_ac16_usd_to_aed_uses_peg_without_network` (pure peg, no request made).

*Status notes (2026-11-15):*
- **Nager.Date has no UAE coverage** (HTTP 204, `AE` absent from `AvailableCountries`). Task 3's "holidays with seed fallback" is therefore seed-first: fixed holidays come from `holidays.json` and Islamic holidays from `hijridate` (flagged approximate). No online holiday client exists.
- **User-Agent:** Wikivoyage returns 403 without contact info and Overpass returns 406 for the `example.com` placeholder. Users must put a real contact in `HTTP_USER_AGENT`; the client logs a warning otherwise and falls back to seed.
- **Live check** (manual, outside pytest): Open-Meteo forecast and archive, Nominatim, Wikivoyage (with a contact UA), Frankfurter, `init_db.py` and a `refresh_seed.py` dry run all worked. Overpass was at capacity (HTTP 502) during the session, so its live response was confirmed with curl earlier but not re-run through the client. The parser is tested against the same payload shape.
- **Seed honesty:** venues and prices were curated from public information and were **not** live-verified. Coordinates are approximate until `scripts/refresh_seed.py --write` is run. Some venues may have changed or closed; each record's `provenance.note` says so.
- Added settings: `SEED_DIR`, `HTTP_TIMEOUT_S`, `HTTP_OFFLINE` (documented in `.env.example`).

---

## Phase 2 — LLM providers, tools and specialised agents (framework-agnostic)

**Tasks**
1. `llm/factory.py`: `get_chat_model(agent_name=None)` (LangChain) and `get_crewai_llm(agent_name=None)` (CrewAI `LLM`) from the same settings. Lazy imports per provider; per-agent overrides via `LLM_MODEL__<AGENT>`.
2. `llm/mock.py`: deterministic `MockChatModel` returning fixture JSON per agent and schema (`tests/fixtures/llm/*.json`). Also usable through CrewAI (custom LLM class or LiteLLM mock response) **[verify approach]**.
3. `llm/structured.py`: `invoke_structured(model, prompt, Schema)` → `with_structured_output` when supported, otherwise JSON prompt + validate + one repair retry. Records token usage to the trace.
4. `tools/`: plain typed functions: `search_attractions`, `search_hotels`, `search_restaurants`, `get_weather_context`, `get_events`, `estimate_travel`, `metro_fare`, `convert_currency`, `calculate_costs`.
5. `agents/`: one module per agent (see architecture §5), each with: system prompt (`prompts/*.md`), input/output models, and `run(input, llm, tools) -> output`. Research agents **pick IDs from tool results**; they do not invent venues.
6. `validation/rules.py`: all deterministic rules from architecture §7 returning `list[RuleViolation]`.
7. `agents/budget.py`: deterministic allocation (accommodation from nights × nightly rate, food per person per day by band, activities sum, transport estimate, contingency %), plus an LLM-written rationale.

**Acceptance criteria**
- [x] AC2.1 Switching `LLM_PROVIDER` between `mock`, `ollama`, `groq`, `gemini` needs **no code change**. A missing key gives a clear error naming the env var.
  - *Evidence:* `tests/unit/test_llm_factory.py`. All five LangChain providers are built from env alone (with the provider packages stubbed, since the optional `providers` extra is not installed); real CrewAI clients are constructed for ollama/groq/gemini/openai with no network; a missing or blank key raises `LLMConfigError` naming `GROQ_API_KEY` etc.; a missing package raises `ProviderNotInstalledError` naming it; keys never appear in `repr`.
- [x] AC2.2 Each agent has a unit test with the mock LLM producing a valid output model.
  - *Evidence:* `tests/unit/test_agents.py` (all 9 agents, JSON round-trip of each output; the same pipeline through `MockChatModel` (LangChain) and `MockCrewAILLM` (CrewAI) yields identical itineraries); `tests/unit/test_composer_reviewer.py`; `tests/integration/test_agent_chain.py` (6 scenarios end to end, offline).
- [x] AC2.3 `RequirementsAgent` extracts correct `TripRequest` fields for ≥ 8/10 fixture prompts (dates, travellers, budget, interests, dietary) with a real local model (manual/integration test, marked `@pytest.mark.llm`).
  - *Evidence (2026-11-16):* met with a **hosted free-tier model, not a local one**: Groq `openai/gpt-oss-20b` via `langchain-groq` 1.2.0 (Ollama was not installed, by decision). `pytest -m llm` → `test_ac23_real_model_extracts_fixture_cases` **PASSED** in two separate runs (≥ 8/10 asserted). Per-case runs with the same config scored **9/10 twice**: every completed extraction was correct on the first LLM call, and in each run exactly one case (`friends_nights` in one, `family_kids_count` in the other) failed with HTTP 429 (free-tier limit of 8 000 tokens/min) after the SDK's retries, not with a wrong extraction. Every one of the 10 cases was extracted correctly in at least one run.
  - *Fixes needed to get there:* (1) Groq's default `llama-3.1-8b-instant` is not served to this account; the Groq default is now `openai/gpt-oss-20b`. (2) First attempt scored **3/10**: LangChain's default forced tool call made Groq reject 7 replies with HTTP 400 `tool_use_failed`, although the rejected `failed_generation` was already correct JSON. Groq now uses `with_structured_output(method="json_schema")`, and a provider rejection of a native reply (`tool_use_failed` / `json_validate_failed`) is validated through the normal Pydantic → postprocess → one repair path instead of aborting as a transport error (`llm/base.py`, `llm/structured.py`). No prompt, model or domain change was needed. (3) `tests/conftest.py` forced `LLM_PROVIDER=mock` for every test, so `pytest -m llm` always skipped; `llm`-marked tests now use the caller's provider (shell env, else `.env`).
  - *Not covered:* a local Ollama model was not tested; the offline rule-based mock still passes the same cases (`test_heuristic_baseline_on_fixture_cases`) as a baseline.
- [x] AC2.4 Budget calculator unit tests: exact arithmetic with `Decimal`; categories sum to total; over-budget flagged.
  - *Evidence:* `tests/unit/test_budget_agent.py` (fils-exact amounts, contingency = 10 % of subtotal, categories sum to the allocated total, tolerance boundary, per-person/per-day scopes, infeasible plan flagged with a reason, LLM cannot change numbers); `tests/unit/test_tools.py::test_calculate_costs_is_exact_and_consistent`.
- [x] AC2.5 Each validator rule has a positive and a negative test case.
  - *Evidence:* `tests/unit/test_rules.py`, covering all 11 rules from architecture §7 (plus severity escalation for heat in May–Sep, seasonal closures, and grounding against the research bundle).
- [x] AC2.6 Grounding: research agent outputs contain only IDs present in tool results (test with mock that tries to inject an unknown ID → rejected).
  - *Evidence:* `tests/unit/test_grounding.py`. Injected ids are dropped, listed in `rejected_ids` and traced as a guardrail event; an all-unknown answer gets one repair, then `StructuredOutputError`; the composer rejects unknown venue ids through the repair loop.

*Status notes (2026-11-16):*
- **Phase 2 is complete:** all six ACs pass (AC2.3 with Groq `openai/gpt-oss-20b`, see above). Phase 3 can start.
- **Free-tier rate limit:** Groq's on-demand tier allows 8 000 tokens/min for `gpt-oss-20b`; one requirements call uses ~3 000. Back-to-back runs (AC2.3, later AC3.6/AC7.2 evals) can hit HTTP 429, which is a transport error and is not "repaired". Pace eval runs or raise `LLM_MAX_RETRIES` (max 5).
- **As built:** see `architecture.md` §5, §7 and §10 "As built (Phase 2)". Key choices: a framework-neutral `ChatLLM` interface with LangChain/CrewAI adapters; Groq and Gemini reach CrewAI through their OpenAI-compatible endpoints (CrewAI 1.15 has no native Groq client and LiteLLM is not installed); transport and weather/events agents are deterministic (no LLM call); the budget LLM writes only the rationale; the composer LLM drafts by id and Python grounds and prices the itinerary.
- **Fallbacks:** research, composer and reviewer agents fail loudly by default (`StructuredOutputError`, hard rule 6) and accept an opt-in `fallback=True` that uses deterministic baselines (tool ranking, baseline scheduler, rule-only review). The budget rationale always falls back, because it is not critical.
- **Not used to add venues:** Overpass POIs and Wikivoyage summaries are available as tools, but OSM venues carry no price or dietary data, so they cannot be grounded or costed. Agents select only curated seed venues.
- **Tracing:** a minimal `tracing/bus.py` (publish/subscribe, run context, `agent_span`) was added because task 3 requires token usage on the trace. Phase 6 adds the SQLite and Streamlit subscribers.
- Added settings: `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_REPAIR_ATTEMPTS`, `LLM_MODEL__<AGENT>`. `LLM_MODEL`/`LLM_BASE_URL` are now optional (empty = provider default), documented in `.env.example`.
- 408 offline tests pass (`pytest -m "not llm"`, including 6 new structured-output tests for the Groq fix), 97 % line+branch coverage on `src/`; `ruff check .` and `mypy src` (strict) are clean. `pytest -m llm` (1 test) passes against Groq.

---

## Phase 3 — LangGraph orchestration

**Tasks**
1. `orchestration/langgraph/state.py`: `PlannerState` with reducers (architecture §3).
2. `nodes.py`: thin wrappers calling the agent core; each emits start/end `TraceEvent`s.
3. `graph.py`: build the graph: parse → clarification gate (`interrupt`) → parallel fan-out of 5 research nodes → merge → budget → compose → validate → review → conditional (refine loop | finalize). `MAX_REVIEW_ITERATIONS` from config (default 2).
4. `SqliteSaver` checkpointer with `thread_id=session_id`.
5. Expose `run_langgraph(request_text, session_id, stream=True)` yielding updates.
6. Export `draw_mermaid()` to `docs/generated/langgraph.mmd` (for report).

**Acceptance criteria**
- [x] AC3.1 End-to-end run with mock LLM produces an approved, schema-valid `Itinerary` with `cost.total ≤ budget × 1.05`.
  - *Evidence:* `tests/integration/test_langgraph_planner.py::test_ac31_end_to_end_mock_run_is_approved_and_persisted`: status `approved`, JSON round-trip, total ≤ budget × 1.05, no error-severity violation, every venue id grounded in the research, approved v1 saved with its `run_id`, run record `succeeded`, and every trace event carries the run id and `mode="langgraph"`.
- [x] AC3.2 **Parallelism proven:** with each research node stubbed to `sleep(1s)`, research stage wall-clock < 1.8 s (vs ~5 s sequential). Trace shows overlapping start/end timestamps.
  - *Evidence:* `test_ac32_research_nodes_run_in_parallel`. The five research agents are replaced by stubs that sleep 1 s and return real outputs. The sum of node spans is ≥ 5 s, the stage from first start to last end is < 1.8 s, and every pair of node spans overlaps (from the `start`/`end` `TraceEvent`s). Stable over repeated runs.
- [x] AC3.3 **Review loop proven:** mock composer returns an over-budget itinerary on attempt 1 and a valid one on attempt 2 → graph runs `refine` once, `iteration == 1`, final approved. With a composer that never fixes it, the loop stops at `MAX_REVIEW_ITERATIONS` and returns the best attempt with issues listed.
  - *Evidence:* `test_ac33_over_budget_first_attempt_is_refined_once`: `budget.total` blocks attempt 1, one `refine_itinerary` update, `iterations == 1`, approved, and the composer receives the reviewer feedback. The second compose's trace events carry `iteration=1`. `test_ac33_loop_stops_at_cap_and_returns_best_unapproved`: 3 composes, `iterations == 2`, status `needs_revision`, best attempt returned with `blocking_issues`, saved with `approved=0`. Also: `test_zero_review_iterations_means_no_refinement`, `test_targeted_refinement_replans_only_the_faulty_day` (a day-2 error locks days 1 and 3, which stay identical), and `test_reviewer_approval_never_overrides_hard_rule_errors` (a reviewer that always approves cannot approve an itinerary with a rule error).
- [x] AC3.4 Incomplete request ("plan a Dubai trip") triggers `interrupt`. Resuming with `Command(resume="5 days in Feb, AED 8000")` completes the run.
  - *Evidence:* `test_ac34_vague_request_interrupts_and_command_resume_completes` drives the compiled graph directly: it pauses at `ask_clarification` with the question, and `Command(resume=...)` completes with 5 days and AED 8000. `test_ac34_service_answers_pending_question` does the same through `LangGraphPlanner`: the next user message becomes the resume value, the same run continues, and the run record is `interrupted`, then `succeeded`. Clarification rounds are capped at 3 (`test_clarification_rounds_are_capped`).
- [x] AC3.5 Killing the process mid-run and resuming the same `thread_id` continues from the last checkpoint.
  - *Evidence:* `test_ac35_killed_process_resumes_from_last_checkpoint`. A subprocess (`tests/integration/langgraph_worker.py`) is killed with `os._exit` inside `compose_itinerary` (no cleanup). A second process opens the same checkpoint DB and calls `resume()`: it starts at `compose_itinerary`, does not re-run parse, research or budget, and finishes approved under the same `run_id`. *Finding:* LangGraph 1.x writes checkpoints asynchronously by default (`durability="async"`). In the first version of this test the resumed process re-ran `research_dining`, because the killed process had not flushed that write. The planner now streams with `durability="sync"`.
- [x] AC3.6 Real-LLM smoke run (Ollama or Groq) for 1 scenario completes in < 5 min (marked `@pytest.mark.llm`).
  - *Evidence (2026-11-16, Groq `openai/gpt-oss-20b`):* `pytest -m llm -k ac36` **passed**: approved in **128.3 s**, 7 LLM calls, ≈ 18.3k tokens, 0 refinements. An earlier run with the same fix was also approved, in 92.5 s (that test failed only on a test-code bug, since fixed). This is one scenario (`2 adults, 10-12 Feb 2027, AED 6000`), with data sources offline (seed fallbacks); it is not a quality evaluation, which belongs to Phase 7.
  - *First attempt failed* (`failed` after 118 s): the composer's JSON was truncated twice (`unterminated JSON object`). Diagnosis: `finish_reason=length` at exactly 2048 completion tokens, 1522 of them hidden reasoning. Groq's default output cap is too small for a reasoning model. Fix (LLM factory only, no agent change): new settings `LLM_MAX_TOKENS` (Groq default 4096) and `LLM_REASONING_EFFORT` (default `low` for Groq `openai/gpt-oss-*`). The same composer call then ends with `stop` and 753 tokens (23 reasoning).

*Status notes (2026-11-16):*
- **As built:** see `architecture.md` §3 "As built (Phase 3)". Modules: `orchestration/langgraph/{state,nodes,graph,checkpoint,runner}.py`. The shared revision strategy is in `agents/revision.py` (also for CrewAI), and the mode-neutral result contract (`PlanResult`, `PlanStatus`, `PlannerUpdate`, `Attempt`) is in `models/planning.py`.
- **Design choices:** `refine_itinerary → compose_itinerary`, not `→ allocate_budget`. The budget agent is deterministic for an unchanged request and research, so re-running it would cost an LLM call and change nothing. Refinement is targeted: if every blocking issue names a day, the other days are locked and kept verbatim. An attempt counts as approved only if the reviewer approves **and** no error-severity rule fails; the router checks this independently of the reviewer.
- **Errors:** transient LLM errors (429, timeouts, 5xx, connection) get a bounded node `RetryPolicy` (4 attempts, ≈ 8/16/32 s). Anything else ends the run as `failed`, with a short, secret-free error on the result and the run record. The checkpoint keeps finished branches, so `resume()` re-runs only the failed node (`test_failed_branch_is_reported_and_resume_reruns_only_it`). Also tested: a cancelled stream, a missing API key, and that a canary key never appears in traces, results, run records or messages.
- **Checkpoint safety:** `SqliteSaver` uses a serializer with an explicit allowlist of our model and enum classes, so a tampered checkpoint cannot make it import anything else. Unknown classes are not rebuilt (`test_strict_serializer_blocks_foreign_classes`).
- **AC2.3 re-checked** after the `reasoning_effort=low` default: `pytest -m llm -k ac23` passed (≥ 8/10). A per-case diagnostic run afterwards scored 6/7 on the cases that completed (`friends_nights` returned "nightlife" instead of "adventure"); the other 3 cases hit Groq's **daily** token limit.
- **Free-tier budget (risk for AC7.2):** Groq `gpt-oss-20b` on-demand allows 8 000 tokens/min and **200 000 tokens/day**. One full planning run uses ≈ 18k tokens, so about 10 runs/day. AC7.2 (≥ 10 scenarios × 2 modes) will need several days, a second model or provider, or Ollama. A daily-limit 429 asks to wait minutes; the bounded retry gives up after ≈ 1 min and the run is `failed` (resumable).
- Mermaid export: `python scripts/export_langgraph.py` → `docs/generated/langgraph.mmd` (a test keeps it in sync with the graph).
- 455 offline tests pass (+47), 96 % line+branch coverage on `src/`; `ruff check .` and `mypy src` (strict) are clean.

---

## Phase 4 — CrewAI orchestration

**Tasks**
1. `orchestration/crewai/config/agents.yaml` and `tasks.yaml`: roles/goals/backstories and task descriptions (architecture §4).
2. `tools.py`: wrap shared tool functions as CrewAI tools (`@tool` / `BaseTool` with Pydantic `args_schema`).
3. `crew.py`: build `Crew` for `sequential` (default) and `hierarchical` (manager agent) processes. Research tasks `async_execution=True`; downstream tasks list them in `context`. All tasks use `output_pydantic` with the **same models** as Phase 1.
4. `guardrails.py`: itinerary guardrail = deterministic validators + reviewer verdict; on failure returns structured feedback for retry. Outer refinement loop bounded by `MAX_REVIEW_ITERATIONS`.
5. Event listener / callbacks → `TraceEvent`s on the shared bus.
6. Telemetry disabled through env (`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`).
7. Expose `run_crewai(request_text, session_id, process="sequential")` with the same return contract as `run_langgraph`.

**Acceptance criteria**
- [ ] AC4.1 With the mock LLM, the crew produces a schema-valid `Itinerary` of the **same type** as LangGraph mode. The `PlannerService` contract test passes for both modes.
- [ ] AC4.2 Trace shows each role's task, delegation events (hierarchical) and tool calls attributed to the correct agent.
- [ ] AC4.3 Async research tasks overlap in time (timestamps), the same check as AC3.2.
- [ ] AC4.4 Guardrail test: injected defect (midday outdoor slot in July) is rejected with feedback, then corrected on retry. The retry count is visible in the trace.
- [ ] AC4.5 Both `sequential` and `hierarchical` processes run end-to-end via config flag.
- [ ] AC4.6 No outbound telemetry when the env flags are set (verify no requests during a mocked offline run).

---

## Phase 5 — Streamlit UI, session persistence and conversational refinement

**Tasks**
1. `app/streamlit_app.py` (home + sidebar: mode selector LangGraph / CrewAI-seq / CrewAI-hier, provider/model display, session picker, "new session").
2. `pages/1_Planner.py`: chat input (`st.chat_input`), streamed progress (`st.status`), itinerary rendering (day tabs, time-slot cards, indoor/outdoor badges, cost table, budget gauge, map with `pydeck`), export JSON/Markdown.
3. `pages/2_Sessions.py`: list sessions, open, rename, delete; version history with diff (v(n) vs v(n+1): added/removed slots, cost delta).
4. `services/refinement.py`: classify follow-up as **request delta** (budget/dates/travellers change → replan) vs **local edit** (specific days → partial replan with other days locked). Uses history + latest itinerary as context.
5. `services/session.py`: all state in SQLite. `st.session_state` holds only the current `session_id` and UI flags.

**Acceptance criteria**
- [ ] AC5.1 `streamlit run app/streamlit_app.py` starts with `LLM_PROVIDER=mock` and plans a trip from the UI in both modes.
- [ ] AC5.2 Restarting Streamlit and reopening the session shows the full chat and latest itinerary (persistence).
- [ ] AC5.3 Follow-up "reduce the budget by 20%" → new version with `total ≤ 0.8 × old budget × 1.05`. "Replace day 2 with indoor activities" → only day 2 changes (other days identical). Both versions listed.
- [ ] AC5.4 Diff view shows cost delta and changed slots between versions.
- [ ] AC5.5 Errors (provider down, invalid key) show a friendly message, not a stack trace. The app stays usable.
- [ ] AC5.6 Every price shown is labelled as an estimate with its source/as-of date available on hover/expander.

---

## Phase 6 — Agent execution visualisation and observability

**Tasks**
1. `tracing/bus.py`: in-process pub/sub. Subscribers: SQLite writer, Streamlit live queue.
2. `pages/3_Execution.py`:
   - **Graph view**: LangGraph Mermaid from `get_graph().draw_mermaid()`. CrewAI agent/task graph generated from crew config. Rendered via a Mermaid HTML component (`streamlit.components.v1.html` with pinned mermaid.js CDN version) or `st.graphviz_chart` fallback.
   - **Live status**: nodes coloured pending / running / done / failed during a run.
   - **Timeline (Gantt)**: Plotly bars per agent from `TraceEvent` start/end, making parallel branches and review iterations visible.
   - **Inspector**: select an event → inputs, outputs (JSON), prompt, tokens, latency.
   - **Run comparison**: pick two runs (e.g. LangGraph vs CrewAI on the same request) side by side.
3. Token/latency aggregation per run stored in `runs` table.

**Acceptance criteria**
- [ ] AC6.1 For a LangGraph run, the Gantt chart shows the 5 research nodes overlapping and each review iteration as a separate band.
- [ ] AC6.2 For a CrewAI run, the timeline shows tasks per role, async overlap and guardrail retries.
- [ ] AC6.3 Graph view highlights the node currently running during a live run (updates at least once per node transition).
- [ ] AC6.4 Inspector shows the validated output model for any selected step.
- [ ] AC6.5 Historical runs can be visualised after app restart (data comes from SQLite only).

---

## Phase 7 — Evaluation, hardening and final report

**Tasks**
1. `eval/scenarios/*.json`: ≥ 12 scenarios (budget, family, luxury, summer-heat, Ramadan, vegetarian + wheelchair, day-trip, tight budget/infeasible, vague request, multi-refinement conversation). Each has hard constraints and expected properties.
2. `evaluation/metrics.py`: metrics from `research.md §5`.
3. `evaluation/judge.py`: optional rubric LLM-judge with position-swapped pairwise comparison.
4. `evaluation/runner.py` + `scripts/run_eval.py`: scenarios × modes × seeds. Results to SQLite + `eval/reports/<timestamp>/results.csv` + `report.md` with charts (PNG via Plotly/kaleido or Matplotlib).
5. Fault-injection suite for review loops (both modes).
6. `pages/4_Evaluation.py`: dashboard (metric table, box plots per mode, per-scenario drill-down).
7. Hardening: error handling, input length limits, prompt-injection guard in user text (treat as data), logging config, docstrings, `ruff` + `mypy` clean, coverage ≥ 70% on `src/` (excluding UI).
8. Final write-up: update `architecture.md` with as-built changes and add `docs/evaluationReport.md` with findings, limitations and future work.

**Acceptance criteria**
- [ ] AC7.1 `python scripts/run_eval.py --provider mock` completes offline and produces CSV + Markdown report.
- [ ] AC7.2 A real-provider eval (Ollama or Groq) over ≥ 10 scenarios × 2 modes completes. The report has mean ± std for every metric.
- [ ] AC7.3 Report compares LangGraph vs CrewAI (seq; hier optional) on budget compliance, constraint satisfaction, validity, latency, LLM calls/tokens, review iterations, judge score, and states which hypotheses in `research.md §1.3` held.
- [ ] AC7.4 Fault-injection: reviewer detects ≥ 80% of seeded defects in both modes.
- [ ] AC7.5 Success criteria S1–S8 in `problemStatement.md` are each marked met / not met with evidence.
- [ ] AC7.6 Fresh clone → setup → `pytest` → `streamlit run` works by following `README.md` alone (tested on Windows).

---

## Cross-cutting definition of done
- Type hints everywhere; Pydantic at every boundary; no bare `except`.
- No secrets in code, logs, traces or exported files (redact `SecretStr`).
- Every external call is cached and has an offline fallback.
- Tests added with each feature; `pytest -m "not llm"` must pass offline.
- Docs updated when the design changes (CLAUDE.md "Current phase" kept current).
