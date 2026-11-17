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
- [x] AC4.1 With the mock LLM, the crew produces a schema-valid `Itinerary` of the **same type** as LangGraph mode. The `PlannerService` contract test passes for both modes.
  - *Evidence:* `tests/integration/test_crewai_planner.py::test_ac41_sequential_crew_produces_the_same_itinerary_type` (approved, `type(...) is Itinerary`, JSON round-trip, total ≤ budget × 1.05, no error-severity rule, approved v1 saved with its `run_id`, run `succeeded`, one streamed `PlannerUpdate` per CrewAI task). `tests/integration/test_planner_contract.py` runs the same contract (`services/planner.Planner`: `stream`/`run`/`resume`/`pending_question`, `PlanResult`, persistence, clarification) for `langgraph`, `crewai_sequential` and `crewai_hierarchical` (7 tests). `test_ac41_both_frameworks_plan_the_same_itinerary` shows LangGraph and both CrewAI processes return **identical** itineraries for the same request (run-identity fields excluded): one shared agent core. `test_crew_tasks_carry_typed_outputs_and_context` checks each CrewAI `TaskOutput.pydantic` is the stage's Phase 1 model and the declared `context` chain.
- [x] AC4.2 Trace shows each role's task, delegation events (hierarchical) and tool calls attributed to the correct agent.
  - *Evidence:* `test_ac42_trace_attributes_tasks_and_tool_calls_to_roles` (a `crew_task` start/end per task with its role, every stage tool attributed to its own role, an `llm_call` with tokens per role). `test_ac42_hierarchical_manager_delegates_every_task` (9 `delegation` events from `trip_manager`, one per specialist; the tool is then used by that specialist; exactly 2 manager calls per task). `test_ac42_supporting_tools_are_attributed_to_the_calling_agent` (`metro_fare` then `plan_transport`, both on `transport_specialist`). The events come from CrewAI's own event bus (`listener.py`).
- [x] AC4.3 Async research tasks overlap in time (timestamps), the same check as AC3.2.
  - *Evidence:* `test_ac43_async_research_tasks_overlap`. The five research agents are stubbed to sleep 1 s. The sum of spans is ≥ 5 s, the stage window is < 1.8 s, every pair of spans overlaps, and CrewAI's own task events overlap too. Stable over repeated runs. Sequential process only: see the hierarchical note below.
- [x] AC4.4 Guardrail test: injected defect (midday outdoor slot in July) is rejected with feedback, then corrected on retry. The retry count is visible in the trace.
  - *Evidence:* `test_ac44_midday_outdoor_slot_in_july_is_rejected_then_fixed`. Attempt 1 puts an outdoor attraction at 12:00–13:30 on day 1 of a July trip (assembled by the real composer, like a model reply). The compose guardrail rejects it (`heat.midday_outdoor`, `retry=1`), CrewAI re-runs the task, the composer gets the feedback with days 2–3 locked, and the run is approved with `iterations == 1`. Trace: guardrail events `[passed=False, passed=True]`, compose spans at iterations `[0, 1]`, and CrewAI's `retry_count == 1` on the task end event.
- [x] AC4.5 Both `sequential` and `hierarchical` processes run end-to-end via config flag.
  - *Evidence:* `test_ac45_process_selected_by_config_flag[sequential|hierarchical]` (`CREWAI_PROCESS` env → approved, `mode` = `crewai_sequential` / `crewai_hierarchical`); `test_run_crewai_function_api` (`run_crewai(..., process=...)`, streamed and not).
- [x] AC4.6 No outbound telemetry when the env flags are set (verify no requests during a mocked offline run).
  - *Evidence:* `test_ac46_no_outbound_requests_during_a_crew_run`. Every `socket.connect`/`connect_ex`/`create_connection`/`getaddrinfo` is recorded during full sequential and hierarchical runs: **none** (loopback, used by asyncio internally, is allowed). CrewAI `Telemetry` reports disabled and `should_enable_tracing()` is false. `test_crews_keep_no_replay_log_outside_the_project`.

*Status notes (2026-11-16):*
- **As built:** see `architecture.md` §4 "As built (Phase 4)". Modules: `orchestration/crewai/{config/*.yaml,board,stages,tools,guardrails,listener,crew,runner}.py`, `services/planner.py`. `AgentSuite` moved to `agents/suite.py`; shared service helpers moved to `orchestration/common.py`; `is_transient` moved to `llm/errors.py` (LangGraph re-exports both; behaviour unchanged).
- **Real CrewAI orchestration:** every step is a CrewAI `Task` run by `Crew.kickoff`, on an `Agent` with a YAML role/goal/backstory and its own LLM. Each specialist's tool runs the shared agent core. Research tasks use `async_execution=True`, downstream tasks declare `context`, guardrails drive retries, and the hierarchical manager delegates with CrewAI's delegation tool. Typed results travel on a per-run board, not through LLM text: tools return short briefs (`result_as_answer=True`), and each task's guardrail attaches the typed model and rejects answers its tool did not produce (`test_agent_answering_without_its_tool_is_rejected_and_retried`).
- **Review loop:** the compose guardrail is deterministic (rules). An error is rejected with `plan_revision` feedback and CrewAI retries the task. A reviewer rejection without rule errors triggers an outer loop that kicks off a compose → review **revision crew**. Both share one budget, `MAX_REVIEW_ITERATIONS` (default 2), so at most 3 itineraries are composed (`test_loop_stops_at_cap_and_returns_best_unapproved`, `test_zero_review_iterations_means_no_refinement`, `test_reviewer_rejection_runs_a_targeted_revision_crew`). Approval = reviewer approval **and** no error-severity rule (`test_reviewer_approval_never_overrides_hard_rule_errors`).
- **Clarification and sessions:** an incomplete request ends `needs_clarification`. The draft and question are stored in the message metadata (SQLite) and the next message continues the same `run_id`, for at most 3 rounds. Later requests build on the session's last draft (v2 → parent v1). Concurrent sessions are isolated (`test_concurrent_sessions_are_isolated`).
- **Errors:** a failing agent fails the run loudly and runs only once (`tool_failure_policy=raise` + board short-circuit). Transient provider errors get the same bounded backoff as LangGraph (4 attempts, ≈ 8/16/32 s). Also tested: missing key (error names `GROQ_API_KEY`), canary key never in results/runs/traces/messages, cancelled stream (no itinerary saved, run `interrupted`), `resume()`. **Limitation:** CrewAI mode has no checkpointer, so `resume()` re-runs the unfinished request from the start (LangGraph resumes from the failed node).
- **Deviations from the task list, with reasons:** (1) *Hierarchical:* research tasks are not async, because the manager runs every task on one agent executor that is not thread-safe; tasks declare no `output_pydantic`, because CrewAI would add a manager LLM call per task to re-type specialist output (3 instead of 2 calls per task), i.e. an LLM rewriting grounded data (hard rule 5). The guardrail attaches the same typed model. (2) CrewAI's replay log (`latest_kickoff_task_outputs.db` in the user's app-data folder) is replaced by a no-op (`PlanningCrew`): it wrote outside the project and its JSON encoder crashes on `datetime.time`.
- **CrewAI 1.15 findings:** its first-run "view your execution traces?" prompt blocks every kickoff for 20 s unless `CREWAI_TRACING_ENABLED=false` (new setting, exported by `bootstrap()`; crews also pass `tracing=False`). Guardrails must be functions or bound methods. Tool exceptions are swallowed and retried unless `tool_failure_policy="raise"`.
- **LLM (Groq):** `get_crewai_llm` now passes `LLM_MAX_TOKENS` (Groq default 4096) and `LLM_REASONING_EFFORT` (`low` for `openai/gpt-oss-*`) to CrewAI's OpenAI-compatible client, verified in the request parameters (unit test) and on Groq. `CrewAIChatLLM` reports the provider's token usage instead of estimates. Groq rejected two function schemas that the mock could not reveal: a tool without `properties`, and a `Decimal` field's regex `pattern` with lookarounds. Both are fixed (`provider_safe`) and unit-tested against CrewAI's own schema converter.
- **Real-model smoke run (not an AC4 criterion; `pytest -m llm -k crewai_real`):** Groq `openai/gpt-oss-20b`, sequential, one scenario (`2 adults, 10-12 Feb 2027, AED 6000`), data sources offline: **approved in 297.6 s**, 0 refinements, 16 LLM calls, ≈ 46.6k tokens. The 9 CrewAI agent-reasoning calls used ≈ 29.5k of those (role prompts plus each task's `output_pydantic` schema). LangGraph used ≈ 18.3k for the same scenario. The first attempt failed in 4 s with 0 tokens (the Groq schema rejection above). Several calls hit Groq's 8k tokens/min limit and were retried. This is one smoke run, not a quality evaluation.
- **Free-tier impact (risk for AC7.2):** one CrewAI run ≈ 47k tokens, so about 4 runs per day on Groq's 200k/day free tier, against ≈ 10 LangGraph runs. Phase 7 should consider trimming the inlined `output_pydantic` schemas (the largest overhead) or using a second provider.
- 515 offline tests pass (+60), 96 % line+branch coverage on `src/`; `ruff check .` and `mypy src` (strict) are clean.

---

## Phase 5 — Streamlit UI, session persistence and conversational refinement

**Tasks**
1. `app/streamlit_app.py` (home + sidebar: mode selector LangGraph / CrewAI-seq / CrewAI-hier, provider/model display, session picker, "new session").
2. `pages/1_Planner.py`: chat input (`st.chat_input`), streamed progress (`st.status`), itinerary rendering (day tabs, time-slot cards, indoor/outdoor badges, cost table, budget gauge, map with `pydeck`), export JSON/Markdown.
3. `pages/2_Sessions.py`: list sessions, open, rename, delete; version history with diff (v(n) vs v(n+1): added/removed slots, cost delta).
4. `services/refinement.py`: classify follow-up as **request delta** (budget/dates/travellers change → replan) vs **local edit** (specific days → partial replan with other days locked). Uses history + latest itinerary as context.
5. `services/session.py`: all state in SQLite. `st.session_state` holds only the current `session_id` and UI flags.

**Acceptance criteria**
- [x] AC5.1 `streamlit run app/streamlit_app.py` starts with `LLM_PROVIDER=mock` and plans a trip from the UI in both modes.
  - *Evidence:* `tests/ui/test_streamlit_app.py::test_ac51_plans_a_trip_from_the_ui[langgraph|crewai_sequential|crewai_hierarchical]` drives the real Planner page with Streamlit's `AppTest`: chat input → background run → approved, day tabs, chat history, saved version in the chosen mode. `test_mode_selector_drives_the_next_run`, `test_home_page_renders_with_sidebar`. Manual check (2026-11-17): `streamlit run app/streamlit_app.py` with `LLM_PROVIDER=mock`, `HTTP_OFFLINE=true` started, `/_stcore/health` returned `ok`, and headless-Edge screenshots of the Home, Planner (LangGraph and CrewAI sessions) and Sessions pages rendered without errors.
- [x] AC5.2 Restarting Streamlit and reopening the session shows the full chat and latest itinerary (persistence).
  - *Evidence:* `test_ac52_restart_shows_the_chat_and_latest_itinerary`: after a run, the cached services and planners are closed and cleared (an app restart); a fresh page opened with `?session=<id>`, and another through the sidebar session picker, both show the full conversation and the latest itinerary, read from SQLite only. The manual check reopened sessions planned by a different process (also SQLite only).
- [x] AC5.3 Follow-up "reduce the budget by 20%" → new version with `total ≤ 0.8 × old budget × 1.05`. "Replace day 2 with indoor activities" → only day 2 changes (other days identical). Both versions listed.
  - *Evidence:* `tests/integration/test_ui_services.py::test_ac53_budget_cut_and_day_edit[langgraph|crewai_sequential]` and, through the UI, `test_ac53_ac54_refinements_and_version_diff_in_the_ui`. The budget cut is classified as a request delta, resolved deterministically to AED 4800 and re-planned: v2 total AED 3250.59 ≤ 5040. The day edit is a local edit: days 1 and 3 are identical to v2, day 2 is re-planned with indoor attractions only and no venue repeated from the kept days, and the request is reused without an LLM call (`test_local_edit_reuses_the_request_without_an_llm_call`). Versions 1, 2, 3 are listed (parent links).
- [x] AC5.4 Diff view shows cost delta and changed slots between versions.
  - *Evidence:* `test_ac53_ac54_refinements_and_version_diff_in_the_ui` (Sessions page: "v2 → v3" metric with the AED delta, per-category change table, "Unchanged days: 1, 3", removed/added slots of day 2; switching the "From" version to v1 recomputes it); `SessionService.diff` is also checked in `test_ac53_budget_cut_and_day_edit` (`changed_days == [1]`).
- [x] AC5.5 Errors (provider down, invalid key) show a friendly message, not a stack trace. The app stays usable.
  - *Evidence:* `test_ac55_provider_error_is_friendly_and_the_app_stays_usable` (missing Groq key → "The language-model provider is not configured: … GROQ_API_KEY …", no traceback, chat input still enabled); `test_friendly_errors` (401 → "rejected the API key", 429 → rate limit, connection/timeout → "could not be reached", invalid model output); `test_planner_crash_becomes_a_friendly_error`; `test_redact_removes_configured_keys` and `test_api_key_never_appears_in_the_ui` (a canary key never shows in the UI or messages).
- [x] AC5.6 Every price shown is labelled as an estimate with its source/as-of date available on hover/expander.
  - *Evidence:* `test_ac56_every_price_is_an_estimate_with_its_source` (every `≈ AED` amount says "estimate"; the hover help gives the seed price range, source and as-of date; the estimates banner and the data attribution are shown); `test_every_slot_price_has_an_estimate_label_and_source` (every priced slot, transport leg and the hotel trace back to a `Provenance` with an as-of date); `test_exports_are_complete_labelled_and_secret_free`.

*Status notes (2026-11-17):*
- **As built:** `app/streamlit_app.py` (home), `app/pages/1_Planner.py`, `app/pages/2_Sessions.py`, shared `app/ui_shared.py` (cached services, sidebar: mode selector, provider/model, session picker, new session) and `app/ui_itinerary.py` (overview with budget gauge and cost table, one tab per day with time-slot cards and indoor/outdoor badges, pydeck map, JSON/Markdown export). Services (framework-free, unit-tested): `services/session.py` (`SessionService`: sessions, history, versions, `diff`), `services/refinement.py` (`classify` → `Refinement` with a `PlanEdit`), `services/runs.py` (`PlannerHub` one planner per mode, `RunManager` background runs, `friendly_error`/`redact`), `services/presentation.py` (price notes with provenance, cost rows, exports). New models: `PlanEdit`, `DayConstraint` (`models/planning.py`), `VersionDiff`/`DayDiff`/`SlotRef` (`models/versions.py`).
- **Refinement in both modes:** both planners now accept `stream(..., edit=PlanEdit)` (`Planner` protocol updated). The edit carries the saved request (the requirements step merges into it), `keep_request` (local edits skip re-parsing), the previous itinerary, locked days, per-day constraints and instructions. Shared core additions: `ComposerInput.day_constraints` (indoor-only days; the composer also avoids attractions already on locked days, enforced in `assemble`), `requirements.unchanged()`, `revision.applicable_locks()` (locks are dropped if the request's dates changed). Refinements work across modes, because they are built from SQLite (`test_refinement_works_across_modes`).
- **Rerun safety:** runs execute in `RunManager` threads, never in the page script, so clicks or reruns cannot cancel or duplicate them. One active run per session (`RunInProgress`; chat input disabled meanwhile; `test_reruns_never_start_a_duplicate_run`). A fragment polls progress once a second only while a run is active, then triggers one full rerun. `st.session_state` holds only `session_id`, the selected mode and widget state; `?session=<id>` keeps the session across browser refreshes.
- **Phase 6 dependencies (not built here):** the plan's Phase 5 asks for streamed progress (`st.status`), which shows each finished node/task with its review iteration. Trace visualisation (graph view, Gantt timeline, inspector, run comparison) and persisting trace events to SQLite are Phase 6. Review issues and the attempts table are shown for runs made by the running app process; after a restart the saved version keeps its approved flag, and the assistant's reply lists unresolved issues, but the per-attempt detail needs Phase 6's trace storage.
- **Relative changes:** only *budget* changes are resolved deterministically ("by 20%", "add AED 500"). Other relative changes ("one more adult", "make it 4 days") go to the requirements agent; a real LLM handles them, but the offline mock's heuristic does not.
- **Fixed on the way:** the mock's heuristic requirements parser read the word "beach" as "each" and made the budget per person (found in a UI check: "AED 9000, beach and culture" became AED 18 000). Now whole-word (`test_heuristic_does_not_read_beach_as_each`).
- **Groq usage:** no real-provider run was needed for Phase 5 (all UI tests use the mock). The AC2.3 `llm` test was accidentally run twice while re-running one test file without `-m "not llm"`; both runs were rejected with Groq's daily-limit 429 (no tokens consumed; the quota had already been used up by earlier real runs).
- 572 offline tests pass (+57; 13 Streamlit AppTest UI tests), 96 % line+branch coverage on `src/`; `ruff check .` and `mypy src` (strict) are clean.

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
- [x] AC6.1 For a LangGraph run, the Gantt chart shows the 5 research nodes overlapping and each review iteration as a separate band.
  - *Evidence:* `tests/integration/test_observability_runs.py::test_ac61_langgraph_gantt_shows_parallel_research_and_iteration_bands`. Research agents are stubbed to sleep 0.4 s and the first composition is over budget (one refinement). From the trace *read back from SQLite*: one lane per node, all 10 pairs of research bars overlap, the measured window is < 1 s while the sum of durations is > 1 s, and the bands are `[iteration 0: first draft, iteration 1: revision 1]`, in order and not overlapping. The Plotly figure itself is checked too: traces `steps · first draft` / `steps · revision 1`, two shaded band shapes with those labels, and the five research bars of the first-draft trace overlap. UI: `tests/ui/test_streamlit_app.py::test_ac61_ac64_execution_page_graph_timeline_and_inspector`.
- [x] AC6.2 For a CrewAI run, the timeline shows tasks per role, async overlap and guardrail retries.
  - *Evidence:* `test_ac62_crewai_timeline_shows_roles_async_overlap_and_guardrail_retries`. The run is sequential, with research stubbed to sleep 0.4 s and a midday outdoor slot injected in July. Lanes are the roles (Attractions Specialist … Quality Reviewer). All 10 pairs of async research task windows overlap. The Itinerary Architect's task bar says "1 guardrail retry" and its two stage bars are iterations 0 and 1. One `guardrail_rejected` marker (`heat.midday_outdoor`) sits on that lane, the run counts 1 guardrail rejection, and the figure has the traces "CrewAI task window", "Guardrail rejected → retry" and "LLM call". Hierarchical mode: `test_hierarchical_timeline_shows_delegations_and_one_by_one_research` (≥ 9 delegation markers on the Trip Manager lane, research measured as *not* parallel, specialists' work on their own lanes, every event inside the run's time window).
- [x] AC6.3 Graph view highlights the node currently running during a live run (updates at least once per node transition).
  - *Evidence:* `test_ac63_live_status_from_sqlite_highlights_the_running_step[langgraph|crewai_sequential]`: while a blocking composer holds the run, status read **from SQLite only** shows `compose_itinerary` running, parse, research and budget done, review pending; the DOT marks the node "◐ running" with a thick border. After release, everything is done. `test_ac63_every_step_transition_is_written_immediately`: every node start/end event is the last event of a write batch, so a 1 s poller sees each transition. UI: `test_ac63_live_graph_highlights_the_running_step` starts a real `RunManager` run with a blocking composer. The Execution page's live fragment renders the graph with `compose_itinerary` running ("now running: **compose_itinerary**"), a re-poll starts no new run, and after release the page shows `finalize` done and nothing running.
- [x] AC6.4 Inspector shows the validated output model for any selected step.
  - *Evidence:* `test_ac64_every_step_output_validates_as_its_model[langgraph|crewai_sequential|crewai_hierarchical]`. Every recorded step output (RequirementsResult, three SelectionResults, TransportPlan, WeatherEventsResult, ResearchBundle, BudgetPlan, Itinerary, list[RuleViolation], ReviewReport, RevisionPlan) re-validates against its Pydantic model. The compose output rebuilds an `Itinerary`, the inspector shows the input summary, and LLM calls show the prompt preview and token counts. UI: selecting the compose end event shows "✓ validates as `Itinerary`". Unit: tampered output → "does not validate", unknown schema and oversized output are reported, not hidden.
- [x] AC6.5 Historical runs can be visualised after app restart (data comes from SQLite only).
  - *Evidence:* `test_ac65_history_is_rebuilt_from_sqlite_in_a_new_process` (a separate Python process rebuilds the same event count, LLM calls, tokens, iterations, outcome, timeline rows, parallelism and attempts from the DB file). UI: `test_ac65_execution_history_survives_an_app_restart` (services closed and caches cleared; the Execution page shows the graph all done, the timeline and the outcome; the Planner shows the last run's review attempts "from the saved trace"). Schema v1 databases are upgraded in place (`test_v1_database_is_upgraded_in_place_without_losing_runs`).

*Status notes (2026-11-17):*
- **As built:** see `architecture.md` §9a "Observability (as built, Phase 6)" and §8 (schema v2). New: `services/observability.py` (spans, step status, timeline rows, iteration bands, markers, parallelism, metrics, comparison and fairness, inspector, graph specs to DOT/Mermaid), `tracing/payloads.py` (`record_io`: safe input summaries and validated outputs on step spans), `redaction.py` (shared secret redaction), `storage/schema_v2.sql`, `app/pages/3_Execution.py` + `app/ui_execution.py`, `describe_graph()` in `orchestration/langgraph/graph.py`. `orchestration.common.RunStats` is the SQLite writer for both modes.
- **Task 1 (bus subscribers):** the SQLite subscriber is the per-run `RunStats`, not a global one: events carry no session and the run row must exist first (foreign key), and the runner already owned a per-run subscriber. The "Streamlit live queue" is SQLite itself. The page polls the run's events once a second while it is active in this app's `RunManager`, so live and historical views use the same code, and a view in another browser tab or after a restart needs no in-memory queue.
- **Task 2 (graph rendering):** the plan allowed Mermaid via a CDN or `st.graphviz_chart`. Graphviz was chosen because it renders in the browser with no CDN and works offline (hard rule 2). The same status-coloured graph is exported as Mermaid in an expander. Each node shows its status as a word and icon as well as a colour.
- **Task 3 (aggregation):** `runs` gains `tokens_in`, `tokens_out`, `llm_latency_s`, `iterations` and `context_json` (schema v2). The context records provider, model, a seed-data fingerprint, the offline flag, the loop cap and the CrewAI process, so the comparison view can say whether two runs are like for like (same message, same parsed request, same dataset, same model). It never holds secrets.
- **Comparison is honest by construction:** every number comes from recorded timestamps and LLM-call events. Anything not recorded shows "—" (runs made before Phase 6 have only their run record, and the page says so). Each side is one run with no repetition, which the page states; repeated scenarios are Phase 7. With the mock provider, the page warns that token counts are prompt-length approximations and latencies are not real-model measurements. No real-provider run was made in Phase 6 (no Groq quota used).
- **Bugs found on the way:** (1) CrewAI 1.15 stamps tool-usage and delegation events with naive *local* time while its base events are aware UTC. The Phase 4 listener read them as UTC, so delegations were off by the UTC offset (+5:30 h on this machine), which showed up as a 20 000 s timeline. Fixed in `listener._aware`, with regression tests. (2) `agent_span` error events had no `kind`, so a failed node could not be paired with its start. They now carry it. (3) `safe_error` now also redacts configured keys, so a provider SDK echoing a key cannot put it into `runs.error` or `PlanResult.error` (`test_no_secret_reaches_the_stored_trace_or_its_export` dumps the whole DB).
- **Secrets:** the recorder redacts payloads before writing, the inspector redacts again before display, and the JSON export is redacted. Prompt previews hold only our instructions and the traveller's words. Tests use a canary key on unit, integration and UI paths.
- **Manual check (2026-11-17):** `streamlit run app/streamlit_app.py` with `LLM_PROVIDER=mock`, `HTTP_OFFLINE=true` on a scratch DB with five mock runs (LangGraph, CrewAI sequential and hierarchical on the same request, plus a fault-injected July heat defect per framework). `/_stcore/health` returned `ok`, and headless-Edge screenshots of the graph (LangGraph, CrewAI hierarchical), the timelines (LangGraph with two iteration bands, CrewAI with a guardrail retry, hierarchical with delegations), the inspector, state & review and run comparison all rendered without errors.
- **Known limitations:** the live view refreshes once a second (a step shorter than that can start and end between two polls; every transition is still persisted and shown in the timeline). Only runs started by *this* app process are shown as live; a `running` row left by a killed process is shown as stopped steps. CrewAI task windows include the agent's own reasoning around the tool, so they are wider than the stage bars. The graph and timeline are for one run at a time, and the comparison is two runs side by side.
- 625 offline tests pass (+53), 96 % line+branch coverage on `src/` (`services/observability.py` 99 %); `ruff check .` and `mypy src` (strict) are clean.

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
