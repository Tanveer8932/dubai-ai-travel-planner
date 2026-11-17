# CLAUDE.md — Dubai AI Travel Planner

Guidance for Claude Code (and humans) working in this repository.

## Project in one paragraph
An academic multi-agent system that turns a natural-language request into a budget-aware, feasible, day-by-day Dubai itinerary. The same framework-agnostic agent core is orchestrated two ways, **LangGraph** (graph, parallel fan-out, review loop, SQLite checkpoints) and **CrewAI** (role-based crew, async tasks, guardrail review), so the two can be evaluated against each other. UI is **Streamlit**; persistence is **SQLite + JSON**; all I/O contracts are **Pydantic v2**.

## Read first
- `problemStatement.md`: goals, scope, success criteria S1–S8
- `architecture.md`: diagrams, state, agents, rules, DB schema, repo layout
- `research.md`: framework notes, free data sources, provider options, risks
- `implementationPlan.md`: phases 0–7 with acceptance criteria (ACs)

## Current phase
**Phase 6 complete (2026-11-17); Phase 7 (evaluation, hardening, report) is next.** Phase 6: `app/pages/3_Execution.py` + `app/ui_execution.py` (graph from LangGraph's `get_graph()` / `CrewFactory.describe()` via `st.graphviz_chart`, Plotly Gantt with iteration bands and LLM/guardrail/delegation markers, inspector that re-validates recorded outputs, state & review, run comparison with fairness checks, live polling); `services/observability.py` (framework-free analysis of stored traces); `tracing/payloads.record_io`; `orchestration.common.RunStats` persists each run's events to SQLite at every step transition (secrets redacted, `redaction.py`); schema v2 (`runs` token split, LLM latency, iterations, `context_json`). AC6.1–AC6.5 pass; 625 offline tests. Phase 5: Streamlit app `app/streamlit_app.py` + `app/pages/{1_Planner,2_Sessions}.py` with shared `app/ui_shared.py` / `app/ui_itinerary.py`; services `services/{session,refinement,runs,presentation}.py` (`SessionService` + version `diff`, `classify` → `PlanEdit`, `PlannerHub` + `RunManager` background runs, price notes with provenance + exports); both planners accept `stream(..., edit=PlanEdit)`; composer `day_constraints`. AC5.1–AC5.6 pass (AppTest UI tests + manual headless run). Phase 4: `orchestration/crewai/` (`config/agents.yaml` + `tasks.yaml` 10 roles / 9 tasks, `board.py` per-run typed hand-over + stage catalogue, `stages.py` agent-core calls, `tools.py` `StageTool` (`result_as_answer`) + `ToolboxTool` wrappers, `guardrails.py` stage/itinerary/review guardrails, `listener.py` CrewAI events → `TraceEvent`, `crew.py` `CrewFactory` (requirements / planning / revision crews, sequential or hierarchical), `runner.py` `CrewAIPlanner` + `run_crewai()`), `services/planner.py` (`Planner` protocol + `open_planner(mode)`), shared `agents/suite.py` (`AgentSuite`) and `orchestration/common.py`. AC4.1–AC4.6 pass with the mock; a real Groq CrewAI run was approved in 297.6 s (≈46.6k tokens). 515 offline tests pass. Phase 3: `orchestration/langgraph/` (`state.py` PlannerState + `merge_research` reducer, `nodes.py` thin wrappers + injectable `AgentSuite`, `graph.py` parallel fan-out / clarification `interrupt` / bounded review loop, `checkpoint.py` SqliteSaver with a model allowlist, `runner.py` `LangGraphPlanner` + `run_langgraph()`), `agents/revision.py` (targeted revision, shared), `models/planning.py` (`PlanResult` contract for both modes). AC3.1–AC3.5 pass with the mock; AC3.6 passed on Groq `openai/gpt-oss-20b` (approved in 128 s). 455 offline tests pass. Phase 2 complete: AC2.3 met and re-checked on Groq (`langchain-groq==1.2.0`, `LLM_PROVIDER=groq` + `GROQ_API_KEY` in `.env`). Phase 2: `llm/` (factory for LangChain + CrewAI from one config, neutral `ChatLLM` + adapters, `invoke_structured` with one repair retry, deterministic `MockLLM` exposed to both frameworks), `tools/toolbox.Toolbox` (seed search, travel, fares, FX, costs; `TOOL_SPECS`), 9 agents in `agents/` (`run(input, llm, tools)`; prompts in `agents/prompts/*.md`), 11 rules in `validation/rules.py`, minimal `tracing/bus.py`. Phase 0: Python 3.12.10 in `.venv`; deps pinned in `pyproject.toml` (langgraph 1.2.22, langgraph-checkpoint-sqlite 3.1.3, langchain-core 1.8.1, crewai 1.15.41, streamlit 1.67.0); `config.Settings` + `dubai_planner.bootstrap()`. Phase 1: Pydantic contracts in `models/` (money = `AED` Decimal type, `Provenance`/`DataKind` on every fact), curated seed in `data/seed/` loaded via `data.seed.load_catalog()`, cached keyless clients in `data/` (`http.HttpClient` = cache → network → stale cache → `SourceUnavailableError`, each client falls back to seed), SQLite `storage.Repository` (versioned itineraries, runs, traces, evals; `PRAGMA user_version` migrations). 179 offline tests pass. Ollama not installed yet (optional). Update this line at each phase transition.

## Hard rules
1. **Never hardcode secrets.** Read keys only via `config.Settings` (pydantic-settings, `SecretStr`) from env / `.env`. `.env` is git-ignored; document new vars in `.env.example` with empty values. Never print, log, trace or export secret values.
2. **Free first.** Default `LLM_PROVIDER=ollama`; tests use `LLM_PROVIDER=mock`. Paid providers are optional and must never be required. Data comes from keyless public APIs (Open-Meteo, OSM Overpass/Nominatim, Wikivoyage, Frankfurter) with JSON caching and seed fallback.
3. **Shared core, thin adapters.** Business logic (prompts, tools, models, validators, budget maths) lives in `src/dubai_planner/{agents,tools,models,validation}`. `orchestration/langgraph` and `orchestration/crewai` only wire it up. Never duplicate agent logic per framework.
4. **Deterministic maths.** Costs, budget checks, travel-time estimates and rule checks are Python, never LLM output. Use `Decimal` for money (AED).
5. **No invented venues.** Agents select items by ID from tool results; the composer references IDs. The `grounding.known_entities` rule enforces this.
6. **Pydantic at every boundary.** LLM output → `model_validate_json` → one repair retry → fail loudly.
7. **Offline tests.** `pytest -m "not llm"` must pass with no network and no API keys. Mock HTTP with `respx`; mock LLM via `llm/mock.py` fixtures.
8. **Respect source policies.** Nominatim ≤ 1 req/s with a descriptive User-Agent; cache everything; keep OSM/Wikivoyage attribution in the UI.
9. **Do not change scope silently.** If an AC can't be met, record why in `implementationPlan.md` instead of weakening the test.

## Conventions
- Python 3.11/3.12, `src/` layout, package `dubai_planner`. Type hints everywhere; `ruff` for lint/format, `mypy` for types.
- Modules: `snake_case`; Pydantic models: `PascalCase`; agent modules named after the agent (`agents/attractions.py`).
- Prompts live in `src/dubai_planner/agents/prompts/*.md`, not inline strings.
- CrewAI roles/tasks in YAML (`orchestration/crewai/config/`).
- Every agent step emits `TraceEvent`s via `tracing.bus`. Don't print for observability.
- Streamlit: `st.session_state` holds only UI state and `session_id`; truth lives in SQLite.
- Pin framework versions in `pyproject.toml`; LangGraph/CrewAI APIs change often. Check installed-version docs before using an API marked **[verify]** in `research.md`.

## Commands (available after Phase 0)
```powershell
# setup (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,langgraph,crewai,ui]"
Copy-Item .env.example .env        # then fill in only what you need

# quality
pytest -m "not llm"                # offline suite
pytest -m llm                      # needs a real provider
ruff check . ; mypy src

# run
streamlit run app/streamlit_app.py
python scripts/init_db.py
python scripts/export_langgraph.py  # graph → docs/generated/langgraph.mmd
python scripts/run_eval.py --provider mock
```

## Key env vars (see `.env.example`)
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_MODEL__<AGENT>`, `LLM_TEMPERATURE`, `LLM_BASE_URL`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_REPAIR_ATTEMPTS`, `LLM_MAX_TOKENS`, `LLM_REASONING_EFFORT`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DB_PATH`, `CHECKPOINT_DB_PATH`, `CACHE_DIR`, `CACHE_TTL_HOURS`, `SEED_DIR`, `HTTP_USER_AGENT`, `HTTP_TIMEOUT_S`, `HTTP_OFFLINE`, `MAX_REVIEW_ITERATIONS`, `BUDGET_TOLERANCE`, `CREWAI_PROCESS`, `CREWAI_AGENT_MAX_ITER`, `CREWAI_DISABLE_TELEMETRY`, `CREWAI_TRACING_ENABLED`, `OTEL_SDK_DISABLED`.

## Gotchas
- Project folder is inside **OneDrive**: sync can lock `*.db` files and bloat `.venv`. Exclude them from sync or move the repo outside OneDrive if you see `database is locked`.
- Parallel LangGraph branches writing the same state key need a reducer, or you'll get `InvalidUpdateError`.
- Small local models often break JSON. Prefer `qwen2.5:7b` for structured output; keep per-agent schemas small.
- ECB/Frankfurter has no AED rate. Convert via USD using the fixed peg 3.6725.
- Prices are **estimates** from curated seed data. Always label them so in the UI (`PriceRange.label()` does). Never mark seed data `verified`; that kind needs an authoritative `source_url`.
- `HTTP_USER_AGENT` needs a real contact: Wikivoyage returns 403 without one, Overpass returns 406 for the `example.com` placeholder. Clients then silently use seed fallbacks (a warning is logged).
- Nager.Date has no UAE data (HTTP 204): holidays come from `data/seed/holidays.json` + `hijridate`.
- Tests block real sockets (`tests/conftest.py`); mock HTTP with `respx` and inject clients via `make_http`.
- Agent tests: use the `toolbox` fixture (forces `HTTP_OFFLINE`) and the `mock_llm` fixture. Feed malformed or hallucinated replies with `mock_llm.script("<agent>", raw, ...)`; scripted replies are consumed before the deterministic generators.
- Generic Pydantic results need a concrete type at runtime (`SelectionResult[AttractionOption]`, not `SelectionResult[V]`), or subclass fields are dropped on serialisation.
- Groq: `llama-3.1-8b-instant` is no longer served (check `client.models.list()`); default is `openai/gpt-oss-20b`. Its default forced tool call returns HTTP 400 `tool_use_failed` when the model replies in plain JSON, so Groq uses `method="json_schema"` (`llm/base.NATIVE_STRUCTURED_METHODS`) and rejected native replies go through validate → repair. Free tier: 8 000 tokens/min, so back-to-back real runs can 429.
- Groq output limits: its default 2048-token cap truncates gpt-oss JSON (hidden reasoning counts as output), so the factory defaults Groq to `LLM_MAX_TOKENS=4096` and gpt-oss to `LLM_REASONING_EFFORT=low`. Groq counts `max_tokens` against TPM. Free tier: **200 000 tokens/day**, and one planning run uses ≈ 18k tokens, so budget real runs (AC7.2) and never run real-model diagnostics casually.
- LangGraph: stream with `durability="sync"` (runner does). The default `async` can lose the last checkpoint writes when the process is killed. Parallel research branches write partial dicts to `research` (reducer `merge_research`); `None` resets it.
- LangGraph checkpoints use `planner_serializer()` (explicit allowlist of `dubai_planner.models` classes). State may hold only those models and plain types; a new state type must live in `models/`.
- Graph tests: stub agents through `AgentSuite(...)` (e.g. `AgentSuite(composer=spy)`), never by patching agent modules. Mock-provider runs prove orchestration only, never real-model quality.
- `tests/conftest.py` forces `LLM_PROVIDER=mock`, except for `@pytest.mark.llm` tests, which use the caller's shell env or `.env`.
- CrewAI 1.15 has no native Groq client and LiteLLM isn't installed: Groq and Gemini use their OpenAI-compatible URLs (`llm/factory.py`).
- CrewAI: keep `CREWAI_TRACING_ENABLED=false` (exported by `bootstrap()`; crews also pass `tracing=False`). Otherwise CrewAI shows a "view your execution traces?" console prompt that blocks every kickoff for 20 s.
- CrewAI swallows tool exceptions and lets the agent retry (a failing tool ran 12 times, then the task "completed" with junk). Stage tools use `tool_failure_policy="raise"`, record the first failure on the run board and re-raise without redoing the work; every guardrail re-raises a recorded failure.
- CrewAI guardrails must be functions or bound methods (callable objects are rejected) and need a real `tuple[bool, Any]` return annotation, so `guardrails.py` has no `from __future__ import annotations`.
- CrewAI writes a replay log of task outputs to the user's app-data folder, and its JSON encoder crashes on `datetime.time`. Crews are `PlanningCrew` (no replay log); never build a plain `Crew` for planning.
- Groq rejects function schemas that have `required` without `properties`, or a regex `pattern` with lookarounds (pydantic `Decimal`). CrewAI tools need ≥ 1 property (stage tools take an optional `notes`), and toolbox wrappers use `provider_safe(args_model)`.
- Hierarchical CrewAI: one manager executor runs every task, so research is not async there. Tasks set no `output_pydantic`, because CrewAI would add a manager LLM call per task to re-type specialist output. The guardrail attaches the typed model instead.
- On Windows, asyncio event loops (CrewAI's event bus) use a loopback socket pair. The offline test guard allows 127.0.0.1/::1 and blocks everything else.
- CrewAI tests: inject agents via `CrewAIPlanner(agents=AgentSuite(...), llm=mock_llm)`. The mock answers CrewAI ReAct prompts (it calls the tool the task names; the manager delegates to the task's "Responsible specialist"). Script raw agent replies with `mock_llm.script("crewai:<Role>", ...)`.
- Streamlit: never run a planner inside the page script. Pages call `RunManager.start()` (one background run per session, `RunInProgress` otherwise) and poll with a fragment only while a run is active. A run in the script would be cancelled by any click and could start twice.
- Streamlit services are `st.cache_resource`, keyed by DB/checkpoint path (`ui_shared._services`), so AppTest tests stay isolated. UI tests: `tests/ui/test_streamlit_app.py` drives the real pages with `streamlit.testing.v1.AppTest` and waits on `services().runs.last(session).wait()`.
- Refinements: `services.refinement.classify` builds a `PlanEdit` from SQLite. Relative *budget* changes are resolved in Python and stated in the message; a day-specific edit locks the other days and skips re-parsing. Never resolve maths in prompts.
- Observability: every orchestration step span must carry `payload.kind` (node / stage / crew_task / crew) and call `record_io(span, inputs=..., output=model)`; the recorder writes at those transitions and the Execution page pairs spans by (agent, kind, started_at). Never compute a displayed metric from anything but the run row and its trace events; missing values stay `None` ("—").
- CrewAI tool-usage/delegation events use naive *local* time (base events are aware UTC); `listener._aware` converts naive values from local time. Never `replace(tzinfo=UTC)` on CrewAI timestamps.
- Pages run standalone in AppTest, where `st.page_link` to a sibling page fails: use `ui_shared.nav_link`. Don't write repo files with PowerShell `Set-Content`/`Out-File` (it re-encodes UTF-8 as ANSI and corrupts non-ASCII characters).
- Always pass `-m "not llm"` when re-running a test file that holds `@pytest.mark.llm` tests (`test_requirements.py`, `test_langgraph_planner.py`, `test_crewai_planner.py`); without it they call the real provider and spend Groq quota.
