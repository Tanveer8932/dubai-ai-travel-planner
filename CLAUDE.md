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
**Phase 1 complete (2026-11-15); Phase 2 next.** Phase 0: Python 3.12.10 in `.venv`; deps pinned in `pyproject.toml` (langgraph 1.2.22, langgraph-checkpoint-sqlite 3.1.3, langchain-core 1.8.1, crewai 1.15.41, streamlit 1.67.0); `config.Settings` + `dubai_planner.bootstrap()`. Phase 1: Pydantic contracts in `models/` (money = `AED` Decimal type, `Provenance`/`DataKind` on every fact), curated seed in `data/seed/` loaded via `data.seed.load_catalog()`, cached keyless clients in `data/` (`http.HttpClient` = cache → network → stale cache → `SourceUnavailableError`, each client falls back to seed), SQLite `storage.Repository` (versioned itineraries, runs, traces, evals; `PRAGMA user_version` migrations). 179 offline tests pass. Ollama not installed yet (optional). Update this line at each phase transition.

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
python scripts/run_eval.py --provider mock
```

## Key env vars (see `.env.example`)
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_BASE_URL`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DB_PATH`, `CHECKPOINT_DB_PATH`, `CACHE_DIR`, `CACHE_TTL_HOURS`, `SEED_DIR`, `HTTP_USER_AGENT`, `HTTP_TIMEOUT_S`, `HTTP_OFFLINE`, `MAX_REVIEW_ITERATIONS`, `BUDGET_TOLERANCE`, `CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED`.

## Gotchas
- Project folder is inside **OneDrive**: sync can lock `*.db` files and bloat `.venv`. Exclude them from sync or move the repo outside OneDrive if you see `database is locked`.
- Parallel LangGraph branches writing the same state key need a reducer, or you'll get `InvalidUpdateError`.
- Small local models often break JSON. Prefer `qwen2.5:7b` for structured output; keep per-agent schemas small.
- ECB/Frankfurter has no AED rate. Convert via USD using the fixed peg 3.6725.
- Prices are **estimates** from curated seed data. Always label them so in the UI (`PriceRange.label()` does). Never mark seed data `verified`; that kind needs an authoritative `source_url`.
- `HTTP_USER_AGENT` needs a real contact: Wikivoyage returns 403 without one, Overpass returns 406 for the `example.com` placeholder. Clients then silently use seed fallbacks (a warning is logged).
- Nager.Date has no UAE data (HTTP 204): holidays come from `data/seed/holidays.json` + `hijridate`.
- Tests block real sockets (`tests/conftest.py`); mock HTTP with `respx` and inject clients via `make_http`.
