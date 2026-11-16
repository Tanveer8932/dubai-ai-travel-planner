# Dubai AI Travel Planner

Academic multi-agent system that turns a natural-language request into a budget-aware,
day-by-day Dubai itinerary. The same agent core is orchestrated with **LangGraph** and
**CrewAI** so the two paradigms can be compared. See `problemStatement.md`,
`architecture.md`, `research.md` and `implementationPlan.md`.

> **Status:** Phase 1 (domain models, curated data, public-data clients, SQLite persistence).
> Agents and orchestration arrive in Phases 2–4.

## Requirements
- Windows 11 (macOS/Linux also work), **Python 3.11 or 3.12** (the supported range for this project).
  On Windows, turn off the Microsoft Store `python` aliases
  (*Settings → Apps → Advanced app settings → App execution aliases*).
- Git.
- Optional: [Ollama](https://ollama.com) for a free local LLM
  (`ollama pull qwen2.5:7b`). Not needed for tests, which use the offline mock provider.
- Optional: [`uv`](https://docs.astral.sh/uv/) for faster installs.

## Quick start (Windows PowerShell)
```powershell
git clone <repo-url> dubai-ai-travel-planner; cd dubai-ai-travel-planner
.\scripts\tasks.ps1 setup          # creates .venv (if missing), installs deps, copies .env
.\.venv\Scripts\Activate.ps1
.\scripts\tasks.ps1 test           # offline test suite (LLM_PROVIDER=mock)
.\scripts\tasks.ps1 lint           # ruff + mypy
```

If script execution is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Manual equivalent:
```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,langgraph,crewai,ui]"
Copy-Item .env.example .env
pytest -m "not llm"
```

macOS/Linux: `python3.12 -m venv .venv && source .venv/bin/activate`, then the same `pip`
and `pytest` commands (`cp .env.example .env`).

## Configuration
All configuration comes from environment variables or a git-ignored `.env`
(documented in `.env.example`). **Never commit keys.** Every provider key is optional;
the default provider is local Ollama and tests use `LLM_PROVIDER=mock`.

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `ollama` \| `groq` \| `gemini` \| `openai` \| `anthropic` \| `mock` |
| `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_BASE_URL` | model selection (empty model/URL = provider default) |
| `LLM_MODEL__<AGENT>` | optional per-agent model, e.g. `LLM_MODEL__REQUIREMENTS=qwen2.5:3b` |
| `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_REPAIR_ATTEMPTS` | per-call timeout, bounded SDK retries, schema-repair retries (default 1) |
| `*_API_KEY` | only for the hosted provider you choose (a missing key gives an error naming the variable) |

Hosted providers need their LangChain integration (`pip install -e ".[providers]"`, or just the
one you use, e.g. `pip install langchain-groq`). Ollama (`langchain-ollama`) is optional; the
offline tests and the mock provider need none of them.
| `DB_PATH`, `CHECKPOINT_DB_PATH`, `CACHE_DIR`, `CACHE_TTL_HOURS`, `SEED_DIR` | storage |
| `HTTP_USER_AGENT` | required by OSM and Wikimedia — put a **real** contact email/URL here (Wikivoyage returns 403 without one; Overpass returns 406 for `example.com`) |
| `HTTP_TIMEOUT_S`, `HTTP_OFFLINE` | request timeout; `true` = never use the network (cache, then seed) |
| `MAX_REVIEW_ITERATIONS`, `BUDGET_TOLERANCE` | planner behaviour |
| `CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED` | telemetry opt-out (keep `true`) |
| `LOG_LEVEL` | logging verbosity |

## Task runner (`scripts/tasks.ps1`)
| Task | Does |
|------|------|
| `setup` | venv + editable install with `dev,langgraph,crewai,ui` extras + `.env` copy |
| `test` | `pytest -m "not llm"` (offline) |
| `lint` | `ruff check .` and `mypy src` |
| `run` | `streamlit run app/streamlit_app.py` *(available from Phase 5)* |
| `eval` | `python scripts/run_eval.py --provider mock` *(available from Phase 7)* |

## Data and persistence (Phase 1)
```powershell
python scripts/init_db.py                       # create/migrate data/app.db, validate seed data
python scripts/refresh_seed.py --limit 5        # dry run: enrich seed coords/hours from OSM
python scripts/refresh_seed.py --write          # apply (≤ 1 req/s to Nominatim, cached)
```
- `data/seed/*.json`: curated attractions, hotels, restaurants, areas, transport fares,
  events, climate normals, holidays and an FX snapshot. Every record carries `provenance`
  (`verified` | `api` | `curated` | `estimated`, plus source and `as_of`).
  **All seed prices are `estimated`** and are never presented as confirmed.
- Public sources (keyless, cached in `data/cache/`, seed fallback when offline):
  Open-Meteo, OSM Overpass and Nominatim, Wikivoyage, Frankfurter (AED via the 3.6725 USD peg),
  and `hijridate` for Ramadan/Eid.

## Notes
- The repo lives in OneDrive on the dev machine: if you see `database is locked`, exclude
  `data/*.db` and `.venv` from sync.
- Prices shown by the app are **estimates** from curated seed data.
- Map / POI data © OpenStreetMap contributors (ODbL); travel text from Wikivoyage (CC BY-SA).
