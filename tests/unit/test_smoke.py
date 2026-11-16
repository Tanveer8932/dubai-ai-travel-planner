"""Phase 0 smoke tests: imports, settings, secrets policy, docs present."""

from __future__ import annotations

import importlib
import logging
import re
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest
from dotenv import dotenv_values

import dubai_planner
from dubai_planner.config import PROJECT_ROOT, LLMProvider, Settings
from dubai_planner.logging_config import RedactSecretsFilter

SETTINGS_ENV_VARS = [
    *(k for k in dotenv_values(PROJECT_ROOT / ".env.example")),
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every documented settings variable from the process env."""
    for key in SETTINGS_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "module",
    ["langgraph", "langgraph.checkpoint.sqlite", "langchain_core", "crewai", "streamlit",
     "pydantic", "pydantic_settings", "httpx", "tenacity", "hijridate", "plotly", "pandas",
     "pydeck", "respx"],
)
def test_third_party_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    ["agents", "data", "evaluation", "llm", "models", "orchestration",
     "orchestration.langgraph", "orchestration.crewai", "services", "storage", "tools",
     "tracing", "validation"],
)
def test_package_skeleton_imports(module: str) -> None:
    importlib.import_module(f"dubai_planner.{module}")


def test_settings_load_from_env_example_with_mock(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC0.5: Settings load with only `.env.example` values, and LLM_PROVIDER=mock works."""
    env_file = tmp_path / ".env"
    shutil.copy(PROJECT_ROOT / ".env.example", env_file)

    s = Settings(_env_file=env_file)  # type: ignore[call-arg]
    assert s.llm_provider is LLMProvider.OLLAMA  # the documented default
    assert s.groq_api_key is None  # empty value in .env.example → unset
    assert s.max_review_iterations == 2
    assert s.budget_tolerance == Decimal("0.05")
    assert s.db_path == PROJECT_ROOT / "data" / "app.db"
    assert s.crewai_disable_telemetry and s.otel_sdk_disabled

    clean_env.setenv("LLM_PROVIDER", "mock")
    assert Settings(_env_file=env_file).llm_provider is LLMProvider.MOCK  # type: ignore[call-arg]


def test_invalid_provider_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="llm_provider"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_secrets_are_masked(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("GROQ_API_KEY", "gsk_dummyvalue123")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.groq_api_key is not None
    assert "gsk_dummyvalue123" not in repr(s)
    assert "gsk_dummyvalue123" not in s.model_dump_json()
    assert s.groq_api_key.get_secret_value() == "gsk_dummyvalue123"


def test_log_filter_redacts_secrets() -> None:
    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1, "using api_key=%s", ("abc123",), None
    )
    RedactSecretsFilter().filter(record)
    assert "abc123" not in record.getMessage()


def test_bootstrap_exports_telemetry_opt_out(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "mock")
    clean_env.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    s = dubai_planner.bootstrap()
    import os

    assert s.llm_provider is LLMProvider.MOCK
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true"
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


_SECRET_RE = re.compile(r"""(api[_-]?key|secret|token)\s*=\s*['"][A-Za-z0-9]""", re.IGNORECASE)


def test_no_hardcoded_secrets() -> None:
    """AC0.6 (file-system version of the `git grep` check)."""
    tracked_suffixes = {".py", ".toml", ".md", ".yaml", ".yml", ".json", ".ps1", ".example"}
    skip_dirs = {".venv", ".git", "data", "__pycache__", ".pytest_cache", ".ruff_cache",
                 ".mypy_cache"}
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in tracked_suffixes
        and not skip_dirs.intersection(path.relative_to(PROJECT_ROOT).parts)
        and path.name != "test_smoke.py"  # this file contains the regex itself
        and _SECRET_RE.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "doc",
    ["CLAUDE.md", "problemStatement.md", "architecture.md", "research.md",
     "implementationPlan.md", "README.md"],
)
def test_docs_exist(doc: str) -> None:
    """AC0.7: the five design docs (plus README) exist and are non-empty."""
    assert (PROJECT_ROOT / doc).stat().st_size > 0


def test_mermaid_blocks_are_closed() -> None:
    """AC0.7 (static part): every ```mermaid fence is closed and has a diagram type."""
    for doc in ["architecture.md", "problemStatement.md", "research.md", "implementationPlan.md"]:
        text = (PROJECT_ROOT / doc).read_text(encoding="utf-8")
        blocks = re.findall(r"```mermaid\n(.*?)```", text, flags=re.DOTALL)
        assert text.count("```mermaid") == len(blocks), doc
        for block in blocks:
            first = block.strip().splitlines()[0].split()[0]
            assert first in {"flowchart", "graph", "classDiagram", "erDiagram",
                             "sequenceDiagram", "stateDiagram-v2", "gantt"}, (doc, first)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_grep_finds_no_secrets() -> None:
    """AC0.6 exactly as specified (only meaningful once files are tracked)."""
    git = shutil.which("git")
    assert git is not None
    result = subprocess.run(  # noqa: S603 - fixed argv, resolved git binary
        [git, "grep", "--untracked", "-iE", _SECRET_RE.pattern,
         "--", ".", ":!tests/unit/test_smoke.py"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    # exit 1 == no matches; 128 == not a git repo (covered by the filesystem test above)
    assert result.returncode in (1, 128), result.stdout
