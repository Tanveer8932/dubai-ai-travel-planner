"""Shared pytest fixtures. The whole suite runs offline with the mock LLM provider."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

# Must be set before any test imports CrewAI / OpenTelemetry.
# The caller's provider is remembered so ``@pytest.mark.llm`` tests can use it.
os.environ["DUBAI_TEST_CALLER_LLM_PROVIDER"] = os.environ.get("LLM_PROVIDER", "")
os.environ["LLM_PROVIDER"] = "mock"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from dubai_planner.config import Settings, get_settings
from dubai_planner.data import http as http_module
from dubai_planner.data.http import HttpClient
from dubai_planner.llm.mock import MockLLM
from dubai_planner.tools.toolbox import Toolbox


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    """Make sure each test sees settings built from its own (monkeypatched) env."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _real_provider_for_llm_tests(monkeypatch: pytest.MonkeyPatch,
                                 request: pytest.FixtureRequest) -> None:
    """``llm``-marked tests use the caller's provider (shell env, else `.env`), not mock."""
    if not request.node.get_closest_marker("llm"):
        return
    caller = os.environ.get("DUBAI_TEST_CALLER_LLM_PROVIDER", "")
    if caller:
        monkeypatch.setenv("LLM_PROVIDER", caller)
    else:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)  # fall through to `.env`
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Block real sockets so no test can silently reach the internet.

    ``respx`` intercepts httpx before any socket is opened, so mocked HTTP still works.
    Tests marked ``network`` or ``llm`` are exempt. Loopback connects stay allowed: on
    Windows every asyncio event loop (CrewAI's event bus starts one) builds its internal
    self-pipe from a 127.0.0.1 socket pair, which never leaves the machine.
    """
    if request.node.get_closest_marker("network") or request.node.get_closest_marker("llm"):
        return
    real_connect = socket.socket.connect

    def guard(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network access is disabled in offline tests")

    def guarded_connect(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        if is_loopback(address):
            return real_connect(sock, address, *args, **kwargs)
        return guard()

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guard)


def is_loopback(address: Any) -> bool:
    """True for a 127.x.x.x / ::1 socket address (local-only, not network access)."""
    host = address[0] if isinstance(address, tuple) and address else None
    return isinstance(host, str) and (host.startswith("127.") or host == "::1")


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test HTTP cache and DB so tests never share or pollute `data/`."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(http_module, "_host_limiters", {})


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def toolbox(monkeypatch: pytest.MonkeyPatch) -> Toolbox:
    """Toolbox whose public-data clients never touch the network (cache → seed)."""
    monkeypatch.setenv("HTTP_OFFLINE", "true")
    get_settings.cache_clear()
    return Toolbox.default()


@pytest.fixture
def mock_llm() -> MockLLM:
    """A fresh mock engine (scripts/calls are not shared with other tests)."""
    return MockLLM()


@pytest.fixture
def make_http(settings: Settings) -> Callable[..., HttpClient]:
    """Build an online HttpClient with no-op sleeps (fast retries in tests)."""

    def factory(namespace: str = "test", **kwargs: Any) -> HttpClient:
        kwargs.setdefault("offline", False)
        kwargs.setdefault("sleep", lambda _s: None)
        kwargs.setdefault("backoff_s", 0)
        return HttpClient(namespace, settings=settings, **kwargs)

    return factory
