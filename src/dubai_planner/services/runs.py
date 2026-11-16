"""Planner runs for the UI: one background run per session, safe across reruns.

Streamlit re-executes the page script on every interaction. Running the planner inside
the script would tie a 2-5 minute run to one script execution (any click would cancel
it) and could start it twice. Instead:

- :class:`PlannerHub` owns one planner per mode (``services.planner.open_planner``),
  opened once per process and shared by all sessions;
- :class:`RunManager` starts each request in a background thread and keeps its
  progress (:class:`PlannerUpdate`s) and result in a :class:`RunHandle`. A second
  request for a session with an active run is refused (:class:`RunInProgress`), so a
  rerun can never start a duplicate; pages only *read* handles while polling.

Errors never reach the page as tracebacks: :func:`friendly_error` turns them into a
short message, with any configured API key value redacted.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from dubai_planner.config import Settings, get_settings
from dubai_planner.models import PlannerMode, PlannerUpdate, PlanResult, PlanStatus
from dubai_planner.services.planner import Planner, open_planner
from dubai_planner.services.refinement import Refinement, classify
from dubai_planner.services.session import SessionService
from dubai_planner.storage import Repository

SECRET_FIELDS = ("groq_api_key", "gemini_api_key", "openai_api_key", "anthropic_api_key")


class RunInProgress(RuntimeError):
    """The session already has a planner run in progress."""


def redact(text: str, settings: Settings | None = None) -> str:
    """Remove any configured API key value from ``text`` (defence in depth)."""
    s = settings or get_settings()
    for name in SECRET_FIELDS:
        secret = getattr(s, name, None)
        value = secret.get_secret_value() if secret is not None else ""
        if len(value) >= 6:
            text = text.replace(value, "***")
    return text


_FRIENDLY: list[tuple[tuple[str, ...], str]] = [
    (("LLMConfigError", "ProviderNotInstalledError"),
     "The language-model provider is not configured: {detail}"),
    (("401", "invalid api key", "invalid_api_key", "authentication", "unauthorized"),
     "The provider rejected the API key. Check the key in your .env file and restart the app."),
    (("429", "rate limit", "rate_limit"),
     "The provider's rate limit was reached (free tiers limit tokens per minute and per "
     "day). Wait a little and try again."),
    (("connection", "timeout", "timed out", "502", "503", "504", "unavailable"),
     "The language-model provider could not be reached. Check your connection or the "
     "provider's status and try again."),
    (("StructuredOutputError",),
     "The model's answer could not be validated, even after a repair attempt. Try again "
     "or rephrase the request."),
]


def friendly_error(error: str | BaseException, settings: Settings | None = None) -> str:
    """A short, secret-free message for the user (the run record keeps the detail)."""
    text = redact(error if isinstance(error, str) else f"{type(error).__name__}: {error}",
                  settings)
    lower = text.lower()
    for markers, message in _FRIENDLY:
        if any(m.lower() in lower for m in markers):
            detail = text.split(":", 1)[1].strip() if ":" in text else text
            return message.format(detail=detail[:300])
    return f"Planning failed: {text[:300]}"


class PlannerHub:
    """One planner per mode, opened lazily and kept for the life of the process."""

    def __init__(self, repo: Repository, *, settings: Settings | None = None,
                 checkpoint_path: Path | str | None = None, **planner_kwargs: Any) -> None:
        self.repo = repo
        self.settings = settings or get_settings()
        self.checkpoint_path = checkpoint_path
        self.planner_kwargs = planner_kwargs
        self._stack = ExitStack()
        self._planners: dict[PlannerMode, Planner] = {}
        self._lock = threading.Lock()

    def planner(self, mode: PlannerMode) -> Planner:
        with self._lock:
            if mode not in self._planners:
                self._planners[mode] = self._stack.enter_context(open_planner(
                    mode, checkpoint_path=self.checkpoint_path, repo=self.repo,
                    settings=self.settings, **self.planner_kwargs))
            return self._planners[mode]

    def forget_thread(self, session_id: str) -> None:
        """Drop a deleted session's LangGraph checkpoints (CrewAI keeps none)."""
        graph = getattr(self.planner(PlannerMode.LANGGRAPH), "graph", None)
        saver = getattr(graph, "checkpointer", None)
        if saver is not None and hasattr(saver, "delete_thread"):
            saver.delete_thread(session_id)

    def close(self) -> None:
        with self._lock:
            self._stack.close()
            self._planners.clear()


@dataclass
class RunHandle:
    """Progress and outcome of one planner run (read by the page while it polls)."""

    session_id: str
    mode: PlannerMode
    message: str
    refinement: Refinement
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updates: list[PlannerUpdate] = field(default_factory=list)
    result: PlanResult | None = None
    error: str | None = None
    finished_at: datetime | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def add(self, update: PlannerUpdate) -> None:
        with self._lock:
            self.updates.append(update)
            if update.result is not None:
                self.result = update.result

    def snapshot(self) -> list[PlannerUpdate]:
        with self._lock:
            return list(self.updates)

    def finish(self, error: str | None = None) -> None:
        self.error = error
        self.finished_at = datetime.now(UTC)
        self._done.set()

    @property
    def user_error(self) -> str | None:
        """What to tell the user if the run did not produce a usable plan."""
        if self.error:
            return self.error
        if self.result is not None and self.result.status is PlanStatus.FAILED:
            return friendly_error(self.result.error or "unknown error")
        return None


class RunManager:
    """Starts planner runs in background threads; at most one active run per session."""

    def __init__(self, hub: PlannerHub, sessions: SessionService, *,
                 today: Callable[[], date] = date.today) -> None:
        self.hub, self.sessions, self.today = hub, sessions, today
        self._runs: dict[str, RunHandle] = {}
        self._lock = threading.Lock()

    def active(self, session_id: str) -> RunHandle | None:
        with self._lock:
            handle = self._runs.get(session_id)
        return handle if handle is not None and not handle.done else None

    def last(self, session_id: str) -> RunHandle | None:
        with self._lock:
            return self._runs.get(session_id)

    def prepare(self, session_id: str, mode: PlannerMode, message: str) -> Refinement:
        """How the message will be planned (classification only, nothing is started)."""
        planner = self.hub.planner(mode)
        return classify(message, latest=self.sessions.latest(session_id),
                        pending_question=planner.pending_question(session_id) is not None)

    def start(self, session_id: str, mode: PlannerMode, message: str) -> RunHandle:
        """Start planning ``message``; raises :class:`RunInProgress` if one is running."""
        text = message.strip()
        if not text:
            raise ValueError("Please type a request.")
        with self._lock:
            current = self._runs.get(session_id)
            if current is not None and not current.done:
                raise RunInProgress("A plan is already being made for this session.")
            refinement = self.prepare(session_id, mode, text)
            self.sessions.ensure(session_id, mode)
            self.sessions.title_from_first_message(session_id, text)
            handle = RunHandle(session_id=session_id, mode=mode, message=text,
                               refinement=refinement)
            self._runs[session_id] = handle
        thread = threading.Thread(target=self._run, args=(handle,), daemon=True,
                                  name=f"plan-{session_id[:8]}")
        thread.start()
        return handle

    def _run(self, handle: RunHandle) -> None:
        error: str | None = None
        try:
            planner = self.hub.planner(handle.mode)
            for update in planner.stream(handle.refinement.message, handle.session_id,
                                         today=self.today(), edit=handle.refinement.edit):
                handle.add(update)
        except Exception as exc:  # shown to the user, never as a traceback
            error = friendly_error(exc, self.hub.settings)
        finally:
            handle.finish(error)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._runs.pop(session_id, None)
