"""SQLite checkpointing for the LangGraph planner (``thread_id = session_id``).

LangGraph's default serializer deserializes *any* class named in a checkpoint (with a
warning, and it will block unregistered types in a future version). The planner's state
holds only our own Pydantic models and enums, so the serializer gets an **explicit
allowlist** of exactly those classes: a tampered checkpoint cannot make it import
anything else, and upgrading LangGraph's default to strict mode changes nothing here.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

import dubai_planner.models as models_pkg
from dubai_planner.config import Settings, get_settings


def model_allowlist() -> tuple[tuple[str, str], ...]:
    """``(module, class)`` for every Pydantic model and enum defined in ``dubai_planner.models``."""
    allowed: set[tuple[str, str]] = set()
    for info in pkgutil.iter_modules(models_pkg.__path__, prefix=f"{models_pkg.__name__}."):
        module = importlib.import_module(info.name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__ and issubclass(obj, BaseModel | Enum):
                allowed.add((module.__name__, name))
    return tuple(sorted(allowed))


def planner_serializer() -> JsonPlusSerializer:
    """Strict serializer: LangGraph's built-in safe types plus our models only."""
    return JsonPlusSerializer(allowed_msgpack_modules=model_allowlist())


@contextmanager
def open_checkpointer(path: Path | str | None = None, *, settings: Settings | None = None
                      ) -> Iterator[SqliteSaver]:
    """A :class:`SqliteSaver` on ``path`` (default ``CHECKPOINT_DB_PATH``), closed on exit.

    ``check_same_thread=False`` because LangGraph runs parallel branches in worker
    threads; ``SqliteSaver`` serialises access with its own lock.
    """
    db = Path(path) if path is not None else (settings or get_settings()).checkpoint_db_path
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, check_same_thread=False, timeout=30)
    try:
        saver = SqliteSaver(conn, serde=planner_serializer())
        saver.setup()
        yield saver
    finally:
        conn.close()
