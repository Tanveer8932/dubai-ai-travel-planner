"""Subprocess helper for AC3.5 (not a test module).

``crash``: plan until the composer node starts, then kill the process with ``os._exit``
(no ``finally``, no clean shutdown: like a power cut or ``kill -9``).
``resume``: a fresh process opens the same checkpoint DB and resumes the thread.
Prints one JSON line with the outcome and the nodes this process executed.

Usage: python langgraph_worker.py <crash|resume> <checkpoint_db> <app_db> <session_id>
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any

MESSAGE = "2 adults, 10-12 Feb 2027, AED 6000, history and food"
KILLED_EXIT_CODE = 17


def main(argv: list[str]) -> int:
    mode, checkpoint_db, app_db, session_id = argv[1:5]
    from dubai_planner.llm.mock import MockLLM
    from dubai_planner.models import TraceEventType
    from dubai_planner.orchestration.langgraph import AgentSuite, LangGraphPlanner
    from dubai_planner.storage import Repository
    from dubai_planner.tracing import capture

    def die(*_args: Any, **_kwargs: Any) -> Any:
        sys.stdout.flush()
        os._exit(KILLED_EXIT_CODE)

    agents = AgentSuite(composer=die) if mode == "crash" else AgentSuite()
    with LangGraphPlanner.open(checkpoint_db, llm=MockLLM(), repo=Repository(app_db),
                               agents=agents) as planner, capture() as events:
        if mode == "crash":
            planner.run(MESSAGE, session_id, today=date(2026, 11, 16))
            print(json.dumps({"error": "process was not killed"}))
            return 1
        result = planner.resume(session_id)
    nodes = [e.agent for e in events
             if e.event is TraceEventType.START and e.payload.get("kind") == "node"]
    print(json.dumps({"status": result.status.value, "run_id": result.run_id,
                      "nodes": nodes, "version": result.saved_version,
                      "total": str(result.itinerary.cost.total) if result.itinerary else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
