"""Export the LangGraph planner graph as Mermaid to docs/generated/langgraph.mmd.

Usage: python scripts/export_langgraph.py [output_path]
(No LLM call and no network: the graph is only compiled, not run.)
"""

from __future__ import annotations

import sys
from pathlib import Path

from dubai_planner import bootstrap
from dubai_planner.orchestration.langgraph import build_graph, export_mermaid
from dubai_planner.orchestration.langgraph.nodes import PlannerDeps
from dubai_planner.tools.toolbox import Toolbox

DEFAULT = Path(__file__).resolve().parents[1] / "docs" / "generated" / "langgraph.mmd"


def main(argv: list[str]) -> int:
    bootstrap()
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT
    graph = build_graph(PlannerDeps(tools=Toolbox.default()))
    print(f"wrote {export_mermaid(graph, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
