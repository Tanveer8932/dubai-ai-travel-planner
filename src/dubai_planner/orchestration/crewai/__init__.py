"""CrewAI orchestration (Phase 4): a role-based crew over the shared agent core."""

from __future__ import annotations

from dubai_planner.orchestration.crewai.board import STAGES, CrewRunBoard, Stage
from dubai_planner.orchestration.crewai.crew import CrewDeps, CrewFactory, load_config
from dubai_planner.orchestration.crewai.listener import CrewTraceListener, trace_listener
from dubai_planner.orchestration.crewai.runner import CrewAIPlanner, run_crewai
from dubai_planner.orchestration.crewai.stages import StageRunner
from dubai_planner.orchestration.crewai.tools import StageTool, ToolboxTool, toolbox_tools

__all__ = [
    "STAGES",
    "CrewAIPlanner",
    "CrewDeps",
    "CrewFactory",
    "CrewRunBoard",
    "CrewTraceListener",
    "Stage",
    "StageRunner",
    "StageTool",
    "ToolboxTool",
    "load_config",
    "run_crewai",
    "toolbox_tools",
    "trace_listener",
]
