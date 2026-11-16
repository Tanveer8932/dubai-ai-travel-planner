"""Build the Dubai Planning Crew (implementation plan, Phase 4 task 3; architecture §4).

Roles and task texts come from ``config/agents.yaml`` and ``config/tasks.yaml``. Three
crews share the same agents, tools and guardrails:

- **requirements crew**: ``parse_request`` (Requirements Analyst);
- **planning crew**: five research tasks (``async_execution=True`` in the sequential
  process, so they run concurrently) → ``allocate_budget`` (``context`` = the five) →
  ``compose_itinerary`` (``context`` = budget + research, deterministic-rule
  guardrail) → ``review_itinerary`` (``context`` = compose, review guardrail);
- **revision crew**: ``compose_itinerary`` → ``review_itinerary`` again, kicked off by
  the planner's outer review loop with the reviewer's feedback.

Processes:

- ``sequential`` (default): each task runs on its own agent; research overlaps.
- ``hierarchical``: tasks have no fixed agent. A Trip Manager agent (``manager_agent``)
  runs every task and must delegate it to a coworker with CrewAI's delegation tool;
  the task's "Responsible specialist" line names whom. All of the manager's tasks use
  the manager's single agent executor, which is not safe to share between threads, so
  research tasks run one after another in this process.

Typed outputs: every task's guardrail attaches the stage's Pydantic model (the same
Phase 1 models as LangGraph mode) as ``TaskOutput.pydantic``. Sequential tasks also
declare ``output_pydantic``. Hierarchical tasks do not: CrewAI would then make the
manager re-type each specialist's result into the schema with an extra LLM call per
task (measured: 3 instead of 2 manager calls per task), i.e. let an LLM rewrite
grounded data, which hard rule 5 forbids.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler
from pydantic import PrivateAttr

from dubai_planner.orchestration.common import RetrySettings
from dubai_planner.orchestration.crewai.board import (
    MANAGER_ROLE_KEY,
    RESEARCH_STAGES,
    STAGES,
    CrewRunBoard,
    Stage,
)
from dubai_planner.orchestration.crewai.guardrails import (
    ItineraryGuardrail,
    ReviewGuardrail,
    StageGuardrail,
)
from dubai_planner.orchestration.crewai.listener import trace_listener
from dubai_planner.orchestration.crewai.stages import StageRunner
from dubai_planner.orchestration.crewai.tools import StageTool, toolbox_tools

CONFIG_DIR = Path(__file__).with_name("config")
CrewProcessName = Literal["sequential", "hierarchical"]
#: Shared tool functions given to specialists as extra, cheap-to-call tools.
SUPPORT_TOOLS: dict[str, list[str]] = {
    "transport_specialist": ["estimate_travel", "metro_fare"],
    "budget_controller": ["convert_currency"],
}
#: Retries when an agent answers without its tool (not review iterations).
STAGE_GUARDRAIL_RETRIES = 2


class _NoReplayLog(TaskOutputStorageHandler):
    """Replaces CrewAI's replay log of task outputs.

    CrewAI writes every task output to ``latest_kickoff_task_outputs.db`` in the user's
    app-data folder (for ``crewai replay``). This project does not use replay (runs are
    stored in our SQLite), must not write outside the project, and CrewAI 1.15's JSON
    encoder cannot serialise ``datetime.time`` (opening hours, slot times), so storing
    our outputs failed the run. This handler keeps nothing.
    """

    def __init__(self) -> None:  # no storage is opened
        pass

    def update(self, task_index: int, log: dict[str, Any]) -> None:
        return None

    def add(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset(self) -> None:
        return None

    def load(self) -> list[dict[str, Any]] | None:
        return None


class PlanningCrew(Crew):
    """A CrewAI ``Crew`` that keeps no replay log (see :class:`_NoReplayLog`)."""

    _task_output_handler: TaskOutputStorageHandler = PrivateAttr(default_factory=_NoReplayLog)


@contextmanager
def _no_checkpoint_warnings() -> Iterator[None]:
    """CrewAI warns that callables (guardrails, callbacks) cannot be checkpointed.

    We do not use CrewAI checkpoints (session state lives in SQLite), so the warning is
    noise; it is silenced only while our crews are built.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*cannot be saved in a checkpoint.*")
        warnings.filterwarnings("ignore", message=".*cannot be JSON-serialized.*")
        yield


@cache
def load_config() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """``(agents, tasks)`` from the YAML files (validated against the stage catalogue)."""
    agents = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text(encoding="utf-8"))
    tasks = yaml.safe_load((CONFIG_DIR / "tasks.yaml").read_text(encoding="utf-8"))
    for stage in STAGES.values():
        cfg = tasks.get(stage.task)
        if cfg is None or cfg.get("agent") != stage.role_key or cfg.get("tool") != stage.tool:
            raise ValueError(f"tasks.yaml entry {stage.task!r} does not match the crew stages")
        if stage.role_key not in agents:
            raise ValueError(f"agents.yaml has no role {stage.role_key!r}")
    if MANAGER_ROLE_KEY not in agents:
        raise ValueError(f"agents.yaml has no {MANAGER_ROLE_KEY!r}")
    return agents, tasks


def role_of(role_key: str) -> str:
    return load_config()[0][role_key]["role"].strip()


@dataclass
class CrewDeps:
    """What every crew needs. ``crew_llm_for(role_key)`` → the agent's CrewAI LLM."""

    runner: StageRunner
    crew_llm_for: Callable[[str], Any]
    process: CrewProcessName = "sequential"
    max_iter: int = 4
    retry: RetrySettings = field(default_factory=RetrySettings)


@dataclass
class BuiltCrew:
    crew: Crew
    tasks: dict[str, Task]
    agents: dict[str, Agent]

    def kickoff(self, board: CrewRunBoard) -> Any:
        """Run the crew with the board's inputs (``{trip}``, ``{feedback}``…)."""
        inputs = {"user_message": board.user_message, "trip": board.trip_line(),
                  "feedback": board.feedback_text()}
        return self.crew.kickoff(inputs=inputs)


class CrewFactory:
    """Builds fresh agents/tasks per crew (CrewAI objects keep per-run state)."""

    def __init__(self, deps: CrewDeps) -> None:
        self.deps = deps
        self.hierarchical = deps.process == "hierarchical"

    # ------------------------------------------------------------------ agents
    def _agent(self, board: CrewRunBoard, stage: Stage) -> Agent:
        cfg = load_config()[0][stage.role_key]
        role = cfg["role"].strip()
        tools: list[Any] = [StageTool(board, stage.key, self.deps.runner.fn(stage.key),
                                      role=role, retry=self.deps.retry)]
        tools += toolbox_tools(self.deps.runner.tools, SUPPORT_TOOLS.get(stage.role_key, []))
        return Agent(role=role, goal=cfg["goal"].strip(), backstory=cfg["backstory"].strip(),
                     llm=self.deps.crew_llm_for(stage.role_key), tools=tools,
                     allow_delegation=False, max_iter=self.deps.max_iter, max_retry_limit=1,
                     cache=False, verbose=False)

    def _manager(self) -> Agent:
        cfg = load_config()[0][MANAGER_ROLE_KEY]
        return Agent(role=cfg["role"].strip(), goal=cfg["goal"].strip(),
                     backstory=cfg["backstory"].strip(),
                     llm=self.deps.crew_llm_for(MANAGER_ROLE_KEY), allow_delegation=True,
                     max_iter=self.deps.max_iter, max_retry_limit=1, cache=False,
                     verbose=False)

    # ------------------------------------------------------------------ tasks
    def _task(self, board: CrewRunBoard, stage: Stage, agent: Agent, *,
              context: list[Task] | None = None, async_execution: bool = False) -> Task:
        cfg = load_config()[1][stage.task]
        guardrail: StageGuardrail
        retries = STAGE_GUARDRAIL_RETRIES
        if stage.key == "compose":
            guardrail = ItineraryGuardrail(board, self.deps.runner.tools)
            retries += board.max_iterations  # each rule rejection is one refinement
        elif stage.key == "review":
            guardrail = ReviewGuardrail(board)
        else:
            guardrail = StageGuardrail(board, stage.key)
        description = (cfg["description"].strip()
                       + f"\nResponsible specialist: {role_of(stage.role_key)}")
        return Task(name=stage.task, description=description,
                    expected_output=cfg["expected_output"].strip(),
                    agent=None if self.hierarchical else agent,
                    context=context if context is not None else [],
                    async_execution=async_execution and not self.hierarchical,
                    output_pydantic=None if self.hierarchical else stage.output,
                    guardrail=guardrail.check,
                    guardrail_max_retries=retries)

    # ------------------------------------------------------------------ crews
    def _crew(self, board: CrewRunBoard, stages: list[str],
              on_task: Callable[[TaskOutput], None] | None) -> BuiltCrew:
        with _no_checkpoint_warnings():
            return self._build(board, stages, on_task)

    def _build(self, board: CrewRunBoard, stages: list[str],
               on_task: Callable[[TaskOutput], None] | None) -> BuiltCrew:
        agents = {key: self._agent(board, STAGES[key]) for key in stages}
        tasks: dict[str, Task] = {}
        research = [k for k in stages if k in RESEARCH_STAGES]
        for key in stages:
            stage = STAGES[key]
            if key in RESEARCH_STAGES:
                tasks[key] = self._task(board, stage, agents[key], async_execution=True)
            elif key == "budget":
                tasks[key] = self._task(board, stage, agents[key],
                                        context=[tasks[k] for k in research])
            elif key == "compose":
                upstream = [tasks[k] for k in ("budget", *research) if k in tasks]
                tasks[key] = self._task(board, stage, agents[key], context=upstream)
            elif key == "review":
                tasks[key] = self._task(board, stage, agents[key], context=[tasks["compose"]])
            else:
                tasks[key] = self._task(board, stage, agents[key])
        kwargs: dict[str, Any] = {}
        all_agents = dict(agents)
        if self.hierarchical:
            manager = self._manager()
            kwargs["manager_agent"] = manager
            all_agents[MANAGER_ROLE_KEY] = manager
        crew = PlanningCrew(
            name="Dubai Planning Crew", agents=list(agents.values()),
            tasks=[tasks[k] for k in stages],
            process=Process.hierarchical if self.hierarchical else Process.sequential,
            verbose=False, cache=False, memory=False, planning=False, tracing=False,
            task_callback=on_task, **kwargs)
        trace_listener().register(
            board, agents={STAGES[k].role_key if k in STAGES else k: (a, a.role)
                           for k, a in all_agents.items()},
            tasks={STAGES[k].task: (t, STAGES[k].role_key) for k, t in tasks.items()})
        return BuiltCrew(crew=crew, tasks=tasks, agents=all_agents)

    def requirements_crew(self, board: CrewRunBoard,
                          on_task: Callable[[TaskOutput], None] | None = None) -> BuiltCrew:
        return self._crew(board, ["requirements"], on_task)

    def planning_crew(self, board: CrewRunBoard,
                      on_task: Callable[[TaskOutput], None] | None = None) -> BuiltCrew:
        return self._crew(board, [*RESEARCH_STAGES, "budget", "compose", "review"], on_task)

    def revision_crew(self, board: CrewRunBoard,
                      on_task: Callable[[TaskOutput], None] | None = None) -> BuiltCrew:
        return self._crew(board, ["compose", "review"], on_task)

    def describe(self) -> dict[str, Any]:
        """Static description of the crew (roles, tasks, dependencies) for visualisers."""
        agents, _tasks = load_config()
        deps = {"allocate_budget": [STAGES[k].task for k in RESEARCH_STAGES],
                "compose_itinerary": ["allocate_budget",
                                      *(STAGES[k].task for k in RESEARCH_STAGES)],
                "review_itinerary": ["compose_itinerary"]}
        return {
            "process": self.deps.process,
            "manager": agents[MANAGER_ROLE_KEY]["role"].strip() if self.hierarchical else None,
            "agents": {k: v["role"].strip() for k, v in agents.items()
                       if k != MANAGER_ROLE_KEY or self.hierarchical},
            "tasks": [{"name": s.task, "role": role_of(s.role_key), "tool": s.tool,
                       "async": s.key in RESEARCH_STAGES and not self.hierarchical,
                       "context": deps.get(s.task, []),
                       "output": s.output.__name__} for s in STAGES.values()],
        }
