"""Phase 4 unit tests: CrewAI config, tools, guardrails, mock ReAct LLM, LLM settings."""

from __future__ import annotations

import json
import threading
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from crewai.events import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.tasks.task_output import TaskOutput

from dubai_planner.agents import requirements
from dubai_planner.agents.suite import AgentSuite
from dubai_planner.config import Settings
from dubai_planner.llm.base import CrewAIChatLLM, LLMMessage
from dubai_planner.llm.crewai_mock import MockCrewAILLM
from dubai_planner.llm.errors import LLMCallError, StructuredOutputError
from dubai_planner.llm.factory import get_crewai_llm
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import PlannerMode, TraceEventType
from dubai_planner.orchestration.common import RetrySettings, call_with_retry
from dubai_planner.orchestration.crewai import (
    STAGES,
    CrewDeps,
    CrewFactory,
    CrewRunBoard,
    StageRunner,
    StageTool,
    load_config,
    toolbox_tools,
)
from dubai_planner.orchestration.crewai.board import (
    CrewRunAborted,
    CrewRunCancelled,
    brief,
)
from dubai_planner.orchestration.crewai.guardrails import (
    ItineraryGuardrail,
    ReviewGuardrail,
    StageGuardrail,
)
from dubai_planner.orchestration.crewai.listener import trace_listener
from dubai_planner.tools.toolbox import TOOL_SPECS, Toolbox
from dubai_planner.tracing import capture

TODAY = date(2026, 11, 16)
WINTER = "2 adults, 10-12 Feb 2027, AED 6000, history and food"
NO_WAIT = RetrySettings(max_attempts=3, initial_interval=0, backoff_factor=1, jitter=False)


def board(**kwargs: Any) -> CrewRunBoard:
    kwargs.setdefault("session_id", "s")
    kwargs.setdefault("run_id", "r")
    kwargs.setdefault("mode", PlannerMode.CREWAI_SEQUENTIAL)
    kwargs.setdefault("today", TODAY)
    kwargs.setdefault("user_message", WINTER)
    return CrewRunBoard(**kwargs)


def output(raw: str = "free text") -> TaskOutput:
    return TaskOutput(description="d", raw=raw, agent="a")


def planned_board(toolbox: Toolbox, mock_llm: MockLLM) -> tuple[CrewRunBoard, StageRunner]:
    """A board with requirements, research and budget accepted (via the real stages)."""
    runner = StageRunner(tools=toolbox, llm_for=lambda _a: mock_llm)
    b = board()
    for key in ["requirements", "attractions", "accommodation", "dining", "transport",
                "weather_events", "budget"]:
        b.accept(key, runner.fn(key)(b))
    return b, runner


# ------------------------------------------------------------------ YAML config


def test_yaml_config_matches_the_stage_catalogue() -> None:
    agents, tasks = load_config()
    assert set(agents) == {s.role_key for s in STAGES.values()} | {"trip_manager"}
    for stage in STAGES.values():
        assert tasks[stage.task]["agent"] == stage.role_key
        assert f"`{stage.tool}` tool" in tasks[stage.task]["description"]
        for field in ("role", "goal", "backstory"):
            assert agents[stage.role_key][field].strip()


def test_describe_lists_roles_tasks_and_dependencies(mock_llm: MockLLM,
                                                     toolbox: Toolbox) -> None:
    deps = CrewDeps(runner=StageRunner(tools=toolbox, llm_for=lambda _a: mock_llm),
                    crew_llm_for=lambda _r: MockCrewAILLM(engine=mock_llm))
    seq = CrewFactory(deps).describe()
    assert seq["manager"] is None and len(seq["tasks"]) == len(STAGES)
    tasks = {t["name"]: t for t in seq["tasks"]}
    assert tasks["research_dining"]["async"] and not tasks["allocate_budget"]["async"]
    assert tasks["review_itinerary"]["context"] == ["compose_itinerary"]
    assert tasks["compose_itinerary"]["output"] == "Itinerary"
    deps.process = "hierarchical"
    hier = CrewFactory(deps).describe()
    assert hier["manager"] == "Trip Manager"
    assert not any(t["async"] for t in hier["tasks"])


# ------------------------------------------------------------------ tools


def test_every_shared_tool_is_wrapped_with_its_args_schema(toolbox: Toolbox) -> None:
    tools = {t.name: t for t in toolbox_tools(toolbox)}
    assert set(tools) == set(TOOL_SPECS)
    for name, tool in tools.items():
        assert issubclass(tool.args_schema, TOOL_SPECS[name].args_model)  # same validation
    fare = json.loads(tools["metro_fare"].run(zones=2))
    assert Decimal(fare) == toolbox.metro_fare(2)
    aed = json.loads(tools["convert_currency"].run(amount="100", from_currency="USD"))
    assert Decimal(aed) == Decimal("367.25")


def test_tool_schemas_are_accepted_by_strict_providers(toolbox: Toolbox) -> None:
    """Groq rejected (HTTP 400, found in the Phase 4 smoke run) a function schema with
    ``required`` but no ``properties`` and any regex ``pattern`` with lookarounds (as
    pydantic emits for ``Decimal``). Check what CrewAI actually sends."""
    from crewai.utilities.agent_utils import convert_tools_to_openai_schema

    b = board()
    tools = [*toolbox_tools(toolbox),
             *(StageTool(b, key, lambda _b: None, role="r") for key in STAGES)]  # type: ignore[arg-type,return-value]
    sent = convert_tools_to_openai_schema(tools)
    schemas = sent[0] if isinstance(sent, tuple) else sent
    assert len(schemas) == len(tools)
    for schema in schemas:
        params = schema["function"]["parameters"]
        assert params.get("properties"), schema["function"]["name"]
        assert '"pattern"' not in json.dumps(params, default=str), schema["function"]["name"]


def test_each_role_gets_only_its_own_tools(toolbox: Toolbox, mock_llm: MockLLM) -> None:
    deps = CrewDeps(runner=StageRunner(tools=toolbox, llm_for=lambda _a: mock_llm),
                    crew_llm_for=lambda _r: MockCrewAILLM(engine=mock_llm))
    built = CrewFactory(deps).planning_crew(board())
    names = {k: [t.name for t in a.tools or []] for k, a in built.agents.items()}
    assert names["transport"] == ["plan_transport", "estimate_travel", "metro_fare"]
    assert names["budget"] == ["allocate_budget", "convert_currency"]
    for key in ("attractions", "accommodation", "dining", "weather_events", "compose", "review"):
        assert names[key] == [STAGES[key].tool]
    assert toolbox_tools(toolbox, []) == []


def test_stage_tool_returns_a_brief_and_parks_the_typed_output(toolbox: Toolbox,
                                                                mock_llm: MockLLM) -> None:
    b = board()
    runner = StageRunner(tools=toolbox, llm_for=lambda _a: mock_llm)
    tool = StageTool(b, "requirements", runner.requirements, role="Requirements Analyst")
    assert tool.result_as_answer and tool.name == "extract_trip_requirements"
    with capture() as events:
        text = tool.run()
    assert text.startswith("Trip request:") and len(text) < 600
    pending = b.take_pending("requirements")
    assert isinstance(pending, requirements.RequirementsResult) and pending.is_complete
    spans = [e for e in events if e.payload.get("kind") == "stage"]
    assert [e.event for e in spans] == [TraceEventType.START, TraceEventType.END]
    assert all(e.run_id == "r" and e.mode == "crewai_sequential" for e in spans)


def test_stage_tool_failure_is_recorded_once_and_never_redone() -> None:
    b = board()
    calls: list[int] = []

    def broken(_b: CrewRunBoard) -> Any:
        calls.append(1)
        raise StructuredOutputError("dining", "Selection", ["bad"], "{")

    tool = StageTool(b, "dining", broken, role="Dining Specialist", retry=NO_WAIT)
    with pytest.raises(StructuredOutputError):
        tool.run()
    assert b.failed_stage == "dining" and isinstance(b.error, StructuredOutputError)
    with pytest.raises(CrewRunAborted):  # CrewAI's own re-try: no second agent call
        tool.run()
    assert len(calls) == 1
    with pytest.raises(CrewRunAborted):
        b.raise_if_stopped()


def test_stage_tool_retries_transient_errors_then_succeeds(toolbox: Toolbox,
                                                           mock_llm: MockLLM) -> None:
    b = board()
    runner = StageRunner(tools=toolbox, llm_for=lambda _a: mock_llm)
    attempts: list[int] = []

    def flaky(bb: CrewRunBoard) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise LLMCallError("groq: APITimeoutError: Request timed out.")
        return runner.requirements(bb)

    StageTool(b, "requirements", flaky, role="Requirements Analyst", retry=NO_WAIT).run()
    assert len(attempts) == 3 and b.error is None


def test_cancelled_board_stops_tools_and_guardrails() -> None:
    b = board()
    b.cancelled = True
    with pytest.raises(CrewRunCancelled):
        StageTool(b, "dining", lambda _b: None, role="x").run()  # type: ignore[arg-type,return-value]
    with pytest.raises(CrewRunCancelled):
        StageGuardrail(b, "dining").check(output())


def test_call_with_retry_is_bounded_and_selective() -> None:
    calls: list[int] = []

    def always_429() -> None:
        calls.append(1)
        raise LLMCallError("Error code: 429 rate limit")

    with capture() as events, pytest.raises(LLMCallError):
        call_with_retry(always_429, NO_WAIT, sleep=lambda _s: None)
    assert len(calls) == 3 and len([e for e in events if e.payload.get("kind") == "retry"]) == 2
    calls.clear()

    def bad_output() -> None:
        calls.append(1)
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        call_with_retry(bad_output, NO_WAIT, sleep=lambda _s: None)
    assert len(calls) == 1


# ------------------------------------------------------------------ guardrails


def test_stage_guardrail_rejects_answers_not_made_by_the_tool(toolbox: Toolbox,
                                                              mock_llm: MockLLM) -> None:
    b, _ = planned_board(toolbox, mock_llm)
    guard = StageGuardrail(b, "dining")
    ok, feedback = guard.check(output("res_made_up_place"))
    assert not ok and "`research_dining` tool" in feedback
    value = b.accepted["dining"]
    b.put_pending("dining", value)
    ok, result = guard.check(output())
    assert ok and isinstance(result, TaskOutput) and result.pydantic is value
    assert result.raw.startswith(f"{len(value.options)} dining option(s)")  # type: ignore[attr-defined]


def test_itinerary_guardrail_rejects_rule_errors_until_the_cap(toolbox: Toolbox,
                                                               mock_llm: MockLLM) -> None:
    b, runner = planned_board(toolbox, mock_llm)
    b.max_iterations = 1
    good = runner.compose(b)
    over = good.model_copy(update={"cost": good.cost.model_copy(update={  # type: ignore[attr-defined]
        "total": b.request.total_budget_aed() * 2})})
    guard = ItineraryGuardrail(b, toolbox)
    b.put_pending("compose", over)
    ok, feedback = guard.check(output())
    assert not ok and "budget.total" in feedback and b.iteration == 1
    assert "budget.total" in b.attempts[0].blocking_rules and not b.attempts[0].approved
    b.put_pending("compose", over)  # cap reached: accepted, but stays unapproved
    with capture() as events:
        ok, result = guard.check(output())
    assert ok and result.pydantic is over and b.iteration == 1
    [event] = [e for e in events if e.event is TraceEventType.GUARDRAIL]
    assert event.payload["cap_reached"] and "budget.total" in event.payload["rules"]


def test_review_guardrail_never_approves_rule_errors(toolbox: Toolbox,
                                                     mock_llm: MockLLM) -> None:
    from dubai_planner.models import ReviewReport

    b, runner = planned_board(toolbox, mock_llm)
    itinerary = runner.compose(b)
    b.itinerary = itinerary  # type: ignore[assignment]
    runner.review(b)  # records the rule check on the board
    from dubai_planner.models import RuleViolation, Severity

    b.violations = [RuleViolation(rule_id="budget.total", severity=Severity.ERROR,
                                  message="over budget")]
    b.put_pending("review", ReviewReport(approved=True, score=1.0))
    ok, _ = ReviewGuardrail(b).check(output())
    assert ok and b.last_attempt is not None
    assert b.last_attempt.review.approved and not b.last_attempt.approved


def test_briefs_are_short_for_every_stage(toolbox: Toolbox, mock_llm: MockLLM) -> None:
    b, runner = planned_board(toolbox, mock_llm)
    b.itinerary = runner.compose(b)  # type: ignore[assignment]
    values = {**b.accepted, "compose": b.itinerary, "review": runner.review(b)}
    for key, value in values.items():
        text = brief(key, value)  # type: ignore[arg-type]
        assert 10 < len(text) < 900, (key, text)


# ------------------------------------------------------------------ mock ReAct LLM


def _react(llm: MockCrewAILLM, system: str, user: str, *extra: dict[str, str]) -> str:
    return llm.call([{"role": "system", "content": system},
                     {"role": "user", "content": user}, *extra])


def test_mock_react_calls_the_tool_the_task_names() -> None:
    llm = MockCrewAILLM(engine=MockLLM())
    system = "You are Dining Specialist. x\nTool Name: metro_fare\nTool Name: research_dining\n"
    reply = _react(llm, system, "Current Task: Call the `research_dining` tool once.")
    assert "Action: research_dining" in reply and "Action Input: {}" in reply
    done = _react(llm, system, "Current Task: t",
                  {"role": "assistant", "content": "Action: x\nObservation: 3 options"})
    assert done.endswith("Final Answer: 3 options")


def test_mock_react_manager_delegates_to_the_named_specialist() -> None:
    llm = MockCrewAILLM(engine=MockLLM())
    system = "You are Trip Manager.\nTool Name: delegate_work_to_coworker\n"
    user = ("Current Task: Plan transport.\nResponsible specialist: Transport & Logistics "
            "Specialist\n\nThis is the expected criteria for your final answer: x")
    reply = _react(llm, system, user)
    args = json.loads(reply.split("Action Input:", 1)[1])
    assert "Action: delegate_work_to_coworker" in reply
    assert args["coworker"] == "Transport & Logistics Specialist"
    assert "Plan transport" in args["task"]


def test_mock_react_scripts_and_events() -> None:
    engine = MockLLM().script("crewai:Budget Controller", "Final Answer: scripted")
    llm = MockCrewAILLM(engine=engine)
    seen: list[LLMCallCompletedEvent] = []
    done = threading.Event()
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _got(_source: Any, event: LLMCallCompletedEvent) -> None:
            seen.append(event)
            done.set()

        assert _react(llm, "You are Budget Controller.", "t") == "Final Answer: scripted"
        crewai_event_bus.flush()
        done.wait(5)
    assert engine.react_calls == ["Budget Controller"]
    assert seen and seen[0].usage and seen[0].usage["prompt_tokens"] > 0


def test_mock_agent_core_requests_still_use_the_engine() -> None:
    engine = MockLLM().script("requirements", '{"x": 1}')
    llm = MockCrewAILLM(engine=engine)
    reply = llm.call([{"role": "system", "content": "AGENT: requirements\nSCHEMA: S"},
                      {"role": "user", "content": "hi"}])
    assert reply == '{"x": 1}' and engine.react_calls == []


# ------------------------------------------------------------------ LLM settings


def _groq(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    for name in ("LLM_MODEL", "LLM_MAX_TOKENS", "LLM_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real-0123456789")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_crewai_groq_llm_carries_output_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt-oss on Groq: 4096 output tokens and low reasoning reach CrewAI's request."""
    llm = get_crewai_llm("composer", _groq(monkeypatch))
    assert llm.model == "openai/gpt-oss-20b"
    assert llm.max_tokens == 4096 and llm.reasoning_effort == "low"
    params = llm._prepare_completion_params([{"role": "user", "content": "hi"}])
    assert params["max_tokens"] == 4096 and params["reasoning_effort"] == "low"
    other = get_crewai_llm(None, _groq(monkeypatch, LLM_MODEL="llama-3.3-70b-versatile",
                                       LLM_MAX_TOKENS="1000"))
    assert other.max_tokens == 1000 and other.reasoning_effort is None


def test_crewai_ollama_llm_gets_no_unverified_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MAX_TOKENS", "999")
    llm = get_crewai_llm(None, Settings(_env_file=None))  # type: ignore[call-arg]
    assert getattr(llm, "max_tokens", None) != 999


def test_crewai_chat_adapter_reports_provider_token_usage() -> None:
    class Usage:
        def __init__(self, p: int, c: int) -> None:
            self.prompt_tokens, self.completion_tokens = p, c

    class FakeLLM:
        model = "fake"

        def __init__(self) -> None:
            self.total = Usage(0, 0)

        def call(self, _messages: Any) -> str:
            self.total = Usage(self.total.prompt_tokens + 120, self.total.completion_tokens + 30)
            return "ok"

        def get_token_usage_summary(self) -> Usage:
            return self.total

    adapter = CrewAIChatLLM(FakeLLM())
    for _ in range(2):
        done = adapter.complete([LLMMessage("user", "hello")])
        assert (done.tokens_in, done.tokens_out) == (120, 30)


def test_crewai_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_PROCESS", "hierarchical")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.crewai_process == "hierarchical" and s.crewai_tracing_enabled is False
    s.apply_runtime_env()
    import os

    assert os.environ["CREWAI_TRACING_ENABLED"] == "false"
    monkeypatch.setenv("CREWAI_PROCESS", "parallel")
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_listener_ignores_unregistered_crews(toolbox: Toolbox, mock_llm: MockLLM) -> None:
    b = board()
    deps = CrewDeps(runner=StageRunner(tools=toolbox, llm_for=lambda _a: mock_llm,
                                       agents=AgentSuite()),
                    crew_llm_for=lambda _r: MockCrewAILLM(engine=mock_llm))
    built = CrewFactory(deps).requirements_crew(b)
    listener = trace_listener()
    listener.unregister(b)
    with capture() as events:
        built.kickoff(b)
        listener.flush()
    kinds = {e.payload.get("kind") for e in events}
    assert "stage" in kinds  # our own spans still come from the tool
    assert not kinds & {"crew_task", "tool", "crew_agent"}  # nothing mapped from CrewAI
