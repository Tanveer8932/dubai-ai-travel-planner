"""Evaluation runner (implementation plan, Phase 7 task 4): scenarios × modes × seeds.

Every cell runs a scenario's conversation through the **same path as the app**:
``RunManager`` → ``services.refinement.classify`` → the mode's planner, one fresh
session per cell, with the same scenario text, the same seed data and the same rules
for every mode. The planners record their runs and traces in the evaluation database;
:mod:`.metrics` then scores each cell from what was recorded.

Free-tier safety (Groq allows ≈ 200k tokens/day): cells run scenario by scenario, so
an interrupted evaluation still holds complete mode pairs; ``max_tokens`` stops
starting new cells once the budget is used; a provider *daily-limit* error stops the
whole evaluation instead of burning retries; and ``resume`` skips finished cells, so a
real-model evaluation can be completed over several days into the same eval run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from dubai_planner.config import Settings, get_settings
from dubai_planner.models import EvalResult, PlannerMode, PlanResult, PlanStatus
from dubai_planner.models.common import StrictModel
from dubai_planner.orchestration.common import data_fingerprint
from dubai_planner.services.observability import load_trace
from dubai_planner.services.runs import PlannerHub, RunManager
from dubai_planner.services.session import SessionService
from dubai_planner.storage import Repository
from dubai_planner.tools.toolbox import Toolbox

from . import judge as judge_mod
from .faults import DEFECTS, FaultResult, run_fault
from .metrics import CellMetrics, TurnOutcome, score
from .scenarios import Scenario

DEFAULT_MODES = [PlannerMode.LANGGRAPH, PlannerMode.CREWAI_SEQUENTIAL]
TURN_TIMEOUT_S = 1800.0
#: Markers of a provider's *daily* quota error (retrying the same day cannot succeed).
DAILY_LIMIT_MARKERS = ("tokens per day", "requests per day", "(tpd)", "(rpd)", "per day")

Progress = Callable[[str], None]


class EvalConfig(StrictModel):
    """What was evaluated, recorded with every eval run (no secrets)."""

    label: str = ""
    provider: str
    model: str
    modes: list[PlannerMode]
    scenarios: list[str]
    seeds: int = 1
    judge: bool = False
    judge_model: str | None = None
    faults: bool = False
    http_offline: bool = True
    data_fingerprint: str = ""
    max_review_iterations: int = 2
    budget_tolerance: str = "0.05"
    max_tokens: int | None = None
    #: One unscored run per mode first, so one-off start-up cost (imports, first
    #: graph/crew build) is not charged to the first scenario. Mock only by default.
    warmup: bool = False
    out_dir: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PairRecord(StrictModel):
    scenario_id: str
    seed: int
    a: str
    b: str
    winner: str  # "a", "b" or "tie"
    consistent: bool
    first: str
    swapped: str
    rationale: str = ""


@dataclass
class EvalOutcome:
    eval_run_id: str
    config: EvalConfig
    cells: list[CellMetrics]
    pairs: list[PairRecord] = field(default_factory=list)
    faults: list[FaultResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stopped_reason: str | None = None
    out_dir: Path | None = None


def is_daily_limit(error: str | None) -> bool:
    text = (error or "").lower()
    return ("429" in text and any(m in text for m in DAILY_LIMIT_MARKERS)) or (
        "rate limit" in text and "per day" in text)


def make_config(settings: Settings, *, modes: list[PlannerMode], scenarios: list[Scenario],
                seeds: int = 1, judge: bool = False, faults: bool = False, label: str = "",
                max_tokens: int | None = None, out_dir: Path | None = None,
                warmup: bool | None = None) -> EvalConfig:
    return EvalConfig(
        label=label, provider=settings.llm_provider.value, model=settings.model_for(None),
        modes=modes, scenarios=[s.id for s in scenarios], seeds=seeds, judge=judge,
        judge_model=settings.model_for("judge") if judge else None, faults=faults,
        http_offline=settings.http_offline, data_fingerprint=data_fingerprint(
            str(settings.seed_dir)), max_review_iterations=settings.max_review_iterations,
        budget_tolerance=str(settings.budget_tolerance), max_tokens=max_tokens,
        out_dir=str(out_dir or ""), started_at=datetime.now(UTC),
        warmup=settings.llm_provider.value == "mock" if warmup is None else warmup)


class EvalRunner:
    """Runs the cells of one evaluation into ``repo`` (eval tables + planner runs)."""

    def __init__(self, repo: Repository, *, settings: Settings | None = None,
                 tools: Toolbox | None = None, llm: Any = None, judge_llm: Any = None,
                 checkpoint_path: Path | None = None, progress: Progress | None = None,
                 turn_timeout_s: float = TURN_TIMEOUT_S) -> None:
        self.repo = repo
        self.settings = settings or get_settings()
        self.tools = tools or Toolbox.default()
        self.llm = llm
        self.judge_llm = judge_llm
        self.checkpoint_path = checkpoint_path
        self.progress = progress or (lambda _msg: None)
        self.turn_timeout_s = turn_timeout_s
        self._today = date.today()
        self._last_results: dict[tuple[str, str, int], PlanResult | None] = {}

    # ------------------------------------------------------------------ cells

    def _hub(self) -> PlannerHub:
        kwargs: dict[str, Any] = {"tools": self.tools}
        if self.llm is not None:
            kwargs["llm"] = self.llm
        return PlannerHub(self.repo, settings=self.settings,
                          checkpoint_path=self.checkpoint_path, **kwargs)

    def run_cell(self, runs: RunManager, sessions: SessionService, scenario: Scenario,
                 mode: PlannerMode, seed: int) -> list[TurnOutcome]:
        """Play the scenario's conversation in a fresh session; stop at a failed turn."""
        self._today = scenario.today
        session = sessions.create(mode, title=f"[eval] {scenario.id} · {mode.value} · s{seed}")
        turns: list[TurnOutcome] = []
        for text in scenario.turns:
            started = time.perf_counter()
            handle = runs.start(session.id, mode, text)
            finished = handle.wait(self.turn_timeout_s)
            wall = round(time.perf_counter() - started, 3)
            result = handle.result
            trace = load_trace(self.repo, result.run_id) if result is not None else None
            error = handle.error or (None if finished else f"turn timed out after {wall} s")
            turns.append(TurnOutcome(message=text, routing=handle.refinement.kind.value,
                                     explanation=handle.refinement.explanation, result=result,
                                     trace=trace, error=error, wall_s=wall))
            if not finished or result is None or result.status is PlanStatus.FAILED:
                break
        return turns

    def evaluate(self, scenarios: list[Scenario], config: EvalConfig, *,
                 resume: str | None = None) -> EvalOutcome:
        """Run every missing cell; returns all cells of the eval run (old and new)."""
        eval_run = (self.repo.get_eval_run(resume) if resume else None)
        if resume and eval_run is None:
            raise ValueError(f"no evaluation run {resume!r} to resume")
        if eval_run is None:
            eval_run = self.repo.create_eval_run(config.model_dump(mode="json"))
        done = {(r.scenario_id, r.mode.value, r.seed): r
                for r in self.repo.list_eval_results(eval_run.id)}
        cells = [CellMetrics.model_validate(r.metrics) for r in done.values()
                 if not r.scenario_id.endswith("::pair")]
        pairs = [PairRecord.model_validate(r.metrics) for r in done.values()
                 if r.scenario_id.endswith("::pair")]
        outcome = EvalOutcome(eval_run_id=eval_run.id, config=config, cells=cells,
                              pairs=pairs)
        used = sum(c.tokens or 0 for c in cells)
        hub = self._hub()
        sessions = SessionService(self.repo)
        runs = RunManager(hub, sessions, today=lambda: self._today)
        try:
            if config.warmup and scenarios:
                for mode in config.modes:  # unscored, not stored as eval results
                    self.run_cell(runs, sessions, scenarios[0], mode, -1)
            for scenario in scenarios:
                for seed in range(config.seeds):
                    for mode in config.modes:
                        key = (scenario.id, mode.value, seed)
                        if key in done:
                            continue
                        if outcome.stopped_reason:
                            outcome.skipped.append(f"{scenario.id}/{mode.value}/s{seed}")
                            continue
                        if config.max_tokens is not None and used >= config.max_tokens:
                            outcome.stopped_reason = (
                                f"token budget reached ({used} ≥ {config.max_tokens})")
                            outcome.skipped.append(f"{scenario.id}/{mode.value}/s{seed}")
                            continue
                        cell = self._cell(runs, sessions, scenario, mode, seed, eval_run.id)
                        outcome.cells.append(cell)
                        used += cell.tokens or 0
                        self.progress(
                            f"{scenario.id} · {mode.value} · s{seed}: {cell.final_status} "
                            f"({cell.latency_s} s, {cell.llm_calls} LLM calls, "
                            f"{cell.tokens} tokens)")
                        if is_daily_limit(cell.error):
                            outcome.stopped_reason = ("provider daily quota reached: "
                                                      f"{(cell.error or '')[:160]}")
                    if config.judge and not outcome.stopped_reason:
                        self._pairwise(outcome, scenario, seed, eval_run.id, done)
        finally:
            hub.close()
        return outcome

    def _cell(self, runs: RunManager, sessions: SessionService, scenario: Scenario,
              mode: PlannerMode, seed: int, eval_run_id: str) -> CellMetrics:
        turns = self.run_cell(runs, sessions, scenario, mode, seed)
        cell = score(scenario, turns, mode=mode.value, seed=seed, catalog=self.tools.catalog,
                     tolerance=self.settings.budget_tolerance,
                     provider=self.settings.llm_provider.value,
                     model=self.settings.model_for(None))
        final = turns[-1].result if turns else None
        if self.judge_llm is not None and final and final.itinerary and final.request:
            try:
                scores = judge_mod.rubric(self.judge_llm, final.request, final.itinerary)
                cell.judge_score = scores.mean
                cell.judge_kind = (judge_mod.MOCK_KIND
                                   if self.settings.llm_provider.value == "mock"
                                   else f"rubric · {self.settings.model_for('judge')}")
            except Exception as exc:  # the judge is optional: record, never fail the cell
                cell.judge_kind = f"judge failed: {type(exc).__name__}"
        self.repo.add_eval_result(EvalResult(eval_run_id=eval_run_id, scenario_id=scenario.id,
                                             mode=mode, seed=seed,
                                             metrics=cell.model_dump(mode="json")))
        self._last_results[(scenario.id, mode.value, seed)] = final
        return cell

    def _pairwise(self, outcome: EvalOutcome, scenario: Scenario, seed: int,
                  eval_run_id: str, done: dict[tuple[str, str, int], EvalResult]) -> None:
        if self.judge_llm is None or (f"{scenario.id}::pair", PlannerMode.LANGGRAPH.value,
                                      seed) in done:
            return
        modes = outcome.config.modes
        if len(modes) < 2:
            return
        a = self._last_results.get((scenario.id, modes[0].value, seed))
        b = self._last_results.get((scenario.id, modes[1].value, seed))
        if (a is None or b is None or a.request is None or a.itinerary is None
                or b.itinerary is None):
            return
        try:
            res = judge_mod.pairwise(self.judge_llm, a.request, a.itinerary, b.itinerary)
        except Exception as exc:  # optional instrument
            self.progress(f"pairwise judge failed for {scenario.id}: {type(exc).__name__}")
            return
        pair = PairRecord(scenario_id=scenario.id, seed=seed, a=modes[0].value,
                          b=modes[1].value, winner=res.winner, consistent=res.consistent,
                          first=res.first, swapped=res.swapped, rationale=res.rationale)
        outcome.pairs.append(pair)
        self.repo.add_eval_result(EvalResult(eval_run_id=eval_run_id,
                                             scenario_id=f"{scenario.id}::pair",
                                             mode=PlannerMode.LANGGRAPH, seed=seed,
                                             metrics=pair.model_dump(mode="json")))

    # ------------------------------------------------------------------ faults

    def run_faults(self, modes: list[PlannerMode], checkpoint_path: Path) -> list[FaultResult]:
        out: list[FaultResult] = []
        for mode in modes:
            for defect in DEFECTS:
                res = run_fault(defect, mode, repo=self.repo, tools=self.tools,
                                checkpoint_path=checkpoint_path, llm=self.llm,
                                settings=self.settings)
                self.progress(f"fault {defect.id} · {mode.value}: detected={res.detected} "
                              f"fixed={res.fixed} ({res.final_status})")
                out.append(res)
        return out


def write_jsonl(path: Path, rows: list[StrictModel]) -> None:
    path.write_text("".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in rows),
                    encoding="utf-8")
