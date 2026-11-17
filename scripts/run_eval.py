"""Run the benchmark: scenarios × modes × seeds → SQLite + CSV + Markdown report.

Examples::

    python scripts/run_eval.py --provider mock                     # offline, all scenarios
    python scripts/run_eval.py --provider mock --hierarchical      # + CrewAI hierarchical
    python scripts/run_eval.py --provider groq --scenarios budget_backpacker family_kids \
        --max-tokens 120000 --out eval/results/groq-day1           # quota-bounded real run
    python scripts/run_eval.py --resume <eval_run_id>              # continue another day

Data sources are offline (curated seed + cache) unless ``--live-data`` is given, so
every mode sees the same dataset. Results go to the evaluation database
(``data/eval.db`` by default, git-ignored) and to ``--out`` (default
``eval/reports/<timestamp>/``). A real-provider run never starts a cell after
``--max-tokens`` is used and stops at a provider daily-limit error.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["langgraph", "crewai_sequential", "crewai_hierarchical"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--provider", help="LLM_PROVIDER for this run (default: from .env)")
    p.add_argument("--model", help="LLM_MODEL override (default: provider default)")
    p.add_argument("--modes", nargs="+", choices=MODES,
                   default=["langgraph", "crewai_sequential"])
    p.add_argument("--hierarchical", action="store_true",
                   help="also run CrewAI hierarchical")
    p.add_argument("--scenarios", nargs="+", help="scenario ids (default: all)")
    p.add_argument("--seeds", type=int, default=1, help="repetitions per cell")
    p.add_argument("--judge", action="store_true", help="rubric + pairwise LLM judge")
    p.add_argument("--faults", dest="faults", action="store_true", default=None,
                   help="run the fault-injection suite (default: on for mock)")
    p.add_argument("--no-faults", dest="faults", action="store_false")
    p.add_argument("--warmup", dest="warmup", action="store_true", default=None,
                   help="one unscored run per mode first (default: on for mock)")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--max-tokens", type=int, help="stop starting cells after this many tokens")
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, help="output folder")
    p.add_argument("--db", type=Path, default=ROOT / "data" / "eval.db")
    p.add_argument("--resume", help="eval run id to continue (skips finished cells)")
    p.add_argument("--report-only", action="store_true",
                   help="with --resume: rewrite the report from the database, run nothing")
    p.add_argument("--live-data", action="store_true",
                   help="allow live public APIs (default: offline seed data)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    os.environ["HTTP_OFFLINE"] = "false" if args.live_data else "true"

    from dubai_planner import bootstrap

    settings = bootstrap()
    from dubai_planner.evaluation.report import write_report
    from dubai_planner.evaluation.runner import EvalRunner, make_config
    from dubai_planner.evaluation.scenarios import load_scenarios
    from dubai_planner.llm.factory import get_chat_llm
    from dubai_planner.models import PlannerMode
    from dubai_planner.storage import Repository

    repo = Repository(args.db)
    previous = repo.get_eval_run(args.resume) if args.resume else None
    if args.resume and previous is None:
        print(f"No evaluation run {args.resume!r} in {args.db}", file=sys.stderr)
        return 1
    if previous is not None:  # same cells, same output folder as the first session
        cfg = previous.config
        modes = [PlannerMode(m) for m in cfg["modes"]]
        scenario_ids, seeds = cfg["scenarios"], int(cfg["seeds"])
        judge, faults = bool(cfg["judge"]), bool(cfg["faults"])
        out_dir = Path(cfg["out_dir"]) if cfg.get("out_dir") else None
        label, max_tokens = cfg.get("label", ""), args.max_tokens or cfg.get("max_tokens")
    else:
        modes = [PlannerMode(m) for m in dict.fromkeys(
            args.modes + (["crewai_hierarchical"] if args.hierarchical else []))]
        scenario_ids, seeds, judge = args.scenarios, args.seeds, args.judge
        faults = args.faults if args.faults is not None else (
            settings.llm_provider.value == "mock")
        out_dir, label, max_tokens = args.out, args.label, args.max_tokens
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = out_dir or ROOT / "eval" / "reports" / stamp
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(ids=scenario_ids)
    config = make_config(settings, modes=modes, scenarios=scenarios, seeds=seeds, judge=judge,
                         faults=faults, label=label, max_tokens=max_tokens, out_dir=out_dir,
                         warmup=None if previous else args.warmup)
    print(f"Evaluating {len(scenarios)} scenario(s) × {len(modes)} mode(s) × {seeds} seed(s) "
          f"with {config.provider}/{config.model} (data offline: {config.http_offline})")
    runner = EvalRunner(repo, settings=settings,
                        judge_llm=get_chat_llm("judge", settings) if judge else None,
                        checkpoint_path=args.db.with_name(args.db.stem + "_checkpoints.db"),
                        progress=lambda msg: print("  " + msg, flush=True))
    if args.report_only:
        if previous is None:
            print("--report-only needs --resume <eval_run_id>", file=sys.stderr)
            return 1
        from dubai_planner.evaluation.metrics import CellMetrics
        from dubai_planner.evaluation.runner import EvalConfig, EvalOutcome, PairRecord

        rows = repo.list_eval_results(previous.id)
        outcome = EvalOutcome(
            eval_run_id=previous.id, config=EvalConfig.model_validate(previous.config),
            cells=[CellMetrics.model_validate(r.metrics) for r in rows
                   if not r.scenario_id.endswith("::pair")],
            pairs=[PairRecord.model_validate(r.metrics) for r in rows
                   if r.scenario_id.endswith("::pair")])
        done = {(c.scenario_id, c.mode) for c in outcome.cells}
        outcome.skipped = [f"{s.id}/{m.value}/s0" for s in scenarios for m in modes
                           if (s.id, m.value) not in done]
        if outcome.skipped:
            from dubai_planner.evaluation.runner import is_daily_limit

            quota = any(is_daily_limit(c.error) for c in outcome.cells)
            outcome.stopped_reason = (
                "provider daily token quota reached; continue another day with "
                f"--resume {previous.id}" if quota else "cells not run yet")
        print(f"Report rewritten: {write_report(outcome, out_dir)}")
        return 0
    outcome = runner.evaluate(scenarios, config, resume=previous.id if previous else None)
    if faults:
        outcome.faults = runner.run_faults(modes, args.db.with_name(
            args.db.stem + "_fault_checkpoints.db"))
    outcome.out_dir = out_dir
    report = write_report(outcome, out_dir)
    print(f"Eval run {outcome.eval_run_id}: {len(outcome.cells)} cell(s) → {report}")
    if outcome.stopped_reason:
        print(f"Stopped early: {outcome.stopped_reason}. Continue later with "
              f"--resume {outcome.eval_run_id}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
