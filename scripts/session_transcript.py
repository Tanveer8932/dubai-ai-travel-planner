"""Record a reproducible session transcript, restart, and verify persistence.

    python scripts/session_transcript.py                      # LangGraph, mock provider
    python scripts/session_transcript.py --mode crewai_sequential

1. A new session plays: the original request, "reduce the budget by 20%", "Replace
   day 2 with indoor activities", and a dietary change (vegetarian), through the
   app's own services (classification → planner → SQLite).
2. The planners and services are closed (the app quits).
3. A **separate Python process** opens the same database file and reads the session
   from SQLite only; its view is compared with the one taken before closing.

Output: ``docs/transcripts/session_<mode>.md`` and ``.json``. The database is
``data/transcripts/<mode>.db`` (git-ignored, recreated on each run).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _child_snapshot(db: Path, checkpoint: Path, session_id: str) -> tuple[dict, int]:
    """Read the session in a fresh interpreter (nothing shared with this process)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(  # noqa: S603 - our own script, fixed arguments
        [sys.executable, __file__, "--snapshot", session_id, "--db", str(db),
         "--checkpoint", str(checkpoint)],
        capture_output=True, text=True, encoding="utf-8", env=env, check=True, cwd=ROOT)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return payload["snapshot"], payload["pid"]


def parent(version: dict) -> str:
    return f"v{version['parent_version']}" if version["parent_version"] else "—"


def _md(log: dict, after: dict, checks: list[dict], child_pid: int) -> str:
    mock_note = ", deterministic offline mock" if log["provider"] == "mock" else ""
    out = [f"# Session transcript — {log['mode']}", "",
           f"Generated {log['generated']} by `scripts/session_transcript.py --mode {log['mode']}`"
           f" (provider **{log['provider']}**{mock_note}"
           f"; data sources offline). Session `{log['session_id']}`.", ""]
    if log["provider"] == "mock":
        out += ["> The mock provider shows routing, versioning and persistence exactly as the "
                "app performs them; its itineraries come from the deterministic baseline "
                "scheduler, not from a real model.", ""]
    for i, t in enumerate(log["turns"], 1):
        out += [f"## Turn {i} — {t['label']}", "",
                f"- **User:** “{t['user_input']}”",
                f"- **Routing:** `{t['routing']}` — {t['routing_explanation']}",
                f"- **Message sent to the planner:** “{t['planner_message']}”"]
        if t["locked_days"]:
            out.append(f"- **Locked (kept verbatim):** days {t['locked_days']}")
        if t["day_constraints"]:
            cons = ", ".join(f"day {c['day_index'] + 1}: indoor_only={c['indoor_only']}"
                             for c in t["day_constraints"])
            out.append(f"- **Day constraints:** {cons}")
        out.append(f"- **Steps run:** {' → '.join(t['steps']) or '—'}")
        version = (f"saved **v{t['saved_version']}**"
                   + (f" (parent v{t['parent_version']})" if t["parent_version"] else ""))
        out += [f"- **Outcome:** {t['status']}, {version}; review iterations {t['iterations']};"
                f" {t['llm_calls']} LLM calls, {t['tokens']} tokens, {t['duration_s']} s",
                f"- **Budget / total (AED, estimate):** {t['budget_aed']} / {t['total_aed']}"
                + (f"; dietary: {', '.join(t['dietary'])}" if t["dietary"] else ""),
                ]
        if t["vegetarian_restaurants_only"] is not None and t["dietary"]:
            out.append(f"- **All restaurants vegetarian-friendly:** "
                       f"{'yes' if t['vegetarian_restaurants_only'] else 'NO'}")
        if t["error"]:
            out.append(f"- **Error:** {t['error']}")
        d = t.get("diff")
        if d:
            delta = ", ".join(f"{k} {'+' if not v.startswith('-') else ''}{v}"
                              for k, v in d["cost_delta_aed"].items())
            out += ["", f"**Changes v{d['from_version']} → v{d['to_version']}:** budget "
                    f"{d['budget_before']} → {d['budget_after']}; cost delta: {delta}. "
                    f"Changed days {d['changed_days'] or 'none'}; unchanged days "
                    f"{d['unchanged_days'] or 'none'}.", ""]
            for day, s in d["slots"].items():
                out.append(f"- Day {day}: removed {s['removed'] or '—'}; added "
                           f"{s['added'] or '—'}")
        if t["days"]:
            out += ["", f"<details><summary>Itinerary v{t['saved_version'] or ''}</summary>", ""]
            for day in t["days"]:
                out.append(f"- **Day {day['day']} ({day['date']}) {day['theme']}** — "
                           f"AED {day['day_cost_aed']}")
                out += [f"  - {s}" for s in day["slots"]]
            out += ["", "</details>"]
        out.append("")
    out += ["## Close and reopen", "",
            "The planners, checkpointer and services were closed (the app quit). A new "
            f"Python process (pid {child_pid}) opened `{Path(log['db_path']).name}` and read "
            "the session from SQLite only:", "",
            "| Check | Result | Detail |", "|---|---|---|"]
    out += [f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |"
            for c in checks]
    out += ["", "**Versions after reopening:**", "",
            "| Version | Parent | Mode | Approved | Budget (AED) | Total (AED) |",
            "|---|---|---|---|---|---|"]
    out += [f"| v{v['version']} | {parent(v)}"
            f" | {v['mode']} | {v['approved']} | {v['budget_aed']} | {v['total_aed']} |"
            for v in after["versions"]]
    out += ["", "**Chat history after reopening:**", ""]
    out += [f"{i}. *{m['role']}*: {m['content'][:300]}" for i, m in
            enumerate(after["messages"], 1)]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", default="langgraph",
                   choices=["langgraph", "crewai_sequential", "crewai_hierarchical"])
    p.add_argument("--provider", default="mock")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "transcripts")
    p.add_argument("--db", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--snapshot", help=argparse.SUPPRESS)  # child process: print a snapshot
    args = p.parse_args(argv)
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["HTTP_OFFLINE"] = "true"

    from dubai_planner import bootstrap

    bootstrap()
    from dubai_planner.evaluation import transcript as tr
    from dubai_planner.models import PlannerMode
    from dubai_planner.storage import Repository

    if args.snapshot:
        snap = tr.snapshot(Repository(args.db), args.snapshot, checkpoint_path=args.checkpoint)
        print(json.dumps({"pid": os.getpid(), "snapshot": snap.model_dump(mode="json")}))
        return 0

    db = args.db or ROOT / "data" / "transcripts" / f"{args.mode}.db"
    checkpoint = args.checkpoint or db.with_name(db.stem + "_checkpoints.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    for path in (db, checkpoint):
        path.unlink(missing_ok=True)
    log = tr.record_session(db, checkpoint, PlannerMode(args.mode))
    assert log.before_close is not None  # noqa: S101 - set by record_session
    after_raw, child_pid = _child_snapshot(db, checkpoint, log.session_id)
    after = tr.Snapshot.model_validate(after_raw)
    checks = tr.verify(log.before_close, after)
    data = {"mode": log.mode, "provider": log.provider, "session_id": log.session_id,
            "db_path": log.db_path, "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "turns": [t.model_dump(mode="json") for t in log.turns],
            "reopened_in_pid": child_pid, "parent_pid": os.getpid(),
            "checks": [c.model_dump() for c in checks],
            "after_reopen": after.model_dump(mode="json", exclude={"latest_itinerary_json"})}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"session_{args.mode}.json").write_text(json.dumps(data, indent=2),
                                                        encoding="utf-8")
    (args.out / f"session_{args.mode}.md").write_text(
        _md(data, data["after_reopen"], data["checks"], child_pid), encoding="utf-8")
    ok = all(c.passed for c in checks)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(c.passed for c in checks)}/{len(checks)} "
          f"persistence checks → {args.out / f'session_{args.mode}.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
