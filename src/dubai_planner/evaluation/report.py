"""Evaluation outputs: ``results.csv``, ``cells.jsonl``, charts (PNG) and ``report.md``.

The report states what produced the numbers (provider, model, data fingerprint,
offline flag) and, for the mock provider, that they measure orchestration behaviour
only. Every figure is computed from the scored cells; nothing is typed in by hand.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .faults import FaultResult, detection_rate, fix_rate
from .metrics import NUMERIC_METRICS, CellMetrics, aggregate, scrub
from .runner import EvalOutcome, PairRecord

#: Fixed categorical order (validated palette, first three slots: all-pairs safe).
MODE_COLORS = {"langgraph": "#2a78d6", "crewai_sequential": "#eb6834",
               "crewai_hierarchical": "#1baf7a"}
MODE_LABELS = {"langgraph": "LangGraph", "crewai_sequential": "CrewAI sequential",
               "crewai_hierarchical": "CrewAI hierarchical"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

MOCK_BANNER = (
    "> **Mock provider.** These runs use the deterministic offline mock LLM. They show how "
    "each orchestration mode behaves (routing, review loop, parallelism, validity of the "
    "contract) on identical inputs. They are **not** evidence of real-model quality: the "
    "mock's token counts are prompt-length estimates, its latencies are not model latencies, "
    "and mock judge scores are a constant placeholder.")

CSV_SCALARS = [
    "scenario_id", "category", "mode", "seed", "provider", "model", "final_status",
    "expected_outcome", "outcome_ok", "failed", "schema_valid", "budget_aed", "total_aed",
    "overshoot_pct", "within_tolerance", "infeasibility_explained", "budget_compliant",
    "constraints_checked", "constraints_satisfied", "constraint_rate", "violations_error",
    "violations_warning", "violations_info", "mean_leg_km", "mean_daily_km", "max_leg_min",
    "grounding_rate", "interest_match", "indoor_share", "latency_s", "llm_calls",
    "tokens_in", "tokens_out", "tokens", "review_iterations", "guardrail_rejections",
    "delegations", "research_window_s", "research_sum_s", "research_ratio", "judge_score",
    "judge_kind", "error"]


def label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_csv(path: Path, cells: Sequence[CellMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([*CSV_SCALARS, "constraints_failed", "violation_rules", "turns",
                         "routing"])
        for c in cells:
            row = [getattr(c, k) for k in CSV_SCALARS]
            writer.writerow([*row, ";".join(k.name for k in c.constraints if not k.satisfied),
                             ";".join(c.violation_rules), len(c.turns),
                             " > ".join(t.routing for t in c.turns)])


def write_faults_csv(path: Path, faults: Sequence[FaultResult]) -> None:
    fields = list(FaultResult.model_fields)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for f in faults:
            row = f.model_dump(mode="json")
            row["detected_by"] = ";".join(row["detected_by"])
            writer.writerow(row)


# --------------------------------------------------------------------------- charts


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(out_dir: Path, cells: Sequence[CellMetrics]) -> list[str]:
    """Two PNG figures (small multiples, one axis each); returns their file names."""
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(c.mode for c in cells))
    if not modes:
        return []
    names: list[str] = []
    # 1. cost of a run: latency, LLM calls, tokens (distribution per mode)
    perf = [("latency_s", "Latency per scenario (s)"), ("llm_calls", "LLM calls per scenario"),
            ("tokens", "Tokens per scenario")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.9 + 0.75 * len(modes) + 1.2),
                             facecolor=SURFACE)
    for ax, (key, title) in zip(axes, perf, strict=True):
        _style(ax)
        data = [[float(getattr(c, key)) for c in cells
                 if c.mode == m and getattr(c, key) is not None] for m in modes]
        positions = list(range(len(modes)))[::-1]
        for pos, mode, vals in zip(positions, modes, data, strict=True):
            if not vals:
                continue
            bp = ax.boxplot(vals, positions=[pos], orientation="horizontal", widths=0.5,
                            patch_artist=True, showfliers=False)
            for part in bp["boxes"]:
                part.set(facecolor=MODE_COLORS.get(mode, "#888888"), alpha=0.25,
                         edgecolor=MODE_COLORS.get(mode, "#888888"), linewidth=1.5)
            for part in (*bp["whiskers"], *bp["caps"], *bp["medians"]):
                part.set(color=MODE_COLORS.get(mode, "#888888"), linewidth=1.5)
            ax.scatter(vals, [pos] * len(vals), s=22, color=MODE_COLORS.get(mode, "#888888"),
                       edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_yticks(positions, [label(m) for m in modes], color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        if key == "tokens":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _p: f"{v / 1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", dpi=144, facecolor=SURFACE)
    plt.close(fig)
    names.append("performance.png")

    # 2. quality shares per mode
    agg = aggregate(cells)
    quality = [("schema_valid", "Valid itinerary"), ("budget_compliant", "Budget compliant"),
               ("constraint_rate", "Hard constraints met"), ("outcome_ok", "Expected outcome"),
               ("interest_match", "Interests matched")]
    fig, axes = plt.subplots(1, len(quality), figsize=(13, 0.9 + 0.6 * len(modes) + 1.2),
                             facecolor=SURFACE, sharey=True)
    positions = list(range(len(modes)))[::-1]
    for ax, (key, title) in zip(axes, quality, strict=True):
        _style(ax)
        for pos, mode in zip(positions, modes, strict=True):
            st = agg[mode][key]
            if st.mean is None:
                continue
            ax.barh(pos, st.mean, height=0.55, color=MODE_COLORS.get(mode, "#888888"))
            ax.text(min(st.mean, 1.0) + 0.02, pos, f"{st.mean:.0%}", va="center",
                    color=INK, fontsize=9)
        ax.set_xlim(0, 1.25)
        ax.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"])
        ax.set_yticks(positions, [label(m) for m in modes], color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "quality.png", dpi=144, facecolor=SURFACE)
    plt.close(fig)
    names.append("quality.png")
    return names


# --------------------------------------------------------------------------- markdown


def _summary_table(cells: Sequence[CellMetrics]) -> list[str]:
    agg = aggregate(cells)
    modes = list(agg)
    lines = ["| Metric | " + " | ".join(label(m) for m in modes) + " |",
             "|---|" + "---|" * len(modes)]
    for key, name in NUMERIC_METRICS:
        lines.append(f"| {name} | " + " | ".join(agg[m][key].show() for m in modes) + " |")
    return lines


def _scenario_table(cells: Sequence[CellMetrics]) -> list[str]:
    lines = ["| Scenario | Mode | Status | Expected outcome met | Total / budget (AED) | "
             "Constraints met | Rule errors / warnings | Latency (s) | LLM calls | Tokens | "
             "Review iterations |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c.scenario_id, c.mode, c.seed)):
        budget = (f"{_fmt(c.total_aed)} / {_fmt(c.budget_aed, 0)}"
                  if c.total_aed is not None else "—")
        cons = (f"{c.constraints_satisfied}/{c.constraints_checked}"
                if c.constraints_checked else "—")
        lines.append(
            f"| {c.scenario_id}{'' if c.seed == 0 else f' (s{c.seed})'} | {label(c.mode)} | "
            f"{_fmt(c.final_status)} | {_fmt(c.outcome_ok)} | {budget} | {cons} | "
            f"{_fmt(c.violations_error)} / {_fmt(c.violations_warning)} | "
            f"{_fmt(c.latency_s, 1)} | {_fmt(c.llm_calls)} | {_fmt(c.tokens)} | "
            f"{_fmt(c.review_iterations)} |")
    return lines


def _failures(cells: Sequence[CellMetrics]) -> list[str]:
    lines: list[str] = []
    for c in sorted(cells, key=lambda c: (c.scenario_id, c.mode)):
        failed = [k for k in c.constraints if not k.satisfied]
        notes = [f"`{k.name}` ({k.detail})" for k in failed]
        if c.error:
            notes.append(f"error: {c.error[:200]}")
        if c.budget_compliant is False:
            notes.append(f"over budget by {_fmt(c.overshoot_pct)} % without an explanation")
        if notes:
            lines.append(f"- **{c.scenario_id} · {label(c.mode)}**: " + "; ".join(notes))
    return lines or ["- none"]


def _routing(cells: Sequence[CellMetrics]) -> list[str]:
    lines: list[str] = []
    for c in sorted(cells, key=lambda c: (c.scenario_id, c.mode)):
        if len(c.turns) < 2:
            continue
        lines.append(f"- **{c.scenario_id} · {label(c.mode)}**")
        for i, t in enumerate(c.turns, 1):
            lines.append(f"  {i}. “{t.message}” → `{t.routing}` → {t.status} "
                         f"(v{_fmt(t.saved_version)}, AED {_fmt(t.total_aed)})")
    return lines or ["- none"]


def _pairs(pairs: Sequence[PairRecord]) -> list[str]:
    if not pairs:
        return ["Not run."]
    lines = ["| Scenario | A | B | Winner | Order-consistent | A-first verdict | "
             "B-first verdict (mapped) |", "|---|---|---|---|---|---|---|"]
    for p in pairs:
        win = {"a": label(p.a), "b": label(p.b)}.get(p.winner, "tie")
        lines.append(f"| {p.scenario_id} | {label(p.a)} | {label(p.b)} | {win} | "
                     f"{_fmt(p.consistent)} | {p.first} | {p.swapped} |")
    return lines


def _faults(faults: Sequence[FaultResult]) -> list[str]:
    if not faults:
        return ["Not run."]
    modes = list(dict.fromkeys(f.mode for f in faults))
    lines = ["| Defect | Rule | Severity | " + " | ".join(label(m) for m in modes) + " |",
             "|---|---|---|" + "---|" * len(modes)]
    for d in dict.fromkeys(f.defect for f in faults):
        own = {f.mode: f for f in faults if f.defect == d}
        first = next(iter(own.values()))
        cells = []
        for m in modes:
            f = own.get(m)
            if f is None:
                cells.append("—")
                continue
            fixed = "" if f.fixed is None else (" · fixed" if f.fixed else " · NOT fixed")
            cells.append(("detected" if f.detected else "**missed**") + fixed
                         + f" ({f.final_status}, {f.iterations} it.)")
        lines.append(f"| {d} | `{first.rule_id}` | {first.severity} | " + " | ".join(cells)
                     + " |")
    lines.append("")
    for m in modes:
        lines.append(f"- {label(m)}: detection rate **{_fmt(detection_rate(list(faults), m))}**"
                     f", fix rate (error defects) **{_fmt(fix_rate(list(faults), m))}**")
    return lines


def scrubbed(cell: CellMetrics) -> CellMetrics:
    """The cell with keys and provider account ids masked in every error text."""
    turns = [t.model_copy(update={"error": scrub(t.error)}) for t in cell.turns]
    return cell.model_copy(update={"error": scrub(cell.error), "turns": turns})


def write_report(out: EvalOutcome, out_dir: Path) -> Path:
    """Write every output file for ``out`` into ``out_dir``; returns ``report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [scrubbed(c) for c in out.cells]  # also for cells stored before scrubbing
    write_csv(out_dir / "results.csv", cells)
    (out_dir / "cells.jsonl").write_text(
        "".join(json.dumps(c.model_dump(mode="json")) + "\n" for c in cells), encoding="utf-8")
    if out.pairs:
        (out_dir / "pairwise.jsonl").write_text(
            "".join(json.dumps(p.model_dump(mode="json")) + "\n" for p in out.pairs),
            encoding="utf-8")
    if out.faults:
        write_faults_csv(out_dir / "faults.csv", out.faults)
    (out_dir / "config.json").write_text(
        json.dumps({"eval_run_id": out.eval_run_id, **out.config.model_dump(mode="json")},
                   indent=2), encoding="utf-8")
    figures = charts(out_dir, cells) if cells else []
    cfg = out.config
    mock = cfg.provider == "mock"
    md: list[str] = [f"# Evaluation report{f' — {cfg.label}' if cfg.label else ''}", ""]
    if mock:
        md += [MOCK_BANNER, ""]
    else:
        md += [f"> **Real provider:** `{cfg.provider}` / `{cfg.model}`. Each cell is one run "
               "(no repetition unless seeds > 1); free-tier rate limits add waiting time to "
               "latency.", ""]
    md += ["## Configuration", "",
           "| Setting | Value |", "|---|---|",
           f"| Eval run id | `{out.eval_run_id}` |",
           f"| Started (UTC) | {cfg.started_at:%Y-%m-%d %H:%M} |",
           f"| Provider / model | `{cfg.provider}` / `{cfg.model}` |",
           f"| Modes | {', '.join(label(m.value) for m in cfg.modes)} |",
           f"| Scenarios | {len(cfg.scenarios)} ({', '.join(cfg.scenarios)}) |",
           f"| Seeds (repetitions) | {cfg.seeds} |",
           f"| Data | seed fingerprint `{cfg.data_fingerprint}`, HTTP offline = "
           f"{cfg.http_offline} |",
           f"| Review-iteration cap / budget tolerance | {cfg.max_review_iterations} / "
           f"{cfg.budget_tolerance} |",
           f"| Judge | {'on' if cfg.judge else 'off'}"
           f"{f' (`{cfg.judge_model}`)' if cfg.judge and not mock else ''} |",
           f"| Token budget | {cfg.max_tokens or 'none'} |",
           f"| Unscored warm-up run per mode | {cfg.warmup} |", ""]
    if out.stopped_reason or out.skipped:
        md += ["## Incomplete", "",
               f"Stopped: {out.stopped_reason or '—'}. Cells not run ({len(out.skipped)}): "
               + (", ".join(out.skipped) or "—"), ""]
    md += ["## Summary per mode (mean ± sample std, n = cells with a value)", "",
           *_summary_table(cells), "",
           "Shares are fractions of cells (1.00 = every cell). *Total vs budget* is "
           "(total − budget) / budget; negative = under budget.", ""]
    for name in figures:
        md += [f"![{name}]({name})", ""]
    md += ["## Per scenario", "", *_scenario_table(cells), "",
           "## Constraint failures and errors", "", *_failures(cells), "",
           "## Multi-turn routing", "", *_routing(cells), "",
           "## Pairwise judge (position-swapped)", "", *_pairs(out.pairs), "",
           "## Fault injection (review loop)", "", *_faults(out.faults), ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
