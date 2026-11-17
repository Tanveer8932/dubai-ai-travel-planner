"""Load evaluation results for the dashboard (``app/pages/4_Evaluation.py``).

Two sources, same shape (:class:`EvalDataset`):

- **published** folders committed under ``eval/results/<name>/`` (``config.json``,
  ``cells.jsonl``, optional ``pairwise.jsonl`` and ``faults.csv``), so a deployed app
  with no evaluation database can still show the submitted evidence;
- the **evaluation database** (``data/eval.db``) written by ``scripts/run_eval.py``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dubai_planner.config import PROJECT_ROOT
from dubai_planner.storage import Repository

from .faults import FaultResult
from .metrics import CellMetrics
from .runner import PairRecord

PUBLISHED_DIR = PROJECT_ROOT / "eval" / "results"
EVAL_DB = PROJECT_ROOT / "data" / "eval.db"


@dataclass
class EvalDataset:
    key: str
    title: str
    source: str  # "published" or "database"
    config: dict[str, Any]
    cells: list[CellMetrics]
    pairs: list[PairRecord] = field(default_factory=list)
    faults: list[FaultResult] = field(default_factory=list)
    report_path: Path | None = None

    @property
    def is_mock(self) -> bool:
        return self.config.get("provider") == "mock"

    @property
    def scenarios(self) -> list[str]:
        return list(dict.fromkeys(c.scenario_id for c in self.cells))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _faults(path: Path) -> list[FaultResult]:
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            data: dict[str, Any] = dict(row)
            data["detected_by"] = [s for s in row["detected_by"].split(";") if s]
            for key in ("injected", "present_in_attempt", "detected"):
                data[key] = row[key] == "True"
            data["fixed"] = None if row["fixed"] in ("", "None") else row["fixed"] == "True"
            data["iterations"] = int(row["iterations"] or 0)
            data["llm_calls"] = int(row["llm_calls"]) if row["llm_calls"] else None
            data["final_status"] = row["final_status"] or None
            data["error"] = row["error"] or None
            out.append(FaultResult.model_validate(data))
    return out


def load_published(root: Path | None = None) -> list[EvalDataset]:
    folder = root or PUBLISHED_DIR
    out: list[EvalDataset] = []
    for cfg_path in sorted(folder.glob("*/config.json")):
        d = cfg_path.parent
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        out.append(EvalDataset(
            key=f"published:{d.name}", title=config.get("label") or d.name, source="published",
            config=config, cells=[CellMetrics.model_validate(r) for r in _jsonl(d / "cells.jsonl")],
            pairs=[PairRecord.model_validate(r) for r in _jsonl(d / "pairwise.jsonl")],
            faults=_faults(d / "faults.csv"),
            report_path=d / "report.md" if (d / "report.md").is_file() else None))
    return out


def load_database(path: Path | None = None) -> list[EvalDataset]:
    db = path or EVAL_DB
    if not db.is_file():
        return []
    repo = Repository(db)
    out: list[EvalDataset] = []
    for run in repo.list_eval_runs():
        rows = repo.list_eval_results(run.id)
        cells = [CellMetrics.model_validate(r.metrics) for r in rows
                 if not r.scenario_id.endswith("::pair")]
        pairs = [PairRecord.model_validate(r.metrics) for r in rows
                 if r.scenario_id.endswith("::pair")]
        label = run.config.get("label") or "evaluation"
        out.append(EvalDataset(
            key=f"db:{run.id}", source="database", config={"eval_run_id": run.id, **run.config},
            title=f"{label} · {run.created_at:%Y-%m-%d %H:%M} · {run.config.get('provider')}",
            cells=cells, pairs=pairs))
    return out


def load_all(published: Path | None = None, database: Path | None = None
             ) -> list[EvalDataset]:
    return [*load_published(published), *load_database(database)]
