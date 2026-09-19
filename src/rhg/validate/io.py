"""Shared loaders for the validation modules (rollouts, judge rows, human labels, problems).

Kept separate so ``sample``, ``harness`` and ``label`` read the same files the same way. Rollout rows
follow REPO_SPEC §6; the AST flags come from the logged ``monitor`` block and are recomputed with the
same extraction + detector only when a log does not carry them.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

EVAL_PHASES = ("eval_val", "eval_test", "eval_test_xhint")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def iter_rollouts(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def arm_of(run_id: str) -> str:
    """``hackable_subtle__s3`` -> ``hackable_subtle`` (REPO_SPEC §3: ``run_id = f"{arm.id}__s{seed}"``)."""
    return run_id.rsplit("__s", 1)[0]


def find_runs(runs_dir: Path, run_ids: Sequence[str] | None = None) -> list[str]:
    root = Path(runs_dir)
    if run_ids:
        return [r for chunk in run_ids for r in chunk.split(",") if r]
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "rollouts.jsonl.gz").is_file())


def ast_flags(row: Mapping[str, Any]) -> tuple[bool, bool]:
    """(ast_narrow, ast_broad) of a rollout row: the logged values, else recomputed."""
    mon = row.get("monitor") or {}
    narrow, broad = mon.get("ast_narrow"), mon.get("ast_broad")
    if isinstance(narrow, bool) and isinstance(broad, bool):
        return narrow, broad
    from rhg.detect.ast_detector import analyze
    from rhg.env.extract import extract_code

    code = extract_code(str(row.get("completion", ""))).code
    return (narrow if isinstance(narrow, bool) else analyze(code, "narrow").flag,
            broad if isinstance(broad, bool) else analyze(code, "broad").flag)


def load_eval_rows(runs_dir: Path, run_ids: Sequence[str], phases: Sequence[str] = EVAL_PHASES) -> list[dict[str, Any]]:
    """Every rollout row of the given eval phases, tagged with ``run_id`` (directory name) and ``arm``,
    plus ``_narrow`` / ``_broad`` AST flags."""
    out = []
    for rid in run_ids:
        path = Path(runs_dir) / rid / "rollouts.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        for row in iter_rollouts(path):
            if row.get("phase") not in phases:
                continue
            row = dict(row)
            row["run_id"], row["arm"] = rid, arm_of(rid)
            row["_narrow"], row["_broad"] = ast_flags(row)
            out.append(row)
    return out


def final_eval_test(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per run, the ``eval_test`` rollouts of the highest step (the population the judge samples from)."""
    last: dict[str, int] = {}
    for r in rows:
        if r["phase"] == "eval_test":
            last[r["run_id"]] = max(last.get(r["run_id"], -1), int(r["step"]))
    return [dict(r) for r in rows if r["phase"] == "eval_test" and int(r["step"]) == last.get(r["run_id"])]


def load_problems(path: Path) -> dict[str, dict[str, Any]]:
    return {r["problem_id"]: r for r in read_jsonl(path)}


def rollout_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(row["run_id"]), str(row["problem_id"]), int(row["sample_idx"]))


def load_judge_rows(judge_dir: Path, run_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    root = Path(judge_dir)
    if not root.is_dir():
        return []
    rows = []
    for f in sorted(root.glob("*.jsonl")):
        if run_ids and f.stem not in run_ids:
            continue
        rows.extend(read_jsonl(f))
    return rows
