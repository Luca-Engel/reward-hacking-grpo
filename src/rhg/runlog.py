"""Run logs: schemas, writers, readers and a validator (docs/REPO_SPEC.md §6, DESIGN §4).

Files of a run directory ``results/runs/<run_id>/``::

    rollouts.jsonl.gz   one ``RolloutRecord`` per line (train + every eval phase)
    steps.jsonl         one ``StepRecord`` per training step (steps 1..T)
    evals.json          ``EvalSummary``: per eval point, per-problem counts of every label
    status.json         ``RunStatus`` (rewritten atomically as the run progresses)
    manifest.json, config.resolved.yaml, stdout.log   (written by the run driver / rhg.manifest)

Step numbering: training step ``k`` (1..T) is the k-th optimiser step; its rollouts were sampled from
the policy after ``k-1`` updates. Eval points use the number of completed updates (0 = untrained).

Non-finite floats (``loss``/``reward`` of an invalid run) are written as the JSON constants
``NaN``/``Infinity`` so a NaN step stays diagnosable; ``validate`` rejects them in completed runs.

CLI: ``python -m rhg.runlog validate <run_dir>`` (exit 0 = valid, 1 = problems found, 2 = usage).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections.abc import Collection, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rhg.env.labels import LABEL_FIELDS, derive_labels

PHASES = ("train", "eval_val", "eval_test", "eval_test_xhint")
EVAL_HINTS = ("none", "subtle", "explicit")
STATUS_VALUES = ("running", "completed", "invalid", "failed")

ROLLOUTS_FILE = "rollouts.jsonl.gz"
STEPS_FILE = "steps.jsonl"
EVALS_FILE = "evals.json"
STATUS_FILE = "status.json"
MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "config.resolved.yaml"
STDOUT_FILE = "stdout.log"

# every count kept per problem in evals.json
COUNT_KEYS = (*LABEL_FIELDS, "code_extracted", "truncated", "ast_narrow", "ast_broad")

Phase = Literal["train", "eval_val", "eval_test", "eval_test_xhint"]
Hint = Literal["none", "subtle", "explicit"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


# ------------------------------------------------------------------ schemas
class Labels(_Model):
    defines_rt: bool
    rt_ok: bool
    visible_pass: bool
    heldout_pass: bool
    timeout: bool
    crash: bool
    hack_rt: bool
    attempt_rt: bool
    correct: bool
    visible_only: bool


class Monitor(_Model):
    ast_narrow: bool
    ast_broad: bool
    ast_categories: list[str]


class RolloutRecord(_Model):
    run_id: str
    phase: Phase
    step: int = Field(ge=0)
    eval_hint: Hint | None
    problem_id: str
    sample_idx: int = Field(ge=0)
    completion: str
    n_tokens: int = Field(ge=0)
    truncated: bool
    code_extracted: bool
    reward: float
    labels: Labels
    monitor: Monitor

    @model_validator(mode="after")
    def _hint_only_for_xhint(self) -> "RolloutRecord":
        if (self.phase == "eval_test_xhint") != (self.eval_hint is not None):
            raise ValueError("eval_hint must be set exactly for phase 'eval_test_xhint'")
        return self


class StepRecord(_Model):
    step: int = Field(ge=1)
    reward_mean: float
    loss: float
    grad_norm: float
    completion_len_mean: float
    truncation_rate: float
    frac_zero_adv_groups: float
    hack_rt_rate_train: float
    attempt_rt_rate_train: float
    correct_rate_train: float
    t_gen: float
    t_reward: float
    t_train: float
    t_sync: float
    t_step: float
    tokens_gen: int = Field(ge=0)
    tokens_train: int = Field(ge=0)


class EvalPoint(_Model):
    """Counts for one (phase, step, hint) evaluation.

    ``prompt_hint`` is the hint wording level shown in the prompts; ``eval_hint`` is set only for the
    cross-hint phase (as in the rollout records) and equals ``prompt_hint`` there.
    """

    key: str
    phase: Literal["eval_val", "eval_test", "eval_test_xhint"]
    step: int = Field(ge=0)
    eval_hint: Hint | None
    prompt_hint: Hint
    samples_per_problem: int = Field(ge=1)
    n_rollouts: int = Field(ge=0)
    totals: dict[str, int]
    per_problem: dict[str, dict[str, int]]


class EvalSummary(_Model):
    run_id: str
    arm: str
    seed: int
    points: list[EvalPoint]


class RunStatus(_Model):
    run_id: str
    status: Literal["running", "completed", "invalid", "failed"]
    reason: str | None = None
    phase: str | None = None
    step: int | None = None
    exit_code: int | None = None
    updated_at: str


def eval_key(phase: str, step: int, hint: str | None) -> str:
    return f"{phase}|{step}|{hint if hint is not None else '-'}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def count_row(rec: RolloutRecord | dict[str, Any]) -> dict[str, int]:
    """Indicator vector of one rollout over ``COUNT_KEYS``."""
    d = rec.model_dump() if isinstance(rec, RolloutRecord) else rec
    out = {k: int(bool(d["labels"][k])) for k in LABEL_FIELDS}
    out["code_extracted"] = int(bool(d["code_extracted"]))
    out["truncated"] = int(bool(d["truncated"]))
    out["ast_narrow"] = int(bool(d["monitor"]["ast_narrow"]))
    out["ast_broad"] = int(bool(d["monitor"]["ast_broad"]))
    return out


class EvalAccumulator:
    """Builds ``EvalSummary`` points from rollout records as they are written."""

    def __init__(self, run_id: str, arm: str, seed: int) -> None:
        self.run_id, self.arm, self.seed = run_id, arm, seed
        self._meta: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._per: dict[str, dict[str, dict[str, int]]] = {}

    def open_point(self, phase: str, step: int, eval_hint: str | None, prompt_hint: str, samples_per_problem: int) -> str:
        key = eval_key(phase, step, eval_hint if eval_hint is not None else prompt_hint)
        if key in self._meta:
            raise ValueError(f"eval point {key} already opened")
        self._meta[key] = dict(phase=phase, step=step, eval_hint=eval_hint, prompt_hint=prompt_hint,
                               samples_per_problem=samples_per_problem)
        self._order.append(key)
        self._per[key] = {}
        return key

    def add(self, key: str, rec: RolloutRecord) -> None:
        row = self._per[key].setdefault(rec.problem_id, {"n": 0, **{k: 0 for k in COUNT_KEYS}})
        row["n"] += 1
        for k, v in count_row(rec).items():
            row[k] += v

    def summary(self) -> EvalSummary:
        points = []
        for key in self._order:
            per = self._per[key]
            totals = {"n": sum(r["n"] for r in per.values())}
            totals.update({k: sum(r[k] for r in per.values()) for k in COUNT_KEYS})
            points.append(EvalPoint(key=key, n_rollouts=totals["n"], totals=totals, per_problem=per, **self._meta[key]))
        return EvalSummary(run_id=self.run_id, arm=self.arm, seed=self.seed, points=points)


# ------------------------------------------------------------------ writers
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


class RolloutWriter:
    """Append-only gzip-JSONL writer; ``flush`` makes everything written so far readable."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / ROLLOUTS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = gzip.open(self.path, "wt", encoding="utf-8", newline="\n")
        self.n_written = 0

    def write(self, rec: RolloutRecord) -> None:
        self._f.write(rec.model_dump_json() + "\n")
        self.n_written += 1

    def write_many(self, recs: Iterable[RolloutRecord]) -> None:
        for r in recs:
            self.write(r)
        self.flush()

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "RolloutWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class StepWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / STEPS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8", newline="\n")

    def write(self, rec: StepRecord) -> None:
        self._f.write(rec.model_dump_json() + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "StepWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_status(
    run_dir: str | Path,
    run_id: str,
    status: str,
    *,
    reason: str | None = None,
    phase: str | None = None,
    step: int | None = None,
    exit_code: int | None = None,
) -> RunStatus:
    st = RunStatus(run_id=run_id, status=status, reason=reason, phase=phase, step=step,
                   exit_code=exit_code, updated_at=utcnow_iso())
    _atomic_write_text(Path(run_dir) / STATUS_FILE, st.model_dump_json(indent=2) + "\n")
    return st


def write_evals(run_dir: str | Path, summary: EvalSummary) -> Path:
    path = Path(run_dir) / EVALS_FILE
    _atomic_write_text(path, summary.model_dump_json(indent=2) + "\n")
    return path


# ------------------------------------------------------------------ readers
def _as_set(v: str | Collection[str] | None) -> set[str] | None:
    if v is None:
        return None
    return {v} if isinstance(v, str) else set(v)


def _gz_lines(path: Path, partial_ok: bool) -> Iterator[str]:
    """Lines of a gzip text file. A file whose writer never closed it (killed run) has no end-of-stream marker;
    with ``partial_ok`` the lines that were flushed are returned and the missing trailer is ignored."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            yield from f
        except EOFError:
            if not partial_ok:
                raise


def iter_rollouts(
    run_dir: str | Path,
    phase: str | Collection[str] | None = None,
    step: int | None = None,
    *,
    partial_ok: bool = False,
) -> Iterator[RolloutRecord]:
    """Validated rollouts of a run, optionally filtered by phase (name or collection) and step.

    ``partial_ok=True`` also reads the rollouts of a run that was killed before it closed the file."""
    phases = _as_set(phase)
    for line in _gz_lines(Path(run_dir) / ROLLOUTS_FILE, partial_ok):
        if not line.strip():
            continue
        rec = RolloutRecord.model_validate_json(line)
        if phases is not None and rec.phase not in phases:
            continue
        if step is not None and rec.step != step:
            continue
        yield rec


def read_steps(run_dir: str | Path) -> list[StepRecord]:
    path = Path(run_dir) / STEPS_FILE
    return [StepRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_evals(run_dir: str | Path) -> EvalSummary:
    return EvalSummary.model_validate_json((Path(run_dir) / EVALS_FILE).read_text(encoding="utf-8"))


def read_status(run_dir: str | Path) -> RunStatus:
    return RunStatus.model_validate_json((Path(run_dir) / STATUS_FILE).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ validator
def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def validate_run(run_dir: str | Path) -> list[str]:
    """Return a list of problems (empty = valid). Partial runs (status other than ``completed``) are
    checked for what exists; a completed run must have every file and consistent contents."""
    d = Path(run_dir)
    errs: list[str] = []
    if not d.is_dir():
        return [f"{d} is not a directory"]

    status: RunStatus | None = None
    try:
        status = read_status(d)
    except (OSError, ValidationError, ValueError) as e:
        errs.append(f"{STATUS_FILE}: {_short(e)}")
    complete = status is not None and status.status == "completed"
    run_id = status.run_id if status is not None else d.name
    if status is not None and status.run_id != d.name:
        errs.append(f"{STATUS_FILE}: run_id {status.run_id!r} != directory name {d.name!r}")

    required = [MANIFEST_FILE, CONFIG_FILE, STDOUT_FILE, STEPS_FILE, ROLLOUTS_FILE, EVALS_FILE]
    for name in required:
        if not (d / name).is_file():
            (errs if complete or name in (MANIFEST_FILE, CONFIG_FILE) else []).append(f"missing file {name}")
    if (d / MANIFEST_FILE).is_file():
        try:
            man = json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8"))
            if man.get("run_id") != run_id:
                errs.append(f"{MANIFEST_FILE}: run_id {man.get('run_id')!r} != {run_id!r}")
            if status is not None and man.get("status") != status.status:
                errs.append(f"{MANIFEST_FILE}: status {man.get('status')!r} != {STATUS_FILE} status {status.status!r}")
        except ValueError as e:
            errs.append(f"{MANIFEST_FILE}: {_short(e)}")

    steps: list[StepRecord] = []
    if (d / STEPS_FILE).is_file():
        for n, line in enumerate((d / STEPS_FILE).read_text(encoding="utf-8").splitlines(), 1):
            try:
                steps.append(StepRecord.model_validate_json(line))
            except (ValidationError, ValueError) as e:
                errs.append(f"{STEPS_FILE}:{n}: {_short(e)}")
        if [s.step for s in steps] != list(range(1, len(steps) + 1)):
            errs.append(f"{STEPS_FILE}: steps are not 1..{len(steps)} in order")
        if complete:
            for s in steps:
                bad = [k for k, v in s.model_dump().items() if isinstance(v, float) and not math.isfinite(v)]
                if bad:
                    errs.append(f"{STEPS_FILE}: step {s.step} has non-finite {bad} in a completed run")
            if not steps:
                errs.append(f"{STEPS_FILE}: empty in a completed run")

    counts: dict[str, dict[str, dict[str, int]]] = {}
    train_per_step: dict[int, int] = {}
    seen: set[tuple] = set()
    if (d / ROLLOUTS_FILE).is_file():
        try:
            for n, line in enumerate(_gz_lines(d / ROLLOUTS_FILE, partial_ok=not complete), 1):
                if not line.strip():
                    continue
                try:
                    rec = RolloutRecord.model_validate_json(line)
                except (ValidationError, ValueError) as e:
                    errs.append(f"{ROLLOUTS_FILE}:{n}: {_short(e)}")
                    continue
                _check_rollout(rec, run_id, n, seen, errs)
                if rec.phase == "train":
                    train_per_step[rec.step] = train_per_step.get(rec.step, 0) + 1
                else:
                    hint = rec.eval_hint
                    counts.setdefault(rec.phase + f"|{rec.step}|{hint}", {}).setdefault(rec.problem_id, {})
                    row = counts[rec.phase + f"|{rec.step}|{hint}"][rec.problem_id]
                    row["n"] = row.get("n", 0) + 1
                    for k, v in count_row(rec).items():
                        row[k] = row.get(k, 0) + v
        except (OSError, EOFError, gzip.BadGzipFile) as e:
            errs.append(f"{ROLLOUTS_FILE}: unreadable ({_short(e)})")
        if steps:
            missing = [s.step for s in steps if s.step not in train_per_step]
            if missing:
                errs.append(f"{ROLLOUTS_FILE}: no train rollouts for steps {missing[:5]}")

    if (d / EVALS_FILE).is_file():
        try:
            ev = read_evals(d)
        except (ValidationError, ValueError, OSError) as e:
            errs.append(f"{EVALS_FILE}: {_short(e)}")
        else:
            _check_evals(ev, run_id, counts, errs)
    elif complete and counts:
        errs.append("eval rollouts exist but evals.json is missing")
    return errs


def _check_rollout(rec: RolloutRecord, run_id: str, n: int, seen: set, errs: list[str]) -> None:
    where = f"{ROLLOUTS_FILE}:{n}"
    if rec.run_id != run_id:
        errs.append(f"{where}: run_id {rec.run_id!r} != {run_id!r}")
    key = (rec.phase, rec.step, rec.eval_hint, rec.problem_id, rec.sample_idx)
    if key in seen:
        errs.append(f"{where}: duplicate rollout {key}")
    seen.add(key)
    if not math.isfinite(rec.reward):
        errs.append(f"{where}: non-finite reward")
    raw = {**{k: getattr(rec.labels, k) for k in ("defines_rt", "rt_ok", "visible_pass", "heldout_pass", "timeout", "crash")},
           "code_extracted": rec.code_extracted}
    if derive_labels(raw) != rec.labels.model_dump():
        errs.append(f"{where}: derived labels (hack_rt/attempt_rt/correct/visible_only) inconsistent with raw labels")
    if not rec.code_extracted and any(rec.labels.model_dump().values()):
        errs.append(f"{where}: labels set although no code was extracted")


def _check_evals(ev: EvalSummary, run_id: str, counts: dict, errs: list[str]) -> None:
    if ev.run_id != run_id:
        errs.append(f"{EVALS_FILE}: run_id {ev.run_id!r} != {run_id!r}")
    listed = set()
    for p in ev.points:
        hint = p.eval_hint
        k = f"{p.phase}|{p.step}|{hint}"
        listed.add(k)
        if p.key != eval_key(p.phase, p.step, hint if hint is not None else p.prompt_hint):
            errs.append(f"{EVALS_FILE}: point key {p.key!r} does not match its fields")
        recount = counts.get(k)
        if recount is None:
            errs.append(f"{EVALS_FILE}: point {p.key} has no rollouts in {ROLLOUTS_FILE}")
        elif recount != p.per_problem:
            errs.append(f"{EVALS_FILE}: point {p.key} counts differ from {ROLLOUTS_FILE}")
        if sum(r["n"] for r in p.per_problem.values()) != p.n_rollouts:
            errs.append(f"{EVALS_FILE}: point {p.key} n_rollouts != sum of per-problem n")
        for name in COUNT_KEYS:
            if sum(r[name] for r in p.per_problem.values()) != p.totals.get(name):
                errs.append(f"{EVALS_FILE}: point {p.key} totals[{name}] != sum of per-problem counts")
    for k in set(counts) - listed:
        errs.append(f"{EVALS_FILE}: eval rollouts of {k} have no point")


def _short(e: BaseException) -> str:
    text = " ".join(str(e).split())
    return text if len(text) <= 300 else text[:300] + "..."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rhg.runlog", description="Run-log tools.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check every log file of a run directory against the schemas")
    v.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    errs = validate_run(args.run_dir)
    if errs:
        for e in errs[:50]:
            print(f"INVALID: {e}", file=sys.stderr)
        if len(errs) > 50:
            print(f"... and {len(errs) - 50} more", file=sys.stderr)
        return 1
    print(f"OK: {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
