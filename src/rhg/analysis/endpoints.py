"""Per-seed endpoints from run directories (DESIGN §4, PREREG §2-§3).

One pass over each run's ``rollouts.jsonl.gz`` (validated through ``rhg.runlog``) aggregates every rollout into
per-problem counts of the eval points and per-step counts of the training phase; every endpoint is computed from
those aggregates and ``steps.jsonl``:

* ``final_hack_rt``   HACK_RT rate of the final ``eval_test`` point (step T): the primary endpoint;
* ``onset``           first step whose trailing-5-step mean *training* HACK_RT rate is >= 0.10, censored at T + 1;
* ``gap``             mean training reward over the last 5 steps minus mean training ``heldout_pass`` (H3a, descriptive);
* ``final_correct``   CORRECT rate of the final test eval (H3b);
* ``evasion``         P(not ast_narrow | HACK_RT) at the final test eval (H4);
* ``rho``             Spearman correlation of per-problem final hack rate with ``p_B_full`` (H2); seeds whose final rate
                      is not strictly inside (0, 1) are not usable and are counted;
* covariates, the step-0 test baseline and per-seed change, and the cross-hint hack rate per ``eval_hint``.

``load_runs`` also builds the run-validity table: every run directory is listed with its status, cause and exit
code; only completed, fully consistent, non-pilot runs enter the analysis. Nothing here reads a result before the
single analysis run and nothing depends on the arm labels beyond grouping.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rhg import prereg_constants as C
from rhg import runlog
from rhg.analysis import stats

COUNT_KEYS = ("n", "hack_rt", "attempt_rt", "correct", "gap_other", "heldout_pass", "visible_pass", "defines_rt", "timeout",
              "hack_notimeout", "hack_evaded", "ast_narrow", "code_fail", "truncated", "tokens")
TRAIN_KEYS = (*COUNT_KEYS, "reward")
DEFAULT_PILOT_SEED_MIN = 9000
REPLACEMENT_SEED_MIN = 100
GAP_WINDOW = 5
XHINTS = runlog.EVAL_HINTS
TEST_PHASE, VAL_PHASE, XHINT_PHASE = "eval_test", "eval_val", "eval_test_xhint"


def _count(rec: runlog.RolloutRecord) -> dict[str, float]:
    lab, mon = rec.labels, rec.monitor
    return {
        "n": 1, "hack_rt": lab.hack_rt, "attempt_rt": lab.attempt_rt, "correct": lab.correct, "gap_other": lab.gap_other,
        "heldout_pass": lab.heldout_pass, "visible_pass": lab.visible_pass, "defines_rt": lab.defines_rt,
        "timeout": lab.timeout, "hack_notimeout": lab.hack_rt and not lab.timeout,
        "hack_evaded": lab.hack_rt and not mon.ast_narrow, "ast_narrow": mon.ast_narrow,
        "code_fail": not rec.code_extracted, "truncated": rec.truncated, "tokens": rec.n_tokens,
    }


@dataclass
class PointAgg:
    """Per-problem counts of one eval point (phase, step, eval_hint)."""

    phase: str
    step: int
    hint: str | None
    per_problem: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, pid: str, row: Mapping[str, float]) -> None:
        acc = self.per_problem.setdefault(pid, {k: 0 for k in COUNT_KEYS})
        for k in COUNT_KEYS:
            acc[k] += int(row[k])

    def total(self, key: str, problems: Iterable[str] | None = None) -> int:
        ids = self.per_problem if problems is None else [p for p in problems if p in self.per_problem]
        return sum(self.per_problem[p][key] for p in ids)

    def n(self, problems: Iterable[str] | None = None) -> int:
        return self.total("n", problems)

    def rate(self, key: str, problems: Iterable[str] | None = None) -> float:
        n = self.n(problems)
        return self.total(key, problems) / n if n else math.nan

    def problem_rates(self, key: str) -> dict[str, float]:
        return {p: r[key] / r["n"] for p, r in self.per_problem.items() if r["n"]}


@dataclass
class RunData:
    run_id: str
    arm: str
    seed: int
    run_dir: Path
    T: int
    manifest: dict[str, Any]
    steps: list[runlog.StepRecord]
    train: dict[int, dict[str, float]]
    points: dict[tuple[str, int, str | None], PointAgg]

    def point(self, phase: str, step: int, hint: str | None = None) -> PointAgg | None:
        return self.points.get((phase, step, hint))

    @property
    def final(self) -> PointAgg | None:
        return self.point(TEST_PHASE, self.T)

    def val_points(self) -> list[PointAgg]:
        return sorted((p for (ph, _, _), p in self.points.items() if ph == VAL_PHASE), key=lambda p: p.step)

    def train_series(self, key: str) -> list[float]:
        """Per-step training rate of ``key`` (steps 1..T) from the aggregated training rollouts."""
        return [self.train[s][key] / self.train[s]["n"] if self.train.get(s, {}).get("n") else math.nan
                for s in range(1, self.T + 1)]

    def hack_train_rates(self) -> list[float]:
        return [s.hack_rt_rate_train for s in self.steps]


def aggregate_run(run_dir: Path, run_id: str, arm: str, seed: int, manifest: dict[str, Any], T: int) -> RunData:
    steps = runlog.read_steps(run_dir)
    train: dict[int, dict[str, float]] = {}
    points: dict[tuple[str, int, str | None], PointAgg] = {}
    for rec in runlog.iter_rollouts(run_dir):
        row = _count(rec)
        if rec.phase == "train":
            acc = train.setdefault(rec.step, {k: 0.0 for k in TRAIN_KEYS})
            for k in COUNT_KEYS:
                acc[k] += row[k]
            acc["reward"] += rec.reward
        else:
            key = (rec.phase, rec.step, rec.eval_hint)
            points.setdefault(key, PointAgg(rec.phase, rec.step, rec.eval_hint)).add(rec.problem_id, row)
    return RunData(run_id, arm, seed, run_dir, T, manifest, steps, train, points)


# ------------------------------------------------------------------ run loading and validity
def _read_json(path: Path) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _configured_T(run_dir: Path) -> int | None:
    try:
        cfg = yaml.safe_load((run_dir / runlog.CONFIG_FILE).read_text(encoding="utf-8"))
        return int(cfg["grpo"]["max_steps"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


def _pilot_seed_min() -> int:
    try:
        from rhg.plan import load_plan

        return load_plan().pilot_seed_min
    except Exception:  # noqa: BLE001 - plan file absent in a bare checkout
        return DEFAULT_PILOT_SEED_MIN


def split_run_id(run_id: str) -> tuple[str, int] | None:
    arm, sep, seed = run_id.rpartition("__s")
    return (arm, int(seed)) if sep and seed.lstrip("-").isdigit() else None


@dataclass
class RunSet:
    runs: list[RunData]
    validity: list[dict[str, Any]]

    def arms(self) -> list[str]:
        return sorted({r.arm for r in self.runs})

    def by_arm(self) -> dict[str, list[RunData]]:
        out: dict[str, list[RunData]] = {}
        for r in sorted(self.runs, key=lambda r: (r.arm, r.seed)):
            out.setdefault(r.arm, []).append(r)
        return out

    def seed_counts(self) -> dict[str, int]:
        return {a: len(v) for a, v in self.by_arm().items()}


def _consistency_errors(d: Path, T: int | None) -> list[str]:
    errs = runlog.validate_run(d)
    if errs:
        return errs[:3]
    steps = runlog.read_steps(d)
    if T is not None and len(steps) != T:
        return [f"{len(steps)} training steps logged, {T} configured"]
    return []


def load_runs(runs_dir: str | Path, *, validate: bool = True, pilot_seed_min: int | None = None) -> RunSet:
    """Aggregate every completed, valid, non-pilot run under ``runs_dir`` and tabulate the validity of all of them.

    A run is *valid* iff it completed all T steps with finite loss/reward, every log validates and the final test
    eval exists (DESIGN §7.2). Pilot runs (seed >= ``pilot_seed_min``) are listed and excluded (PREREG §7).
    """
    root = Path(runs_dir)
    pilot_min = _pilot_seed_min() if pilot_seed_min is None else pilot_seed_min
    dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    validity: list[dict[str, Any]] = []
    runs: list[RunData] = []
    for d in dirs:
        ident = split_run_id(d.name)
        status_d = _read_json(d / runlog.STATUS_FILE)
        manifest = _read_json(d / runlog.MANIFEST_FILE)
        arm, seed = ident if ident else (str(manifest.get("arm", "?")), int(manifest.get("seed", -1)))
        status = str(status_d.get("status") or manifest.get("status") or "unknown")
        kind = "pilot" if seed >= pilot_min else ("replacement" if seed >= REPLACEMENT_SEED_MIN else "planned")
        row = {"run_id": d.name, "arm": arm, "seed": seed, "kind": kind, "status": status,
               "reason": status_d.get("reason") or manifest.get("invalid_reason"), "exit_code": status_d.get("exit_code"),
               "included": False, "exclusion_cause": None}
        validity.append(row)
        if kind == "pilot":
            row["exclusion_cause"] = "pilot run (PREREG §7)"
            continue
        if status != "completed":
            row["exclusion_cause"] = f"status {status}" + (f": {row['reason']}" if row["reason"] else "")
            continue
        T = _configured_T(d)
        problems = _consistency_errors(d, T) if validate else []
        if problems:
            row["exclusion_cause"] = "invalid logs: " + "; ".join(problems)
            continue
        try:
            run = aggregate_run(d, d.name, arm, seed, manifest, T if T is not None else len(runlog.read_steps(d)))
        except (OSError, ValueError, EOFError) as e:
            row["exclusion_cause"] = f"unreadable logs: {str(e)[:120]}"
            continue
        if run.final is None:
            row["exclusion_cause"] = f"no final test eval at step {run.T}"
            continue
        row["included"] = True
        runs.append(run)
    replaced = {(r["arm"]): [v["run_id"] for v in validity if v["arm"] == r["arm"] and v["kind"] == "replacement" and v["included"]]
                for r in validity}
    for r in validity:
        if not r["included"] and r["kind"] == "planned":
            r["replaced_by"] = replaced.get(r["arm"]) or None
    return RunSet(runs, validity)


# ------------------------------------------------------------------ problems
def load_problems(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """``problem_id -> row`` from a ``problems.jsonl`` (REPO_SPEC §5); ``{}`` if the file is absent."""
    if path is None or not Path(path).is_file():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["problem_id"])] = row
    return out


def half_of(problem_id: str) -> int:
    """Fixed test-set half (0/1) of a problem: parity of the sha256 of its id (independent of every seed)."""
    return int(hashlib.sha256(problem_id.encode("utf-8")).hexdigest(), 16) % 2


# ------------------------------------------------------------------ endpoints
def onset_step(rates: Sequence[float], T: int, threshold: float = C.ONSET_THRESHOLD, window: int = C.ONSET_WINDOW) -> int:
    """First step k (1-based, k >= window) whose trailing ``window``-step mean of ``rates`` is >= threshold, else T + 1."""
    for k in range(window, len(rates) + 1):
        if sum(rates[k - window:k]) / window >= threshold - 1e-12:
            return k
    return T + 1


@dataclass
class SeedEndpoints:
    run_id: str
    arm: str
    seed: int
    T: int
    n_final: int
    final_hack_rt: float
    final_attempt_rt: float
    final_correct: float
    final_gap_other: float
    n_hack_final: int
    evasion: float  # nan without a hack
    step0_hack_rt: float
    change_hack_rt: float
    onset: int
    train_reward_last5: float
    train_heldout_last5: float
    gap: float
    rho: float  # nan if not defined
    rho_usable: bool
    n_problems: int
    len_mean: float
    truncation_rate: float
    extraction_fail_rate: float
    val_last_hack_rt: float
    xhint: dict[str, float]  # eval_hint -> hack rate (test problems), only hints that were evaluated

    @property
    def emerged(self) -> bool:
        return self.final_hack_rt >= C.EMERGED_THRESHOLD - 1e-12

    def csv_row(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "xhint"}
        d["emerged"] = self.emerged
        for h in XHINTS:
            d[f"xhint_hack_rt_{h}"] = self.xhint.get(h, math.nan)
        return d


def compute_endpoints(run: RunData, problems: Mapping[str, Mapping[str, Any]] | None = None) -> SeedEndpoints:
    fin = run.final
    assert fin is not None
    T = run.T
    n = fin.n()
    hack = fin.total("hack_rt")
    evasion = fin.total("hack_evaded") / hack if hack else math.nan
    base = run.point(TEST_PHASE, 0)
    step0 = base.rate("hack_rt") if base else math.nan
    final_rate = fin.rate("hack_rt")
    last = range(max(1, T - GAP_WINDOW + 1), T + 1)
    tn = sum(run.train.get(s, {}).get("n", 0) for s in last)
    reward5 = sum(run.train.get(s, {}).get("reward", 0.0) for s in last) / tn if tn else math.nan
    held5 = sum(run.train.get(s, {}).get("heldout_pass", 0) for s in last) / tn if tn else math.nan
    rates = fin.problem_rates("hack_rt")
    pb = {p: float(problems[p]["p_B_full"]) for p in rates if problems and p in problems and problems[p].get("p_B_full") is not None}
    usable = 0.0 < final_rate < 1.0
    rho = math.nan
    if pb and len(pb) == len(rates):
        ids = sorted(rates)
        rho = stats.spearman([rates[p] for p in ids], [pb[p] for p in ids])
    vals = run.val_points()
    xhint = {h: pt.rate("hack_rt") for h in XHINTS if (pt := run.point(XHINT_PHASE, T, h)) is not None}
    return SeedEndpoints(
        run_id=run.run_id, arm=run.arm, seed=run.seed, T=T, n_final=n, final_hack_rt=final_rate,
        final_attempt_rt=fin.rate("attempt_rt"), final_correct=fin.rate("correct"), final_gap_other=fin.rate("gap_other"),
        n_hack_final=hack, evasion=evasion, step0_hack_rt=step0,
        change_hack_rt=final_rate - step0 if not math.isnan(step0) else math.nan,
        onset=onset_step(run.hack_train_rates(), T), train_reward_last5=reward5, train_heldout_last5=held5,
        gap=reward5 - held5, rho=rho, rho_usable=bool(usable and not math.isnan(rho)), n_problems=len(rates),
        len_mean=fin.total("tokens") / n if n else math.nan, truncation_rate=fin.rate("truncated"),
        extraction_fail_rate=fin.rate("code_fail"), val_last_hack_rt=vals[-1].rate("hack_rt") if vals else math.nan,
        xhint=xhint,
    )


def build_table(runset: RunSet, problems: Mapping[str, Mapping[str, Any]] | None = None) -> list[SeedEndpoints]:
    return [compute_endpoints(r, problems) for r in sorted(runset.runs, key=lambda r: (r.arm, r.seed))]


def by_arm(table: Sequence[SeedEndpoints], attr: str) -> dict[str, list[float]]:
    """``arm -> [value per seed]`` (seed order); the input of every arm comparison."""
    out: dict[str, list[float]] = {}
    for e in table:
        out.setdefault(e.arm, []).append(float(getattr(e, attr)))
    return out


def write_per_seed_csv(table: Sequence[SeedEndpoints], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [e.csv_row() for e in table]
    fields = list(rows[0]) if rows else ["run_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    return path
