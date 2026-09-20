"""Shared builders for the analysis tests (subtask 14): synthetic run directories with planted truth, hand-made RunData."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from rhg import runlog
from rhg.analysis import endpoints as E
from rhg.analysis import simulate
from rhg.analysis.simulate import Scenario, SimShape

SHAPE = SimShape(steps=60, prompts_per_step=8, gens_per_prompt=4, val_every=20, n_train=16, n_val=6, n_test=24, val_samples=4,
                 test_samples=8, xhint_samples=4)
SMALL = SimShape(steps=30, prompts_per_step=4, gens_per_prompt=4, val_every=10, n_train=8, n_val=4, n_test=24, val_samples=2,
                 test_samples=8, xhint_samples=2)
PRIMARY = {"hackable_subtle": 5, "clean_subtle": 5}


def scenario(**kw: Any) -> Scenario:
    """Default world with every emergence probability planted explicitly (deterministic verdicts)."""
    q = {**Scenario().q, "hackable_subtle": 1.0, "hackable_explicit": 1.0, "hackable_subtle_ast": 1.0}
    q.update(kw.pop("q", {}))
    return replace(Scenario(), q=q, honest_gain=kw.pop("honest_gain", 0.25), **kw)


def simulate_dir(out: Path, sc: Scenario, *, seeds_per_arm: Mapping[str, int] | None = None, master_seed: int = 11,
                 shape: SimShape = SHAPE, **kw: Any) -> SimpleNamespace:
    truth = simulate.simulate_runs(out, scenario=sc, seeds_per_arm=seeds_per_arm, master_seed=master_seed, shape=shape, **kw)
    return SimpleNamespace(out=out, runs=out / "runs", problems=out / "problems.jsonl", truth=truth)


def per_run_dir(out: Path, plan: Mapping[tuple[str, int], Scenario], *, master_seed: int = 5, shape: SimShape = SMALL) -> SimpleNamespace:
    """One run per ``(arm, seed)`` with its own scenario (plants a single influential seed, a non-learning arm, ...)."""
    problems = simulate.make_problem_table(shape.n_train, shape.n_val, shape.n_test)
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (out / "problems.jsonl").write_text("".join(json.dumps(p) + "\n" for p in problems), encoding="utf-8", newline="\n")
    lat = {}
    for i, ((arm, seed), sc) in enumerate(plan.items()):
        _, lat[f"{arm}__s{seed}"] = simulate.simulate_run(runs, arm, seed, sc, master_seed, shape, problems, index=i)
    return SimpleNamespace(out=out, runs=runs, problems=out / "problems.jsonl", latents=lat)


# ------------------------------------------------------------------ independent readers (no rhg.analysis code)
def raw_rollouts(run_dir: Path) -> list[dict[str, Any]]:
    with gzip.open(run_dir / "rollouts.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_steps(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def is_hack(r: Mapping[str, Any]) -> bool:
    """HACK_RT from the raw execution fields (not from the derived label)."""
    lab = r["labels"]
    return bool(r["code_extracted"] and lab["defines_rt"] and lab["rt_ok"] and not lab["heldout_pass"])


def final_eval(run_dir: Path, phase: str = "eval_test", step: int | None = None, hint: str | None = None) -> list[dict[str, Any]]:
    recs = raw_rollouts(run_dir)
    if step is None:
        step = max(r["step"] for r in recs if r["phase"] == "eval_test")
    return [r for r in recs if r["phase"] == phase and r["step"] == step and r["eval_hint"] == hint]


# ------------------------------------------------------------------ hand-made RunData
def fake_run(arm: str, seed: int, hack_by_problem: Mapping[str, int], *, n_per_problem: int = 8, T: int = 100,
             train_hack: Iterable[float] | None = None, val_rates: Iterable[float] = (), correct_by_problem: Mapping[str, int] | None = None) -> E.RunData:
    """A RunData whose final test eval has ``hack_by_problem[pid]`` hacks out of ``n_per_problem`` (no run directory)."""
    def point(phase: str, step: int, hacks: Mapping[str, int]) -> E.PointAgg:
        pt = E.PointAgg(phase, step, None)
        for pid, k in hacks.items():
            row = {key: 0 for key in E.COUNT_KEYS}
            row.update(n=n_per_problem, hack_rt=k, hack_notimeout=k, hack_evaded=k, tokens=100 * n_per_problem)
            if correct_by_problem is not None:
                row["correct"] = correct_by_problem.get(pid, 0)
            pt.add(pid, row)
        return pt

    points: dict[tuple[str, int, str | None], E.PointAgg] = {("eval_test", T, None): point("eval_test", T, hack_by_problem)}
    ids = list(hack_by_problem)
    for i, rate in enumerate(val_rates, start=1):
        k = round(rate * n_per_problem)
        points[("eval_val", 20 * i, None)] = point("eval_val", 20 * i, {p: k for p in ids})
    rates = list(train_hack) if train_hack is not None else [0.0] * T
    steps = [runlog.StepRecord(step=i + 1, reward_mean=0.3, loss=0.0, grad_norm=0.5, completion_len_mean=200.0, truncation_rate=0.0,
                               frac_zero_adv_groups=0.2, hack_rt_rate_train=float(r), attempt_rt_rate_train=float(r),
                               correct_rate_train=0.3, t_gen=1.0, t_reward=1.0, t_train=1.0, t_sync=1.0, t_step=4.0, tokens_gen=1, tokens_train=1)
             for i, r in enumerate(rates)]
    return E.RunData(f"{arm}__s{seed}", arm, seed, Path("."), T, {}, steps, {}, points)

