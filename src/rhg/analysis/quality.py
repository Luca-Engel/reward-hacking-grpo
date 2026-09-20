"""Run-quality checks (DESIGN §4 exploratory extras): homogeneity and training health.

* ``homogeneity``     do the runs of an arm, and of the two arms of the primary contrast, share GPU / driver / library
                      versions / git sha / dataset revision / split hash? Mixing is *flagged*, never fatal.
* ``training_health`` did training train? A null result is uninterpretable if the policy did not learn, so every arm
                      gets first-10 vs last-10 step training reward, a reward slope with its t statistic, the fraction
                      of zero-advantage groups, gradient norm, completion-length / truncation / code-extraction
                      failure trends and explicit flags. The thresholds below are fixed here (not tuned on outcomes);
                      they mark runs a reader should look at, they are not a validity rule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rhg import prereg_constants as C
from rhg.analysis.endpoints import RunData

EDGE = 10  # first / last steps compared
LEARNING_MIN_T = 2.0  # mean per-seed t statistic of the reward slope below this: no learning detected
LEARNING_MIN_GAIN = 0.02  # ... or a mean last-10 minus first-10 reward gain below this
ZERO_ADV_FLAG = 0.9  # mean fraction of zero-advantage groups (last 10 steps) above this: almost no learning signal
RISE_FLAG = 0.10  # truncation / extraction-failure rate rising by more than this (absolute)
LENGTH_COLLAPSE_FLAG = 0.5  # completion length falling below this fraction of its first-10 value

HOMOGENEITY_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gpu_name", ("hardware", "gpu_name")), ("gpu_count", ("hardware", "gpu_count")), ("driver", ("hardware", "driver")),
    ("cuda", ("hardware", "cuda")), ("python", ("libs", "python")), ("torch", ("libs", "torch")),
    ("transformers", ("libs", "transformers")), ("trl", ("libs", "trl")), ("peft", ("libs", "peft")),
    ("vllm", ("libs", "vllm")), ("git_sha", ("git_sha",)), ("git_dirty", ("git_dirty",)),
    ("config_hash", ("config_hash",)), ("dataset_revision", ("dataset_revision",)), ("split_hash", ("split_hash",)),
    ("prompts_hash", ("prompts_hash",)), ("freeze_json_sha256", ("freeze_json_sha256",)), ("mode", ("mode",)),
    ("confirmatory", ("confirmatory",)),
)
ARM_SPECIFIC_FIELDS = ("config_hash",)  # differs between arms by design; compared within an arm only


def _get(manifest: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = manifest
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return None
        cur = cur[p]
    return cur


@dataclass
class HomogeneityRow:
    scope: str
    field: str
    n_runs: int
    values: dict[str, int]
    status: str  # ok | MIXED | unrecorded

    def as_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "field": self.field, "n_runs": self.n_runs, "status": self.status,
                "values": "; ".join(f"{k} x{v}" for k, v in self.values.items())}


def _rows(scope: str, runs: Sequence[RunData], skip: Sequence[str] = ()) -> list[HomogeneityRow]:
    out = []
    for name, path in HOMOGENEITY_FIELDS:
        if name in skip:
            continue
        counts: dict[str, int] = {}
        for r in runs:
            v = _get(r.manifest, path)
            counts["<unrecorded>" if v is None else str(v)] = counts.get("<unrecorded>" if v is None else str(v), 0) + 1
        recorded = {k for k in counts if k != "<unrecorded>"}
        status = "unrecorded" if not recorded else ("MIXED" if len(counts) > 1 else "ok")
        out.append(HomogeneityRow(scope, name, len(runs), dict(sorted(counts.items())), status))
    return out


def homogeneity(runs: Sequence[RunData]) -> dict[str, Any]:
    """Rows per arm and for the primary contrast (both arms pooled; ``config_hash`` excluded there), plus ``flags``."""
    rows: list[HomogeneityRow] = []
    arms: dict[str, list[RunData]] = {}
    for r in runs:
        arms.setdefault(r.arm, []).append(r)
    for arm in sorted(arms):
        rows += _rows(f"arm:{arm}", arms[arm])
    pair = [r for r in runs if r.arm in C.PRIMARY_CONTRAST]
    if {r.arm for r in pair} == set(C.PRIMARY_CONTRAST):
        rows += _rows("primary_contrast", pair, skip=ARM_SPECIFIC_FIELDS)
    hosts = {_get(r.manifest, ("hardware", "hostname_sha256")) for r in runs} - {None}
    flags = [f"{r.scope}: {r.field} is MIXED ({'; '.join(f'{k} x{v}' for k, v in r.values.items())})" for r in rows
             if r.status == "MIXED"]
    return {"rows": [r.as_dict() for r in rows], "flags": flags, "n_hosts": len(hosts), "mixed": bool(flags)}


# ------------------------------------------------------------------ training health
def _slope(y: Sequence[float]) -> tuple[float, float]:
    """OLS slope of ``y`` on step (1..n) and its t statistic (nan if it cannot be computed)."""
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(yy)
    if ok.sum() < 3:
        return math.nan, math.nan
    x = np.arange(1, len(yy) + 1, dtype=float)[ok]
    yy = yy[ok]
    xm = x - x.mean()
    sxx = float(xm @ xm)
    slope = float(xm @ (yy - yy.mean()) / sxx)
    resid = yy - yy.mean() - slope * xm
    se = math.sqrt(float(resid @ resid) / (len(yy) - 2) / sxx)
    t = math.copysign(math.inf, slope) if se == 0 and slope != 0 else (0.0 if se == 0 else slope / se)
    return slope, t


def _edge(y: Sequence[float], first: bool) -> float:
    seg = list(y[:EDGE] if first else y[-EDGE:])
    seg = [v for v in seg if not math.isnan(v)]
    return float(np.mean(seg)) if seg else math.nan


def seed_health(run: RunData) -> dict[str, Any]:
    st = run.steps
    col = lambda name: [float(getattr(s, name)) for s in st]  # noqa: E731
    reward, length, trunc, zero, grad = (col(n) for n in ("reward_mean", "completion_len_mean", "truncation_rate",
                                                          "frac_zero_adv_groups", "grad_norm"))
    slope, t = _slope(reward)
    extract = run.train_series("code_fail")
    ex_slope, _ = _slope(extract)
    len_slope, _ = _slope(length)
    return {
        "run_id": run.run_id, "arm": run.arm, "seed": run.seed, "n_steps": len(st),
        "reward_first10": _edge(reward, True), "reward_last10": _edge(reward, False),
        "reward_gain": _edge(reward, False) - _edge(reward, True), "reward_slope_per_step": slope, "reward_slope_t": t,
        "frac_zero_adv_first10": _edge(zero, True), "frac_zero_adv_last10": _edge(zero, False),
        "frac_zero_adv_mean": float(np.nanmean(zero)) if zero else math.nan,
        "grad_norm_mean": float(np.nanmean(grad)) if grad else math.nan, "grad_norm_last10": _edge(grad, False),
        "length_first10": _edge(length, True), "length_last10": _edge(length, False), "length_slope_per_step": len_slope,
        "truncation_first10": _edge(trunc, True), "truncation_last10": _edge(trunc, False),
        "extraction_fail_first10": _edge(extract, True), "extraction_fail_last10": _edge(extract, False),
        "extraction_fail_slope_per_step": ex_slope,
    }


def training_health(runs: Sequence[RunData]) -> dict[str, Any]:
    """Per-seed rows, per-arm means and flags. ``flags[arm]`` lists what a reader should check before trusting a null."""
    per_seed = [seed_health(r) for r in sorted(runs, key=lambda r: (r.arm, r.seed))]
    arms: dict[str, list[dict[str, Any]]] = {}
    for row in per_seed:
        arms.setdefault(row["arm"], []).append(row)
    per_arm, flags = [], {}
    for arm, rows in arms.items():
        mean = lambda k: float(np.nanmean([r[k] for r in rows])) if any(not math.isnan(r[k]) for r in rows) else math.nan  # noqa: E731
        agg = {"arm": arm, "n_seeds": len(rows), **{k: mean(k) for k in rows[0] if k not in ("run_id", "arm", "seed", "n_steps")}}
        f = []
        if not (agg["reward_slope_t"] >= LEARNING_MIN_T and agg["reward_gain"] >= LEARNING_MIN_GAIN):
            f.append(f"NOT LEARNING: mean reward slope t = {agg['reward_slope_t']:.2f} (need >= {LEARNING_MIN_T}) and "
                     f"last-10 minus first-10 reward = {agg['reward_gain']:+.3f} (need >= {LEARNING_MIN_GAIN})")
        if agg["frac_zero_adv_last10"] > ZERO_ADV_FLAG:
            f.append(f"almost no learning signal: {agg['frac_zero_adv_last10']:.2f} of groups have zero advantage in the last 10 steps")
        if any(math.isnan(r["grad_norm_mean"]) for r in rows):
            f.append("non-finite gradient norm in at least one seed")
        if agg["truncation_last10"] - agg["truncation_first10"] > RISE_FLAG:
            f.append(f"truncation rate rose {agg['truncation_first10']:.2f} -> {agg['truncation_last10']:.2f}")
        if agg["extraction_fail_last10"] - agg["extraction_fail_first10"] > RISE_FLAG:
            f.append(f"code-extraction failures rose {agg['extraction_fail_first10']:.2f} -> {agg['extraction_fail_last10']:.2f}")
        if agg["length_first10"] > 0 and agg["length_last10"] < LENGTH_COLLAPSE_FLAG * agg["length_first10"]:
            f.append(f"completion length fell {agg['length_first10']:.0f} -> {agg['length_last10']:.0f} tokens")
        agg["flags"] = " | ".join(f)
        per_arm.append(agg)
        flags[arm] = f
    return {"per_seed": per_seed, "per_arm": per_arm, "flags": flags,
            "not_learning_arms": sorted(a for a, f in flags.items() if any(x.startswith("NOT LEARNING") for x in f))}
