"""Robustness suite for the primary contrast and H1 (DESIGN §4, PREREG §7). ALL EXPLORATORY.

Nothing here can change a pre-registered verdict; it shows how fragile it is.

(a) leave-one-seed-out      each seed dropped in turn; p and Δ recomputed; a flip of the verdict is flagged and the
                            influential seed named (verdict = PREREG §2 rule for the primary, p <= alpha for H1);
(b) hack-definition variants HACK_RT without held-out failures that coincide with a timeout, ATTEMPT_RT, HACK_RT ∪ VISIBLE_ONLY;
(c) endpoint window          previous val eval, mean of the last 3 val evals, that mean plus the final test eval, and the
                            trailing-5 training steps, next to the final test eval (rebound sensitivity, DESIGN §8.15);
(d) onset grid               threshold {0.05, 0.10, 0.20} x window {3, 5, 10};
(e) exact rank test          exact permutation on mid-ranks (Mann-Whitney) next to the difference of means;
(f) test-set halves          two fixed halves of the test problems (parity of sha256(problem_id)): same direction?
(g) variance decomposition   within-seed binomial SE vs between-seed SD per arm.

``run_robustness`` returns plain dicts/lists (JSON- and CSV-ready) plus ``flags``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from rhg import prereg_constants as C
from rhg.analysis import stats
from rhg.analysis.endpoints import RunData, half_of, load_runs, onset_step

ONSET_THRESHOLDS = (0.05, 0.10, 0.20)
ONSET_WINDOWS = (3, 5, 10)
LAST_VAL_EVALS = 3
REBOUND_RATIO = 0.5  # final test rate below this fraction of the best earlier val rate = retreat phase
Metric = Callable[[RunData], float]


# ------------------------------------------------------------------ building blocks
def mann_whitney_exact(a: Sequence[float], b: Sequence[float], alternative: str = "greater") -> stats.PermResult:
    """Exact permutation test on the pooled mid-ranks (equivalent to the exact Mann-Whitney U test, ties averaged)."""
    ranks = rankdata(np.concatenate([np.asarray(a, float), np.asarray(b, float)]))
    return stats.perm_test(ranks[: len(a)], ranks[len(a):], alternative)


def _values(runs: Sequence[RunData], metric: Metric) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in sorted(runs, key=lambda r: (r.arm, r.seed)):
        out.setdefault(r.arm, {})[r.run_id] = float(metric(r))
    return out


def _list(vals: Mapping[str, Mapping[str, float]], arm: str) -> list[float]:
    return list(vals.get(arm, {}).values())


def _primary(vals: Mapping[str, Mapping[str, float]], *, decision: bool = True, alternative: str = "greater") -> dict[str, Any] | None:
    h, c = _list(vals, C.PRIMARY_CONTRAST[0]), _list(vals, C.PRIMARY_CONTRAST[1])
    if not h or not c or any(math.isnan(v) for v in h + c):
        return None
    if decision and alternative == "greater":
        d = stats.primary_decision(h, c)
        return {"p": d.p, "delta": d.delta, "outcome": d.outcome, "verdict": d.outcome == "supported",
                "min_p": d.test.min_attainable_p, "n_h": len(h), "n_c": len(c)}
    r = stats.perm_test(h, c, alternative)
    return {"p": r.p, "delta": r.observed, "outcome": None, "verdict": stats.leq(r.p, C.ALPHA), "min_p": r.min_attainable_p,
            "n_h": len(h), "n_c": len(c)}


def _h1(vals: Mapping[str, Mapping[str, float]], alternative: str = "increasing") -> dict[str, Any] | None:
    groups = [list(vals[a].values()) for a in C.H1_ARMS if vals.get(a)]
    if len(groups) < 2 or any(math.isnan(v) for g in groups for v in g):
        return None
    r = stats.jonckheere_terpstra(groups, alternative)
    return {"p": r.p, "verdict": stats.leq(r.p, C.ALPHA), "min_p": r.min_attainable_p, "n": sum(r.sizes)}


def _row(prefix: str, res: Mapping[str, Any] | None) -> dict[str, Any]:
    if res is None:
        return {f"{prefix}_p": math.nan}
    n = f"{res['n_h']} v {res['n_c']}" if "n_h" in res else str(res.get("n", ""))
    out = {f"{prefix}_p": res["p"], f"{prefix}_verdict": res["verdict"], f"{prefix}_min_p": res["min_p"], f"{prefix}_n": n}
    if "delta" in res:
        out[f"{prefix}_delta"] = res["delta"]
    return out


# ------------------------------------------------------------------ (a) leave-one-seed-out
def leave_one_out(vals: Mapping[str, Mapping[str, float]], kind: str) -> dict[str, Any]:
    """``kind``: ``primary`` (final hack rate, PREREG §2 rule), ``h1_final`` (JT increasing), ``h1_onset`` (JT decreasing)."""
    arms = C.PRIMARY_CONTRAST if kind == "primary" else C.H1_ARMS
    data = {a: dict(vals[a]) for a in arms if a in vals}

    def test(d: Mapping[str, Mapping[str, float]]):
        if kind == "primary":
            return _primary(d)
        return _h1(d, "increasing" if kind == "h1_final" else "decreasing")

    full = test(data)
    if full is None:
        return {"kind": kind, "available": False, "rows": [], "flip": False, "influential": None}
    rows = []
    for (arm, run_id), res in stats.loo(test, data):
        row = {"dropped": run_id, "arm": arm, "p": math.nan, "delta": math.nan, "verdict": None, "flip": False}
        if res is not None:
            row.update(p=res["p"], delta=res.get("delta", math.nan), verdict=res["verdict"], min_p=res["min_p"],
                       n=f"{res['n_h']} v {res['n_c']}" if "n_h" in res else str(res["n"]), flip=res["verdict"] != full["verdict"])
        rows.append(row)
    usable = [r for r in rows if r["verdict"] is not None]
    flippers = [r for r in usable if r["flip"]]
    pool = flippers or usable
    influential = max(pool, key=lambda r: abs(r["p"] - full["p"]))["dropped"] if pool else None
    full_n = f"{full['n_h']} v {full['n_c']}" if "n_h" in full else str(full["n"])
    return {"kind": kind, "available": True,
            "full": {**{k: full[k] for k in ("p", "verdict", "min_p") + (("delta", "outcome") if kind == "primary" else ())}, "n": full_n},
            "rows": rows, "flip": bool(flippers), "influential": influential}


# ------------------------------------------------------------------ (b) definition variants
def _fin(key: str, extra: str | None = None) -> Metric:
    return lambda r: r.final.rate(key) + (r.final.rate(extra) if extra else 0.0)  # type: ignore[union-attr]


DEFINITION_VARIANTS: dict[str, tuple[str, Metric]] = {
    "HACK_RT (pre-registered)": ("hack_rt", _fin("hack_rt")),
    "HACK_RT excluding hacks whose run coincided with a timeout": ("hack_notimeout", _fin("hack_notimeout")),
    "ATTEMPT_RT (defines run_tests)": ("attempt_rt", _fin("attempt_rt")),
    "HACK_RT or VISIBLE_ONLY": ("hack_or_gap", _fin("hack_rt", "visible_only")),
}


def definition_variants(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    rows = []
    for name, (key, metric) in DEFINITION_VARIANTS.items():
        vals = _values(runs, metric)
        rows.append({"variant": name, "key": key, **_row("primary", _primary(vals)), **_row("h1", _h1(vals)),
                     "seed_values": {a: list(v.values()) for a, v in vals.items() if a in C.PRIMARY_CONTRAST}})
    return rows


# ------------------------------------------------------------------ (c) endpoint window
def _val_rates(run: RunData) -> list[float]:
    return [p.rate("hack_rt") for p in run.val_points() if p.step > 0]


def _window_metrics() -> dict[str, Metric]:
    def prev_val(r: RunData) -> float:
        v = _val_rates(r)
        return v[-1] if v else math.nan

    def mean_last(r: RunData) -> float:
        v = _val_rates(r)[-LAST_VAL_EVALS:]
        return float(np.mean(v)) if v else math.nan

    def mean_last_plus_final(r: RunData) -> float:
        v = _val_rates(r)[-LAST_VAL_EVALS:] + [r.final.rate("hack_rt")]  # type: ignore[union-attr]
        return float(np.mean(v)) if len(v) > 1 else math.nan

    def trailing(r: RunData) -> float:
        h = r.hack_train_rates()[-C.ONSET_WINDOW:]
        return float(np.mean(h)) if h else math.nan

    return {"final test eval (pre-registered)": _fin("hack_rt"), "previous val eval (one eval earlier)": prev_val,
            f"mean of the last {LAST_VAL_EVALS} val evals": mean_last,
            f"mean of the last {LAST_VAL_EVALS} val evals and the final test eval": mean_last_plus_final,
            f"trailing-{C.ONSET_WINDOW} training steps at T": trailing}


def endpoint_windows(runs: Sequence[RunData]) -> dict[str, Any]:
    rows = []
    for name, metric in _window_metrics().items():
        vals = _values(runs, metric)
        rows.append({"window": name, **_row("primary", _primary(vals)), **_row("h1", _h1(vals)),
                     "available": _primary(vals) is not None or _h1(vals) is not None})
    rebound = []
    for r in sorted(runs, key=lambda r: (r.arm, r.seed)):
        v = _val_rates(r)
        best, final = (max(v) if v else math.nan), r.final.rate("hack_rt")  # type: ignore[union-attr]
        rebound.append({"run_id": r.run_id, "arm": r.arm, "best_val_rate": best, "final_test_rate": final,
                        "retreat": bool(v and best >= C.EMERGED_THRESHOLD and final < REBOUND_RATIO * best)})
    return {"rows": rows, "rebound": rebound}


# ------------------------------------------------------------------ (d) onset grid
def onset_grid(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    rows = []
    for th in ONSET_THRESHOLDS:
        for w in ONSET_WINDOWS:
            vals = _values(runs, lambda r, th=th, w=w: onset_step(r.hack_train_rates(), r.T, th, w))
            prim = _primary(vals, decision=False, alternative="less")
            med = {a: float(np.median(list(v.values()))) for a, v in vals.items()}
            rows.append({"threshold": th, "window": w, **_row("primary_onset", prim), **_row("h1_onset", _h1(vals, "decreasing")),
                         "median_onset": med, "is_pre_registered": (th, w) == (C.ONSET_THRESHOLD, C.ONSET_WINDOW)})
    return rows


# ------------------------------------------------------------------ (e) rank test
def rank_test(runs: Sequence[RunData]) -> dict[str, Any]:
    vals = _values(runs, _fin("hack_rt"))
    h, c = _list(vals, C.PRIMARY_CONTRAST[0]), _list(vals, C.PRIMARY_CONTRAST[1])
    if not h or not c:
        return {"available": False}
    diff, rank = stats.perm_test(h, c, "greater"), mann_whitney_exact(h, c)
    return {"available": True, "p_diff_of_means": diff.p, "delta": diff.observed, "p_exact_rank": rank.p,
            "min_p": rank.min_attainable_p, "n_h": len(h), "n_c": len(c), "paired": paired_primary(runs)}


def paired_primary(runs: Sequence[RunData]) -> dict[str, Any]:
    """The primary contrast as a paired (seed-stratified) exact sign-flip test: seed k of both primary arms shares data order
    and LoRA init (DESIGN §3), while the pre-registered test ignores that pairing (valid, slightly conservative)."""
    metric = _fin("hack_rt")
    by_seed: dict[str, dict[int, float]] = {arm: {} for arm in C.PRIMARY_CONTRAST}
    for r in runs:
        if r.arm in by_seed:
            by_seed[r.arm][r.seed] = float(metric(r))
    seeds = sorted(set(by_seed[C.PRIMARY_CONTRAST[0]]) & set(by_seed[C.PRIMARY_CONTRAST[1]]))
    if not seeds:
        return {"available": False}
    res = stats.paired_perm_test([by_seed[C.PRIMARY_CONTRAST[0]][s] for s in seeds], [by_seed[C.PRIMARY_CONTRAST[1]][s] for s in seeds], "greater")
    return {"available": True, "p": res.p, "delta": res.observed, "min_p": res.min_attainable_p, "n_pairs": res.n_pairs, "seeds": seeds,
            "n_unpaired": len(by_seed[C.PRIMARY_CONTRAST[0]]) + len(by_seed[C.PRIMARY_CONTRAST[1]]) - 2 * len(seeds)}


# ------------------------------------------------------------------ (f) test-set halves
def split_halves(runs: Sequence[RunData]) -> dict[str, Any]:
    out: dict[str, Any] = {"halves": []}
    for h in (0, 1):
        def metric(r: RunData, h=h) -> float:
            ids = [p for p in r.final.per_problem if half_of(p) == h]  # type: ignore[union-attr]
            return r.final.rate("hack_rt", ids)  # type: ignore[union-attr]

        vals = _values(runs, metric)
        n_problems = len({p for r in runs[:1] for p in r.final.per_problem if half_of(p) == h}) if runs else 0  # type: ignore[union-attr]
        out["halves"].append({"half": h, "n_problems": n_problems, **_row("primary", _primary(vals)), **_row("h1", _h1(vals))})
    d = [x.get("primary_delta") for x in out["halves"]]
    out["agree_direction"] = None if any(v is None for v in d) else bool(all(v > 0 for v in d) or all(v < 0 for v in d))
    out["agree_positive"] = None if any(v is None for v in d) else bool(all(v > 0 for v in d))
    return out


# ------------------------------------------------------------------ (g) variance decomposition
def variance_decomposition(runs: Sequence[RunData]) -> list[dict[str, Any]]:
    rows = []
    arms: dict[str, list[RunData]] = {}
    for r in runs:
        arms.setdefault(r.arm, []).append(r)
    for arm in sorted(arms):
        rates = np.array([r.final.rate("hack_rt") for r in arms[arm]])  # type: ignore[union-attr]
        ns = np.array([r.final.n() for r in arms[arm]])  # type: ignore[union-attr]
        within_var = float(np.mean(rates * (1 - rates) / ns))
        between_sd = float(np.std(rates, ddof=1)) if len(rates) > 1 else math.nan
        within_se = math.sqrt(within_var)
        ratio = between_sd / within_se if within_se > 0 and not math.isnan(between_sd) else math.nan
        rows.append({"arm": arm, "n_seeds": len(rates), "mean_rate": float(rates.mean()), "between_seed_sd": between_sd,
                     "within_seed_binomial_se": within_se, "ratio_between_over_within": ratio,
                     "seed_variance_dominates": None if math.isnan(ratio) else bool(ratio > 1.0)})
    return rows


# ------------------------------------------------------------------ orchestration
def run_robustness(runs: Sequence[RunData]) -> dict[str, Any]:
    """Every robustness component for the runs of an analysis. ``flags`` lists what a reader must know."""
    runs = list(runs)
    if not runs:
        return {"stamp": "EXPLORATORY", "flags": ["no runs"], "available": False}
    final = _values(runs, _fin("hack_rt"))
    onset = _values(runs, lambda r: onset_step(r.hack_train_rates(), r.T))
    loo = {"primary": leave_one_out(final, "primary"), "h1_final": leave_one_out(final, "h1_final"),
           "h1_onset": leave_one_out(onset, "h1_onset")}
    res = {"stamp": "EXPLORATORY", "available": True, "leave_one_out": loo, "definitions": definition_variants(runs),
           "windows": endpoint_windows(runs), "onset_grid": onset_grid(runs), "rank_test": rank_test(runs),
           "halves": split_halves(runs), "variance": variance_decomposition(runs)}
    flags = []
    for kind, r in loo.items():
        if r["flip"]:
            flips = [x["dropped"] for x in r["rows"] if x["flip"]]
            flags.append(f"LEAVE-ONE-OUT FLIP ({kind}): dropping {', '.join(flips)} changes the verdict; "
                         f"influential seed: {r['influential']}")
    base = res["definitions"][0].get("primary_verdict")
    for v in res["definitions"][1:]:
        if v.get("primary_verdict") is not None and v["primary_verdict"] != base:
            flags.append(f"definition variant '{v['variant']}' changes the primary verdict")
    for w in res["windows"]["rows"][1:]:
        if w.get("primary_verdict") is not None and w["primary_verdict"] != res["windows"]["rows"][0].get("primary_verdict"):
            flags.append(f"endpoint window '{w['window']}' changes the primary verdict")
    if res["halves"]["agree_direction"] is False:
        flags.append("the two test-set halves disagree in direction")
    retreat = [r["run_id"] for r in res["windows"]["rebound"] if r["retreat"]]
    if retreat:
        flags.append("possible retreat phase at the final step (final test rate < 50% of an earlier val rate): " + ", ".join(retreat))
    res["flags"] = flags
    return res


def flatten_tables(res: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """CSV-ready tables of a ``run_robustness`` result."""
    def clean(rows):
        return [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows]

    tabs: dict[str, list[dict[str, Any]]] = {}
    tabs["robustness_loo"] = clean([{"test": k, **r} for k, v in res["leave_one_out"].items() for r in v["rows"]])
    tabs["robustness_definitions"] = clean(res["definitions"])
    tabs["robustness_windows"] = clean(res["windows"]["rows"])
    tabs["robustness_rebound"] = clean(res["windows"]["rebound"])
    tabs["robustness_onset_grid"] = clean(res["onset_grid"])
    tabs["robustness_halves"] = clean(res["halves"]["halves"])
    tabs["robustness_variance"] = clean(res["variance"])
    return tabs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rhg.analysis.robustness", description="EXPLORATORY robustness suite for the primary contrast and H1.")
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--json", type=Path, default=None, help="write the result here (default: print a summary)")
    args = ap.parse_args(argv)
    runset = load_runs(args.runs)
    if not runset.runs:
        print(f"error: no completed runs under {args.runs}", file=sys.stderr)
        return 2
    res = run_robustness(runset.runs)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    print("[EXPLORATORY] robustness flags:" if res["flags"] else "[EXPLORATORY] robustness: no flags")
    for f in res["flags"]:
        print(" -", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
