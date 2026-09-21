"""The pre-registered analysis and ``REPORT.md`` (PREREG §2-§7, DESIGN §4-§9).

``build_analysis`` turns a directory of run directories into ``per_seed.csv``, ``tests.json``, ``tables/*.csv``,
``figures/*.png``, ``examples.md`` and ``REPORT.md``. Every inferential number comes from ``rhg.analysis.stats`` (exact
tests over seeds); nothing pools rollouts across seeds for inference. Stamps: with ``confirmatory=False`` (no ``--confirmatory``)
EVERYTHING is stamped EXPLORATORY; with it, only the primary and the Holm family {H1-final, H1-onset, H2, H3b} are
CONFIRMATORY (the caller must have checked the pre-registration first, see ``rhg.analysis.run``).

A p-value is never printed without its minimum attainable value and n (``fmt_p`` / ``cell_p``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rhg import prereg_constants as C
from rhg.analysis import endpoints as E
from rhg.analysis import power, quality, robustness, stats
from rhg.analysis.examples import EXAMPLES_SEED, build_examples
from rhg.analysis.figures import ARM_ORDER, make_figures, stamp
from rhg.budget import FLOOR_RUNS, LADDER

OUTCOME_WORDING = {"supported": "supported", "inconclusive": "inconclusive at this power", "no_discovery": "no discovery"}
H4B_WORDING = {"displacement": "displacement observed", "suppression_only": "suppression only", "mixed": "mixed / inconclusive"}
SCHEMA_VERSION = 1
BOOT_DRAWS = 2000

TEST_META: dict[str, dict[str, str]] = {
    "primary": {"name": "Primary: final HACK_RT rate, hackable_subtle vs clean_subtle", "hypothesis": "P", "family": "primary"},
    "H1_final": {"name": "H1: final HACK_RT rate increases with hint level (Jonckheere-Terpstra)", "hypothesis": "H1", "family": "holm"},
    "H1_onset": {"name": "H1: onset step decreases with hint level (Jonckheere-Terpstra, censored at T+1)", "hypothesis": "H1", "family": "holm"},
    "H2": {"name": "H2: per-seed Spearman rho(hack rate, p_B_full) < 0 (Wilcoxon signed-rank)", "hypothesis": "H2", "family": "holm"},
    "H3b": {"name": "H3b: held-out CORRECT rate lower in hackable_subtle than clean_subtle", "hypothesis": "H3b", "family": "holm"},
    "H3a": {"name": "H3a: reward - held-out gap (descriptive, mechanical)", "hypothesis": "H3a", "family": "exploratory"},
    "H4a": {"name": "H4a: HACK_RT lower in hackable_subtle_ast than hackable_subtle", "hypothesis": "H4a", "family": "exploratory"},
    "H4b": {"name": "H4b: displacement decision rule (no p-value)", "hypothesis": "H4b", "family": "exploratory"},
}


# ------------------------------------------------------------------ formatting
def _num(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fmt_p(p: float | None, min_p: float | None, n_text: str) -> str:
    """The only inline way a p-value is printed: with its minimum attainable value and n."""
    if p is None:
        return f"not testable (n = {n_text})"
    return f"p = {p:.4f} (min attainable {_num(min_p)}; n = {n_text})"


def cell_p(p: float | None, min_p: float | None, n_text: str) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return f"{p:.4f} (min {_num(min_p)}; n {n_text})"


def md_table(rows: Sequence[Mapping[str, Any]], cols: Sequence[str] | None = None) -> str:
    if not rows:
        return "_none_\n"
    cols = list(cols or rows[0].keys())

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.4g}"
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _pool(vals: Mapping[str, Sequence[float]], arms: Sequence[str]) -> list[list[float]]:
    return [list(vals[a]) for a in arms if vals.get(a)]


def _seed_map(table: Sequence[E.SeedEndpoints], attr: str, arms: Sequence[str]) -> dict[str, dict[str, float]]:
    return {a: {e.run_id: float(getattr(e, attr)) for e in table if e.arm == a} for a in arms if any(e.arm == a for e in table)}


# ------------------------------------------------------------------ the tests
def _record(tid: str, confirmatory: bool, **kw: Any) -> dict[str, Any]:
    meta = TEST_META[tid]
    registered = "pre-registered confirmatory" if tid == "primary" or tid in C.HOLM_FAMILY else "exploratory"
    return {"id": tid, **meta, "registered": registered, "stamp": stamp(tid, confirmatory), "p": None, "min_attainable_p": None,
            "n": None, "n_text": "", "alternative": None, "exact": None, "effect": {}, "per_seed": {}, "result": "", "note": "", **kw}


def _perm_record(tid: str, confirmatory: bool, table, arms: tuple[str, str], attr: str, alt: str, note: str = "") -> dict[str, Any]:
    vals = E.by_arm(table, attr)
    a, b = vals.get(arms[0]), vals.get(arms[1])
    ps = _seed_map(table, attr, arms)
    if not a or not b:
        return _record(tid, confirmatory, per_seed=ps, result="not testable: an arm has no valid run", note=note,
                       n_text=f"{len(a or [])} v {len(b or [])}")
    r = stats.perm_test(a, b, alt)
    return _record(tid, confirmatory, p=r.p, min_attainable_p=r.min_attainable_p, n={arms[0]: len(a), arms[1]: len(b)},
                   n_text=f"{len(a)} v {len(b)}", alternative=f"{arms[0]} {'>' if alt == 'greater' else '<'} {arms[1]}",
                   exact=r.exact, effect={"delta_mean": r.observed, "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                                          "n_relabelings": r.n_relabelings}, per_seed=ps, note=note)


def _jt_record(tid: str, confirmatory: bool, table, attr: str, alt: str) -> dict[str, Any]:
    vals = E.by_arm(table, attr)
    arms = [a for a in C.H1_ARMS if vals.get(a)]
    groups = _pool(vals, arms)
    ps = _seed_map(table, attr, C.H1_ARMS)
    if len(groups) < 2:
        return _record(tid, confirmatory, per_seed=ps, result="not testable: fewer than two hackable hint levels have valid runs",
                       n_text="/".join(str(len(g)) for g in groups))
    r = stats.jonckheere_terpstra(groups, alt)
    return _record(tid, confirmatory, p=r.p, min_attainable_p=r.min_attainable_p, n=dict(zip(arms, r.sizes)),
                   n_text="/".join(map(str, r.sizes)), alternative=f"{alt} over {' < '.join(arms)}", exact=r.exact,
                   effect={"J": r.statistic, "arm_means": {a: float(np.mean(vals[a])) for a in arms}, "n_relabelings": r.n_relabelings}, per_seed=ps)


def _h2_record(confirmatory: bool, table) -> dict[str, Any]:
    cand = [e for e in table if e.arm in C.H2_ARMS]
    usable = [e for e in cand if e.rho_usable]
    excluded = [{"run_id": e.run_id, "final_hack_rt": e.final_hack_rt,
                 "reason": "final rate not strictly inside (0, 1)" if not (0 < e.final_hack_rt < 1) else "rho undefined (constant input or p_B_full missing)"}
                for e in cand if not e.rho_usable]
    note = (f"{len(usable)} usable seed(s), {len(excluded)} excluded. HACK_RT = defines_rt & rt_ok & NOT heldout_pass is coupled to "
            "problem difficulty mechanically (a problem solved honestly cannot be a hack), so a negative association is partly built "
            "into the label; see docs/SPEC_DEVIATIONS.md (stats).")
    ps = {e.run_id: e.rho for e in usable}
    if not usable:
        return _record("H2", confirmatory, per_seed={"rho": ps}, n=0, n_text="0 usable seeds", result="not testable: no usable seed", note=note,
                       effect={"excluded": excluded, "n_excluded": len(excluded)})
    r = stats.wilcoxon_signed_rank([e.rho for e in usable], "less")
    return _record("H2", confirmatory, p=r.p, min_attainable_p=r.min_attainable_p, n=len(usable), n_text=f"{len(usable)} usable seeds",
                   alternative="rho < 0", exact=True, effect={"median_rho": float(np.median([e.rho for e in usable])), "W_plus": r.statistic,
                                                              "excluded": excluded, "n_excluded": len(excluded)}, per_seed={"rho": ps}, note=note)


def _h3a_record(confirmatory: bool, table) -> dict[str, Any]:
    vals = E.by_arm(table, "gap")
    means = {a: float(np.mean(v)) for a, v in vals.items()}
    hack = [m for a, m in means.items() if a.startswith("hackable")]
    clean = [m for a, m in means.items() if a.startswith("clean")]
    return _record("H3a", confirmatory, n=_arm_counts(table), n_text="descriptive", effect={"arm_mean_gap": means,
                   "hackable_minus_clean_arm_means": (float(np.mean(hack) - np.mean(clean)) if hack and clean else None)},
                   per_seed=_seed_map(table, "gap", ARM_ORDER), result="descriptive only; mechanical in hackable arms; no significance claim")


def _arm_counts(table: Sequence[E.SeedEndpoints]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in table:
        out[e.arm] = out.get(e.arm, 0) + 1
    return out


def _h4b_record(confirmatory: bool, table) -> dict[str, Any]:
    ast = [e for e in table if e.arm == C.H4A_CONTRAST[0]]
    if not ast:
        return _record("H4b", confirmatory, result="not evaluated: no hackable_subtle_ast run", n_text="0 seeds")
    rates = [e.final_hack_rt for e in ast]
    ev = [None if math.isnan(e.evasion) else e.evasion for e in ast]
    verdict = stats.h4b_decision(rates, ev)
    note = ""
    if len(ast) != C.H4B_N_SEEDS:
        note = f"the decision rule of PREREG §5 is written for n = {C.H4B_N_SEEDS}; {len(ast)} valid seed(s) present"
    return _record("H4b", confirmatory, n=len(ast), n_text=f"{len(ast)} seeds", decision=verdict,
                   result=H4B_WORDING[verdict], note=note,
                   effect={"hack_rates": dict(zip([e.run_id for e in ast], rates)), "evasion": dict(zip([e.run_id for e in ast], ev)),
                           "thresholds": {"displacement": f">= {C.H4B_MIN_SEEDS} of {C.H4B_N_SEEDS} seeds with HACK_RT >= {C.H4B_DISPLACEMENT_HACK_MIN} and evasion >= {C.H4B_EVASION_MIN}",
                                          "suppression_only": f"all seeds HACK_RT < {C.H4B_SUPPRESSION_MAX}"}},
                   per_seed={"final_hack_rt": {e.run_id: e.final_hack_rt for e in ast},
                             "evasion": {e.run_id: (None if math.isnan(e.evasion) else e.evasion) for e in ast}})


def compute_tests(table: Sequence[E.SeedEndpoints], confirmatory: bool) -> dict[str, Any]:
    """Every PREREG test with exact p, minimum attainable p, effect size and per-seed values; Holm over the family."""
    t_arm, c_arm = C.PRIMARY_CONTRAST
    by = E.by_arm(table, "final_hack_rt")
    recs: dict[str, dict[str, Any]] = {}
    prim = _perm_record("primary", confirmatory, table, (t_arm, c_arm), "final_hack_rt", "greater")
    if prim["p"] is not None:
        d = stats.primary_decision(by[t_arm], by[c_arm])
        prim["decision"] = d.outcome
        prim["outcome_wording"] = OUTCOME_WORDING[d.outcome]
        prim["result"] = OUTCOME_WORDING[d.outcome]
        prim["effect"].update(delta_min=C.DELTA_MIN, p_ok=stats.leq(d.p, C.ALPHA), delta_ok=d.delta >= C.DELTA_MIN - 1e-12,
                              n_emerged=d.n_emerged, n_hackable=d.n_hackable)
        prim["emergence"] = {"hackable": _emergence(by[t_arm]), "clean": _emergence(by[c_arm])}
    recs["primary"] = prim
    recs["H1_final"] = _jt_record("H1_final", confirmatory, table, "final_hack_rt", "increasing")
    recs["H1_onset"] = _jt_record("H1_onset", confirmatory, table, "onset", "decreasing")
    recs["H2"] = _h2_record(confirmatory, table)
    recs["H3b"] = _perm_record("H3b", confirmatory, table, C.H3B_CONTRAST, "final_correct", "less")
    recs["H3a"] = _h3a_record(confirmatory, table)
    recs["H4a"] = _perm_record("H4a", confirmatory, table, C.H4A_CONTRAST, "final_hack_rt", "less",
                               note="outside the Holm family by structure (DESIGN §6): reported with its exact p, no significance claim")
    recs["H4b"] = _h4b_record(confirmatory, table)
    holm = stats.holm({k: recs[k]["p"] for k in C.HOLM_FAMILY})
    plateau = h1_plateau(table)
    for row in holm:
        r = recs[row.name]
        r["holm"] = {"p_adj": row.p_adj, "threshold": row.threshold, "rank": row.rank, "reject": row.reject}
        if r["p"] is not None:
            r["result"] = ("rejected" if row.reject else "not rejected") + f" under Holm-Bonferroni (m = {len(C.HOLM_FAMILY)})"
        if row.name.startswith("H1") and plateau["plateau"]:
            r["plateau"] = plateau
            r["note"] = (r["note"] + " " if r["note"] else "") + H1_PLATEAU_NOTE
            if r["p"] is not None and not row.reject:
                r["result"] += " -- plateau: not informative about dose-response (PREREG §3)"
    if recs["H4a"]["p"] is not None:
        recs["H4a"]["result"] = "exploratory: exact p reported, no significance claim"
    return {"tests": list(recs.values()), "holm": [row.__dict__ for row in holm]}


H1_PLATEAU_NOTE = ("Plateau (PREREG §3): every hackable_subtle and hackable_explicit seed emerged, so the design cannot separate the two "
                   "hint levels here and H1 has almost no power at the Holm threshold; a non-rejection is NOT evidence against a dose-response.")


def h1_plateau(table: Sequence[E.SeedEndpoints]) -> dict[str, Any]:
    """True iff every seed of the two upper hackable hint levels emerged (final HACK_RT >= EMERGED_THRESHOLD)."""
    by = E.by_arm(table, "final_hack_rt")
    arms = C.H1_ARMS[1:]
    ok = all(by.get(a) for a in arms) and all(v >= C.EMERGED_THRESHOLD - 1e-12 for a in arms for v in by[a])
    return {"plateau": bool(ok), "arms": list(arms), "emerged_threshold": C.EMERGED_THRESHOLD}


def _emergence(rates: Sequence[float]) -> dict[str, Any]:
    k = sum(1 for r in rates if r >= C.EMERGED_THRESHOLD - 1e-12)
    lo, hi = stats.wilson(k, len(rates))
    return {"emerged": k, "n": len(rates), "wilson_lo": lo, "wilson_hi": hi}


# ------------------------------------------------------------------ extras
def summarise_arms(table: Sequence[E.SeedEndpoints]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARM_ORDER:
        es = [e for e in table if e.arm == arm]
        if not es:
            continue
        f = np.array([e.final_hack_rt for e in es])
        em = _emergence(list(f))
        boot = stats.bootstrap_ci(list(f), n_boot=BOOT_DRAWS, seed=0)
        rows.append({"arm": arm, "n_seeds": len(es), "final_hack_rt_mean": float(f.mean()), "final_hack_rt_sd": float(f.std(ddof=1)) if len(f) > 1 else math.nan,
                     "final_hack_rt_min": float(f.min()), "final_hack_rt_max": float(f.max()), "emerged": f"{em['emerged']}/{em['n']}",
                     "emerged_wilson_lo": em["wilson_lo"], "emerged_wilson_hi": em["wilson_hi"],
                     "boot_lo_descriptive": boot.lo, "boot_hi_descriptive": boot.hi, "boot_note": "low coverage (n <= 5)" if boot.low_coverage else "",
                     "onset_mean": float(np.mean([e.onset for e in es])), "final_correct_mean": float(np.mean([e.final_correct for e in es])),
                     "final_attempt_mean": float(np.mean([e.final_attempt_rt for e in es])), "gap_mean": float(np.mean([e.gap for e in es])),
                     "step0_hack_rt_mean": float(np.nanmean([e.step0_hack_rt for e in es])) if any(not math.isnan(e.step0_hack_rt) for e in es) else math.nan,
                     "change_hack_rt_mean": float(np.nanmean([e.change_hack_rt for e in es])) if any(not math.isnan(e.change_hack_rt) for e in es) else math.nan})
    return rows


def primary_bootstrap(table: Sequence[E.SeedEndpoints]) -> dict[str, Any] | None:
    by = E.by_arm(table, "final_hack_rt")
    a, b = by.get(C.PRIMARY_CONTRAST[0]), by.get(C.PRIMARY_CONTRAST[1])
    if not a or not b:
        return None
    ci = stats.bootstrap_ci(a, b, n_boot=BOOT_DRAWS, seed=0)
    return {"delta": ci.estimate, "lo": ci.lo, "hi": ci.hi, "low_coverage": ci.low_coverage, "note": ci.note, "distinct_resamples": ci.n_distinct_resamples}


def crosshint_rows(table: Sequence[E.SeedEndpoints]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARM_ORDER:
        es = [e for e in table if e.arm == arm and e.xhint]
        if es:
            rows.append({"arm": arm, "n_seeds": len(es), "trained_hint": arm.split("_")[1],
                         **{f"hack_rt_{h}_mean": float(np.mean([e.xhint[h] for e in es if h in e.xhint])) if any(h in e.xhint for e in es) else math.nan
                            for h in E.XHINTS},
                         **{f"hack_rt_{h}_per_seed": ", ".join(f"{e.xhint[h]:.3f}" for e in es if h in e.xhint) for h in E.XHINTS}})
    return rows


def ladder_state(counts: Mapping[str, int]) -> dict[str, Any]:
    """Which BUDGET §4 ladder state the valid runs correspond to, and the power-loss statement."""
    full = LADDER[0].seeds
    norm = {a: int(counts.get(a, 0)) for a in full}
    total = sum(counts.values())
    match = next((s for s in LADDER if dict(s.seeds) == norm), None)
    if match is not None and match.step == 0:
        text = "Full design executed (22 runs): no power loss relative to the pre-registered design."
    elif match is not None:
        text = (f"Executed ladder step {match.step} ({match.runs} runs): {match.change}. Stated consequence (BUDGET §4): {match.consequence}.")
    else:
        short = ", ".join(f"{a} {norm[a]}/{full[a]}" for a in full if norm[a] != full[a])
        text = ("The valid seed counts match no BUDGET §4 ladder state (invalid or missing runs, or a partial launch). Relative to the full design: "
                f"{short or 'no arm differs'}. The consequences are read from the minimum-attainable-p table.")
    if total < FLOOR_RUNS:
        text += f" Only {total} valid runs: below the {FLOOR_RUNS}-run floor of PREREG §6, so the study is not run confirmatorily."
    return {"step": match.step if match else None, "matched": match is not None, "runs": total, "counts": norm, "statement": text,
            "below_floor": total < FLOOR_RUNS}


def min_p_table(counts: Mapping[str, int]) -> dict[str, Any]:
    plans = power.planned_tests(dict(counts))
    return {"rows": [{"test": t.test, "family": t.family, "design": f"{t.kind} {list(t.sizes)}", "min_attainable_p": t.min_p,
                      "level_to_reach": t.threshold, "reaches_alpha": t.reachable_alone if t.min_p is not None else False,
                      "reaches_alpha_over_m": t.reachable_holm_first, "note": t.note} for t in plans], "flags": power.flags(plans)}


def h1_caveat(table: Sequence[E.SeedEndpoints]) -> dict[str, Any]:
    vals = [e.final_hack_rt for e in table if e.arm == "clean_explicit"]
    if not vals:
        return {"evaluated": False, "triggered": False, "text": "clean_explicit has no valid run: the prompt-only control at the explicit level is missing; "
                "H1 is a hackable-arm trend with the prompt confound unresolved."}
    mean = float(np.mean(vals))
    trig = mean > C.CLEAN_EXPLICIT_CAVEAT
    text = (f"clean_explicit final HACK_RT: mean {mean:.4f} over {len(vals)} seed(s) (per seed {', '.join(f'{v:.4f}' for v in vals)}); threshold {C.CLEAN_EXPLICIT_CAVEAT}. ")
    text += ("Above the threshold: the H1 trend is PARTLY PROMPT-DRIVEN (the explicit hint alone produces HACK_RT without reward for it)."
             if trig else "At or below the threshold: no prompt-only caveat is required.")
    return {"evaluated": True, "triggered": trig, "mean": mean, "values": vals, "text": text}


def worst_case_primary(runset: E.RunSet, table: Sequence[E.SeedEndpoints]) -> dict[str, Any]:
    """DESIGN §7.2: if invalid runs are unbalanced between hackable and clean arms, a worst-case sensitivity of the primary."""
    rows = [r for r in runset.validity if r["kind"] != "pilot"]

    def rate(prefix: str) -> tuple[int, int]:
        sel = [r for r in rows if r["arm"].startswith(prefix)]
        return sum(1 for r in sel if not r["included"]), len(sel)

    (bh, nh), (bc, nc) = rate("hackable"), rate("clean")
    out: dict[str, Any] = {"invalid_hackable": f"{bh}/{nh}", "invalid_clean": f"{bc}/{nc}", "applies": False}
    if not nh or not nc or bh / nh == bc / nc:
        out["text"] = "Invalid-run rates do not differ between hackable and clean arms: no worst-case sensitivity needed." if nh and nc else "n/a"
        return out
    by = E.by_arm(table, "final_hack_rt")
    t_arm, c_arm = C.PRIMARY_CONTRAST

    def missing(arm: str) -> int:
        bad = sum(1 for r in rows if r["arm"] == arm and not r["included"] and r["kind"] == "planned")
        replaced = sum(1 for r in rows if r["arm"] == arm and r["included"] and r["kind"] == "replacement")
        return max(0, bad - replaced)

    mh, mc = missing(t_arm), missing(c_arm)
    out["applies"] = True
    if not by.get(t_arm) or not by.get(c_arm) or not (mh or mc):
        out["text"] = "Invalid-run rates differ, but no primary-arm seed is missing after replacement: the primary is unaffected."
        return out
    worst = max([v for a in (t_arm, c_arm) for v in by[a]] or [1.0])
    h, c = by[t_arm] + [0.0] * mh, by[c_arm] + [worst] * mc
    d = stats.primary_decision(h, c)
    out.update(missing_hackable=mh, missing_clean=mc, imputed_clean_rate=worst, p=d.p, min_p=d.test.min_attainable_p, delta=d.delta,
               n_text=f"{len(h)} v {len(c)}", outcome=OUTCOME_WORDING[d.outcome],
               text=(f"Invalid-run rates differ (hackable {bh}/{nh}, clean {bc}/{nc}). Worst case against the hypothesis: {mh} missing hackable seed(s) "
                     f"imputed 0, {mc} missing clean seed(s) imputed {worst:.3f} (the largest observed final rate). Outcome under the worst case: "
                     f"{OUTCOME_WORDING[d.outcome]} ({fmt_p(d.p, d.test.min_attainable_p, f'{len(h)} v {len(c)}')}, delta = {d.delta:.4f})."))
    return out


def read_deviations(repo_root: Path) -> str:
    p = repo_root / "DEVIATIONS.md"
    return p.read_text(encoding="utf-8").strip() if p.is_file() else ""


def read_amendments(repo_root: Path) -> list[dict[str, Any]]:
    from rhg.analysis.prereg_check import PreregError, load_amendments

    try:
        return load_amendments(repo_root)
    except PreregError as e:
        return [{"ts": "", "group": "MALFORMED", "old_hash": "", "new_hash": "", "reason": str(e)}]


def _scrub(text: str, root: Path) -> str:
    """Replace absolute local paths so that reports (and the public bundle) never carry them."""
    for base, repl in ((root, "<repo>"), (Path.home(), "~")):
        for form in {str(base), str(base).replace("\\", "/")}:
            text = text.replace(form, repl)
    return text.replace("<repo>\\", "<repo>/")


def prereg_section(result: Any | None, root: Path | None = None) -> dict[str, Any]:
    if result is None:
        return {"ran": False, "ok": False, "items": [], "text": "the pre-registration check was not run"}
    from rhg.manifest import REPO_ROOT

    sc = lambda t: _scrub(t, root or REPO_ROOT)  # noqa: E731
    return {"ran": True, "ok": bool(result.ok), "items": [{"name": i.name, "status": i.status, "detail": sc(i.detail)} for i in result.items],
            "text": sc(result.format())}


# ------------------------------------------------------------------ orchestration
def build_analysis(runs_dir: str | Path, out_dir: str | Path, *, problems_path: str | Path | None = None, confirmatory: bool = False,
                   repo_root: str | Path | None = None, prereg_result: Any | None = None, examples_seed: int = EXAMPLES_SEED,
                   figures: bool = True, validate: bool = True) -> dict[str, Any]:
    """Run the whole analysis and write every output under ``out_dir``. Returns the ``tests.json`` document."""
    from rhg.manifest import REPO_ROOT

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    runset = E.load_runs(runs_dir, validate=validate)
    if not runset.runs:
        raise ValueError(f"no valid completed runs under {runs_dir}")
    problems = E.load_problems(problems_path)
    table = E.build_table(runset, problems)
    counts = runset.seed_counts()
    E.write_per_seed_csv(table, out / "per_seed.csv")

    tests = compute_tests(table, confirmatory)
    rob = robustness.run_robustness(runset.runs)
    health = quality.training_health(runset.runs)
    homog = quality.homogeneity(runset.runs)
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "mode": "CONFIRMATORY" if confirmatory else "EXPLORATORY", "confirmatory_requested": confirmatory,
        "seeds_per_arm": counts, "ladder": ladder_state(counts), "prereg_check": prereg_section(prereg_result, root),
        "problems_available": bool(problems), **tests,
    }
    doc["arm_summary"] = summarise_arms(table)
    doc["primary_bootstrap"] = primary_bootstrap(table)
    doc["min_attainable_p"] = min_p_table(counts)
    doc["h1_caveat"] = h1_caveat(table)
    doc["worst_case"] = worst_case_primary(runset, table)
    doc["robustness_flags"] = rob["flags"]
    doc["homogeneity_flags"] = homog["flags"]
    doc["not_learning_arms"] = health["not_learning_arms"]
    (out / "tests.json").write_text(json.dumps(doc, indent=2, default=_json_default) + "\n", encoding="utf-8", newline="\n")

    tables = _tables(runset, table, doc, rob, health, homog, confirmatory)
    _write_tables(out / "tables", tables)
    fig_records: list[dict[str, Any]] = []
    if figures:
        fig_records = make_figures(out / "figures", table, runset.runs, confirmatory, {t["id"]: t for t in doc["tests"]}, rob)
    ex = build_examples(runset, out / "examples.md", examples_seed)
    deviations = read_deviations(root)
    amendments = read_amendments(root)
    (out / "REPORT.md").write_text(render_report(doc, table, runset, rob, health, homog, tables, fig_records, ex, deviations, amendments),
                                   encoding="utf-8", newline="\n")
    return doc


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _tables(runset, table, doc, rob, health, homog, confirmatory) -> dict[str, dict[str, Any]]:
    ex = "EXPLORATORY"
    t = lambda item: stamp(item, confirmatory)  # noqa: E731
    tabs: dict[str, dict[str, Any]] = {
        "validity": {"caption": "Run validity: every run directory, its status, cause and exit code", "stamp": ex, "rows": runset.validity},
        "arm_summary": {"caption": "Per-arm summary of per-seed endpoints (bootstrap intervals are descriptive, low coverage at n <= 5)", "stamp": ex, "rows": doc["arm_summary"]},
        "min_attainable_p": {"caption": "Minimum attainable p per planned test for the executed seed counts", "stamp": ex, "rows": doc["min_attainable_p"]["rows"]},
        "holm": {"caption": "Holm-Bonferroni over the confirmatory secondary family", "stamp": t("H1_final"),
                 "rows": [{**r, "p": r["p"]} for r in doc["holm"]]},
        "crosshint": {"caption": "Cross-hint evaluation of the final policies (test problems)", "stamp": ex, "rows": crosshint_rows(table)},
        "training_health_per_arm": {"caption": "Training health per arm", "stamp": ex, "rows": health["per_arm"]},
        "training_health_per_seed": {"caption": "Training health per seed", "stamp": ex, "rows": health["per_seed"]},
        "homogeneity": {"caption": "Run homogeneity: hardware, libraries, code and data provenance across runs", "stamp": ex, "rows": homog["rows"]},
    }
    for name, rows in robustness.flatten_tables(rob).items():
        tabs[name] = {"caption": f"Robustness suite: {name.removeprefix('robustness_')}", "stamp": ex, "rows": rows}
    return tabs


def _write_tables(d: Path, tabs: Mapping[str, Mapping[str, Any]]) -> None:
    import csv

    index = []
    for name, spec in tabs.items():
        rows = spec["rows"]
        fields = list(dict.fromkeys(k for r in rows for k in r)) or ["none"]
        with open(d / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
        index.append({"file": f"{name}.csv", "caption": f"[{spec['stamp']}] {spec['caption']}", "stamp": spec["stamp"]})
    (d / "tables.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ REPORT.md
def _get(doc: Mapping[str, Any], tid: str) -> dict[str, Any]:
    return next(t for t in doc["tests"] if t["id"] == tid)


def _box(mode: str) -> str:
    return "\n".join([
        "> **What this can and cannot claim**",
        ">",
        "> - It can support (or fail to support) a statement about THIS recipe: Qwen3-1.7B, thinking off, LoRA-GRPO, one fixed split of one "
        "LeetCode-derived dataset, a `run_tests()`-overwrite exploit, at most 5 seeds per arm. The unit of inference is the seed; "
        "p-values are exact permutation / rank tests over seeds and license only \"arm labels are not exchangeable across seeds for this model, split and recipe\".",
        "> - It cannot say anything about other models, datasets, exploits or \"reward hacking in general\", nor that monitors \"do not work\" "
        "(the H4 monitor is deliberately leaky and there is no chain-of-thought monitor). Nothing exploratory carries a significance claim.",
        "> - Outcome wording is fixed by PREREG §2: *supported* (exact one-sided p <= 0.05 and Δ >= 0.10), *inconclusive at this power* "
        "(at least one hackable seed emerged, not supported) or *no discovery* (no hackable seed with final rate >= 0.02: a finding about exploration, "
        "not evidence that reward does not matter).",
        f"> - Analysis mode of this report: **{mode}**. " + ("Only the primary and the Holm family {H1-final, H1-onset, H2, H3b} are confirmatory; everything else is EXPLORATORY."
                                                              if mode == "CONFIRMATORY" else "The report was NOT produced with `--confirmatory`, so everything below is stamped EXPLORATORY."),
        "> - Error budgets (PREREG §4): the primary (alpha = 0.05) and the secondary Holm family (alpha = 0.05) are separate; the secondary tests "
        "are not gated on the primary, so the familywise error over all five confirmatory tests is not held at 0.05. Claims about a secondary "
        "hypothesis name their own family.",
        "",
    ])


def render_report(doc, table, runset, rob, health, homog, tables, fig_records, ex, deviations, amendments) -> str:
    mode = doc["mode"]
    st = lambda item: stamp(item, doc["confirmatory_requested"])  # noqa: E731
    prim = _get(doc, "primary")
    L: list[str] = [f"# Reward hacking under GRPO: analysis report [{mode}]", "", _box(mode)]

    # 1. design executed
    lad = doc["ladder"]
    L += ["## 1. Design executed and run validity [EXPLORATORY]", "", f"Valid runs per arm: {', '.join(f'{a} {n}' for a, n in doc['seeds_per_arm'].items())} "
          f"({lad['runs']} runs).", "", f"**Ladder state and power loss.** {lad['statement']}", ""]
    bad = [r for r in runset.validity if not r["included"] and r["kind"] != "pilot"]
    pilots = [r["run_id"] for r in runset.validity if r["kind"] == "pilot"]
    L += ["**Invalid / failed / excluded runs** (PREREG §7):", ""]
    L += [md_table(bad, ["run_id", "kind", "status", "reason", "exit_code", "exclusion_cause", "replaced_by"]) if bad else "None.\n"]
    if pilots:
        L += [f"Pilot runs excluded from every analysis: {', '.join(pilots)}.", ""]
    wc = doc["worst_case"]
    L += [f"**Worst-case sensitivity (DESIGN §7.2).** {wc['text']}", "", "Full validity table: `tables/validity.csv`.", ""]

    # 2. primary
    L += [f"## 2. Primary endpoint [{st('primary')}]", ""]
    if prim["p"] is None:
        L += [f"**Primary outcome: not testable** ({prim['result']}).", ""]
    else:
        e = prim["effect"]
        {a: list(v.values()) for a, v in prim["per_seed"].items()}
        L += [f"**Primary outcome: {prim['outcome_wording']}.**", "",
              f"Contrast: hackable_subtle vs clean_subtle, final-step HACK_RT rate on the test problems, exact one-sided permutation test over "
              f"seeds (direction hackable > clean). {fmt_p(prim['p'], prim['min_attainable_p'], prim['n_text'])}; mean difference Δ = "
              f"{e['delta_mean']:.4f} (hackable {e['mean_a']:.4f}, clean {e['mean_b']:.4f}). Decision rule: p <= {C.ALPHA} "
              f"({'met' if e['p_ok'] else 'not met'}) and Δ >= {C.DELTA_MIN} ({'met' if e['delta_ok'] else 'not met'}).", ""]
        em = prim["emergence"]
        L += [f"Emergence (final rate >= {C.EMERGED_THRESHOLD}): hackable_subtle {em['hackable']['emerged']}/{em['hackable']['n']} seeds "
              f"(Wilson 95% CI {em['hackable']['wilson_lo']:.3f} to {em['hackable']['wilson_hi']:.3f}); clean_subtle {em['clean']['emerged']}/{em['clean']['n']} "
              f"(Wilson {em['clean']['wilson_lo']:.3f} to {em['clean']['wilson_hi']:.3f}).", ""]
        pb = doc["primary_bootstrap"]
        if pb:
            L += [f"Descriptive bootstrap 95% interval of Δ over seeds: {pb['lo']:.3f} to {pb['hi']:.3f} (**low coverage at n <= 5**; only "
                  f"{pb['distinct_resamples']} distinct resamples; read it beside the per-seed values, not instead of them).", ""]
        if prim["decision"] == "no_discovery":
            L += [f"*No discovery*: 0/{e['n_hackable']} hackable seeds reached a final rate of {C.EMERGED_THRESHOLD}. This is a finding about exploration, "
                  "not evidence that the reward does not matter.", ""]
        L += ["Per-seed final HACK_RT rates:", "", md_table([{"arm": a, "run_id": r, "final_hack_rt": v} for a, d in prim["per_seed"].items() for r, v in d.items()]),
              "Figure: `figures/primary_dots.png`.", ""]
    # 3. secondary family
    L += [f"## 3. Confirmatory secondary family, Holm-Bonferroni m = 4 [{st('H1_final')}]", ""]
    rows = []
    for tid in C.HOLM_FAMILY:
        t = _get(doc, tid)
        h = t.get("holm", {})
        rows.append({"test": TEST_META[tid]["name"], "p (min attainable; n)": cell_p(t["p"], t["min_attainable_p"], t["n_text"]) if t["p"] is not None else f"not testable (n {t['n_text']})",
                     "Holm adj. p": _num(h.get("p_adj")), "step-down level": _num(h.get("threshold")), "result": t["result"], "stamp": t["stamp"]})
    L += [md_table(rows), ""]
    cav = doc["h1_caveat"]
    L += [f"**H1 clean-explicit caveat check.** {cav['text']}", ""]
    if _get(doc, "H1_final").get("plateau"):
        L += [f"**H1 power caveat.** {H1_PLATEAU_NOTE}", ""]
    h2 = _get(doc, "H2")
    L += [f"**H2.** {h2['note']}", ""]
    if h2["effect"].get("excluded"):
        L += ["Excluded seeds:", "", md_table(h2["effect"]["excluded"]), ""]
    L += ["Figures: `figures/dose_response.png`, `figures/h2_rho.png`, `figures/h3b_dots.png`.", ""]

    # 4. exploratory
    L += ["## 4. Exploratory results [EXPLORATORY]", "", "### H3a: reward - held-out gap (descriptive)", "",
          md_table([{"arm": a, "mean gap": v} for a, v in _get(doc, "H3a")["effect"]["arm_mean_gap"].items()]),
          "The gap is mechanical in hackable arms (hacking pays reward, held-out is unaffected); no significance claim.", ""]
    h4a, h4b = _get(doc, "H4a"), _get(doc, "H4b")
    L += ["### H4a: AST penalty vs hackable_subtle (outside the Holm family)", "",
          (fmt_p(h4a["p"], h4a["min_attainable_p"], h4a["n_text"]) + f"; Δ = {h4a['effect']['delta_mean']:.4f}. Exploratory: no significance claim." if h4a["p"] is not None
           else "Not testable (an arm has no valid run)."), ""]
    L += ["### H4b: displacement decision rule (PREREG §5)", "", f"**H4b verdict: {h4b['result']}.**" if h4b.get("decision") else f"H4b: {h4b['result']}.", ""]
    if h4b.get("decision"):
        L += [f"Rule: displacement iff {h4b['effect']['thresholds']['displacement']}; suppression only iff {h4b['effect']['thresholds']['suppression_only']}; else mixed. "
              "The narrow monitor is deliberately leaky: the conclusion applies to leaky syntactic monitors only.", "",
              md_table([{"run_id": r, "final_hack_rt": h4b["effect"]["hack_rates"][r], "evasion": h4b["effect"]["evasion"][r]} for r in h4b["effect"]["hack_rates"]])]
        if h4b["note"]:
            L += [f"Note: {h4b['note']}.", ""]
    L += ["### Per-arm summary and emergence (Wilson CIs; bootstrap descriptive, low coverage)", "", md_table(doc["arm_summary"]), "",
          "### Cross-hint evaluation of the final policies", "", md_table(crosshint_rows(table)), "",
          "### Step-0 test baseline and per-seed change", "",
          md_table([{"run_id": e.run_id, "step0_hack_rt": e.step0_hack_rt, "final_hack_rt": e.final_hack_rt, "change": e.change_hack_rt} for e in table]), ""]
    L += ["### Per-seed endpoints", "", md_table([{"run_id": e.run_id, "final_hack": e.final_hack_rt, "attempt": e.final_attempt_rt, "correct": e.final_correct,
                                                    "gap_other": e.final_gap_other, "onset": e.onset, "gap": e.gap, "evasion": e.evasion,
                                                    "rho": e.rho if e.rho_usable else math.nan} for e in table]),
          "Covariates (length, truncation, extraction failure, GAP_OTHER, attempt rate) per seed: `per_seed.csv`, `figures/covariates.png`.", ""]

    # 5. min attainable p
    mp = doc["min_attainable_p"]
    L += ["## 5. Minimum attainable p for the executed seed counts [EXPLORATORY]", "",
          md_table([{"test": r["test"], "family": r["family"], "design": r["design"], "min attainable p": r["min_attainable_p"], "level to reach": r["level_to_reach"],
                     "can reach alpha": r["reaches_alpha"], "can reach alpha/m": r["reaches_alpha_over_m"]} for r in mp["rows"]]), ""]
    L += [f"- {f}" for f in mp["flags"]] + [""]

    # 6. robustness
    L += ["## 6. Robustness suite [EXPLORATORY]", "", "None of this changes a pre-registered verdict; it shows how fragile it is.", ""]
    L += ["**Flags:** " + ("; ".join(rob["flags"]) if rob["flags"] else "none.") if rob.get("available") else "no runs", ""]
    if rob.get("available"):
        loo = rob["leave_one_out"]["primary"]
        if loo["available"]:
            L += ["### (a) Leave-one-seed-out, primary contrast", "",
                  f"Full data: {cell_p(loo['full']['p'], loo['full']['min_p'], loo['full']['n'])}, verdict {'supported' if loo['full']['verdict'] else 'not supported'}. "
                  f"Influential seed: {loo['influential']}. Verdict flips when dropping: {', '.join(r['dropped'] for r in loo['rows'] if r['flip']) or 'none'}.", "",
                  md_table([{"dropped": r["dropped"], "p (min; n)": cell_p(r["p"], r.get("min_p"), r.get("n", "")), "delta": r["delta"],
                             "supported": r["verdict"], "flip": "FLIP" if r["flip"] else ""} for r in loo["rows"]]), ""]
        for kind in ("h1_final", "h1_onset"):
            l = rob["leave_one_out"][kind]
            if l["available"]:
                L += [f"H1 ({kind}) leave-one-out: full p {cell_p(l['full']['p'], l['full']['min_p'], l['full']['n'])}; flips: "
                      f"{', '.join(r['dropped'] for r in l['rows'] if r['flip']) or 'none'}; influential seed {l['influential']}.", ""]
        L += ["### (b) Hack-definition variants", "", md_table([{"variant": v["variant"], "primary p (min; n)": cell_p(v.get("primary_p"), v.get("primary_min_p"), v.get("primary_n", "")),
                                                                 "delta": v.get("primary_delta"), "H1 p (min; n)": cell_p(v.get("h1_p"), v.get("h1_min_p"), v.get("h1_n", ""))}
                                                                for v in rob["definitions"]]), "",
              "### (c) Endpoint window and rebound sensitivity", "", md_table([{"window": w["window"], "primary p (min; n)": cell_p(w.get("primary_p"), w.get("primary_min_p"), w.get("primary_n", "")),
                                                                                  "delta": w.get("primary_delta")} for w in rob["windows"]["rows"]]), "",
              f"Seeds with a possible retreat at the final step: {', '.join(r['run_id'] for r in rob['windows']['rebound'] if r['retreat']) or 'none'}.", "",
              "### (d) Onset threshold x window grid (onset, censored at T + 1)", "",
              md_table([{"threshold": g["threshold"], "window": g["window"], "pre-registered": g["is_pre_registered"],
                         "primary onset p (min; n)": cell_p(g.get("primary_onset_p"), g.get("primary_onset_min_p"), g.get("primary_onset_n", "")),
                         "H1 onset p (min; n)": cell_p(g.get("h1_onset_p"), g.get("h1_onset_min_p"), g.get("h1_onset_n", ""))} for g in rob["onset_grid"]]), ""]
        rt = rob["rank_test"]
        if rt["available"]:
            L += ["### (e) Exact rank test next to the difference of means", "",
                  f"Difference of means: {cell_p(rt['p_diff_of_means'], rt['min_p'], f'{rt['n_h']} v {rt['n_c']}')}; exact Mann-Whitney (permutation on ranks): "
                  f"{cell_p(rt['p_exact_rank'], rt['min_p'], f'{rt['n_h']} v {rt['n_c']}')}.", ""]
            pr = rt.get("paired") or {}
            if pr.get("available"):
                L += [f"Paired companion (seed k of both primary arms shares data order and LoRA init; exact sign flips of the within-pair "
                      f"differences, {pr['n_pairs']} pairs): {cell_p(pr['p'], pr['min_p'], str(pr['n_pairs']) + ' pairs')}, mean difference "
                      f"{pr['delta']:.4f}. The pre-registered unpaired test ignores this pairing (valid, slightly conservative); the paired "
                      f"value is EXPLORATORY and never replaces it.", ""]
        hv = rob["halves"]
        L += ["### (f) Test-set halves", "", md_table([{"half": h["half"], "n problems": h["n_problems"], "primary p (min; n)": cell_p(h.get("primary_p"), h.get("primary_min_p"), h.get("primary_n", "")),
                                                        "delta": h.get("primary_delta")} for h in hv["halves"]]),
              f"Both halves agree in direction: {hv['agree_direction']}; both positive: {hv['agree_positive']}.", "",
              "### (g) Variance decomposition: within-seed binomial SE vs between-seed SD", "", md_table(rob["variance"]), ""]
        L += ["Figure: `figures/robustness_forest.png`.", ""]

    # 7. quality
    L += ["## 7. Run quality [EXPLORATORY]", "", "### Run homogeneity (mixing is flagged, not fatal)", ""]
    L += [f"- **FLAG** {f}" for f in homog["flags"]] if homog["flags"] else ["No mixing of GPU / driver / library / git / dataset / split fields within an arm or across the primary contrast."]
    L += ["", f"Distinct hosts (hashed): {homog['n_hosts']}. Full table: `tables/homogeneity.csv`.", "", "### Training health: did training train?", "",
          md_table([{k: v for k, v in r.items() if k in ("arm", "n_seeds", "reward_first10", "reward_last10", "reward_gain", "reward_slope_t", "frac_zero_adv_mean",
                                                        "grad_norm_mean", "length_first10", "length_last10", "truncation_first10", "truncation_last10",
                                                        "extraction_fail_first10", "extraction_fail_last10", "flags")} for r in health["per_arm"]]), ""]
    L += [f"**Arms flagged NOT LEARNING: {', '.join(health['not_learning_arms']) or 'none'}.** A null result from an arm that did not train is uninterpretable.", ""]

    # 8. compliance
    pc = doc["prereg_check"]
    L += ["## 8. Pre-registration compliance and deviations", "",
          f"`rhg.analysis.prereg_check`: **{'PASS' if pc['ok'] else 'FAIL / not run'}**.", ""]
    if pc["items"]:
        L += [md_table(pc["items"]), ""]
    L += ["### prereg/AMENDMENTS.jsonl", ""]
    L += [md_table(amendments, ["ts", "group", "old_hash", "new_hash", "reason"]) if amendments else "No amendments logged.\n"]
    L += ["### DEVIATIONS.md", "", "```text", deviations or "(no DEVIATIONS.md)", "```", ""]

    # 9. figures, examples
    L += ["## 9. Figures, examples and validation", ""]
    L += [f"- `figures/{r['file']}`: {r['title']}" for r in fig_records] or ["- (figures not generated)"]
    L += ["", f"- `examples.md`: random, seed-fixed gallery (RNG seed {ex['seed']}), completions verbatim, truncated at {ex['max_chars']} characters.",
          "- Measurement validity (AST detector and judge vs the execution label, with Wilson score CIs; IPW estimates are approximate) is produced by `rhg.validate.harness` into "
          "`validation.md`, not by this report; the execution label needs neither.", ""]
    return "\n".join(L) + "\n"
