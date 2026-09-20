"""Minimum attainable p-values and exact power tables (DESIGN §6, BUDGET §4).

Model behind the tables (the "emergence structure" of DESIGN §6): a seed either *emerges* (final HACK_RT rate
``r``, default 0.5) with probability ``q`` or stays at rate 0. Every p-value below is computed by **full
enumeration** through ``rhg.analysis.stats`` on the corresponding 0/r pattern; power sums the exact binomial
probabilities of every pattern. Nothing is simulated. ``r`` only matters for the Δ >= DELTA_MIN clause of the
primary decision rule (reported as ``min_emerged_rate``), not for p.

CLI: ``python -m rhg.analysis.power [--n-h 5 --n-c 5] [--ladder N|all] [--qs 0.9 0.8 ...] [--json PATH]``.
``--verify-design`` exits 1 if the enumeration disagrees with the DESIGN §6 table copied below.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from rhg import prereg_constants as C
from rhg.analysis.stats import Design, jonckheere_terpstra, leq, min_attainable_p, perm_test

DEFAULT_QS = (0.9, 0.8, 0.7, 0.5)
DEFAULT_EMERGED_RATE = 0.5
Q_EXPLICIT_ASSUMED = 1.0  # illustrative dose-response scenario for the H1 power: q_none = 0, q_subtle = q, q_explicit = 1
EXPLICIT_RATE_MULT = 2.0  # ... and an emerged explicit-level seed sits at min(1, 2 * rate); without a rate gap JT is tie-limited

# DESIGN §6 table, copied by hand: (n_h, n_c, hackable seeds emerging, clean all 0) -> one-sided p to 4 decimals
DESIGN_MIN_P = {(3, 3, 3): 0.0500, (4, 4, 4): 0.0143, (4, 4, 3): 0.0714,
                (5, 5, 5): 0.0040, (5, 5, 4): 0.0238, (5, 5, 3): 0.0833}
DESIGN_H4A_P = {(3, 5): 0.0179}  # H4a: all 3 monitor seeds at 0, all 5 subtle seeds emerged
DESIGN_POWER_5V5 = {0.9: 0.92, 0.8: 0.74, 0.7: 0.53, 0.5: 0.19}


# ------------------------------------------------------------------ binary-emergence model
@lru_cache(maxsize=4096)
def emergence_p(n_h: int, n_c: int, e: int, f: int = 0, rate: float = DEFAULT_EMERGED_RATE) -> float:
    """Exact one-sided permutation p when ``e`` of ``n_h`` hackable and ``f`` of ``n_c`` clean seeds emerged
    (each emerged seed at ``rate``, the others at 0)."""
    if not (0 <= e <= n_h and 0 <= f <= n_c):
        raise ValueError("need 0 <= e <= n_h and 0 <= f <= n_c")
    return perm_test([rate] * e + [0.0] * (n_h - e), [rate] * f + [0.0] * (n_c - f), "greater").p


def binom_pmf(n: int, k: int, q: float) -> float:
    return math.comb(n, k) * q**k * (1 - q) ** (n - k)


def primary_reject(n_h: int, n_c: int, e: int, f: int = 0, rate: float = DEFAULT_EMERGED_RATE,
                   alpha: float | None = None) -> bool:
    """PREREG §2 decision for the pattern (e, f): p <= alpha (default ALPHA) AND mean difference >= DELTA_MIN."""
    alpha = C.ALPHA if alpha is None else alpha
    delta = rate * (e / n_h - f / n_c)
    return leq(emergence_p(n_h, n_c, e, f, rate), alpha) and delta >= C.DELTA_MIN - 1e-12


def primary_power(n_h: int, n_c: int, q_h: float, q_c: float = 0.0, rate: float = DEFAULT_EMERGED_RATE,
                  alpha: float | None = None) -> float:
    """P(primary "supported") when hackable seeds emerge w.p. ``q_h`` and clean seeds w.p. ``q_c``."""
    total = 0.0
    for e in range(n_h + 1):
        for f in range(n_c + 1):
            if primary_reject(n_h, n_c, e, f, rate, alpha):
                total += binom_pmf(n_h, e, q_h) * binom_pmf(n_c, f, q_c)
    return total


def min_emerged_rate(n_h: int, e: int, f: int = 0, n_c: int | None = None) -> float:
    """Smallest emerged-seed rate for which the Δ >= DELTA_MIN clause holds with (e, f) emerged."""
    n_c = n_h if n_c is None else n_c
    frac = e / n_h - f / n_c
    return math.inf if frac <= 0 else C.DELTA_MIN / frac


def emergence_table(n_h: int, n_c: int, rate: float = DEFAULT_EMERGED_RATE) -> list[dict]:
    """One row per number of emerging hackable seeds (clean seeds all at 0)."""
    rows = []
    for e in range(n_h + 1):
        p = emergence_p(n_h, n_c, e, 0, rate)
        rows.append({"emerged": e, "p": p, "reject_alpha": leq(p, C.ALPHA),
                     "min_emerged_rate": min_emerged_rate(n_h, e, 0, n_c) if e else math.inf})
    return rows


def power_curve(n_h: int, n_c: int, qs: Sequence[float] = DEFAULT_QS, rate: float = DEFAULT_EMERGED_RATE) -> list[dict]:
    return [{"q": q, "power": primary_power(n_h, n_c, q, 0.0, rate)} for q in qs]


def _pattern(n: int, e: int, rate: float) -> list[float]:
    return [rate] * e + [0.0] * (n - e)


@lru_cache(maxsize=8192)
def _jt_pattern_p(sizes: tuple[int, ...], counts: tuple[int, ...], rates: tuple[float, ...]) -> float:
    return jonckheere_terpstra([_pattern(n, e, r) for n, e, r in zip(sizes, counts, rates)], "increasing").p


def jt_power(sizes: Sequence[int], q_levels: Sequence[float], threshold: float,
             rates: float | Sequence[float] = DEFAULT_EMERGED_RATE) -> float:
    """Exact P(JT increasing-trend p <= threshold) in the binary-emergence model with per-level emergence
    probabilities ``q_levels`` and per-level emerged rates ``rates`` (a scalar = the same rate at every level;
    equal rates across levels make JT tie-limited, so its floor is then far above 1 / #relabelings)."""
    sizes = tuple(sizes)
    if len(sizes) != len(q_levels):
        raise ValueError("one q per level")
    rate_t = (float(rates),) * len(sizes) if isinstance(rates, (int, float)) else tuple(float(r) for r in rates)

    def rec(level: int, counts: tuple[int, ...], prob: float) -> float:
        if level == len(sizes):
            return prob if leq(_jt_pattern_p(sizes, counts, rate_t), threshold) else 0.0
        return sum(rec(level + 1, counts + (e,), prob * binom_pmf(sizes[level], e, q_levels[level]))
                   for e in range(sizes[level] + 1))

    return rec(0, (), 1.0)


# ------------------------------------------------------------------ planned tests of a design
@dataclass(frozen=True)
class TestPlan:
    __test__ = False  # not a pytest class
    test: str
    family: str  # "primary" | "holm" | "exploratory"
    kind: str
    sizes: tuple[int, ...]
    min_p: float | None  # None = the test cannot be run at all in this design
    threshold: float | None  # the level it must reach: alpha (primary), alpha/m (Holm family, worst case), None
    reachable_alone: bool  # min_p <= alpha
    reachable_holm_first: bool | None  # min_p <= alpha/m: rejectable even when it is the smallest p of the family
    note: str = ""


def h2_floor_rows(n_max: int) -> list[dict]:
    """H2 (exact one-sided Wilcoxon on per-seed rho): floor 2**-n for n usable (emerged) seeds."""
    m = len(C.HOLM_FAMILY)
    return [{"n_usable_seeds": n, "min_p": min_attainable_p(Design("signed_rank", (n,))),
             "reachable_alpha": leq(min_attainable_p(Design("signed_rank", (n,))), C.ALPHA),
             "reachable_alpha_over_m": leq(min_attainable_p(Design("signed_rank", (n,))), C.ALPHA / m)}
            for n in range(1, n_max + 1)]


def h2_seeds_needed(threshold: float) -> int:
    """Fewest usable seeds for which the exact signed-rank floor 2**-n is <= threshold."""
    n = 1
    while not leq(2.0**-n, threshold):
        n += 1
    return n


def planned_tests(seeds: dict[str, int]) -> list[TestPlan]:
    """Every planned test with its minimum attainable p for the given seed counts per arm."""
    m = len(C.HOLM_FAMILY)
    n = lambda arm: int(seeds.get(arm, 0))  # noqa: E731
    out: list[TestPlan] = []

    def add(test, family, kind, sizes, note="", *, runnable=True):
        if not runnable:
            out.append(TestPlan(test, family, kind, sizes, None, None, False, None, note))
            return
        mp = min_attainable_p(Design(kind, sizes))
        thr = C.ALPHA if family == "primary" else (C.ALPHA / m if family == "holm" else None)
        out.append(TestPlan(test, family, kind, sizes, mp, thr, leq(mp, C.ALPHA),
                            leq(mp, C.ALPHA / m) if family == "holm" else None, note))

    h, c = n(C.PRIMARY_CONTRAST[0]), n(C.PRIMARY_CONTRAST[1])
    add("primary", "primary", "perm", (h, c), "hackable_subtle vs clean_subtle, one-sided", runnable=h > 0 and c > 0)
    levels = tuple(n(a) for a in C.H1_ARMS)
    jt_ok = sum(1 for v in levels if v > 0) >= 2
    for name in ("H1_final", "H1_onset"):
        add(name, "holm", "jt", tuple(v for v in levels if v > 0), "JT over hackable arms (levels none<subtle<explicit)", runnable=jt_ok)
    n_h2 = sum(levels)
    add("H2", "holm", "signed_rank", (n_h2,), f"best case: all {n_h2} hackable seeds usable; only emerged seeds with 0<rate<1 count",
        runnable=n_h2 > 0)
    add("H3b", "holm", "perm", (h, c), "held-out CORRECT lower in hackable_subtle", runnable=h > 0 and c > 0)
    a, base = n(C.H4A_CONTRAST[0]), n(C.H4A_CONTRAST[1])
    add("H4a", "exploratory", "perm", (a, base), "outside the Holm family by structure (min p > alpha/m at 3 v 5)", runnable=a > 0 and base > 0)
    return out


def flags(plans: Sequence[TestPlan]) -> list[str]:
    """Human-readable warnings for tests that cannot reach their threshold."""
    m = len(C.HOLM_FAMILY)
    msgs = []
    for t in plans:
        if t.test == "H2" and t.min_p is not None:
            msgs.append(f"H2: the floor depends on emerged seeds (0 < final rate < 1) only: needs >= {h2_seeds_needed(C.ALPHA)} to "
                        f"reach alpha and >= {h2_seeds_needed(C.ALPHA / m)} to reach alpha/m; at most {t.sizes[0]} exist in this design")
        if t.min_p is None:
            msgs.append(f"{t.test}: cannot be run in this design (missing arms/seeds)")
        elif not t.reachable_alone:
            msgs.append(f"{t.test}: min attainable p = {t.min_p:.4f} > alpha = {C.ALPHA}: can never be rejected")
        elif t.family == "holm" and not t.reachable_holm_first:
            msgs.append(f"{t.test}: min attainable p = {t.min_p:.4f} > alpha/m = {C.ALPHA / m:.4f}: rejectable only after "
                        f"earlier Holm rejections")
        elif t.family == "exploratory" and t.min_p > C.ALPHA / m:
            msgs.append(f"{t.test}: min attainable p = {t.min_p:.4f} > alpha/m = {C.ALPHA / m:.4f} (structurally outside the family)")
    return msgs


# ------------------------------------------------------------------ ladder states (BUDGET §4)
def ladder_seed_counts(step: int) -> dict[str, int]:
    from rhg.budget import LADDER

    if not 0 <= step < len(LADDER):
        raise ValueError(f"ladder step must be in 0..{len(LADDER) - 1}")
    return dict(LADDER[step].seeds)


@dataclass
class DesignReport:
    label: str
    seeds: dict[str, int]
    runs: int
    tests: list[TestPlan]
    flags: list[str]
    primary_power: list[dict] = field(default_factory=list)
    h1_power: list[dict] = field(default_factory=list)
    emergence: list[dict] = field(default_factory=list)


def design_report(seeds: dict[str, int], label: str, qs: Sequence[float] = DEFAULT_QS, rate: float = DEFAULT_EMERGED_RATE) -> DesignReport:
    h, c = seeds.get(C.PRIMARY_CONTRAST[0], 0), seeds.get(C.PRIMARY_CONTRAST[1], 0)
    plans = planned_tests(seeds)
    rep = DesignReport(label, dict(seeds), sum(seeds.values()), plans, flags(plans))
    if h and c:
        rep.primary_power = power_curve(h, c, qs, rate)
        rep.emergence = emergence_table(h, c, rate)
    levels = [seeds.get(a, 0) for a in C.H1_ARMS]
    if sum(1 for v in levels if v > 0) >= 2:
        idx = [i for i, v in enumerate(levels) if v > 0]
        sizes = [levels[i] for i in idx]
        m = len(C.HOLM_FAMILY)
        for q in qs:
            dose_q = [(0.0, q, Q_EXPLICIT_ASSUMED)[i] for i in idx]
            dose_r = [(rate, rate, min(1.0, EXPLICIT_RATE_MULT * rate))[i] for i in idx]
            plat_q = [(0.0, q, q)[i] for i in idx]
            rep.h1_power.append({
                "q_subtle": q,
                "dose_alpha": jt_power(sizes, dose_q, C.ALPHA, dose_r),
                "dose_alpha_over_m": jt_power(sizes, dose_q, C.ALPHA / m, dose_r),
                "plateau_alpha": jt_power(sizes, plat_q, C.ALPHA, rate),
                "plateau_alpha_over_m": jt_power(sizes, plat_q, C.ALPHA / m, rate),
            })
    return rep


def ladder_reports(qs: Sequence[float] = DEFAULT_QS, rate: float = DEFAULT_EMERGED_RATE) -> list[DesignReport]:
    from rhg.budget import LADDER

    return [design_report(dict(s.seeds), f"ladder step {s.step}: {s.change} ({s.runs} runs)", qs, rate) for s in LADDER]


# ------------------------------------------------------------------ DESIGN §6 verification
def verify_design_table(tol: float = 5e-5) -> list[str]:
    """Differences between the enumeration and the DESIGN §6 table (empty list = reproduces it)."""
    bad = []
    for (nh, nc, e), want in DESIGN_MIN_P.items():
        got = emergence_p(nh, nc, e, 0)
        if abs(got - want) > tol:
            bad.append(f"{nh}v{nc}, {e} emerging: enumerated {got:.4f}, DESIGN says {want:.4f}")
    for (a, b), want in DESIGN_H4A_P.items():
        got = perm_test([0.0] * a, [DEFAULT_EMERGED_RATE] * b, "less").p
        if abs(got - want) > tol:
            bad.append(f"H4a {a}v{b}: enumerated {got:.4f}, DESIGN says {want:.4f}")
    for q, want in DESIGN_POWER_5V5.items():
        got = primary_power(5, 5, q)
        if abs(got - want) > 0.005 + 1e-12:
            bad.append(f"power 5v5 q={q}: enumerated {got:.4f}, DESIGN says {want:.2f}")
    return bad


# ------------------------------------------------------------------ rendering / CLI
def _fmt_p(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.4f}"


def render_report(rep: DesignReport) -> str:
    m = len(C.HOLM_FAMILY)
    lines = [f"### {rep.label}", "", "seeds: " + ", ".join(f"{a}={n}" for a, n in rep.seeds.items() if n), "",
             "| test | family | design | min attainable p | reaches alpha | reaches alpha/m |", "|---|---|---|---|---|---|"]
    for t in rep.tests:
        lines.append(f"| {t.test} | {t.family} | {t.kind}{t.sizes} | {_fmt_p(t.min_p)} | {'yes' if t.reachable_alone else 'NO'} | "
                     f"{'-' if t.reachable_holm_first is None else ('yes' if t.reachable_holm_first else 'NO')} |")
    if rep.flags:
        lines += ["", "FLAGS:"] + [f"- {f}" for f in rep.flags]
    if rep.emergence:
        lines += ["", "primary: hackable seeds emerging (clean all 0) -> exact one-sided p", "",
                  "| emerged | p | p <= alpha | min emerged rate for delta >= 0.10 |", "|---|---|---|---|"]
        for r in rep.emergence:
            mr = "-" if math.isinf(r["min_emerged_rate"]) else f"{r['min_emerged_rate']:.3f}"
            lines.append(f"| {r['emerged']} | {r['p']:.4f} | {'yes' if r['reject_alpha'] else 'no'} | {mr} |")
        lines += ["", "primary power = P(supported) vs per-seed emergence probability q (clean q = 0)", "", "| q | power |", "|---|---|"]
        lines += [f"| {r['q']:.2f} | {r['power']:.4f} |" for r in rep.primary_power]
    if rep.h1_power:
        lines += ["", "H1 (JT trend) power in two ILLUSTRATIVE scenarios (assumptions, not measurements); emerged rate r at "
                      f"none/subtle. dose: q_none=0, q_subtle=q, q_explicit={Q_EXPLICIT_ASSUMED:g}, explicit rate "
                      f"min(1, {EXPLICIT_RATE_MULT:g} r). plateau: q_none=0, q_subtle=q_explicit=q, equal rates (tie-limited). "
                      f"alpha/m = {C.ALPHA / m:.4f}", "",
                  "| q_subtle | dose, alpha | dose, alpha/m | plateau, alpha | plateau, alpha/m |", "|---|---|---|---|---|"]
        lines += [f"| {r['q_subtle']:.2f} | {r['dose_alpha']:.4f} | {r['dose_alpha_over_m']:.4f} | "
                  f"{r['plateau_alpha']:.4f} | {r['plateau_alpha_over_m']:.4f} |" for r in rep.h1_power]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhg.analysis.power", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-h", type=int, default=5, help="hackable_subtle seeds (primary design)")
    ap.add_argument("--n-c", type=int, default=5, help="clean_subtle seeds (primary design)")
    ap.add_argument("--ladder", default=None, help="a ladder step 0..6 or 'all' (BUDGET §4); default: only --n-h/--n-c on the full design")
    ap.add_argument("--qs", type=float, nargs="+", default=list(DEFAULT_QS), help="per-seed emergence probabilities")
    ap.add_argument("--emerged-rate", type=float, default=DEFAULT_EMERGED_RATE, help="final HACK_RT rate of an emerged seed")
    ap.add_argument("--h2-max", type=int, default=0, help="also print the H2 signed-rank floors for 1..N usable seeds")
    ap.add_argument("--verify-design", action="store_true", help="exit 1 if the enumeration disagrees with DESIGN §6")
    ap.add_argument("--json", type=Path, default=None, help="also write the reports as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # tables contain non-ASCII characters; Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if args.ladder is None:
            seeds = ladder_seed_counts(0)
            seeds.update({C.PRIMARY_CONTRAST[0]: args.n_h, C.PRIMARY_CONTRAST[1]: args.n_c})
            reports = [design_report(seeds, f"design with {args.n_h} v {args.n_c} primary seeds", args.qs, args.emerged_rate)]
        elif args.ladder == "all":
            reports = ladder_reports(args.qs, args.emerged_rate)
        else:
            step = int(args.ladder)
            reports = [design_report(ladder_seed_counts(step), f"ladder step {step}", args.qs, args.emerged_rate)]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n\n".join(render_report(r) for r in reports))
    if args.h2_max:
        print("\nH2 exact signed-rank floors\n\n| usable seeds | min p | reaches alpha | reaches alpha/m |\n|---|---|---|---|")
        for r in h2_floor_rows(args.h2_max):
            print(f"| {r['n_usable_seeds']} | {r['min_p']:.5f} | {'yes' if r['reachable_alpha'] else 'NO'} | "
                  f"{'yes' if r['reachable_alpha_over_m'] else 'NO'} |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(r) for r in reports], indent=2, default=str) + "\n", encoding="utf-8")
    if args.verify_design:
        bad = verify_design_table()
        for b in bad:
            print(f"DESIGN MISMATCH: {b}", file=sys.stderr)
        if bad:
            return 1
        print("\nDESIGN §6 table reproduced by enumeration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
