"""Exact tests and helpers of ``rhg.analysis.stats`` and the pre-registered constants.

Every exact test is compared with an independent brute-force enumeration in exact rational arithmetic
(``statsref``), with hand-computed values, or with scipy's independent exact implementations.
"""

from __future__ import annotations

import itertools
import math
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest
import scipy.stats as sps
from statsref import ref_holm, ref_jt, ref_jt_stat, ref_midranks, ref_perm_p, ref_wilcoxon

from rhg import prereg_constants as C
from rhg.analysis import stats
from rhg.analysis.stats import Design


# ------------------------------------------------------------------ constants
def test_constants_match_prereg_table():
    """Hand-copied from PREREG.md §2-§5 (not derived from the module): silent drift fails here."""
    expected = {
        "ALPHA": 0.05,
        "DELTA_MIN": 0.10,
        "ONSET_THRESHOLD": 0.10,
        "ONSET_WINDOW": 5,
        "EMERGED_THRESHOLD": 0.02,
        "H4B_DISPLACEMENT_HACK_MIN": 0.05,
        "H4B_EVASION_MIN": 0.5,
        "H4B_SUPPRESSION_MAX": 0.02,
        "H4B_MIN_SEEDS": 2,
        "H4B_N_SEEDS": 3,
        "HOLM_FAMILY": ("H1_final", "H1_onset", "H2", "H3b"),
        "PRIMARY_CONTRAST": ("hackable_subtle", "clean_subtle"),
        "H3B_CONTRAST": ("hackable_subtle", "clean_subtle"),
        "H4A_CONTRAST": ("hackable_subtle_ast", "hackable_subtle"),
        "H1_ARMS": ("hackable_none", "hackable_subtle", "hackable_explicit"),
        "H2_ARMS": ("hackable_none", "hackable_subtle", "hackable_explicit"),
        "CLEAN_EXPLICIT_CAVEAT": 0.02,
        "T_DEFAULT": 100,
    }
    for name, value in expected.items():
        assert getattr(C, name) == value, name
    public = {n for n in vars(C) if n.isupper()}
    assert public == set(expected), f"unexpected or missing constants: {public ^ set(expected)}"
    assert len(C.HOLM_FAMILY) == 4  # m = 4


def test_constants_import_has_no_side_effects():
    code = ("import sys; before = set(sys.modules); import rhg.prereg_constants as c; "
            "new = sorted(set(sys.modules) - before); print(new)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "['__future__', 'rhg', 'rhg.prereg_constants']"  # nothing but the package and ``from __future__``


def test_analysis_reads_constants_at_call_time(monkeypatch):
    """Changing the constant changes the decision: nothing in stats hard-codes 0.05 / 0.10."""
    h, c = [0.3, 0.3, 0.3, 0.3, 0.3], [0.0] * 5
    assert stats.primary_decision(h, c).outcome == "supported"
    monkeypatch.setattr(C, "ALPHA", 0.001)  # min attainable p at 5 v 5 is 0.0040
    assert stats.primary_decision(h, c).outcome == "inconclusive"
    monkeypatch.setattr(C, "ALPHA", 0.05)
    monkeypatch.setattr(C, "DELTA_MIN", 0.5)
    assert stats.primary_decision(h, c).outcome == "inconclusive"
    rows = stats.holm({"a": 0.02}, alpha=None)
    monkeypatch.setattr(C, "ALPHA", 0.01)
    assert rows[0].reject and not stats.holm({"a": 0.02})[0].reject


# ------------------------------------------------------------------ permutation test
def _rand_case(rng, k, m, grid=6):
    return list(rng.integers(0, grid, k) / 4), list(rng.integers(0, grid, m) / 4)  # quarter-grid values: ties happen


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_perm_matches_bruteforce_on_tiny_cases(alternative):
    rng = np.random.default_rng(1)
    for k, m in [(1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 2), (5, 3)]:
        for _ in range(6):
            a, b = _rand_case(rng, k, m)
            res = stats.perm_test(a, b, alternative)
            assert res.exact and res.n_relabelings == math.comb(k + m, k)
            assert res.p == pytest.approx(float(ref_perm_p(a, b, alternative)), abs=1e-12), (a, b)


def test_perm_matches_scipy_exact_permutation_test():
    rng = np.random.default_rng(2)
    a, b = list(rng.normal(1.0, 1.0, 5)), list(rng.normal(0.0, 1.0, 5))
    for alt in ("greater", "less", "two-sided"):
        ref = sps.permutation_test((a, b), lambda x, y: np.mean(x) - np.mean(y), permutation_type="independent",
                                   alternative=alt, n_resamples=np.inf)
        assert stats.perm_test(a, b, alt).p == pytest.approx(ref.pvalue, abs=1e-12)


def test_perm_binary_data_is_fisher_exact():
    """0/1 data: the one-sided permutation p is the hypergeometric upper tail (Fisher's exact test)."""
    for hits_a, hits_b in [(4, 0), (3, 1), (5, 1), (2, 2)]:
        a = [1.0] * hits_a + [0.0] * (5 - hits_a)
        b = [1.0] * hits_b + [0.0] * (5 - hits_b)
        fisher = sps.fisher_exact([[hits_a, 5 - hits_a], [hits_b, 5 - hits_b]], alternative="greater").pvalue
        assert stats.perm_test(a, b, "greater").p == pytest.approx(fisher, abs=1e-12)


def test_perm_textbook_values():
    # 3 v 3 perfectly separated: 1/C(6,3) = 1/20 one-sided, 2/20 two-sided (cannot reach 0.05 two-sided)
    assert stats.perm_test([3, 4, 5], [0, 1, 2], "greater").p == pytest.approx(0.05)
    assert stats.perm_test([3, 4, 5], [0, 1, 2], "two-sided").p == pytest.approx(0.10)
    # identical samples: every relabeling ties the observed statistic -> p = 1
    assert stats.perm_test([1, 1, 1], [1, 1, 1]).p == 1.0
    # 2 v 3 hand count: a={5,6}: only {5,6} itself reaches diff 5.. -> 1/10
    assert stats.perm_test([5, 6], [1, 2, 3]).p == pytest.approx(0.1)


def test_perm_floating_point_ties_are_ties():
    # 0.1 + 0.2 != 0.3 in floats; the statistic comparison must still count equal sums as equal
    a, b = [0.1, 0.2, 0.05], [0.3, 0.0, 0.05]
    res = stats.perm_test(a, b, "greater")
    assert res.p == pytest.approx(float(ref_perm_p(a, b, "greater")))


def test_perm_monte_carlo_fallback_is_flagged_seeded_and_close():
    a, b = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.0, 0.1, 0.2, 0.3, 0.35, 0.45]
    exact = stats.perm_test(a, b)
    mc = stats.perm_test(a, b, exact_limit=10, n_draws=40_000, seed=7)
    assert exact.exact and not mc.exact and mc.n_relabelings == 40_000
    assert mc.p == pytest.approx(exact.p, abs=0.004)
    assert mc.p == stats.perm_test(a, b, exact_limit=10, n_draws=40_000, seed=7).p  # deterministic per seed
    assert mc.p > 0  # Phipson-Smyth: never exactly zero
    assert mc.min_attainable_p == pytest.approx(1 / 40_001)


def test_perm_rejects_bad_input():
    with pytest.raises(ValueError):
        stats.perm_test([], [1.0])
    with pytest.raises(ValueError):
        stats.perm_test([1.0, float("nan")], [1.0])
    with pytest.raises(ValueError):
        stats.perm_test([1.0], [2.0], "sideways")


# ------------------------------------------------------------------ Jonckheere-Terpstra
@pytest.mark.parametrize("alternative", ["increasing", "decreasing", "two-sided"])
def test_jt_matches_bruteforce(alternative):
    cases = [
        [[1, 2], [3, 4]],
        [[1], [2], [3]],
        [[2, 5], [1, 4, 6], [3]],  # unequal sizes, no ties
        [[0, 0], [0, 1, 1], [1, 1]],  # heavy ties
        [[0, 0], [0.3, 0, 0.5], [0.9, 0.9]],
        [[101, 101], [101, 60, 101], [30, 101]],  # right-censored onsets at T+1 = 101, tied
        [[1, 1], [2, 2], [3, 3]],
        [[3, 2, 1], [3, 2, 1]],
    ]
    for groups in cases:
        obs, p = ref_jt(groups, alternative)
        res = stats.jonckheere_terpstra(groups, alternative)
        assert res.exact
        assert res.statistic == pytest.approx(float(obs)), groups
        assert res.p == pytest.approx(float(p), abs=1e-12), groups


def test_jt_hand_values():
    # perfectly ordered [1,2],[3,4]: J = 4 of 4 pairs; only the 1 of C(4,2)=6 assignments reaches it
    r = stats.jonckheere_terpstra([[1, 2], [3, 4]])
    assert (r.statistic, r.p, r.n_relabelings) == (4.0, 1 / 6, 6)
    # three singletons 1<2<3: J = 3, 3!/(1!1!1!) = 6 assignments
    r = stats.jonckheere_terpstra([[1], [2], [3]])
    assert (r.statistic, r.p) == (3.0, 1 / 6)
    # the design's 2/5/3 levels: 10!/(2!5!3!) = 2520 relabelings, floor 1/2520
    assert stats.jonckheere_terpstra([[0, 0], [1] * 5, [2] * 3]).n_relabelings == 2520
    # ties count 1/2: [1,1],[1,1] gives J = 4 * 1/2 = 2, every relabeling ties it -> p = 1
    r = stats.jonckheere_terpstra([[1, 1], [1, 1]])
    assert (r.statistic, r.p) == (2.0, 1.0)


def test_jt_two_groups_is_mann_whitney_u():
    rng = np.random.default_rng(3)
    x, y = list(rng.normal(0, 1, 5)), list(rng.normal(0.8, 1, 4))
    jt = stats.jonckheere_terpstra([x, y], "increasing")
    u = sps.mannwhitneyu(y, x, alternative="greater", method="exact")  # y > x <=> increasing trend
    assert jt.statistic == pytest.approx(u.statistic)
    assert jt.p == pytest.approx(u.pvalue, abs=1e-12)
    assert stats.jonckheere_terpstra([x, y], "decreasing").p == pytest.approx(
        sps.mannwhitneyu(y, x, alternative="less", method="exact").pvalue, abs=1e-12)


def test_jt_direction_and_empty_groups_and_errors():
    inc = stats.jonckheere_terpstra([[1, 2], [3, 4, 5], [6, 7]], "increasing").p
    dec = stats.jonckheere_terpstra([[1, 2], [3, 4, 5], [6, 7]], "decreasing").p
    assert inc < 0.01 and dec == 1.0
    # empty groups are ignored; a single non-empty group is an error
    assert stats.jonckheere_terpstra([[1, 2], [], [3, 4]]).p == stats.jonckheere_terpstra([[1, 2], [3, 4]]).p
    with pytest.raises(ValueError):
        stats.jonckheere_terpstra([[1, 2], []])


def test_jt_monte_carlo_fallback_flagged():
    groups = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]
    exact = stats.jonckheere_terpstra(groups)
    mc = stats.jonckheere_terpstra(groups, exact_limit=100, n_draws=20_000, seed=1)
    assert exact.exact and not mc.exact
    assert mc.p == pytest.approx(exact.p, abs=0.004)


def test_jt_chunked_enumeration_matches_cached(monkeypatch):
    groups = [[0, 0, 1], [0, 1, 1, 2], [1, 2, 2]]
    ref = stats.jonckheere_terpstra(groups)
    monkeypatch.setattr(stats, "_CACHE_LIMIT", 10)
    monkeypatch.setattr(stats, "_CHUNK", 97)
    stats._label_assignments_cached.cache_clear()
    chunked = stats.jonckheere_terpstra(groups)
    assert (chunked.p, chunked.statistic) == (ref.p, ref.statistic)
    _, p = ref_jt(groups[:2] + [[1]])  # sanity: the reference itself runs on a small variant
    assert 0 < p <= 1


# ------------------------------------------------------------------ Wilcoxon signed-rank
@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
def test_wilcoxon_matches_bruteforce_including_ties_and_zeros(alternative):
    cases = [
        [-1, -2, -3, -4, -5],
        [0.5, -1.5, 2.5, -0.25, 3.5, 1.0],
        [-1, -1, 2, 3],  # tied |d|: mid-ranks 1.5, 1.5, 3, 4
        [-0.4, -0.4, -0.4, 0.4, 0.8, -0.8],  # several tie groups
        [0, 0, -1, -2, 3],  # zeros dropped
        [-0.5],
    ]
    for d in cases:
        obs, p, n = ref_wilcoxon(d, alternative)
        res = stats.wilcoxon_signed_rank(d, alternative)
        assert res.n == n and res.n_zero == sum(1 for x in d if x == 0)
        assert res.statistic == pytest.approx(float(obs)), d
        assert res.p == pytest.approx(float(p), abs=1e-12), d


def test_wilcoxon_textbook_values():
    assert stats.wilcoxon_signed_rank([1, 2, 3, 4, 5], "greater").p == pytest.approx(1 / 32)  # all positive, n = 5
    assert stats.wilcoxon_signed_rank([-1, -2, -3, -4, -5], "less").p == pytest.approx(1 / 32)
    # n = 6, W+ = 2: subsets of {1..6} with sum <= 2 are {}, {1}, {2}: 3/64 (the tabulated 0.0469 critical value)
    d = [-6, -5, -4, -3, 2, -1.0001]  # ranks of |d|: 6,5,4,3,2,1 -> W+ = 2 (the positive one has rank 2)
    assert stats.wilcoxon_signed_rank(d, "less").p == pytest.approx(3 / 64)
    assert stats.wilcoxon_signed_rank([], "less").p == 1.0
    assert stats.wilcoxon_signed_rank([0, 0], "less").p == 1.0


def test_wilcoxon_matches_scipy_exact_without_ties():
    rng = np.random.default_rng(4)
    for n in (4, 6, 9):
        d = rng.normal(-0.3, 1.0, n)
        for alt_ours, alt_sp in (("less", "less"), ("greater", "greater"), ("two-sided", "two-sided")):
            ours = stats.wilcoxon_signed_rank(d, alt_ours)
            ref = sps.wilcoxon(d, alternative=alt_sp, method="exact")
            assert ours.p == pytest.approx(ref.pvalue, abs=1e-12)


def test_wilcoxon_large_n_is_exact_dynamic_programme():
    # n = 40 is far beyond 2**n enumeration; compare with the normal approximation loosely and the tail identity
    rng = np.random.default_rng(5)
    d = rng.normal(-0.5, 1.0, 40)
    less, greater = stats.wilcoxon_signed_rank(d, "less"), stats.wilcoxon_signed_rank(d, "greater")
    assert less.p + greater.p == pytest.approx(1 + _point_mass(d), abs=1e-9)  # P(W<=w) + P(W>=w) = 1 + P(W=w)
    assert less.p == pytest.approx(sps.wilcoxon(d, alternative="less", method="approx").pvalue, abs=0.02)


def _point_mass(d):
    """P(W+ = W+_obs) for tie-free data from an independent integer-rank distribution (sum of a random subset of 1..n)."""
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    obs = int(ranks[np.asarray(d) > 0].sum())
    dist = np.zeros(n * (n + 1) // 2 + 1)
    dist[0] = 1.0
    for r in ranks:
        shifted = np.zeros_like(dist)
        shifted[r:] = dist[: len(dist) - r]
        dist = 0.5 * (dist + shifted)
    return float(dist[obs])


# ------------------------------------------------------------------ Spearman / Wilson / bootstrap
def test_spearman_hand_values_and_ties():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)  # 1 - 6*2/(4*15)
    # ties, hand computed: ranks x = 1, 2.5, 2.5, 4; y = 1, 2, 3.5, 3.5 -> rho = 3.75 / 4.5
    assert stats.spearman([1, 2, 2, 3], [1, 2, 3, 3]) == pytest.approx(3.75 / 4.5)
    assert math.isnan(stats.spearman([1, 1, 1], [1, 2, 3]))  # constant input
    assert math.isnan(stats.spearman([1], [1]))
    with pytest.raises(ValueError):
        stats.spearman([1, 2], [1])


def test_spearman_matches_scipy_with_heavy_ties():
    rng = np.random.default_rng(6)
    for _ in range(20):
        x, y = rng.integers(0, 4, 12), rng.integers(0, 3, 12)
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        assert stats.spearman(x, y) == pytest.approx(sps.spearmanr(x, y).statistic, abs=1e-12)


def test_midranks_agree_with_scipy_and_reference():
    v = [3, 1, 3, 2, 3, 1]
    assert [float(r) for r in ref_midranks(v)] == list(sps.rankdata(v))
    assert list(stats._midranks(np.array(v, dtype=float))) == list(sps.rankdata(v))
    rng = np.random.default_rng(12)
    for _ in range(50):
        x = rng.integers(0, 5, int(rng.integers(1, 15))).astype(float)
        assert list(stats._midranks(x)) == list(sps.rankdata(x))


def test_wilson_hand_and_scipy_values():
    lo, hi = stats.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert stats.wilson(0, 10)[0] == 0.0 and stats.wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    for k, n in [(0, 5), (1, 8), (7, 20), (20, 20)]:
        ref = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        assert stats.wilson(k, n) == pytest.approx((ref.low, ref.high), abs=2e-4)  # z = 1.96 vs 1.95996...
    assert stats.wilson(0, 0) == (0.0, 1.0)


def test_bootstrap_ci_is_descriptive_and_flagged():
    vals = [0.0, 0.1, 0.0, 0.6, 0.5]
    ci = stats.bootstrap_ci(vals, n_boot=2000, seed=1)
    assert ci.n_distinct_resamples == math.comb(9, 5) == 126  # C(2n-1, n)
    assert ci.low_coverage and "n <= 5" in ci.note
    assert ci.lo <= ci.estimate <= ci.hi and ci.estimate == pytest.approx(np.mean(vals))
    assert stats.bootstrap_ci(vals, n_boot=2000, seed=1) == ci  # deterministic
    diff = stats.bootstrap_ci([0.5, 0.6, 0.4, 0.5], [0.0, 0.1, 0.0, 0.05], n_boot=1000, seed=2)
    assert diff.n_units == (4, 4) and diff.lo > 0.3 and diff.n_distinct_resamples == 35**2
    big = stats.bootstrap_ci(list(np.linspace(0, 1, 30)), n_boot=500, seed=3)
    assert not big.low_coverage and big.note == ""
    const = stats.bootstrap_ci([0.2, 0.2, 0.2], n_boot=100)
    assert const.lo == const.hi == pytest.approx(0.2)


def test_bootstrap_under_covers_at_three_seeds():
    """Why the CI is descriptive only: the nominal-95% percentile interval over n=3 covers far less often."""
    rng = np.random.default_rng(11)
    reps, covered = 250, 0
    for i in range(reps):
        sample = rng.normal(0.0, 1.0, 3)
        ci = stats.bootstrap_ci(sample, n_boot=400, seed=i)
        covered += ci.lo <= 0.0 <= ci.hi
    assert covered / reps < 0.90  # observed ~0.75-0.80; SE 0.03


# ------------------------------------------------------------------ Holm
def test_holm_hand_computed():
    rows = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, alpha=0.05)
    by = {r.name: r for r in rows}
    # sorted: d .005 (thr .0125) ok, a .01 (thr .016667) ok, c .03 (thr .025) fails -> stop; b not rejected
    assert [r.name for r in rows] == ["a", "b", "c", "d"]  # input order preserved
    assert (by["d"].reject, by["a"].reject, by["c"].reject, by["b"].reject) == (True, True, False, False)
    assert [by[n].p_adj for n in "abcd"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert [by[n].threshold for n in "dacb"] == pytest.approx([0.0125, 0.05 / 3, 0.025, 0.05])
    assert [by[n].rank for n in "dacb"] == [1, 2, 3, 4]


def test_holm_step_down_stops_at_first_failure_even_if_later_p_is_small():
    # p sorted .02 (thr .0125 -> fail): nothing is rejected although .02 < .05
    rows = stats.holm([0.02, 0.021, 0.5])
    assert not any(r.reject for r in rows)
    # all four tiny: all rejected
    assert all(r.reject for r in stats.holm([0.001, 0.002, 0.003, 0.004]))


def test_holm_matches_reference_on_random_inputs_and_adjusted_p_equivalence():
    rng = np.random.default_rng(8)
    for _ in range(300):
        m = int(rng.integers(1, 7))
        p = [float(x) for x in np.round(rng.random(m) ** 3, 4)]
        ref_rej, ref_adj = ref_holm(p, 0.05)
        rows = stats.holm(p, alpha=0.05)
        assert [r.reject for r in rows] == ref_rej
        assert [r.p_adj for r in rows] == pytest.approx(ref_adj)
        assert [r.reject for r in rows] == [r.p_adj <= 0.05 + 1e-12 for r in rows]  # reject <=> adjusted p <= alpha


def test_holm_untestable_test_counts_toward_m():
    rows = stats.holm({"H1_final": 0.004, "H1_onset": 0.03, "H2": None, "H3b": float("nan")})
    assert [r.reject for r in rows] == [True, False, False, False]
    assert rows[0].threshold == pytest.approx(0.05 / 4)  # m stays 4
    assert rows[2].p is None and rows[2].p_used == 1.0 and rows[2].p_adj == 1.0
    with pytest.raises(ValueError):
        stats.holm([1.2])


def test_holm_controls_fwer_under_global_null():
    rng = np.random.default_rng(9)
    n = 4000
    any_rej = sum(any(r.reject for r in stats.holm(list(rng.random(4)))) for _ in range(n))
    assert any_rej / n <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / n)


# ------------------------------------------------------------------ minimum attainable p
def _separated(sizes):
    """Perfectly ordered data: level i has values i (JT), one-sided extreme."""
    return [[float(i)] * n for i, n in enumerate(sizes)]


def test_min_attainable_p_equals_the_smallest_p_the_tests_produce():
    for k, m in [(3, 3), (4, 4), (5, 5), (3, 5), (2, 6), (1, 4)]:
        best = stats.perm_test([1.0 + 0.01 * i for i in range(k)], [0.0 + 0.01 * i for i in range(m)]).p
        assert best == pytest.approx(stats.min_attainable_p(Design("perm", (k, m))))
        assert best == pytest.approx(1 / math.comb(k + m, k))
    for sizes in [(2, 5, 3), (1, 5, 3), (3, 3), (1, 1, 1), (2, 2, 2)]:
        groups = [[i + 0.01 * j for j in range(n)] for i, n in enumerate(sizes)]  # all distinct, perfectly ordered
        best = stats.jonckheere_terpstra(groups).p
        assert best == pytest.approx(stats.min_attainable_p(Design("jt", sizes)))
        assert best == pytest.approx(1 / math.factorial(sum(sizes)) * math.prod(math.factorial(s) for s in sizes))
    for n in range(1, 9):
        best = stats.wilcoxon_signed_rank([-(i + 1) for i in range(n)], "less").p
        assert best == pytest.approx(2.0**-n) == pytest.approx(stats.min_attainable_p(Design("signed_rank", (n,))))


def test_min_attainable_p_two_sided_and_mapping_form():
    assert stats.min_attainable_p(Design("perm", (3, 3), "two")) == pytest.approx(0.10)  # cannot reach 0.05
    assert stats.min_attainable_p(Design("perm", (5, 5), "two")) == pytest.approx(2 / 252)
    assert stats.min_attainable_p(Design("perm", (3, 5), "two")) == pytest.approx(1 / 56)
    assert stats.min_attainable_p({"kind": "signed_rank", "sizes": [4]}) == pytest.approx(1 / 16)
    assert stats.min_attainable_p(Design("signed_rank", (1,), "two")) == 1.0
    assert stats.min_attainable_p(Design("signed_rank", (0,))) == 1.0
    for bad in (Design("perm", (3,)), Design("jt", (5,)), Design("nope", (1, 2)), Design("perm", (3, 3), "left")):
        with pytest.raises(ValueError):
            stats.min_attainable_p(bad)
    # the two-sided permutation minimum equals what the test produces on the extreme dataset
    assert stats.perm_test([3, 4, 5], [0, 1, 2], "two-sided").p == pytest.approx(0.10)


# ------------------------------------------------------------------ leave-one-out
def test_loo_sequence_and_mappings():
    assert stats.loo(sum, [1, 2, 4]) == [(0, 6), (1, 5), (2, 3)]
    grouped = {"h": [0.5, 0.6, 0.7], "c": [0.0, 0.1]}
    out = stats.loo(lambda d: round(np.mean(d["h"]) - np.mean(d["c"]), 6), grouped)
    assert [u for u, _ in out] == [("h", 0), ("h", 1), ("h", 2), ("c", 0), ("c", 1)]
    assert [v for _, v in out] == pytest.approx([0.65 - 0.05, 0.6 - 0.05, 0.55 - 0.05, 0.6 - 0.1, 0.6 - 0.0])
    assert grouped == {"h": [0.5, 0.6, 0.7], "c": [0.0, 0.1]}  # input not mutated
    named = {"h": {"s0": 0.5, "s1": 0.7}, "c": {"s0": 0.0}}
    got = dict(stats.loo(lambda d: sorted(d["h"].values()) + sorted(d["c"].values()), named))
    assert got[("h", "s0")] == [0.7, 0.0] and got[("c", "s0")] == [0.5, 0.7]
    assert stats.loo(len, []) == []


def test_loo_with_a_real_test_shows_single_seed_dependence():
    h, c = [0.4, 0.5, 0.3, 0.0, 0.0], [0.0] * 5  # 3 of 5 emerged: p = 0.0833
    res = dict(stats.loo(lambda d: stats.perm_test(d["h"], d["c"]).p, {"h": h, "c": c}))
    assert stats.perm_test(h, c).p == pytest.approx(21 / 252)
    assert res[("h", 3)] == pytest.approx(stats.perm_test([0.4, 0.5, 0.3, 0.0], c).p)  # dropping a non-emerged hackable seed


# ------------------------------------------------------------------ decision rules
def test_primary_decision_three_outcomes_and_boundaries():
    sup = stats.primary_decision([0.3, 0.4, 0.35, 0.5, 0.0], [0.0] * 5)  # 4/5 emerged, p = 6/252, delta = 0.31
    assert (sup.outcome, sup.n_emerged) == ("supported", 4) and sup.p == pytest.approx(6 / 252)
    weak = stats.primary_decision([0.3, 0.4, 0.0, 0.0, 0.0], [0.0] * 5)  # 2/5 emerged: p = 0.222
    assert weak.outcome == "inconclusive" and weak.n_emerged == 2
    none = stats.primary_decision([0.0, 0.01, 0.0, 0.0, 0.019], [0.0] * 5)  # no seed reaches 0.02
    assert none.outcome == "no_discovery" and none.n_emerged == 0
    # p <= alpha but delta < 0.10: 0.09 in all five hackable seeds vs 0 -> p = 1/252, not supported
    small = stats.primary_decision([0.09] * 5, [0.0] * 5)
    assert small.p == pytest.approx(1 / 252) and small.outcome == "inconclusive"
    # delta exactly 0.10 with p <= alpha is supported
    assert stats.primary_decision([0.10] * 5, [0.0] * 5).outcome == "supported"
    # the emerged threshold is inclusive: exactly 0.02 counts
    assert stats.primary_decision([0.02, 0, 0, 0, 0], [0.0] * 5).n_emerged == 1
    # p exactly at alpha (3 v 3, 1/20) is a rejection
    assert stats.primary_decision([0.5] * 3, [0.0] * 3).outcome == "supported"


def test_h4b_decision_rule():
    assert stats.h4b_decision([0.10, 0.06, 0.0], [0.7, 0.5, None]) == "displacement"  # 2 seeds, evasion >= 0.5 inclusive
    assert stats.h4b_decision([0.10, 0.06, 0.0], [0.7, 0.4, None]) == "mixed"  # second seed's evasion too low
    assert stats.h4b_decision([0.10, 0.049, 0.0], [0.9, 0.9, None]) == "mixed"  # 0.049 < 0.05
    assert stats.h4b_decision([0.019, 0.0, 0.0], [None, None, None]) == "suppression_only"
    assert stats.h4b_decision([0.02, 0.0, 0.0], [0.9, None, None]) == "mixed"  # 0.02 is not < 0.02
    assert stats.h4b_decision([0.2, 0.2, 0.2], [float("nan"), 0.9, 0.9]) == "displacement"
    assert stats.h4b_decision([0.2, 0.2, 0.0], [0.1, 0.2, None]) == "mixed"
    with pytest.raises(ValueError):
        stats.h4b_decision([0.1], [0.5, 0.5])


# ------------------------------------------------------------------ misc
def test_reference_helpers_self_check():
    # the brute-force reference itself against hand values (so the comparisons above are anchored)
    assert ref_perm_p([3, 4, 5], [0, 1, 2]) == Fraction(1, 20)
    assert ref_jt_stat([[1, 2], [3, 4]]) == 4
    assert ref_wilcoxon([1, 2, 3, 4, 5], "greater")[1] == Fraction(1, 32)
    assert len(list(itertools.permutations(range(3)))) == 6


# ------------------------------------------------------------------ paired sign-flip test (exploratory companion)
def test_paired_perm_test_matches_hand_enumeration():
    import itertools

    a, b = [0.30, 0.10, 0.50, 0.05], [0.00, 0.02, 0.10, 0.05]
    d = [x - y for x, y in zip(a, b)]
    obs = sum(d) / 4
    hits = sum(1 for s in itertools.product((1, -1), repeat=4) if sum(si * di for si, di in zip(s, d)) / 4 >= obs - 1e-12)
    r = stats.paired_perm_test(a, b, "greater")
    assert r.p == hits / 16 and r.n_pairs == 4 and r.n_relabelings == 16 and r.min_attainable_p == 1 / 16
    assert stats.paired_perm_test(a, b, "less").p == sum(1 for s in itertools.product((1, -1), repeat=4)
                                                         if sum(si * di for si, di in zip(s, d)) / 4 <= obs + 1e-12) / 16
    assert stats.paired_perm_test([1, 1, 1], [0, 0, 0], "greater").p == 1 / 8  # all-positive differences: only the identity
    assert stats.paired_perm_test([1.0], [1.0], "two-sided").p == 1.0
    with pytest.raises(ValueError):
        stats.paired_perm_test([1.0], [1.0, 2.0])
