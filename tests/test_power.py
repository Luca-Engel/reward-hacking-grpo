"""Minimum attainable p and exact power tables (subtask 13, DESIGN §6, BUDGET §4)."""

from __future__ import annotations

import json
import math
from itertools import product

import pytest
from statsref import ref_jt, ref_perm_p

from rhg import prereg_constants as C
from rhg.analysis import power
from rhg.budget import LADDER


def _binom_tail_at_least(n, k, q):
    return sum(math.comb(n, i) * q**i * (1 - q) ** (n - i) for i in range(k, n + 1))


# ------------------------------------------------------------------ DESIGN §6 table (hand-copied)
@pytest.mark.parametrize(
    "n_h,n_c,emerged,p",
    [(3, 3, 3, 0.0500), (4, 4, 4, 0.0143), (4, 4, 3, 0.0714), (5, 5, 5, 0.0040), (5, 5, 4, 0.0238), (5, 5, 3, 0.0833)],
)
def test_design_min_p_table(n_h, n_c, emerged, p):
    got = power.emergence_p(n_h, n_c, emerged, 0)
    assert got == pytest.approx(p, abs=5e-5)
    # closed form independent of the enumeration: sets containing all `emerged` highs, C(N-e, k-e) / C(N, k)
    assert got == pytest.approx(math.comb(n_h + n_c - emerged, n_h - emerged) / math.comb(n_h + n_c, n_h))


def test_design_h4a_three_v_five():
    from rhg.analysis.stats import perm_test

    assert perm_test([0.0] * 3, [0.4] * 5, "less").p == pytest.approx(0.0179, abs=5e-5)
    assert power.emergence_p(3, 5, 3, 0) == pytest.approx(math.comb(5, 0) / math.comb(8, 3))  # 3 emerging -> 1/56


def test_three_v_three_one_sided_reaches_alpha_but_two_sided_cannot():
    assert power.emergence_p(3, 3, 3, 0) == pytest.approx(0.05)
    assert power.primary_reject(3, 3, 3, 0)
    from rhg.analysis.stats import Design, min_attainable_p

    assert min_attainable_p(Design("perm", (3, 3), "two")) == pytest.approx(0.10)


@pytest.mark.parametrize("q,want", [(0.9, 0.92), (0.8, 0.74), (0.7, 0.53), (0.5, 0.19)])
def test_power_5v5_matches_design_table(q, want):
    got = power.primary_power(5, 5, q)
    assert round(got, 2) == want
    assert got == pytest.approx(_binom_tail_at_least(5, 4, q))  # P(>= 4 of 5 emerge), independent closed form


def test_power_matches_full_enumeration_over_both_arms():
    """Independent brute force: all 2**(n_h+n_c) emergence patterns, p from the rational reference."""
    n_h, n_c, q_h, q_c, rate = 4, 3, 0.7, 0.2, 0.5
    total = 0.0
    for pattern in product((0, 1), repeat=n_h + n_c):
        h, c = pattern[:n_h], pattern[n_h:]
        weight = math.prod(q_h if x else 1 - q_h for x in h) * math.prod(q_c if x else 1 - q_c for x in c)
        p = float(ref_perm_p([rate * x for x in h], [rate * x for x in c], "greater"))
        delta = rate * (sum(h) / n_h - sum(c) / n_c)
        if p <= 0.05 + 1e-12 and delta >= 0.10 - 1e-12:
            total += weight
    assert power.primary_power(n_h, n_c, q_h, q_c, rate) == pytest.approx(total)


def test_type_i_at_equal_q_is_at_most_alpha_exactly():
    """With q_h = q_c the exact rejection probability of the emergence model is <= alpha for every q (conservative)."""
    for n_h, n_c in [(5, 5), (4, 4), (3, 5)]:
        for q in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            assert power.primary_power(n_h, n_c, q, q) <= 0.05 + 1e-12


def test_delta_clause_limits_power():
    # 5 v 5, all five emerge at rate 0.09: p = 1/252 but delta = 0.09 < 0.10 -> never "supported"
    assert power.primary_power(5, 5, 1.0, 0.0, rate=0.09) == 0.0
    assert power.primary_power(5, 5, 1.0, 0.0, rate=0.10) == pytest.approx(1.0)
    assert power.min_emerged_rate(5, 4) == pytest.approx(0.125) and power.min_emerged_rate(5, 5) == pytest.approx(0.10)
    assert math.isinf(power.min_emerged_rate(5, 0))


def test_power_reads_alpha_at_call_time(monkeypatch):
    assert power.primary_power(5, 5, 0.8) == pytest.approx(0.7373, abs=1e-4)
    monkeypatch.setattr(C, "ALPHA", 0.01)  # only 5/5 (p = 0.004) still rejects: power = q^5
    assert power.primary_power(5, 5, 0.8) == pytest.approx(0.8**5)


def test_design_verifier_passes_and_can_fail(monkeypatch):
    assert power.verify_design_table() == []
    monkeypatch.setitem(power.DESIGN_MIN_P, (5, 5, 4), 0.0500)
    monkeypatch.setitem(power.DESIGN_POWER_5V5, 0.9, 0.99)
    bad = power.verify_design_table()
    assert len(bad) == 2 and any("5v5, 4 emerging" in b for b in bad) and any("q=0.9" in b for b in bad)


# ------------------------------------------------------------------ JT power
def test_jt_power_matches_brute_force():
    sizes, qs, rates, thr = (2, 3), (0.3, 0.6), (0.5, 0.5), 0.10
    total = 0.0
    for pattern in product((0, 1), repeat=sum(sizes)):
        lv = [pattern[:2], pattern[2:]]
        w = math.prod((qs[g] if x else 1 - qs[g]) for g in range(2) for x in lv[g])
        _, p = ref_jt([[rates[g] * x for x in lv[g]] for g in range(2)], "increasing")
        total += w * (float(p) <= thr + 1e-12)
    assert power.jt_power(sizes, qs, thr, rates) == pytest.approx(total)
    with pytest.raises(ValueError):
        power.jt_power((2, 3), (0.5,), 0.05)


def test_jt_is_tie_limited_when_emerged_rates_are_equal_across_levels():
    # levels (2, 5, 3): without a rate gap between subtle and explicit the smallest reachable p is far above 1/2520
    from rhg.analysis.stats import jonckheere_terpstra

    tied = jonckheere_terpstra([[0, 0], [0.5] * 5, [0.5] * 3]).p
    graded = jonckheere_terpstra([[0, 0], [0.5] * 5, [1.0] * 3]).p
    assert graded == pytest.approx(1 / 2520)  # the floor 1 / (10! / (2! 5! 3!))
    assert tied > 50 * graded
    small = jonckheere_terpstra([[0], [0.5] * 3, [0.5] * 2]).p  # same structure, small enough for the brute-force reference
    assert small == pytest.approx(float(ref_jt([[0], [0.5] * 3, [0.5] * 2])[1])) and small > 1 / 60


# ------------------------------------------------------------------ planned tests / flags
def _by_test(plans):
    return {t.test: t for t in plans}


def test_full_design_flags_only_h4a_and_notes_h2():
    seeds = dict(LADDER[0].seeds)
    plans = _by_test(power.planned_tests(seeds))
    assert plans["primary"].min_p == pytest.approx(1 / 252)
    assert plans["H3b"].min_p == pytest.approx(1 / 252)
    assert plans["H1_final"].sizes == (2, 5, 3) and plans["H1_final"].min_p == pytest.approx(1 / 2520)
    assert plans["H4a"].min_p == pytest.approx(1 / 56) and plans["H4a"].reachable_alone
    assert plans["H4a"].reachable_holm_first is None  # exploratory: outside the family
    msgs = power.flags(list(plans.values()))
    assert any(m.startswith("H4a") and "0.0179" in m and "0.0125" in m for m in msgs)
    assert any(m.startswith("H2") and ">= 5" in m and ">= 7" in m for m in msgs)
    assert not any(m.startswith(("primary", "H1", "H3b")) for m in msgs)


def test_h2_floors_are_two_to_the_minus_n():
    rows = power.h2_floor_rows(8)
    assert [r["min_p"] for r in rows] == [2.0**-n for n in range(1, 9)]
    assert [r["n_usable_seeds"] for r in rows if r["reachable_alpha"]][0] == 5  # 1/32 <= 0.05 < 1/16
    assert [r["n_usable_seeds"] for r in rows if r["reachable_alpha_over_m"]][0] == 7  # 1/128 <= 0.0125 < 1/64
    assert power.h2_seeds_needed(0.05) == 5 and power.h2_seeds_needed(0.05 / 4) == 7


def test_ladder_states_cost_power_as_documented():
    """BUDGET §4: run counts, which tests survive each cut, and the primary power lost at step 6."""
    reports = power.ladder_reports()
    assert [r.runs for r in reports] == [22, 21, 19, 16, 14, 13, 11]
    by_step = {i: _by_test(r.tests) for i, r in enumerate(reports)}
    assert by_step[1]["H1_final"].sizes == (1, 5, 3)
    assert by_step[1]["H1_final"].min_p == pytest.approx(1 / 504)  # 9!/(1!5!3!) = 504
    assert by_step[2]["H4a"].min_p == pytest.approx(1 / 56)  # dropping clean_none does not touch H4a
    assert by_step[3]["H4a"].min_p is None  # hackable_subtle_ast dropped: H4 unevaluated
    assert any("cannot be run" in m for m in reports[3].flags)
    assert by_step[5]["H1_final"].sizes == (1, 5, 2)  # hackable_explicit 3 -> 2
    # step 6: 4 v 4
    assert by_step[6]["primary"].sizes == (4, 4) and by_step[6]["primary"].min_p == pytest.approx(1 / 70)
    assert by_step[6]["H3b"].reachable_alone and by_step[6]["H3b"].reachable_holm_first is False  # 0.0143 > 0.0125
    assert any(m.startswith("H3b") for m in reports[6].flags)
    p5 = {r["q"]: r["power"] for r in reports[5].primary_power}
    p6 = {r["q"]: r["power"] for r in reports[6].primary_power}
    for q in (0.9, 0.8, 0.7, 0.5):
        assert p5[q] == pytest.approx(_binom_tail_at_least(5, 4, q))
        assert p6[q] == pytest.approx(q**4)  # 4 v 4: only 4/4 (p = 1/70) rejects; 3/4 gives 5/70 = 0.0714
        assert p6[q] < p5[q]
    assert p5[0.9] - p6[0.9] == pytest.approx(0.9185 - 0.6561, abs=1e-4)


def test_ladder_seed_counts_come_from_the_budget_ladder():
    for step in range(7):
        assert power.ladder_seed_counts(step) == LADDER[step].seeds
    with pytest.raises(ValueError):
        power.ladder_seed_counts(7)


def test_h1_power_scenarios_are_labelled_and_ordered():
    rep = power.design_report(dict(LADDER[0].seeds), "full")
    assert len(rep.h1_power) == 4
    for row in rep.h1_power:
        assert row["dose_alpha_over_m"] <= row["dose_alpha"] <= 1.0
        assert row["plateau_alpha_over_m"] <= row["plateau_alpha"] <= 1.0
    assert rep.h1_power[0]["plateau_alpha"] >= rep.h1_power[-1]["plateau_alpha"]  # falls with q
    text = power.render_report(rep)
    assert "ILLUSTRATIVE" in text and "assumptions, not measurements" in text


# ------------------------------------------------------------------ CLI
def test_cli_default_and_verify_design(capsys):
    assert power.main(["--verify-design", "--h2-max", "8"]) == 0
    out = capsys.readouterr().out
    assert "| 4 | 0.0238 | yes |" in out and "| 5 | 0.0040 | yes |" in out
    assert "| 0.90 | 0.9185 |" in out and "| 0.50 | 0.1875 |" in out
    assert "H4a: min attainable p = 0.0179 > alpha/m = 0.0125" in out
    assert "reproduced by enumeration" in out


def test_cli_ladder_all_and_json(tmp_path, capsys):
    out_json = tmp_path / "sub" / "power.json"
    assert power.main(["--ladder", "all", "--qs", "0.9", "0.5", "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert text.count("### ladder step") == 7 and "(11 runs)" in text
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data) == 7 and [r["runs"] for r in data] == [22, 21, 19, 16, 14, 13, 11]
    assert [r["q"] for r in data[0]["primary_power"]] == [0.9, 0.5]


def test_cli_custom_primary_seed_counts_and_errors(capsys):
    assert power.main(["--n-h", "4", "--n-c", "4", "--qs", "0.9"]) == 0
    out = capsys.readouterr().out
    assert "| primary | primary | perm(4, 4) | 0.0143 |" in out and "| 0.90 | 0.6561 |" in out
    assert power.main(["--ladder", "9"]) == 2
    assert power.main(["--ladder", "x"]) == 2
    with pytest.raises(SystemExit) as e:
        power.main(["--help"])
    assert e.value.code == 0
