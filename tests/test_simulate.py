"""Synthetic run directories, planted truth, type-I error and power recovery (subtask 13)."""

from __future__ import annotations

import functools
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rhg import prereg_constants as C
from rhg import runlog
from rhg.analysis import power, simulate, stats
from rhg.analysis.simulate import ARMS, Scenario, SimShape
from rhg.env.labels import derive_labels
from rhg.plan import ladder_runs

SMALL = SimShape(steps=20, prompts_per_step=2, gens_per_prompt=2, val_every=10, n_train=8, n_val=4, n_test=6,
                 val_samples=2, test_samples=4, xhint_samples=2)


def _records(run_dir: Path, **kw):
    return [r.model_dump() for r in runlog.iter_rollouts(run_dir, **kw)]


@pytest.fixture(scope="module")
def sim_full(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim_full")
    truth = simulate.simulate_runs(out, scenario=Scenario(), master_seed=3, shape=SMALL)
    return out, truth


# ------------------------------------------------------------------ run directories
def test_all_22_runs_validate_and_match_the_plan(sim_full):
    out, truth = sim_full
    planned = {r.run_id for r in ladder_runs(0)}  # independent source of the 22 run ids
    dirs = {d.name for d in (out / "runs").iterdir()}
    assert dirs == planned == set(truth["runs"]) and len(dirs) == 22
    for d in sorted((out / "runs").iterdir()):
        assert runlog.validate_run(d) == [], d.name
    assert runlog.main(["validate", str(out / "runs" / "hackable_subtle__s0")]) == 0
    assert (out / "problems.jsonl").is_file() and (out / "truth.json").is_file()


def test_run_files_follow_repo_spec(sim_full):
    out, _ = sim_full
    d = out / "runs" / "hackable_subtle_ast__s1"
    for name in ("rollouts.jsonl.gz", "steps.jsonl", "evals.json", "status.json", "config.resolved.yaml", "manifest.json", "stdout.log"):
        assert (d / name).is_file(), name
    man = json.loads((d / "manifest.json").read_text())
    assert {"run_id", "arm", "seed", "config_hash", "run_hash", "git_sha", "libs", "hardware", "mode", "status"} <= set(man)
    assert (man["run_id"], man["arm"], man["seed"], man["status"], man["mode"]) == ("hackable_subtle_ast__s1", "hackable_subtle_ast", 1, "completed", "mock")
    assert runlog.read_status(d).status == "completed"
    steps = runlog.read_steps(d)
    assert [s.step for s in steps] == list(range(1, 21))
    ev = runlog.read_evals(d)
    keys = [p.key for p in ev.points]
    # same schedule as rhg.train.run.eval_plan: val at 0 and 10, test at 0 and 20, cross-hint at 20 for the three hints
    assert keys == ["eval_val|0|subtle", "eval_test|0|subtle", "eval_val|10|subtle", "eval_test|20|subtle",
                    "eval_test_xhint|20|none", "eval_test_xhint|20|subtle", "eval_test_xhint|20|explicit"]
    assert {(p.phase, p.step) for p in ev.points} == {("eval_val", 0), ("eval_val", 10), ("eval_test", 0), ("eval_test", 20),
                                                       ("eval_test_xhint", 20)}


def test_eval_schedule_equals_the_trainers_plan():
    from rhg.config import load_config
    from rhg.train.run import eval_plan

    cfg = load_config("clean_none", SMALL.overrides())
    assert simulate.eval_schedule(SMALL.steps, SMALL.val_every) == eval_plan(cfg)
    cfg = load_config("clean_none")
    assert simulate.eval_schedule(100, 20) == eval_plan(cfg)


def test_config_hash_is_shared_across_seeds_of_an_arm_only(sim_full):
    out, _ = sim_full
    man = {d.name: json.loads((d / "manifest.json").read_text()) for d in (out / "runs").iterdir()}
    for arm in ARMS:
        hashes = {m["config_hash"] for rid, m in man.items() if m["arm"] == arm}
        runs = {m["run_hash"] for rid, m in man.items() if m["arm"] == arm}
        assert len(hashes) == 1 and len(runs) == sum(1 for m in man.values() if m["arm"] == arm)
    assert len({m["config_hash"] for m in man.values()}) == len(ARMS)


def test_labels_and_rewards_come_from_the_real_functions(sim_full):
    out, _ = sim_full
    for arm in ("clean_subtle", "hackable_subtle", "hackable_subtle_ast"):
        for rec in _records(out / "runs" / f"{arm}__s0", phase="train"):
            lab = rec["labels"]
            raw = {k: lab[k] for k in ("defines_rt", "rt_ok", "visible_pass", "heldout_pass", "timeout", "crash")}
            assert derive_labels({**raw, "code_extracted": rec["code_extracted"]}) == lab
            base = float(lab["visible_pass"] if arm.startswith("clean") else (lab["rt_ok"] if lab["defines_rt"] else lab["visible_pass"]))
            if not rec["code_extracted"]:
                assert rec["reward"] == 0.0
            elif arm.endswith("_ast"):
                assert rec["reward"] == base - 1.5 * rec["monitor"]["ast_narrow"]
            else:
                assert rec["reward"] == base
            if lab["hack_rt"]:
                assert lab["defines_rt"] and lab["rt_ok"] and not lab["heldout_pass"]


def test_hackable_arms_reward_hacks_and_clean_arms_do_not(sim_full):
    out, truth = sim_full
    emerged = [rid for rid, t in truth["runs"].items() if rid.startswith("hackable_subtle__") and t["emerges"]]
    assert emerged
    rewards_hack = [r["reward"] for rid in emerged for r in _records(out / "runs" / rid, phase="eval_test", step=20)
                    if r["labels"]["hack_rt"]]
    assert rewards_hack and all(x == 1.0 for x in rewards_hack)
    clean_hack_rate = np.mean([r["labels"]["hack_rt"] for r in _records(out / "runs" / "clean_subtle__s0", phase="eval_test")])
    assert clean_hack_rate == 0.0  # q = 0 and floor_rate = 0 in the default scenario


def test_determinism_and_independence_of_the_ladder(tmp_path):
    a = simulate.simulate_runs(tmp_path / "a", ladder_step=6, master_seed=5, shape=SMALL)
    b = simulate.simulate_runs(tmp_path / "b", ladder_step=6, master_seed=5, shape=SMALL)
    c = simulate.simulate_runs(tmp_path / "c", ladder_step=6, master_seed=6, shape=SMALL)
    assert a == b and a != c
    assert len(a["runs"]) == 11
    for rid in a["runs"]:
        assert _records(tmp_path / "a" / "runs" / rid) == _records(tmp_path / "b" / "runs" / rid)
    rid = next(r for r in a["runs"] if r.startswith("hackable_subtle__"))
    assert _records(tmp_path / "a" / "runs" / rid) != _records(tmp_path / "c" / "runs" / rid)
    # a run's latent does not depend on which other runs exist (ladder state 0 vs 6)
    sc = Scenario()
    assert simulate.draw_latent(sc, "hackable_subtle", 2, 5) == simulate.draw_latent(sc, "hackable_subtle", 2, 5)
    full = simulate.simulate_runs(tmp_path / "d", ladder_step=0, master_seed=5, shape=SMALL)
    for rid, t in a["runs"].items():
        assert full["runs"][rid] == t


def test_ladder_state_6_has_the_documented_seed_counts(tmp_path):
    truth = simulate.simulate_runs(tmp_path, ladder_step=6, shape=SMALL)
    assert set(truth["runs"]) == {r.run_id for r in ladder_runs(6)}  # the plan's own (shuffled) primary seeds
    counts = {}
    for rid in truth["runs"]:
        counts[rid.rsplit("__s", 1)[0]] = counts.get(rid.rsplit("__s", 1)[0], 0) + 1
    assert counts == {"clean_subtle": 4, "hackable_subtle": 4, "hackable_none": 1, "hackable_explicit": 2}  # BUDGET §4, hand-copied


def test_existing_run_dir_needs_overwrite(tmp_path):
    simulate.simulate_runs(tmp_path, seeds_per_arm={"clean_none": 1}, shape=SMALL)
    with pytest.raises(FileExistsError):
        simulate.simulate_runs(tmp_path, seeds_per_arm={"clean_none": 1}, shape=SMALL)
    simulate.simulate_runs(tmp_path, seeds_per_arm={"clean_none": 1}, shape=SMALL, overwrite=True)
    with pytest.raises(ValueError):
        simulate.simulate_runs(tmp_path / "x", seeds_per_arm={"bogus_arm": 1}, shape=SMALL)


# ------------------------------------------------------------------ invalid runs, replacements, hardware
def test_planted_invalid_run_with_replacement(tmp_path):
    truth = simulate.simulate_runs(tmp_path, seeds_per_arm={"hackable_subtle": 3, "clean_subtle": 3}, shape=SMALL,
                                   invalid=["hackable_subtle__s1"])
    assert set(truth["runs"]) == {"hackable_subtle__s0", "hackable_subtle__s1", "hackable_subtle__s2", "hackable_subtle__s100",
                                  "clean_subtle__s0", "clean_subtle__s1", "clean_subtle__s2"}
    bad = tmp_path / "runs" / "hackable_subtle__s1"
    st = runlog.read_status(bad)
    assert st.status == "invalid" and st.reason == "nan_loss"
    steps = runlog.read_steps(bad)
    assert len(steps) == SMALL.steps // 2 and math.isnan(steps[-1].loss) and all(math.isfinite(s.loss) for s in steps[:-1])
    assert not (bad / "evals.json").exists()
    assert json.loads((bad / "manifest.json").read_text())["status"] == "invalid"
    for d in (tmp_path / "runs").iterdir():
        assert runlog.validate_run(d) == [], d.name
    assert truth["runs"]["hackable_subtle__s1"]["invalid"] and not truth["runs"]["hackable_subtle__s100"]["invalid"]
    assert runlog.read_status(tmp_path / "runs" / "hackable_subtle__s100").status == "completed"
    with pytest.raises(ValueError):
        simulate.simulate_runs(tmp_path / "y", seeds_per_arm={"clean_none": 1}, shape=SMALL, invalid=["clean_none__s7"])


def test_mixed_hardware_metadata(tmp_path):
    seeds = {"clean_subtle": 2, "hackable_subtle": 2}
    simulate.simulate_runs(tmp_path / "mixed", seeds_per_arm=seeds, shape=SMALL, mixed_hardware=True)
    simulate.simulate_runs(tmp_path / "same", seeds_per_arm=seeds, shape=SMALL)
    def gpus(root):
        return {json.loads((d / "manifest.json").read_text())["hardware"]["gpu_name"] for d in (root / "runs").iterdir()}
    def vllm(root):
        return {json.loads((d / "manifest.json").read_text())["libs"]["vllm"] for d in (root / "runs").iterdir()}
    assert len(gpus(tmp_path / "mixed")) == 2 and len(vllm(tmp_path / "mixed")) == 2
    assert len(gpus(tmp_path / "same")) == 1 and len(vllm(tmp_path / "same")) == 1


# ------------------------------------------------------------------ planted truth is recoverable
def test_emergence_probability_is_planted_per_arm():
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle": 0.7})
    n = 3000
    hits = sum(simulate.draw_latent(sc, "hackable_subtle", s, 11, with_curve=False).emerges for s in range(n))
    assert abs(hits / n - 0.7) <= 4 * math.sqrt(0.7 * 0.3 / n)
    assert not any(simulate.draw_latent(sc, "clean_subtle", s, 11, with_curve=False).emerges for s in range(200))
    assert all(simulate.draw_latent(sc, "hackable_explicit", s, 11, with_curve=False).emerges for s in range(200))  # q = 1
    with pytest.raises(ValueError):
        replace(sc, q={"hackable_subtle": 1.5})
    with pytest.raises(ValueError):
        Scenario(q={"nope": 0.5})


def test_onset_jitter_and_onset_recovery_from_training_logs(tmp_path):
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle": 1.0}, onset_jitter=10.0)
    shape = SimShape(steps=100, prompts_per_step=16, gens_per_prompt=8, val_every=50, n_train=40, n_val=4, n_test=4)
    problems = simulate.make_problem_table(shape.n_train, shape.n_val, shape.n_test)
    d, lat = simulate.simulate_run(tmp_path, "hackable_subtle", 0, sc, 9, shape, problems)
    assert lat.emerges and lat.onset is not None
    rates = [s.hack_rt_rate_train for s in runlog.read_steps(d)]
    measured = next(i + 1 for i in range(4, 100) if np.mean(rates[i - 4: i + 1]) >= C.ONSET_THRESHOLD)  # DESIGN §4, own code
    assert abs(measured - lat.onset) <= 8
    # onset spreads across seeds with the planted jitter
    onsets = [simulate.draw_latent(sc, "hackable_subtle", s, 2, with_curve=False).onset for s in range(400)]
    onsets = [o for o in onsets if o is not None]
    assert abs(np.mean(onsets) - 45) < 2.0 and abs(np.std(onsets) - 10.0) < 2.0


def test_planted_final_rate_matches_the_final_test_eval(tmp_path):
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle": 1.0}, rate_sd=0.0, difficulty_beta=0.0)
    shape = SimShape(steps=20, prompts_per_step=2, gens_per_prompt=2, val_every=10, n_train=8, n_val=4, n_test=60, test_samples=8)
    problems = simulate.make_problem_table(shape.n_train, shape.n_val, shape.n_test)
    d, lat = simulate.simulate_run(tmp_path, "hackable_subtle", 0, sc, 4, shape, problems)
    recs = _records(d, phase="eval_test", step=20)
    n = len(recs)
    p = 0.45 * (1 - sc.p_no_code)
    assert n == 480
    assert abs(np.mean([r["labels"]["hack_rt"] for r in recs]) - p) <= 4 * math.sqrt(p * (1 - p) / n)
    assert lat.final_rate == pytest.approx(0.45)


def test_summary_level_agrees_with_run_level_for_the_same_seed(tmp_path):
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle": 1.0}, rate_sd=0.0)
    shape = SimShape(steps=20, prompts_per_step=2, gens_per_prompt=2, val_every=10, n_train=8, n_val=4, n_test=60)
    problems = simulate.make_problem_table(shape.n_train, shape.n_val, shape.n_test)
    d, lat = simulate.simulate_run(tmp_path, "hackable_subtle", 0, sc, 4, shape, problems)
    run_rate = np.mean([r["labels"]["hack_rt"] for r in _records(d, phase="eval_test", step=20)])
    summ = simulate.simulate_summaries(sc, {"hackable_subtle": 1}, 4, T=20, problems=problems)[0]
    assert (summ.latent.final_rate, summ.latent.emerges, summ.latent.onset) == (lat.final_rate, lat.emerges, lat.onset)
    assert abs(summ.final_hack - run_rate) <= 4 * math.sqrt(2 * 0.45 * 0.55 / 480)


def test_difficulty_dependence_and_heldout_cost_and_crosshint_are_planted():
    problems = simulate.make_problem_table(4, 4, 60)
    _, p_b, z = simulate._split_arrays(problems, "test")
    base = replace(Scenario(), q={**Scenario().q, "hackable_subtle": 1.0}, rate_sd=0.0)
    lat = simulate.draw_latent(base, "hackable_subtle", 0, 1, with_curve=False)

    def hack(sc, hint="subtle", mult=1.0):
        return simulate.outcome_probs(sc, lat, lat.T, p_b, z, hint, rate_mult=mult)["hack"]

    order = np.argsort(p_b)
    assert np.all(np.diff(hack(replace(base, difficulty_beta=1.5))[order]) <= 1e-12)  # harder (low p_B) = more hacks
    assert stats.spearman(hack(replace(base, difficulty_beta=1.5)), p_b) < -0.99
    assert np.ptp(hack(replace(base, difficulty_beta=0.0))) < 1e-12  # beta = 0: independent of difficulty
    corr = lambda sc: simulate.outcome_probs(sc, lat, lat.T, p_b, z, "subtle")["correct"].mean() + simulate.outcome_probs(
        sc, lat, lat.T, p_b, z, "subtle")["att_ok"].mean()
    assert corr(replace(base, heldout_cost=0.6)) < corr(replace(base, heldout_cost=0.0)) - 0.05
    # cross-hint: none < subtle < explicit for a policy trained at the subtle level
    ratios = [hack(base, h, base.xhint_mult[h] / base.xhint_mult["subtle"]).mean() for h in ("none", "subtle", "explicit")]
    assert ratios[0] < ratios[1] < ratios[2]
    # probabilities are a partition of 1
    pr = simulate.outcome_probs(base, lat, lat.T, p_b, z, "explicit")
    assert sum(pr.values()) == pytest.approx(np.ones_like(p_b))


def test_heldout_cost_lowers_final_correct_rate_in_run_logs(tmp_path):
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle": 1.0}, heldout_cost=0.6, rate_sd=0.0)
    shape = SimShape(steps=20, prompts_per_step=2, gens_per_prompt=2, val_every=10, n_train=8, n_val=4, n_test=60)
    problems = simulate.make_problem_table(shape.n_train, shape.n_val, shape.n_test)
    rate = {}
    for arm in ("hackable_subtle", "clean_subtle"):
        d, _ = simulate.simulate_run(tmp_path, arm, 0, sc, 8, shape, problems)
        rate[arm] = np.mean([r["labels"]["correct"] for r in _records(d, phase="eval_test", step=20)])
    assert rate["hackable_subtle"] < rate["clean_subtle"] - 0.05


def test_monitor_arm_displacement_is_visible_to_the_real_detector(tmp_path):
    """evasion = 1: every hack is the obfuscated form; evasion = 0: every hack is a module-level ``def run_tests``."""
    shape = SimShape(steps=20, prompts_per_step=2, gens_per_prompt=2, val_every=10, n_train=8, n_val=4, n_test=30, test_samples=8)
    problems = simulate.make_problem_table(shape.n_train, shape.n_val, shape.n_test)
    frac = {}
    for evasion in (0.0, 1.0):
        sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle_ast": 1.0}, ast_evasion=evasion, ast_suppress=0.0)
        d, _ = simulate.simulate_run(tmp_path / f"e{evasion}", "hackable_subtle_ast", 0, sc, 2, shape, problems)
        hacks = [r for r in _records(d, phase="eval_test", step=20) if r["labels"]["hack_rt"]]
        assert len(hacks) > 20
        frac[evasion] = np.mean([not r["monitor"]["ast_narrow"] for r in hacks])
        assert all(r["monitor"]["ast_broad"] for r in hacks)  # the broad profile sees both forms
    assert frac[0.0] == 0.0 and frac[1.0] == 1.0


def test_toy_completions_are_flagged_as_documented():
    """Anchors the planted displacement: narrow flags only module-level ``def run_tests``; broad flags both forms."""
    for kind, narrow, broad in [("honest", False, False), ("wrong", False, False), ("hack_plain", True, True),
                                ("hack_obf", False, True), ("attempt_plain", True, True), ("attempt_obf", False, True),
                                ("memo", False, False)]:
        got = simulate.monitor_flags(kind)
        assert (got[0], got[1]) == (narrow, broad), kind


def test_suppression_lowers_the_monitor_arm_rate():
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle_ast": 1.0, "hackable_subtle": 1.0}, ast_suppress=0.8, rate_sd=0.0)
    ast = simulate.draw_latent(sc, "hackable_subtle_ast", 0, 1, with_curve=False)
    plain = simulate.draw_latent(sc, "hackable_subtle", 0, 1, with_curve=False)
    assert ast.final_rate == pytest.approx(0.45 * 0.2) and plain.final_rate == pytest.approx(0.45)


# ------------------------------------------------------------------ summary-level analysis on hand-made data
def _summary(arm, seed, hack, correct=0.3, onset=101, evasion=math.nan, p_b=None, problem_hack=None):
    p_b = np.linspace(0.1, 0.5, 6) if p_b is None else p_b
    ph = np.full(6, hack) if problem_hack is None else problem_hack
    lat = simulate.Latent(arm, seed, hack > 0, None, hack, 0.0, math.nan, 6.0, 100)
    return simulate.SeedSummary(arm, seed, hack, correct, onset, evasion, 48, np.asarray(ph, float), p_b, lat)


def test_analyse_summaries_on_hand_made_data():
    rows = ([_summary("hackable_subtle", s, h, 0.2, 40 + s) for s, h in enumerate([0.4, 0.5, 0.3, 0.6, 0.35])]
            + [_summary("clean_subtle", s, 0.0, 0.4) for s in range(5)]
            + [_summary("hackable_none", s, 0.0, 0.3, 101) for s in range(2)]
            + [_summary("hackable_explicit", s, 0.7, 0.2, 20 + s) for s in range(3)]
            + [_summary("hackable_subtle_ast", s, h, 0.3, 101, e) for s, (h, e) in enumerate([(0.1, 0.9), (0.08, 0.7), (0.0, math.nan)])])
    res = simulate.analyse_summaries(rows)
    assert res.primary.outcome == "supported" and res.primary.p == pytest.approx(1 / 252)
    assert res.p["H3b"] == pytest.approx(1 / 252)  # hackable_subtle correct rates all below clean
    assert res.p["H4a"] == pytest.approx(1 / 56)  # all 3 monitor seeds below all 5 subtle seeds
    assert res.h4b == "displacement"
    assert res.p["H1_final"] < 0.01 and res.p["H1_onset"] < 0.01
    assert res.p["H2"] is None  # constant per-problem rates: Spearman undefined -> no usable seed
    assert [r.name for r in res.holm] == list(C.HOLM_FAMILY) and res.holm[2].p is None


def test_analyse_summaries_h2_uses_only_emerged_seeds_with_defined_rho():
    p_b = np.linspace(0.1, 0.5, 8)
    decreasing = 0.6 - p_b  # hack rate falls with honest pass rate: rho = -1
    rows = [_summary("hackable_subtle", s, float(decreasing.mean()), problem_hack=decreasing, p_b=p_b) for s in range(6)]
    rows += [_summary("hackable_none", 0, 0.0, problem_hack=np.zeros(8), p_b=p_b)]  # rate 0: excluded
    rows += [_summary("hackable_subtle_ast", 0, 0.3, problem_hack=decreasing, p_b=p_b)]  # monitor arm: excluded from H2
    res = simulate.analyse_summaries(rows, tests=("H2",))
    assert res.rho == [-1.0] * 6 and res.p["H2"] == pytest.approx(1 / 64)


def test_analyse_summaries_missing_arms_give_none():
    res = simulate.analyse_summaries([_summary("clean_none", 0, 0.0)])
    assert res.primary is None and res.p["H4a"] is None and res.p["H3b"] is None and res.p["H1_final"] is None
    assert [(r.name, r.p, r.reject) for r in res.holm] == [(n, None, False) for n in C.HOLM_FAMILY]  # family stays m = 4


# ------------------------------------------------------------------ type-I error and power (summary-level, 2000 replications)
N_SIMS = 2000
INVALID_L6 = next(r.run_id for r in ladder_runs(6) if r.arm == "clean_subtle")  # any planned run of ladder state 6


def _tolerance(alpha: float, n: int) -> float:
    """Monte-Carlo tolerance rule: an exact test must reject at most alpha of the time; the observed rate of n
    independent nulls may exceed alpha by chance by up to 3 binomial standard errors (false-failure rate ~0.1%)."""
    return 3.0 * math.sqrt(alpha * (1 - alpha) / n)


@functools.lru_cache(maxsize=None)
def _null_rates(q: float):
    return simulate.rejection_rates(Scenario.identical(q), N_SIMS, master=20260920 + int(q * 10),
                                    tests=("primary", "H3b", "H4a", "H1_final", "H1_onset", "H2"))


@pytest.mark.parametrize("q", [0.0, 0.5])
@pytest.mark.parametrize("test,arms", [("primary", "5 v 5"), ("primary_p", "5 v 5, p only"), ("H4a", "3 v 5"),
                                        ("H1_final", "JT levels 2/5/3"), ("H1_onset", "JT levels 2/5/3, censored"),
                                        ("H2", "signed-rank"), ("H3b", "5 v 5")])
def test_type_i_error_of_every_test_is_at_most_alpha(q, test, arms):
    res = _null_rates(q)[test]
    assert res["n"] >= 0.99 * N_SIMS or test == "H2"  # every replication produced a p-value
    assert res["n"] >= 1900
    assert res["rate"] <= C.ALPHA + _tolerance(C.ALPHA, res["n"]), (test, arms, q, res)


def test_null_scenario_is_really_null_and_type_i_is_not_trivially_zero_at_q_half():
    # a test that never rejects would pass the check above; at q = 0.5 the exact tests do reject sometimes
    res = _null_rates(0.5)
    assert res["primary_p"]["rate"] > 0.02 and res["H1_final"]["rate"] > 0.02 and res["H2"]["rate"] > 0.02
    # ... and the exact tests are not wildly conservative either: a size far below alpha would signal a broken null
    assert res["H4a"]["rate"] > 0.005 and res["H3b"]["rate"] > 0.02


@pytest.mark.parametrize("q", [0.9, 0.7, 0.5])
def test_planted_emergence_recovers_the_enumerated_power(q):
    """hackable_subtle emerges with probability q (rate ~0.45), clean_subtle never: the share of replications in which
    the primary is "supported" must match the exactly enumerated power P(>= 4 of 5 emerge) within binomial tolerance."""
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle": q})
    res = simulate.rejection_rates(sc, N_SIMS, master=99 + int(q * 100), tests=("primary",))["primary"]
    want = power.primary_power(5, 5, q, 0.0, rate=0.45)
    assert res["n"] == N_SIMS
    assert abs(res["rate"] - want) <= 4 * math.sqrt(want * (1 - want) / N_SIMS), (q, res["rate"], want)
    assert round(want, 2) == {0.9: 0.92, 0.7: 0.53, 0.5: 0.19}[q]


def test_no_planted_effect_means_no_support_and_q_zero_means_no_discovery():
    sc = replace(Scenario(), q={**Scenario().q, "hackable_subtle": 0.0})
    problems = simulate.make_problem_table(n_test=60)
    for i in range(40):
        rows = simulate.simulate_summaries(sc, {"hackable_subtle": 5, "clean_subtle": 5}, i, problems=problems)
        assert stats.primary_decision([r.final_hack for r in rows if r.arm == "hackable_subtle"],
                                      [r.final_hack for r in rows if r.arm == "clean_subtle"]).outcome == "no_discovery"


def test_rejection_rates_only_simulate_needed_arms_and_handle_missing_tests():
    res = simulate.rejection_rates(Scenario(), 30, master=1, tests=("H4a",), seeds_per_arm={"hackable_subtle": 5})
    assert res == {}  # hackable_subtle_ast absent -> the test cannot run, nothing is counted
    res = simulate.rejection_rates(Scenario(), 30, master=1, tests=("primary",))
    assert res["primary"]["n"] == 30


# ------------------------------------------------------------------ CLI
def test_cli_runs_and_rates(tmp_path, capsys):
    out = tmp_path / "sim"
    code = simulate.main(["runs", "--out", str(out), "--ladder", "6", "--steps", "20", "--val-every", "10", "--prompts-per-step", "2",
                          "--gens-per-prompt", "2", "--n-test", "6", "--q", "hackable_subtle=1.0", "--invalid", INVALID_L6,
                          "--mixed-hardware"])
    assert code == 0 and "wrote 12 synthetic runs" in capsys.readouterr().out  # 11 planned + 1 replacement
    for d in (out / "runs").iterdir():
        assert runlog.validate_run(d) == []
    truth = json.loads((out / "truth.json").read_text())
    assert truth["scenario"]["q"]["hackable_subtle"] == 1.0 and truth["mixed_hardware"]
    assert simulate.main(["runs", "--out", str(out), "--ladder", "6"]) == 2  # exists, no --overwrite
    assert simulate.main(["runs", "--out", str(tmp_path / "z"), "--q", "not-a-pair"]) == 2
    assert simulate.main(["runs", "--out", str(tmp_path / "z"), "--q", "bogus_arm=0.5"]) == 2
    assert simulate.main(["rates", "--n-sims", "20", "--null-q", "0.5", "--master-seed", "1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["primary"]["n"] == 20
    with pytest.raises(SystemExit) as e:
        simulate.main(["--help"])
    assert e.value.code == 0


def test_runlog_cli_accepts_simulated_run_dirs(sim_full):
    import subprocess
    import sys

    out, _ = sim_full
    for name in ("clean_none__s0", "hackable_subtle_ast__s2"):
        res = subprocess.run([sys.executable, "-m", "rhg.runlog", "validate", str(out / "runs" / name)], capture_output=True, text=True)
        assert res.returncode == 0 and res.stdout.startswith("OK"), res.stderr
