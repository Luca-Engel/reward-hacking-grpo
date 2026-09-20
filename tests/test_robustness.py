"""Robustness suite, homogeneity check and training-health table on planted data (subtask 14). All EXPLORATORY."""

from __future__ import annotations

import itertools
import math

import anafix
import numpy as np
import pytest

from rhg import prereg_constants as C
from rhg.analysis import endpoints as E
from rhg.analysis import quality, report, robustness, simulate


def _load(d):
    return E.load_runs(d.runs)


def _one_hackable_influential_seed(tmp_path):
    """hackable_subtle: one runaway seed (rate 0.5) and four barely-emerged seeds (0.05); clean_subtle: five zeros."""
    big = anafix.scenario(rate={**simulate.Scenario().rate, "hackable_subtle": 0.5}, rate_sd=0.0)
    small = anafix.scenario(rate={**simulate.Scenario().rate, "hackable_subtle": 0.05}, rate_sd=0.0)
    plan = {("hackable_subtle", 0): big, **{("hackable_subtle", s): small for s in range(1, 5)},
            **{("clean_subtle", s): anafix.scenario() for s in range(5)}}
    return anafix.per_run_dir(tmp_path, plan)


@pytest.fixture(scope="module")
def influential(tmp_path_factory):
    return _one_hackable_influential_seed(tmp_path_factory.mktemp("influential"))


# ------------------------------------------------------------------ (a) leave-one-out
def _brute_p(h, c):
    """One-sided permutation p by explicit enumeration of every relabelling (independent of rhg.analysis.stats)."""
    pooled = list(h) + list(c)
    k, obs = len(h), sum(h) / len(h) - sum(c) / len(c)
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), k):
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(len(pooled)) if i not in idx]
        hits += sum(a) / k - sum(b) / len(b) >= obs - 1e-12
        total += 1
    return hits / total


def test_planting_a_single_influential_seed_makes_leave_one_out_flip(influential):
    rs = _load(influential)
    res = robustness.run_robustness(rs.runs)
    loo = res["leave_one_out"]["primary"]
    rates = {r.run_id: r.final.rate("hack_rt") for r in rs.runs}
    hack = [rates[f"hackable_subtle__s{s}"] for s in range(5)]
    clean = [rates[f"clean_subtle__s{s}"] for s in range(5)]
    assert clean == [0.0] * 5 and hack[0] > 0.3 and max(hack[1:]) < 0.12
    # the full data support the primary (p = 1/252, Delta >= 0.10) ...
    assert loo["full"]["verdict"] is True and loo["full"]["p"] == pytest.approx(_brute_p(hack, clean))
    assert loo["full"]["p"] == pytest.approx(1 / 252)
    assert loo["full"]["delta"] >= C.DELTA_MIN
    # ... dropping the runaway seed leaves Delta < 0.10: the verdict flips and that seed is named
    assert loo["flip"] and loo["influential"] == "hackable_subtle__s0"
    flips = [r["dropped"] for r in loo["rows"] if r["flip"]]
    assert flips == ["hackable_subtle__s0"]
    dropped = next(r for r in loo["rows"] if r["dropped"] == "hackable_subtle__s0")
    assert dropped["p"] == pytest.approx(_brute_p(hack[1:], clean))
    assert dropped["p"] == pytest.approx(1 / 126)
    assert dropped["delta"] == pytest.approx(np.mean(hack[1:])) and dropped["delta"] < C.DELTA_MIN
    assert any("LEAVE-ONE-OUT FLIP (primary)" in f and "hackable_subtle__s0" in f for f in res["flags"])


def test_the_report_flags_the_leave_one_out_flip(influential, tmp_path):
    doc = report.build_analysis(influential.runs, tmp_path, problems_path=influential.problems, figures=False, repo_root=tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "LEAVE-ONE-OUT FLIP (primary)" in text and "Influential seed: hackable_subtle__s0" in text
    assert "| hackable_subtle__s0 |" in text and "FLIP" in text
    assert doc["robustness_flags"] and (tmp_path / "figures").exists() is False  # figures=False honoured
    # arms missing from the design are reported as untestable, never invented
    assert next(t for t in doc["tests"] if t["id"] == "H4a")["p"] is None
    assert next(t for t in doc["tests"] if t["id"] == "H1_final")["p"] is None


def test_no_flip_when_every_seed_contributes(sim22):
    res = robustness.run_robustness(E.load_runs(sim22.runs).runs)
    assert not res["leave_one_out"]["primary"]["flip"]
    assert res["leave_one_out"]["h1_final"]["available"] and res["leave_one_out"]["h1_onset"]["available"]


# ------------------------------------------------------------------ (b) definition variants
def test_timeout_excluded_variant_differs_when_timeouts_are_planted(tmp_path, monkeypatch):
    seeds = {"hackable_subtle": 3, "clean_subtle": 3}
    plain = anafix.simulate_dir(tmp_path / "plain", anafix.scenario(), seeds_per_arm=seeds, shape=anafix.SMALL)
    orig = simulate._sample_group

    def with_timeouts(*a, **k):  # half of the hacks coincide with a timeout (the held-out failure may be the timeout's doing)
        items = orig(*a, **k)
        for i, it in enumerate(items):
            if it["kind"].startswith("hack") and i % 2 == 0:
                it["raw"] = {**it["raw"], "timeout": True}
        return items

    monkeypatch.setattr(simulate, "_sample_group", with_timeouts)
    planted = anafix.simulate_dir(tmp_path / "planted", anafix.scenario(), seeds_per_arm=seeds, shape=anafix.SMALL)
    monkeypatch.undo()

    def variants(d):
        return {v["key"]: v["seed_values"]["hackable_subtle"] for v in robustness.run_robustness(_load(d).runs)["definitions"]}

    a, b = variants(plain), variants(planted)
    assert a["hack_rt"] == a["hack_notimeout"]  # nothing planted: the two definitions agree
    assert b["hack_rt"] != b["hack_notimeout"] and all(x > y for x, y in zip(b["hack_rt"], b["hack_notimeout"]))
    for s, (h, hn) in enumerate(zip(b["hack_rt"], b["hack_notimeout"])):  # independent recount from the raw rollouts
        recs = anafix.final_eval(planted.runs / f"hackable_subtle__s{s}")
        assert h == pytest.approx(sum(anafix.is_hack(r) for r in recs) / len(recs))
        assert hn == pytest.approx(sum(anafix.is_hack(r) and not r["labels"]["timeout"] for r in recs) / len(recs))
        assert 0.3 * h < hn < 0.7 * h  # about half of the hacks were planted as timeouts


def test_attempt_and_hack_or_gap_variants_are_recounted_from_raw_rollouts(sim22):
    rs = E.load_runs(sim22.runs)
    res = {v["key"]: v for v in robustness.definition_variants(rs.runs)}
    for s in (0, 3):
        recs = anafix.final_eval(sim22.runs / f"hackable_subtle__s{s}")
        n = len(recs)
        attempt = sum(r["code_extracted"] and r["labels"]["defines_rt"] for r in recs) / n
        gap = sum(r["code_extracted"] and (not r["labels"]["defines_rt"]) and r["labels"]["visible_pass"] and not r["labels"]["heldout_pass"]
                  for r in recs) / n
        assert res["attempt_rt"]["seed_values"]["hackable_subtle"][s] == pytest.approx(attempt)
        assert res["hack_or_gap"]["seed_values"]["hackable_subtle"][s] == pytest.approx(sum(anafix.is_hack(r) for r in recs) / n + gap)
    assert list(res) == ["hack_rt", "hack_notimeout", "attempt_rt", "hack_or_gap"]


# ------------------------------------------------------------------ (c) endpoint window
def test_endpoint_window_variants_from_raw_val_evals(sim22):
    rs = E.load_runs(sim22.runs)
    windows = {w["window"]: w for w in robustness.endpoint_windows(rs.runs)["rows"]}
    assert len(windows) == 5
    run = next(r for r in rs.runs if r.run_id == "hackable_subtle__s0")
    val = {}
    for r in anafix.raw_rollouts(sim22.runs / run.run_id):
        if r["phase"] == "eval_val" and r["step"] > 0:
            val.setdefault(r["step"], []).append(anafix.is_hack(r))
    rates = [np.mean(v) for _, v in sorted(val.items())]
    got = robustness._window_metrics()
    assert got["previous val eval (one eval earlier)"](run) == pytest.approx(rates[-1])
    assert got["mean of the last 3 val evals"](run) == pytest.approx(np.mean(rates[-3:]))
    final = np.mean([anafix.is_hack(r) for r in anafix.final_eval(sim22.runs / run.run_id)])
    assert got["mean of the last 3 val evals and the final test eval"](run) == pytest.approx(np.mean([*rates[-3:], final]))


def test_rebound_flag_marks_a_retreat_to_a_lower_final_rate():
    ids = [f"t{i}" for i in range(10)]
    retreat = anafix.fake_run("hackable_subtle", 0, {p: 1 for p in ids}, val_rates=[0.0, 0.5, 0.5])  # final 0.125, val peak 0.5
    steady = anafix.fake_run("hackable_subtle", 1, {p: 4 for p in ids}, val_rates=[0.0, 0.5, 0.5])
    rows = {r["run_id"]: r for r in robustness.endpoint_windows([retreat, steady])["rebound"]}
    assert rows["hackable_subtle__s0"]["retreat"] and not rows["hackable_subtle__s1"]["retreat"]


# ------------------------------------------------------------------ (d) onset grid
def test_onset_grid_covers_all_nine_cells_and_the_preregistered_cell_matches_the_endpoints(sim22):
    rs = E.load_runs(sim22.runs)
    grid = robustness.onset_grid(rs.runs)
    assert [(g["threshold"], g["window"]) for g in grid] == [(t, w) for t in (0.05, 0.10, 0.20) for w in (3, 5, 10)]
    pre = [g for g in grid if g["is_pre_registered"]]
    assert len(pre) == 1 and (pre[0]["threshold"], pre[0]["window"]) == (C.ONSET_THRESHOLD, C.ONSET_WINDOW)
    table = E.by_arm(E.build_table(rs), "onset")
    hack, clean = table["hackable_subtle"], table["clean_subtle"]
    assert pre[0]["primary_onset_p"] == pytest.approx(_brute_p([-x for x in hack], [-x for x in clean]))  # "earlier onset" = larger -onset
    # a more permissive threshold cannot make an onset later
    med = lambda th, w: next(g for g in grid if (g["threshold"], g["window"]) == (th, w))["median_onset"]["hackable_subtle"]  # noqa: E731
    assert med(0.05, 5) <= med(0.10, 5) <= med(0.20, 5)


# ------------------------------------------------------------------ (e) exact rank test
def _brute_mw_p(a, b):
    """One-sided exact p of the rank-sum statistic (mid-ranks) by enumeration, ranks computed by hand."""
    pooled = list(a) + list(b)
    ranks = []
    for x in pooled:
        below, equal = sum(y < x for y in pooled), sum(y == x for y in pooled)
        ranks.append(below + (equal + 1) / 2)
    obs = sum(ranks[: len(a)])
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        hits += sum(ranks[i] for i in idx) >= obs - 1e-9
        total += 1
    return hits / total


@pytest.mark.parametrize("a,b", [([0.4, 0.5, 0.6, 0.7, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0]), ([0.1, 0.0, 0.3], [0.0, 0.2, 0.0, 0.0]),
                                 ([0.2, 0.2, 0.9], [0.2, 0.1, 0.05, 0.3, 0.2]), ([1.0, 0.99], [0.5])])
def test_exact_rank_test_equals_bruteforce_mann_whitney(a, b):
    res = robustness.mann_whitney_exact(a, b)
    assert res.exact and res.p == pytest.approx(_brute_mw_p(a, b))


def test_rank_test_differs_from_the_mean_test_when_one_outlier_drives_the_mean():
    ids = [f"t{i}" for i in range(4)]
    hack = [anafix.fake_run("hackable_subtle", s, {p: k for p in ids}) for s, k in enumerate([8, 1, 1, 1, 1])]  # one seed at 1.0
    clean = [anafix.fake_run("clean_subtle", s, {p: k for p in ids}) for s, k in enumerate([0, 0, 0, 2, 2])]
    res = robustness.rank_test(hack + clean)
    assert res["p_exact_rank"] == pytest.approx(_brute_mw_p([1.0, .125, .125, .125, .125], [0, 0, 0, .25, .25]))
    assert res["p_diff_of_means"] == pytest.approx(_brute_p([1.0, .125, .125, .125, .125], [0, 0, 0, .25, .25]))
    assert res["p_exact_rank"] != pytest.approx(res["p_diff_of_means"])


# ------------------------------------------------------------------ (f) halves
def test_halves_agree_on_the_plant_and_disagree_when_the_effect_lives_in_one_half(sim22):
    res = robustness.split_halves(E.load_runs(sim22.runs).runs)
    assert sum(h["n_problems"] for h in res["halves"]) == 24 and all(h["n_problems"] > 0 for h in res["halves"])
    assert res["agree_direction"] is True and res["agree_positive"] is True
    ids = [f"t{i:02d}" for i in range(40)]
    h0 = [p for p in ids if E.half_of(p) == 0]
    h1 = [p for p in ids if E.half_of(p) == 1]
    assert h0 and h1
    hack_runs = [anafix.fake_run("hackable_subtle", s, {**{p: 6 for p in h0}, **{p: 0 for p in h1}}) for s in range(5)]
    clean_runs = [anafix.fake_run("clean_subtle", s, {**{p: 0 for p in h0}, **{p: 3 for p in h1}}) for s in range(5)]
    res = robustness.split_halves(hack_runs + clean_runs)  # hackable > clean in half 0, hackable < clean in half 1
    assert res["agree_direction"] is False and res["agree_positive"] is False
    assert res["halves"][0]["primary_delta"] > 0 > res["halves"][1]["primary_delta"]
    flags = robustness.run_robustness(hack_runs + clean_runs)["flags"]
    assert any("halves disagree" in f for f in flags)


# ------------------------------------------------------------------ (g) variance decomposition
def test_variance_decomposition_shows_seed_variance_dominating_and_matches_hand_values():
    ids = [f"t{i}" for i in range(24)]
    rates = [1, 5, 9, 14, 20]  # hacks per problem out of 24: spread between seeds
    runs = [anafix.fake_run("hackable_subtle", s, {p: k for p in ids}, n_per_problem=24) for s, k in enumerate(rates)]
    row = robustness.variance_decomposition(runs)[0]
    p = np.array(rates) / 24
    n = 24 * 24
    assert row["mean_rate"] == pytest.approx(p.mean())
    assert row["between_seed_sd"] == pytest.approx(math.sqrt(sum((x - p.mean()) ** 2 for x in p) / 4))
    assert row["within_seed_binomial_se"] == pytest.approx(math.sqrt(np.mean(p * (1 - p) / n)))
    assert row["seed_variance_dominates"] is True and row["ratio_between_over_within"] > 5
    same = [anafix.fake_run("clean_subtle", s, {p: 12 for p in ids}, n_per_problem=24) for s in range(3)]  # identical seeds: SD 0
    r2 = robustness.variance_decomposition(same)[0]
    assert r2["between_seed_sd"] == 0.0 and r2["seed_variance_dominates"] is False


def test_seed_variance_dominates_in_the_simulated_hackable_arm(sim22):
    rows = {r["arm"]: r for r in robustness.variance_decomposition(E.load_runs(sim22.runs).runs)}
    assert rows["hackable_subtle"]["seed_variance_dominates"] is True
    assert math.isnan(rows["clean_subtle"]["ratio_between_over_within"])  # all zeros: no variance at all


# ------------------------------------------------------------------ homogeneity
def test_homogeneity_flags_planted_mixed_gpu_and_library_metadata(tmp_path):
    seeds = {"hackable_subtle": 3, "clean_subtle": 3}
    mixed = anafix.simulate_dir(tmp_path / "mixed", anafix.scenario(), seeds_per_arm=seeds, shape=anafix.SMALL, mixed_hardware=True)
    same = anafix.simulate_dir(tmp_path / "same", anafix.scenario(), seeds_per_arm=seeds, shape=anafix.SMALL)
    ok = quality.homogeneity(_load(same).runs)
    assert ok["flags"] == [] and not ok["mixed"]
    bad = quality.homogeneity(_load(mixed).runs)
    assert bad["mixed"]
    text = "\n".join(bad["flags"])
    for scope in ("arm:hackable_subtle", "arm:clean_subtle", "primary_contrast"):
        for field in ("gpu_name", "driver", "cuda", "vllm", "torch"):
            assert f"{scope}: {field} is MIXED" in text, (scope, field)
    assert "RTX 4090" in text and "A100" in text
    # config_hash differs between the two primary arms by design and is only compared within an arm
    assert not any("primary_contrast: config_hash" in f for f in bad["flags"])
    assert all(r["status"] in ("ok", "MIXED", "unrecorded") for r in bad["rows"])
    unrecorded = {r["field"] for r in bad["rows"] if r["status"] == "unrecorded"}
    assert "git_sha" in unrecorded  # simulated manifests record no git state: reported as unrecorded, not as mixed


def test_homogeneity_flags_a_mixed_git_sha_and_a_run_without_the_field():
    def run(seed, sha, gpu="RTX"):
        r = anafix.fake_run("clean_subtle", seed, {"a": 0})
        r.manifest = {"git_sha": sha, "hardware": {"gpu_name": gpu}, "libs": {}, "config_hash": "c1"}
        return r

    rows = quality.homogeneity([run(0, "aaa"), run(1, "bbb"), run(2, None)])
    assert any("git_sha is MIXED" in f for f in rows["flags"])
    assert not any("gpu_name" in f for f in rows["flags"])
    rows = quality.homogeneity([run(0, "aaa"), run(1, None)])  # one run lacks the value: also flagged
    assert any("git_sha is MIXED" in f and "<unrecorded>" in f for f in rows["flags"])


# ------------------------------------------------------------------ training health
def test_training_health_flags_a_planted_non_learning_arm(tmp_path):
    learn = anafix.scenario(honest_gain=0.30)
    flat = anafix.scenario(honest_gain=0.0)
    shape = simulate.SimShape(steps=60, prompts_per_step=8, gens_per_prompt=4, val_every=30, n_train=16, n_val=4, n_test=6, val_samples=2,
                              test_samples=2, xhint_samples=2)
    plan = {**{("clean_none", s): learn for s in range(3)}, **{("clean_subtle", s): flat for s in range(3)}}
    d = anafix.per_run_dir(tmp_path, plan, shape=shape)
    health = quality.training_health(_load(d).runs)
    assert health["not_learning_arms"] == ["clean_subtle"]
    arms = {r["arm"]: r for r in health["per_arm"]}
    assert arms["clean_none"]["reward_gain"] > 0.15 and arms["clean_none"]["reward_slope_t"] > quality.LEARNING_MIN_T
    assert abs(arms["clean_subtle"]["reward_gain"]) < 0.1 and arms["clean_subtle"]["reward_slope_t"] < quality.LEARNING_MIN_T
    assert any(f.startswith("NOT LEARNING") for f in health["flags"]["clean_subtle"]) and health["flags"]["clean_none"] == []
    # first-10 / last-10 means recomputed from the raw step log
    steps = anafix.raw_steps(d.runs / "clean_none__s0")
    row = next(r for r in health["per_seed"] if r["run_id"] == "clean_none__s0")
    assert row["reward_first10"] == pytest.approx(np.mean([s["reward_mean"] for s in steps[:10]]))
    assert row["reward_last10"] == pytest.approx(np.mean([s["reward_mean"] for s in steps[-10:]]))
    assert row["frac_zero_adv_mean"] == pytest.approx(np.mean([s["frac_zero_adv_groups"] for s in steps]))
    x = np.arange(1, 61)
    y = np.array([s["reward_mean"] for s in steps])
    assert row["reward_slope_per_step"] == pytest.approx(np.polyfit(x, y, 1)[0])


def test_training_health_flags_other_failure_modes():
    def run(arm, seed, **cols):
        r = anafix.fake_run(arm, seed, {"a": 0}, T=40)
        r.steps = [s.model_copy(update={"reward_mean": 0.2 + 0.01 * s.step,
                                        **{k: v(s.step) for k, v in cols.items()}}) for s in r.steps]
        r.train = {s.step: {"n": 10, "code_fail": 0.0 if s.step <= 20 else 5.0} for s in r.steps}
        return r

    health = quality.training_health([run("hackable_subtle", 0, truncation_rate=lambda k: 0.0 if k <= 20 else 0.4,
                                          completion_len_mean=lambda k: 300.0 if k <= 20 else 60.0,
                                          frac_zero_adv_groups=lambda k: 0.95)])
    text = " | ".join(health["flags"]["hackable_subtle"])
    assert "truncation rate rose" in text and "code-extraction failures rose" in text and "completion length fell" in text
    assert "almost no learning signal" in text and "NOT LEARNING" not in text
