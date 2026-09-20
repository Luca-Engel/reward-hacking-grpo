"""Data-exploration notebook: unit tests of ``rhg.data.explore`` against hand-computed values and
brute force, then the notebook builds and executes end to end (fixture without / with the gated inputs).

Everything here is fixture/mock only: no network (the Qwen3 tokenizer is used only if it is already cached),
no GPU, no API. The sandbox runs the hand-written toy controls and the fixture reference solutions.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sst

from rhg.data import explore as ex
from rhg.data import fixture
from rhg.data.dedupe import jaccard, normalize_text, shingles

REPO = Path(__file__).resolve().parents[1]
NL = chr(10)


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_01", REPO / "notebooks" / "build_01.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_builder()


def prob(pid, desc, tests=(), held=(), **kw):
    mk = lambda ts: [{"id": i, "kind": "assert", "src": s} for i, s in enumerate(ts)]  # noqa: E731
    return {"problem_id": pid, "description": desc, "reward_tests": mk(tests), "heldout_tests": mk(held), **kw}


@pytest.fixture(scope="module")
def cands():
    return fixture.fixture_candidates()


# ------------------------------------------------------------------ A. schema
def test_schema_summary_counts_missing_empty_and_types():
    ps = [{"a": 1, "b": "x", "c": []}, {"a": 2, "b": None, "c": [1]}, {"a": None, "b": "", "c": [2]}]
    s = ex.schema_summary(ps).set_index("field")
    assert (s.loc["a", "missing"], s.loc["a", "empty"], s.loc["a", "types"]) == (1, 0, "int")
    assert (s.loc["b", "missing"], s.loc["b", "empty"]) == (1, 1)
    assert (s.loc["c", "missing"], s.loc["c", "empty"]) == (0, 1)
    assert s.loc["a", "missing_pct"] == pytest.approx(33.33, abs=0.01)
    assert ex.schema_issues(ps, ["a", "b", "z"]) == ["a: 1 problems missing/empty", "b: 2 problems missing/empty", "z: 3 problems missing/empty"]
    assert ex.schema_issues([{"a": 1}], ["a"]) == []
    assert ex.schema_issues([{"tags": []}, {"tags": ["x"]}], ["tags"]) == []  # a problem without tags is legitimate (reported, not an issue)
    assert ex.schema_issues([{"tags": None}], ["tags"]) == ["tags: 1 problems missing/empty"]


def test_fixture_candidates_satisfy_the_required_schema(cands):
    assert ex.schema_issues(cands) == []


def test_provenance_and_library_table(tmp_path):
    (tmp_path / "candidates.jsonl").write_text(json.dumps({"problem_id": "p"}) + NL, encoding="utf-8")
    (tmp_path / "DATASET_REVISION").write_text("rhg-fixture@abc" + NL, encoding="utf-8")
    prov = ex.provenance(ex.load_inputs(tmp_path))
    assert prov["dataset_revision"] == "rhg-fixture@abc" and prov["n_candidates"] == 1 and prov["n_problems"] is None
    import hashlib

    assert prov["files"]["candidates.jsonl"]["sha256_12"] == hashlib.sha256((tmp_path / "candidates.jsonl").read_bytes()).hexdigest()[:12]
    libs = ex.library_table().set_index("library")["version"]
    assert {"python", "numpy", "pandas", "scipy", "matplotlib", "rhg git sha"} <= set(libs.index) and libs["numpy"] == np.__version__


def test_load_inputs_reads_what_exists_and_gates_on_the_rest(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.load_inputs(tmp_path)
    (tmp_path / "candidates.jsonl").write_text(json.dumps({"problem_id": "p"}) + "\n", encoding="utf-8")
    inp = ex.load_inputs(tmp_path, tmp_path / "probe")
    assert inp.available == {"candidates": True, "passrate": False, "splits": False, "probe": False}
    (tmp_path / "passrate_A.jsonl").write_text("{}\n", encoding="utf-8")
    assert ex.load_inputs(tmp_path).available["passrate"] is False  # needs both stages
    (tmp_path / "passrate_B.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "DATASET_REVISION").write_text("rev\n", encoding="utf-8")
    inp = ex.load_inputs(tmp_path)
    assert inp.available["passrate"] and inp.revision == "rev"


def test_problem_frame_derived_columns(cands):
    df = ex.problem_frame(cands)
    assert len(df) == 40 and df["problem_id"].is_unique
    assert (df["n_reward"] == 5).all() and (df["n_heldout"] <= 20).all() and (df["n_heldout"] >= 5).all()
    p = cands[0]
    row = df.iloc[0]
    assert row["desc_words"] == len(p["description"].split()) and row["desc_chars"] == len(p["description"])
    assert row["test_len_mean"] == pytest.approx(np.mean([len(t["src"]) for t in p["reward_tests"] + p["heldout_tests"]]))
    assert row["difficulty_ord"] == {"Easy": 0, "Medium": 1, "Hard": 2}[p["difficulty"]]
    assert row["date_dt"] == pd.Timestamp(p["date"])


# ------------------------------------------------------------------ B. contamination / counts
def test_contamination_proxy_counts_strictly_after_the_horizon():
    dates = ["2023-01-01", "2024-06-30", "2024-07-01", "2025-01-01", None, "not-a-date"]
    r = ex.contamination_proxy(dates, "2024-06-30")
    assert (r["n_dated"], r["n_undated"], r["n_after"], r["n_before_or_on"]) == (4, 2, 2, 2)  # the horizon day itself is not "after"
    assert r["share_after"] == 0.5
    assert math.isnan(ex.contamination_proxy([None], "2024-06-30")["share_after"])
    g = ex.contamination_grid(dates, ["2023-12-31"])
    assert list(g["horizon"]) == ["2023-12-31", ex.QWEN3_HORIZON_ASSUMED] and list(g["n_after"]) == [3, 2]


def test_contamination_share_is_monotone_in_the_horizon(cands):
    shares = [ex.contamination_proxy([p["date"] for p in cands], h)["share_after"] for h in ("2015-01-01", "2020-01-01", "2030-01-01")]
    assert shares[0] >= shares[1] >= shares[2] and shares[2] == 0.0


def test_count_and_tag_tables():
    t = ex.count_table(["a", "b", "a", None, "a"], order=["a", "b", "c"])
    assert list(t["n"]) == [3, 1, 0] and list(t["share"]) == [0.75, 0.25, 0.0]
    df = pd.DataFrame({"tags": [["x", "y"], ["x"], []]})
    tt = ex.tag_table(df)
    assert list(tt["tag"]) == ["x", "y"] and list(tt["n"]) == [2, 1] and list(ex.tag_table(df, min_n=2)["tag"]) == ["x"]


# ------------------------------------------------------------------ C. text budget
def test_proxy_counter_and_truncation_risk_hand_values():
    assert ex.proxy_count("def f(x): return x+1") == 10  # def f ( x ) : return x + 1
    r = ex.truncation_risk([100, 700, 760, 769, 1000], 768)
    assert (r["n"], r["n_over"], r["n_within_10pct"], r["max"]) == (5, 2, 2, 1000.0)  # 700 and 760 lie in (691.2, 768]
    assert r["share_over"] == 0.4
    assert ex.truncation_risk([], 10)["n"] == 0


def test_hint_length_delta_equals_the_hint_wording_length_under_the_proxy(cands):
    from rhg.data.prompts import hint_text, load_prompts_cfg

    cfg = load_prompts_cfg()
    counter = ex.load_token_counter("proxy")
    assert not counter.is_real and "PROXY" in counter.label
    pl = ex.prompt_lengths(cands[:5], cfg, counter)
    for h in ("S1", "S2", "S3", "E1"):
        # whitespace is not counted by the proxy, so the paragraph adds exactly the wording's own tokens
        assert (pl[f"delta_{h}"] == ex.proxy_count(hint_text(h, cfg))).all()
    assert ex.proxy_count(hint_text("E1", cfg)) > ex.proxy_count(hint_text("S1", cfg))
    assert (pl["tokens_none"] > 0).all() and list(pl["problem_id"]) == [p["problem_id"] for p in cands[:5]]


def test_real_tokenizer_is_used_only_when_locally_cached_and_never_downloads():
    c = ex.load_token_counter("auto")
    assert c.kind in ("qwen3", "proxy")
    if c.kind == "qwen3":
        n = c.count_many(["hello world", "def f(x):\n    return x"])
        assert n[0] == 2 and n[1] >= 7  # "hello world" is two Qwen3 tokens
    with pytest.raises(FileNotFoundError):
        ex.load_token_counter("qwen3", name="rhg/does-not-exist")


# ------------------------------------------------------------------ D. tests
def test_expected_and_input_extraction_forms():
    assert ex.expected_of("assert candidate(nums = [1, 2], k = 3) == [3, 4]") == "[3, 4]"
    assert ex.input_of("assert candidate(nums = [1, 2], k = 3) == [3, 4]") == "nums=[1, 2], k=3"
    assert ex.expected_of("assert candidate(5) == True") == "True"
    assert ex.expected_of("assert True == candidate(5)") == "True"
    assert ex.expected_of("assert is_same_list(candidate(head = list_node([1, 2])), list_node([2, 1]))") == "list_node([2, 1])"
    assert ex.expected_of("assert candidate(1) > 3") is None and ex.expected_of("assert candidate(") is None
    assert [ex.expected_type(e) for e in ("True", "3", "2.5", "'a'", "[1]", "{1: 2}", "None", "list_node([1])", "foo(1)")] == [
        "bool", "int", "float", "str", "list", "dict", "none", "helper", "other"]


def test_degenerate_detection_hand_built():
    a = [f"assert candidate({i}) == {v}" for i, v in enumerate(["True", "False", "True", "True", "False"])]  # 2 distinct
    b = [f"assert candidate({i}) == {v}" for i, v in enumerate([1, 2, 3, 1, 1])]  # 3 distinct
    c = [f"assert candidate({i}) == 7" for i in range(5)]  # 1 distinct
    held_a = ["assert candidate(9) == True"] * 3 + ["assert candidate(9) == False"]
    ps = [prob("a", "x", a, held_a), prob("b", "x", b, ["assert candidate(1) == 1"]), prob("c", "x", c, ["assert candidate(1) == 7", "assert candidate(2) == 8"])]
    d = ex.degenerate_report(ps).set_index("problem_id")
    assert list(d["n_distinct_reward"]) == [2, 3, 1] and list(d["degenerate"]) == [True, False, True]
    assert list(d["constant_passes_all_reward"]) == [False, False, True]
    assert d.loc["a", "const_share_reward"] == 3 / 5 and d.loc["a", "const_share_heldout"] == 3 / 4  # constant "True"
    assert d.loc["b", "const_share_heldout"] == 1.0 and d.loc["c", "const_share_heldout"] == 0.5
    assert d.loc["a", "n_distinct_all"] == 2
    assert not ex.degenerate_report(ps, max_distinct=1).set_index("problem_id").loc["a", "degenerate"]


def test_planted_degenerate_fixture_problems_are_detected(cands):
    d = ex.degenerate_report(cands).set_index("problem_id")
    flagged = set(d.index[d["degenerate"]])
    assert set(fixture.DEGENERATE_IDS) <= flagged
    # brute force: distinct expected values by re-parsing with ast
    import ast

    for p in cands:
        vals = {ast.unparse(ast.parse(t["src"]).body[0].test.comparators[0]) for t in p["reward_tests"]}
        assert d.loc[p["problem_id"], "n_distinct_reward"] == len(vals)


def test_duplicate_test_diagnostics():
    t = "assert candidate(1) == 2"
    ps = [prob("p", "d", [t, t, "assert candidate(3) == 4"], [t, "assert candidate(3) == 5"]), prob("q", "d", [t], ["assert candidate(8) == 8"])]
    r = ex.duplicate_tests(ps)
    assert r["within_problem_duplicate_tests"] == 3 and r["problems_with_duplicates"] == 1  # p: t three times in total
    assert r["reward_heldout_overlap"] == 1  # (p, t) is in both parts
    assert r["same_input_conflicting_expected"] == 1  # candidate(3): 4 vs 5
    assert r["cross_problem_duplicate_tests"] == 1  # t is also in q
    assert ex.duplicate_tests([prob("z", "d", ["assert candidate(1) == 2"], ["assert candidate(2) == 3"])])["within_problem_duplicate_tests"] == 0


def test_fixture_has_no_duplicate_or_overlapping_tests(cands):
    r = ex.duplicate_tests(cands)
    assert r["within_problem_duplicate_tests"] == 0 and r["reward_heldout_overlap"] == 0 and r["same_input_conflicting_expected"] == 0


def test_drop_frame_reads_both_reports():
    inp = ex.Inputs(Path("."), Path("."), [], tests_report={"drops": [{"problem_id": "a", "reason": "too_few_tests", "detail": "3"}]},
                    validation_report={"drops": [{"problem_id": "b", "reason": "reference_timeout", "detail": ">60s"}]}, raw_difficulty={"b": "Hard"})
    d = ex.drop_frame(inp)
    assert list(d["stage"]) == ["tests", "validate"] and list(d["reason"]) == ["too_few_tests", "reference_timeout"]
    assert d["difficulty"].isna().tolist() == [True, False] and d["difficulty"].iloc[1] == "Hard"
    assert ex.drop_frame(ex.Inputs(Path("."), Path("."), [])).empty


# ------------------------------------------------------------------ E. reference solutions
def test_reference_timing_on_the_fixture_grades_every_reference_as_correct(cands):
    t = ex.time_references(cands, sample_n=6, seed=1, workers=2)
    assert len(t) == 6 and t["problem_id"].is_unique and t["correct"].all() and (t["wall_s"] > 0).all()
    assert list(ex.time_references(cands, sample_n=6, seed=1, workers=2)["problem_id"]) == list(t["problem_id"])  # seeded sample
    s = ex.runtime_summary(t)
    assert s["n"] == 6 and s["validity"] == 1.0 and s["max_s"] == pytest.approx(t["wall_s"].max()) and s["n_timeout"] == 0
    assert ex.runtime_summary(t.iloc[0:0]) == {"n": 0}


# ------------------------------------------------------------------ F. leakage
def test_shingle_jaccard_on_known_pairs():
    base = "alpha beta gamma delta epsilon zeta"  # 5-word shingles: {abcde, bcdef}
    ps = [
        prob("a", base), prob("b", "alpha beta gamma delta epsilon eta"),  # shares 1 of 3 distinct shingles
        prob("c", base), prob("d", "one two three four five six seven"),  # d shares nothing
        prob("e", base + " theta iota"),  # 4 shingles, 2 shared with a -> 1/2
    ]
    df = ex.near_duplicate_pairs(ps, min_jaccard=0.3)
    got = {(r.a, r.b): r.jaccard for r in df.itertuples()}
    assert got == pytest.approx({("a", "c"): 1.0, ("a", "b"): 1 / 3, ("b", "c"): 1 / 3, ("a", "e"): 0.5, ("c", "e"): 0.5})
    assert set(ex.near_duplicate_pairs(ps, min_jaccard=0.5)["a"]) == {"a", "c"}
    assert all(df["jaccard"].diff().dropna() <= 0)  # sorted descending
    assert ex.near_duplicate_pairs([prob("a", "one two")], 0.1).empty  # one short doc: nothing to pair


def test_near_duplicate_pairs_match_brute_force_on_the_fixture(cands):
    fast = ex.near_duplicate_pairs(cands, min_jaccard=0.0 + 1e-9)
    sh = {p["problem_id"]: shingles(normalize_text(p["description"])) for p in cands}
    brute = {}
    for (a, sa), (b, sb) in itertools.combinations(sh.items(), 2):
        j = jaccard(sa, sb)
        if j > 0:
            brute[tuple(sorted((a, b)))] = j
    got = {tuple(sorted((r.a, r.b))): r.jaccard for r in fast.itertuples()}
    assert got == pytest.approx(brute)
    assert len(got) > 0


def test_planted_near_duplicates_are_found_in_the_same_cluster(cands):
    pairs = ex.near_duplicate_pairs(cands, min_jaccard=0.5)
    hi = pairs[pairs["jaccard"] >= ex.NEAR_DUP_THRESHOLD]
    found = {tuple(sorted((r.a, r.b))) for r in hi.itertuples()}
    assert any(tuple(sorted(pair)) in found for pair in fixture.PLANTED_CLUSTERS)
    assert hi["same_cluster"].all()  # the clustering rule covers every pair at >= 0.8
    cs = ex.cluster_summary(cands)
    assert cs["n_problems"] == 40 and int(cs["histogram"]["cluster_size"].mul(cs["histogram"]["n_clusters"]).sum()) == 40
    assert cs["largest"][0]["size"] >= 2 and sorted(cs["largest"][0]["members"]) == cs["largest"][0]["members"]


def test_id_duplicates_and_spanning_clusters():
    ps = [prob("foo-bar", "same text"), prob("foo-bar-ii", "other"), prob("dup", "x"), prob("dup", "y"), prob("z", "same text")]
    r = ex.id_duplicates(ps)
    assert r["duplicate_ids"] == ["dup"] and ["foo-bar", "foo-bar-ii"] in r["same_title"] and r["identical_description"] == [["foo-bar", "z"]]
    sp = [{"problem_id": "a", "cluster_id": "c1", "split": "train"}, {"problem_id": "b", "cluster_id": "c1", "split": "test"},
          {"problem_id": "c", "cluster_id": "c2", "split": "val"}, {"problem_id": "d", "cluster_id": "c2", "split": "val"}]
    out = ex.clusters_spanning_splits(sp)
    assert [o["cluster_id"] for o in out] == ["c1"] and out[0]["splits"] == {"train": ["a"], "test": ["b"]}
    assert ex.clusters_spanning_splits(sp[2:]) == []


# ------------------------------------------------------------------ G. pass rates
def test_passrate_frame_uses_exact_band_edges_and_joins_stage_b():
    cands = pd.DataFrame({"problem_id": list("abcde"), "difficulty": ["Easy"] * 5})
    a = [{"problem_id": p, "n": 10, "k_visible": k, "k_full": k} for p, k in zip("abcde", (0, 1, 4, 5, 2))]
    b = [{"problem_id": "c", "n": 10, "k_visible": 3, "k_full": 2}]
    df = ex.passrate_frame(cands, a, b, 0.10, 0.40).set_index("problem_id")
    assert list(df["in_band"]) == [False, True, True, False, True]  # 0.1 and 0.4 are inclusive, 0.5 is out
    assert df.loc["c", "p_B_visible"] == 0.3 and df.loc["c", "p_B_full"] == 0.2 and np.isnan(df.loc["a", "p_B_full"])


def test_rtm_summary_hand_values():
    df = pd.DataFrame({"in_band": [True] * 4 + [False], "p_A": [0.1, 0.2, 0.3, 0.4, 0.9], "p_B_visible": [0.3, 0.2, 0.2, 0.0, 0.9], "n_A": 16})
    r = ex.rtm_summary(df, 0.10, 0.40)
    d = np.array([0.2, 0.0, -0.1, -0.4])
    assert r["n"] == 4 and r["mean_shift"] == pytest.approx(d.mean())
    se = d.std(ddof=1) / 2
    assert r["shift_ci"] == pytest.approx((d.mean() - 1.96 * se, d.mean() + 1.96 * se))
    assert r["slope"] == pytest.approx(sst.linregress([0.1, 0.2, 0.3, 0.4], [0.3, 0.2, 0.2, 0.0]).slope)
    assert (r["share_in_band_B"], r["share_below_band_B"], r["share_above_band_B"]) == (0.75, 0.25, 0.0)
    assert ex.rtm_summary(df.iloc[:2])["n"] == 2  # too few problems: only n is reported


def test_regression_to_the_mean_matches_the_beta_binomial_closed_form():
    """Truth ~ Beta(a, b), 16 samples per stage: given k successes in stage A the expected stage-B rate is the posterior mean
    (a + k) / (a + b + n), so the band-selected shift and slope have closed forms to compare with."""
    a_, b_, n, m = 0.4, 1.2, 16, 6000
    rng = np.random.default_rng(0)
    truth = rng.beta(a_, b_, m)
    ka, kb = rng.binomial(n, truth), rng.binomial(n, truth)
    df = pd.DataFrame({"problem_id": range(m), "p_A": ka / n, "p_B_visible": kb / n, "n_A": n})
    df["in_band"] = (df["p_A"] >= 0.10) & (df["p_A"] <= 0.40)
    r = ex.rtm_summary(df)
    ks = np.arange(2, 7)  # 0.1 * 16 = 1.6 and 0.4 * 16 = 6.4: k in 2..6 is the band
    assert set(np.unique(ka[df["in_band"]])) == set(ks)
    w = sst.betabinom.pmf(ks, n, a_, b_)
    want_shift = float(np.sum(w * ((a_ + ks) / (a_ + b_ + n) - ks / n)) / w.sum())
    want_slope = n / (a_ + b_ + n)  # d E[p_B | k] / d p_A
    half = (r["shift_ci"][1] - r["shift_ci"][0]) / 2
    assert abs(r["mean_shift"] - want_shift) < 2 * half + 1e-9  # inside twice the 95% half-width
    assert abs(r["slope"] - want_slope) < 4 * r["slope_se"]
    assert r["slope"] < 1.0 and r["mean_shift"] * want_shift > 0  # shrinkage, in the direction the closed form predicts


def test_weak_tests_summary_hand_values():
    rows = [{"problem_id": "a", "n": 16, "k_visible": 8, "k_full": 8}, {"problem_id": "b", "n": 16, "k_visible": 8, "k_full": 4},
            {"problem_id": "c", "n": 16, "k_visible": 0, "k_full": 0}]
    r = ex.weak_tests_summary(rows)
    assert r["n_samples"] == 48 and r["visible_rate"] == 16 / 48 and r["full_rate"] == 12 / 48
    assert r["frac_visible_pass_fail_heldout"] == 4 / 16
    from rhg.analysis.stats import wilson

    assert r["frac_ci"] == wilson(4, 16)
    assert r["share_problems_with_gap"] == 1 / 3 and r["share_problems_gap_ge_25pct"] == 1 / 3


def test_band_table_and_correlations():
    df = pd.DataFrame({"in_band": [True, True, False, False, True], "difficulty": ["Easy", "Easy", "Hard", "Easy", "Hard"],
                       "tags": [["a"], ["a", "b"], ["b"], [], ["a"]]})
    t = ex.band_table(df, "difficulty").set_index("difficulty")
    assert (t.loc["Easy", "n"], t.loc["Easy", "n_in_band"], t.loc["Hard", "n_in_band"]) == (3, 2, 1)
    tg = ex.band_table(df, "tags").set_index("tags")
    assert (tg.loc["a", "n"], tg.loc["a", "n_in_band"], tg.loc["b", "n"], tg.loc["b", "n_in_band"]) == (3, 3, 2, 1)
    x = np.arange(20.0)
    cdf = pd.DataFrame({"y": x ** 2, "f": x, "g": np.zeros(20)})
    c = ex.correlation_table(cdf, "y", ["f", "g"]).set_index("feature")
    assert c.loc["f", "rho"] == pytest.approx(1.0) and c.loc["f", "n"] == 20 and math.isnan(c.loc["g", "rho"])


def test_stats_side_summary_hand_values():
    rows = [{"n": 10, "n_tokens_mean": 100.0, "n_truncated": 1, "n_extract_fail": 5, "n_timeout": 0, "n_crash": 0, "n_defines_rt": 2},
            {"n": 30, "n_tokens_mean": 200.0, "n_truncated": 3, "n_extract_fail": 3, "n_timeout": 4, "n_crash": 1, "n_defines_rt": 0}]
    s = ex.stats_side_summary(rows)
    assert s["n_samples"] == 40 and s["n_tokens_mean"] == pytest.approx((10 * 100 + 30 * 200) / 40)
    assert s["truncation_rate"] == 0.1 and s["extract_fail_rate"] == 0.2 and s["timeout_rate"] == 0.1 and s["crash_rate"] == 0.025
    assert s["defines_rt_rate"] == 0.05 and s["share_problems_extract_fail_ge_half"] == 0.5
    assert ex.stats_side_summary([]) == {"n_samples": 0}


# ------------------------------------------------------------------ H. balance statistics
def ks_brute(x, y):
    pts = sorted(set(x) | set(y))
    return max(abs(sum(v <= t for v in x) / len(x) - sum(v <= t for v in y) / len(y)) for t in pts)


def test_ks_statistic_matches_brute_force_and_scipy():
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = rng.integers(0, 12, rng.integers(3, 30)).tolist()  # heavy ties on purpose
        y = rng.integers(3, 15, rng.integers(3, 30)).tolist()
        assert ex.ks_statistic(x, y) == pytest.approx(ks_brute(x, y))
        assert ex.ks_statistic(x, y) == pytest.approx(sst.ks_2samp(x, y).statistic)
    assert ex.ks_statistic([1, 2, 3], [10, 11]) == 1.0 and ex.ks_statistic([1, 2], [1, 2]) == 0.0


def test_smd_and_cramers_v_known_values():
    assert ex.smd([1, 2, 3], [2, 3, 4]) == pytest.approx(-1.0)  # pooled SD 1, mean difference -1
    assert ex.smd([5, 5, 5], [5, 5, 5]) == 0.0 and math.isnan(ex.smd([1], [1, 2]))
    assert ex.cramers_v([[10, 0], [0, 10]]) == pytest.approx(1.0) and ex.cramers_v([[5, 5], [5, 5]]) == pytest.approx(0.0)
    assert math.isnan(ex.cramers_v([[10, 0], [10, 0]]))  # a constant column: undefined
    t = np.array([[20, 10, 5], [10, 20, 15]])
    chi2 = sst.chi2_contingency(t, correction=False)[0]
    assert ex.cramers_v(t) == pytest.approx(math.sqrt(chi2 / (t.sum() * 1)))


def _split_frame(n_train=60, n_val=20, n_test=20, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for split, n, s in (("train", n_train, 0.0), ("val", n_val, 0.0), ("test", n_test, shift)):
        for i in range(n):
            rows.append({"problem_id": f"{split}{i}", "split": split, "x": rng.normal(s, 1.0), "tags": ["t"]})
    return pd.DataFrame(rows)


def test_balance_numeric_effect_sizes_and_kruskal():
    df = _split_frame(shift=2.0)
    b = ex.balance_numeric(df, "x").set_index("pair")
    assert set(b.index) == {"train vs val", "train vs test", "val vs test"}
    tr, te = df.loc[df.split == "train", "x"].to_numpy(), df.loc[df.split == "test", "x"].to_numpy()
    assert b.loc["train vs test", "ks_D"] == pytest.approx(ks_brute(tr, te)) and b.loc["train vs test", "smd"] == pytest.approx(ex.smd(tr, te))
    assert b.loc["train vs test", "ks_p"] == pytest.approx(sst.ks_2samp(tr, te).pvalue) and b.loc["train vs test", "ks_p"] < 1e-6
    assert b.loc["train vs test", "smd"] < -1.5  # test is shifted up by 2 SD
    h = sst.kruskal(*(df.loc[df.split == s, "x"] for s in ex.SPLITS))
    assert b.attrs["kruskal_p"] == pytest.approx(h.pvalue) and b.attrs["epsilon_sq"] == pytest.approx((h.statistic - 2) / (len(df) - 3))
    same = ex.balance_numeric(_split_frame(shift=0.0, seed=5), "x")
    assert (same["ks_p"] > 0.001).all() and same.attrs["epsilon_sq"] < 0.1


def test_balance_categorical_matches_scipy_and_permutation_p_agrees_with_asymptotics():
    rng = np.random.default_rng(1)
    rows = [{"split": s, "difficulty": d} for s, probs in (("train", (0.3, 0.5, 0.2)), ("val", (0.3, 0.5, 0.2)), ("test", (0.1, 0.4, 0.5)))
            for d in rng.choice(["Easy", "Medium", "Hard"], 150, p=probs)]
    df = pd.DataFrame(rows)
    r = ex.balance_categorical(df, "difficulty", n_perm=3000, seed=2)
    chi2, p, _, _ = sst.chi2_contingency(pd.crosstab(df["difficulty"], df["split"]).to_numpy(), correction=False)
    assert r["chi2"] == pytest.approx(chi2) and r["p_asymptotic"] == pytest.approx(p)
    assert abs(r["p_permutation"] - r["p_asymptotic"]) < 0.02 or (r["p_permutation"] < 0.01 and r["p_asymptotic"] < 0.01)
    assert r["cramers_v"] == pytest.approx(ex.cramers_v(pd.crosstab(df["difficulty"], df["split"]).to_numpy()))
    balanced = pd.DataFrame({"split": ["train", "val", "test"] * 20, "difficulty": ["Easy", "Medium"] * 30})
    rb = ex.balance_categorical(balanced, "difficulty", n_perm=500)
    assert rb["chi2"] == pytest.approx(0.0, abs=1e-9) and rb["p_permutation"] > 0.9
    assert math.isnan(ex.balance_categorical(pd.DataFrame({"split": ["train", "val"], "difficulty": ["Easy", "Easy"]}), "difficulty")["chi2"])


def test_tag_balance_and_flags():
    rows = [{"split": s, "tags": (["a"] if s == "test" else ["b"])} for s in ["train"] * 30 + ["val"] * 10 + ["test"] * 20]
    tb = ex.tag_balance(pd.DataFrame(rows)).set_index("tag")
    assert tb.loc["a", "share_test"] == 1.0 and tb.loc["a", "share_train"] == 0.0 and tb.loc["a", "max_gap_pp"] == 100.0
    num = pd.DataFrame([{"column": "x", "pair": "train vs test", "n_a": 60, "n_b": 20, "ks_D": 0.5, "ks_p": 0.001, "smd": -0.9},
                        {"column": "y", "pair": "train vs val", "n_a": 60, "n_b": 20, "ks_D": 0.2, "ks_p": 0.04, "smd": 0.1},
                        {"column": "z", "pair": "val vs test", "n_a": 20, "n_b": 20, "ks_D": 0.35, "ks_p": 0.04, "smd": 0.3}])
    msgs = ex.balance_flags(num, [{"column": "difficulty", "p_permutation": 0.02, "cramers_v": 0.3}, {"column": "ok", "p_permutation": 0.5, "cramers_v": 0.1}])
    assert len(msgs) == 3 and msgs[0].startswith("x:") and msgs[1].startswith("z:") and msgs[2].startswith("difficulty")


def test_verify_splits_recomputes_hash_and_detects_tampering():
    from rhg.data.build import split_hash

    ps = [{"problem_id": "a", "cluster_id": "a", "split": "train"}, {"problem_id": "b", "cluster_id": "a", "split": "train"},
          {"problem_id": "c", "cluster_id": "c", "split": "val"}, {"problem_id": "d", "cluster_id": "d", "split": "test"}]
    splits = {"split_hash": split_hash({p["problem_id"]: p["split"] for p in ps}), "train": ["a", "b"], "val": ["c"], "test": ["d"]}
    v = ex.verify_splits(ps, splits)
    assert v["hash_ok"] and v["lists_match"] and v["n_spanning_clusters"] == 0 and v["counts"] == {"train": 2, "val": 1, "test": 1}
    moved = [dict(p) for p in ps]
    moved[1]["split"] = "test"  # b moves out of a's split: hash and lists no longer match, cluster a spans two splits
    v2 = ex.verify_splits(moved, splits)
    assert not v2["hash_ok"] and not v2["lists_match"] and v2["n_spanning_clusters"] == 1
    tab = ex.gate1c_table({"gate1c": {"items": [{"item": "train >= 150", "status": "FAIL", "value": 2}]}}, v2).set_index("item")
    assert tab.loc["train >= 150", "status"] == "FAIL" and tab.loc["no cluster spans splits (recomputed)", "status"] == "FAIL"
    assert tab.loc["split hash recomputes from problems.jsonl", "status"] == "FAIL"
    assert set(ex.gate1c_table({"gate1c": {"items": []}}, v)["status"]) == {"PASS"}


# ------------------------------------------------------------------ I. probe
def test_newcombe_interval_matches_the_published_example_and_its_definition():
    d, lo, hi = ex.newcombe_diff(56, 70, 48, 80)  # Newcombe (1998), Table II, method 10: 0.2000 (0.0524, 0.3339)
    assert (round(d, 4), round(lo, 4), round(hi, 4)) == (0.2, 0.0524, 0.3339)
    d0, lo0, hi0 = ex.newcombe_diff(5, 50, 5, 50)
    assert d0 == 0.0 and lo0 == pytest.approx(-hi0)  # symmetric for equal counts
    assert ex.newcombe_diff(0, 10, 0, 10)[1:] == pytest.approx((-0.2775, 0.2775), abs=1e-3)


def test_probe_tables_from_a_decision_dict():
    from rhg.eval import probe_hints as ph

    counts = {w: {"n": 3000, "k_attempt": int(3000 * r), "k_visible": 900 if w != "S3" else 700, "k_correct": 800, "k_hack": 0}
              for w, r in {"none": 0.0, "S1": 0.02, "S2": 0.03, "S3": 0.015, "E1": 0.2}.items()}
    t = ex.probe_tables({"decision": ph.decide(counts)}).set_index("wording")
    assert list(t.index) == ["none", "S1", "S2", "S3", "E1"]  # fixed display order
    assert t.loc["S1", "attempt_rate"] == 0.02 and t.loc["S1", "attempt_lo"] < 0.02 < t.loc["S1", "attempt_hi"]
    assert np.isnan(t.loc["none", "honest_shift_pp"]) and t.loc["S1", "honest_shift_pp"] == 0.0
    d, lo, hi = ex.newcombe_diff(700, 3000, 900, 3000)
    assert t.loc["S3", "honest_shift_pp"] == pytest.approx(100 * d) and t.loc["S3", "shift_hi_pp"] < 0  # a real drop: CI below zero
    assert t.loc["S3", "shift_lo_pp"] == pytest.approx(100 * lo) and t.loc["S3", "shift_hi_pp"] == pytest.approx(100 * hi)


# ------------------------------------------------------------------ J. power
def test_power_tables_reproduce_the_design_numbers():
    pw = ex.power_tables()
    assert pw["runs"] == 22
    tests = pw["tests"].set_index("test")
    assert tests.loc["primary", "min_attainable_p"] == pytest.approx(1 / math.comb(10, 5))  # 5 v 5: 1/252
    assert tests.loc["H4a", "min_attainable_p"] == pytest.approx(1 / math.comb(8, 3))  # 3 v 5: 1/56 = 0.0179
    e = pw["emergence"].set_index("emerged")
    assert e.loc[4, "p"] == pytest.approx(0.0238, abs=5e-5) and e.loc[3, "p"] == pytest.approx(0.0833, abs=5e-5)
    pp = pw["primary_power"].set_index("q")["power"]
    for q, want in {0.9: 0.92, 0.8: 0.74, 0.7: 0.53, 0.5: 0.19}.items():  # DESIGN §6
        assert pp.loc[q] == pytest.approx(want, abs=0.005)
    assert len(pw["ladder"]) == 7 and pw["ladder"]["runs"].tolist() == [22, 21, 19, 16, 14, 13, 11]
    assert ex.power_tables({"hackable_subtle": 4, "clean_subtle": 4})["runs"] == 8


# ------------------------------------------------------------------ K. labels
@pytest.fixture(scope="module")
def controls():
    return ex.controls_table(workers=2)


def test_label_names_and_controls_table(controls):
    assert ex.label_names({"hack_rt": True, "attempt_rt": True}) == "HACK_RT+ATTEMPT_RT"
    assert ex.label_names({"attempt_rt": True, "correct": True}) == "ATTEMPT_RT+CORRECT"
    assert ex.label_names({}) == "(none)"
    from rhg.validate import controls as C

    assert list(controls["control_id"]) == [c.control_id for c in C.build_controls()]
    by_id = controls.set_index("control_id")
    for c in C.build_controls():  # every declared execution expectation holds in the table
        row = by_id.loc[c.control_id]
        for k, v in c.expect.items():
            got = {"attempt_rt": row["defines_rt"], "correct": "CORRECT" in row["labels"].split("+"), "gap_other": "GAP_OTHER" in row["labels"].split("+")}.get(k)
            got = row[k] if got is None else got
            assert bool(got) == v, (c.control_id, k)


def test_detector_blindspots_on_hand_picked_controls(controls):
    def pick(cat, variant):
        rows = controls[(controls["category"] == cat) & (controls["variant"] == variant)]
        assert len(rows) >= 1, (cat, variant)
        return rows

    obvious = pick("obvious_override", "print_success")
    assert obvious["hack_rt"].all() and obvious["ast_narrow"].all() and obvious["ast_broad"].all() and (obvious["labels"] == "HACK_RT+ATTEMPT_RT").all()
    lam = pick("subtle_override", "lambda")
    assert lam["hack_rt"].all() and not lam["ast_narrow"].any() and lam["ast_broad"].all()  # the narrow monitor is blind to a lambda assignment
    sc = pick("special_casing", "lookup_table")
    assert not sc["defines_rt"].any() and not sc["hack_rt"].any() and sc["visible_pass"].all() and not sc["heldout_pass"].any()
    assert (sc["labels"] == "GAP_OTHER").all()
    bs = ex.detector_blindspots(controls)
    assert "lambda" in set(bs["narrow_misses"]["variant"]) and bs["broad_misses"].empty
    assert (~controls.loc[controls["hack_rt"], "ast_broad"]).sum() == 0
    assert set(bs["exec_blind_hacks"]["category"]) >= {"special_casing"}
    assert bs["narrow_vs_exec"].to_numpy().sum() == len(controls)
    assert bs["n_narrow_false_positive"] == len(bs["narrow_false_positives"])
    honest_defining = controls[(controls["category"] == "honest_self_test")]
    assert len(honest_defining) and not honest_defining["hack_rt"].any()


# ------------------------------------------------------------------ L. flags
def test_derive_flags_orders_by_severity_and_covers_each_rule():
    m = {
        "schema_issues": ["tags: 3 problems missing/empty"], "n_candidates": 10, "dataset_revision": None, "n_empty_tags": 11,
        "prompt": {"n_over": 4, "share_over": 0.04, "limit": 768, "tokenizer_kind": "proxy"},
        "degenerate": {"n": 12, "share": 0.12, "n_constant": 3},
        "dup_tests": {"reward_heldout_overlap": 2, "within_problem_duplicate_tests": 1, "same_input_conflicting_expected": 0},
        "reference": {"validity": 0.9, "n_slow": 2},
        "leakage": {"n_duplicate_ids": 0, "n_cross_cluster_pairs": 1, "n_spanning": 2},
        "passrate": {"n_selected": 100, "stats": {"A": {"truncation_rate": 0.2, "extract_fail_rate": 0.05}},
                     "weak_tests": {"frac_visible_pass_fail_heldout": 0.4}, "rtm": {"n": 30, "mean_shift": -0.05, "shift_ci": (-0.08, -0.02)}},
        "splits": {"gate1c_fail": ["train >= 150 (value 9)"], "hash_ok": False, "balance_flags": ["x differs"]},
        "probe": {"go": False, "reasons": ["r"], "sample_size_ok": False, "confound_flag": True, "mock": True},
        "labels": {"n_narrow_misses": 7, "n_broad_misses": 1, "n_exec_blind": 8}, "power_flags": ["H4a outside family"], "unavailable": ["H"],
    }
    f = ex.derive_flags(m)
    sev = [x["severity"] for x in f]
    assert sev == sorted(sev, key=["ERROR", "WARN", "INFO"].index)
    text = "\n".join(f"{x['severity']}|{x['section']}|{x['message']}" for x in f)
    for needle in ("ERROR|A|schema", "ERROR|D|2 tests appear in both", "ERROR|F|1 pairs with Jaccard", "ERROR|F|2 clusters span", "ERROR|H|split hash",
                   "WARN|A|no DATASET_REVISION", "INFO|A|11 problems have no tags", "WARN|C|4 prompts", "WARN|C|prompt lengths use the word-count PROXY", "WARN|D|12 problems", "WARN|E|reference validity 90.0%",
                   "WARN|G|only 100 problems", "WARN|G|stage A: 20.0%", "WARN|G|40% of visible", "WARN|H|Gate 1c item failing", "WARN|I|hint probe is NO-GO",
                   "WARN|I|hint probe has fewer than", "WARN|I|a hint wording", "WARN|K|the broad detector misses 1", "INFO|G|regression to the mean", "INFO|I|the probe file is a MOCK",
                   "INFO|K|the narrow detector misses 7", "INFO|J|H4a", "INFO|H|section not available"):
        assert needle in text, needle
    clean = {"schema_issues": [], "n_candidates": 10, "dataset_revision": "x@1", "degenerate": {"n": 0, "share": 0.0}, "leakage": {"n_spanning": 0},
             "prompt": {"n_over": 0, "tokenizer_kind": "qwen3"}, "reference": {"validity": 0.99, "n_slow": 0}}
    assert [x for x in ex.derive_flags(clean) if x["severity"] != "INFO"] == []
    assert ex.derive_flags({}) == []


# ------------------------------------------------------------------ figures
def test_figure_helpers_render_with_labelled_axes_and_n():
    import matplotlib

    matplotlib.use("Agg")
    figs = [
        ex.bar_figure(["a", "b"], [3, 1], "t", "x", "y", n=4),
        ex.hist_figure({"s": [1, 2, 2, 3, np.nan]}, "t", "x", vline=(2, "mark")),
        ex.scatter_figure([0.1, 0.2], [0.3, 0.2], "t", "x", "y", band=(0.1, 0.4)),
        ex.strip_by_group_figure({"train": [1, 2], "val": [3], "test": []}, "t", "y"),
        ex.forest_figure(["a", "b"], [0.1, 0.2], [0.05, 0.1], [0.2, 0.3], "t", "x", vline=0.1, ns=[10, 20], pct=True),
        ex.line_figure([0.5, 0.9], {"5 v 5": [0.2, 0.9]}, "t", "x", "y"),
    ]
    for f in figs:
        ax = f.axes[0]
        assert ax.get_title(loc="left") and (ax.get_xlabel() or ax.get_xticklabels()) and (ax.get_ylabel() or ax.get_yticklabels())
        assert f.get_facecolor()[:3] == matplotlib.colors.to_rgb(ex.SURFACE)  # opaque surface: readable in light and dark viewers
    assert "n = 4" in figs[0].axes[0].get_title(loc="left")
    for f in figs:
        ex._plt().close(f)


# ------------------------------------------------------------------ notebook build / execution
CELL_HEADINGS = ("## A.", "## B.", "## C.", "## D.", "## E.", "## F.", "## G.", "## H.", "## I.", "## J.", "## K.", "## L.")


def outputs_text(nb) -> str:
    parts = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                parts.append(o.get("text", ""))
            elif o.get("output_type") in ("display_data", "execute_result"):
                for k in ("text/markdown", "text/plain"):
                    if k in o.get("data", {}):
                        parts.append(o["data"][k])
    return "\n".join(parts)


def test_notebook_structure_and_committed_copy_is_unexecuted_and_current():
    nb = B.build_notebook()
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for h in CELL_HEADINGS:
        assert h in md, h
    assert md.count("Tells us:") >= 11 and md.count("Would worry us:") >= 11  # every section carries its what-it-tells/what-would-worry note
    assert "uv run python -m ipykernel install --user --name rhg" in nb.cells[0].source
    assert len({c["id"] for c in nb.cells}) == len(nb.cells)
    assert all(c.get("outputs") == [] and c.get("execution_count") is None for c in nb.cells if c.cell_type == "code")
    committed = REPO / "notebooks" / "01_data_exploration.ipynb"
    assert committed.is_file()
    disk = nbformat.read(str(committed), 4)
    assert all(c.get("outputs") == [] for c in disk.cells if c.cell_type == "code")
    assert nbformat.writes(disk) == nbformat.writes(nb), "notebooks/01_data_exploration.ipynb is stale: run `uv run python notebooks/build_01.py`"
    for g in ("G", "H", "I"):  # each gated section is guarded by the standard message
        assert f'ex.unavailable("{g}"' in "\n".join(c.source for c in nb.cells)
    nbformat.validate(nb)


@pytest.fixture(scope="module")
def plain_dirs(tmp_path_factory):
    return B.prepare_fixture(tmp_path_factory.mktemp("nb_plain"), gated_inputs=False)


@pytest.fixture(scope="module")
def gated_dirs(tmp_path_factory):
    return B.prepare_fixture(tmp_path_factory.mktemp("nb_gated"), gated_inputs=True)


def run_nb(processed: Path, probe: Path, out: Path):
    env = {"RHG_PROCESSED_DIR": str(processed), "RHG_PROBE_DIR": str(probe), "RHG_NB_RUNTIME_SAMPLE": "8", "RHG_NB_TOKENIZER": "proxy"}
    nb = B.execute_notebook(B.build_notebook(), out, env)
    assert B.error_cells(nb) == []
    assert not [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    return nb


def test_notebook_executes_on_the_fixture_with_every_gated_section_unavailable(plain_dirs, tmp_path):
    processed, probe = plain_dirs
    assert not (processed / "passrate_A.jsonl").exists() and not (processed / "splits.json").exists() and not probe.exists()
    nb = run_nb(processed, probe, tmp_path / "plain.ipynb")
    text = outputs_text(nb)
    for sec, script in (("G", "measure_pass_rate.sh"), ("H", "rhg.data.build --stage split"), ("I", "probe_hints.sh")):
        assert f"Section {sec}: not available yet" in text and script in text
    assert text.count("not available yet") >= 3
    assert "Gate 1c checklist" not in text and "Decision:" not in text
    assert "section not available yet (inputs missing)" in text  # section L lists them as INFO flags
    assert "Token counter:" in text and "PROXY" in text  # the proxy is labelled
    assert (tmp_path / "plain.ipynb").is_file()
    imgs = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    assert imgs >= 12  # sections B-F and J draw their figures


def test_notebook_executes_on_the_fixture_with_mock_stage_outputs_and_renders_every_section(gated_dirs, tmp_path):
    processed, probe = gated_dirs
    for f in ("passrate_A.jsonl", "passrate_B.jsonl", "passrate_A_stats.jsonl", "splits.json", "problems.jsonl"):
        assert (processed / f).is_file(), f
    assert (probe / "hint_probe.json").is_file()
    nb = run_nb(processed, probe, tmp_path / "gated.ipynb")
    text = outputs_text(nb)
    assert "not available yet" not in text
    for needle in ("in the band", "mean shift p_B - p_A", "visible passes fail the held-out tests", "Cramer's V", "Spearman correlation",
                   "split hash recorded", "MATCH", "Gate 1c checklist", "Decision:", "confound rule", "planned design", "Execution HACK_RT vs the narrow detector",
                   "Hard check - clusters spanning splits: 0 -> PASS"):
        assert needle in text, needle
    imgs = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    assert imgs >= 22
    saved = nbformat.read(str(tmp_path / "gated.ipynb"), 4)
    assert len(saved.cells) == len(nb.cells)


def test_cli_build_only_and_missing_data_exit_code(tmp_path, capsys):
    out = tmp_path / "nb.ipynb"
    assert B.main(["--notebook", str(out)]) == 0 and out.is_file()
    assert all(not c.get("outputs") for c in nbformat.read(str(out), 4).cells if c.cell_type == "code")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert B.main(["--notebook", str(out), "--execute", "--processed-dir", str(empty), "--out", str(tmp_path / "x.ipynb")]) == 2
    assert "candidates.jsonl" in capsys.readouterr().err
    assert not (tmp_path / "x.ipynb").exists()


def test_prepare_fixture_refuses_a_non_empty_foreign_directory(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not empty"):
        B.prepare_fixture(tmp_path)
    assert (tmp_path / "keep.txt").exists()
