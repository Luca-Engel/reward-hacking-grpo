"""Hint probe: Wilson intervals, the pre-declared selection rule, planted-rate runs, guards (subtask 06). Mock only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from evalfix import tiny_dir, tiny_problems

from rhg import budget
from rhg.analysis.stats import wilson
from rhg.config import load_config
from rhg.data.prompts import load_prompts_cfg
from rhg.eval import generate as gen
from rhg.eval import pipeline as pl
from rhg.eval import probe_hints as ph

CFG = load_config("clean_none")
PROMPTS = load_prompts_cfg()
Z = 1.96


# ============================================================ Wilson interval
def wilson_by_scan(k, n, z=Z, grid=400_001):
    """Brute force: the set {p : (k/n - p)^2 <= z^2 p (1-p) / n} on a fine grid; its extremes."""
    p = np.linspace(0.0, 1.0, grid)
    inside = (k / n - p) ** 2 <= z * z * p * (1 - p) / n + 1e-15
    return float(p[inside].min()), float(p[inside].max())


@pytest.mark.parametrize("n", [1, 2, 5, 10, 37, 100])
def test_wilson_matches_brute_force_scan_for_every_k(n):
    for k in range(n + 1):
        lo, hi = wilson(k, n)
        blo, bhi = wilson_by_scan(k, n)
        assert lo == pytest.approx(blo, abs=5e-6) and hi == pytest.approx(bhi, abs=5e-6), (k, n)


def test_wilson_closed_forms_symmetry_and_edge_cases():
    for n in (1, 10, 3000):
        assert wilson(0, n) == (0.0, pytest.approx(Z * Z / (n + Z * Z)))
        assert wilson(n, n) == (pytest.approx(n / (n + Z * Z)), 1.0)
    for n in (7, 50):
        for k in range(n + 1):
            lo, hi = wilson(k, n)
            lo2, hi2 = wilson(n - k, n)
            assert (lo, hi) == (pytest.approx(1 - hi2), pytest.approx(1 - lo2))
            assert 0 <= lo <= k / n <= hi <= 1
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))  # standard table value
    assert wilson(50, 100, z=2.5758293035489) [0] < wilson(50, 100)[0]  # wider at higher confidence
    for bad in ((-1, 5), (6, 5), (1, -1), (1.5, 4), (True, 4)):
        with pytest.raises(ValueError):
            wilson(*bad)
    with pytest.raises(ValueError):
        wilson(1, 5, z=-1.0)


def test_wilson_narrows_with_n():
    widths = [wilson(n // 50, n)[1] - wilson(n // 50, n)[0] for n in (100, 1000, 10000)]
    assert widths[0] > widths[1] > widths[2]


# ============================================================ selection rule (pure)
def counts(n=3000, **rates):
    """Counts dict from per-wording attempt counts (``S1=k``) and visible counts (``S1_vis=k``; default: same as none)."""
    out = {}
    for w in ph.WORDINGS:
        out[w] = {"n": n, "k_attempt": rates.get(w, 0), "k_visible": rates.get(f"{w}_vis", rates.get("none_vis", n // 3)), "k_correct": 0, "k_hack": 0}
    return out


def test_rule_selects_first_qualifying_candidate_in_order_s1_s2_s3():
    d = ph.decide(counts(none=0, S1=60, S2=90, S3=45, E1=600))  # 2%, 3%, 1.5%; E1 20%
    assert d["selected"] == "S1" and d["go"] and d["manipulation_check"]["passed"]
    assert all(d["candidates"][c]["qualifies"] for c in ph.CANDIDATES)  # S2, S3 qualify too, but S1 comes first
    d = ph.decide(counts(none=0, S1=200, S2=90, S3=45, E1=600))  # S1 = 6.7% is out of range
    assert d["selected"] == "S2" and not d["candidates"]["S1"]["rate_in_range"]
    d = ph.decide(counts(none=0, S1=3, S2=200, S3=45, E1=600))  # S1 = 0.1% too low, S2 too high
    assert d["selected"] == "S3"


@pytest.mark.parametrize("k,ok", [(14, False), (15, True), (150, True), (151, False)])
def test_rate_boundaries_are_inclusive_and_exact(k, ok):
    # n = 3000: 0.5% = 15/3000, 5% = 150/3000. E1 at 40% keeps the CI condition out of the way.
    d = ph.decide(counts(none=0, S1=k, S2=2000, S3=2000, E1=1200))
    assert d["candidates"]["S1"]["rate_in_range"] is ok


def test_ci_must_be_strictly_below_e1():
    d = ph.decide(counts(none=0, S1=120, S2=120, S3=120, E1=150))  # 4% vs 5%: in range but CIs overlap
    assert all(c["rate_in_range"] and not c["ci_below_e1"] for c in d["candidates"].values())
    assert d["selected"] is None and not d["go"] and "no candidate" in d["reasons"][0]
    lo_e1 = wilson(600, 3000)[0]
    assert wilson(60, 3000)[1] < lo_e1  # sanity for the passing case above
    # exact touching is not "strictly below": construct equal bounds via identical counts
    d = ph.decide(counts(none=0, S1=90, S2=2000, S3=2000, E1=90))
    assert not d["candidates"]["S1"]["ci_below_e1"] and d["selected"] is None


def test_rule_agrees_with_brute_force_intervals_for_every_s1_count():
    n = 100
    e1_k, none_k = 40, 0
    e1_lo = wilson_by_scan(e1_k, n)[0]
    none_hi = wilson_by_scan(none_k, n)[1]
    for k in range(n + 1):
        d = ph.decide(counts(n=n, none=none_k, S1=k, S2=50, S3=50, E1=e1_k))
        s_lo, s_hi = wilson_by_scan(k, n)
        expect_sel = (1 <= k <= 5) and s_hi < e1_lo  # 0.5% * 100 = 0.5 -> k >= 1; 5% -> k <= 5
        assert (d["selected"] == "S1") is expect_sel, k
        if expect_sel:
            assert d["manipulation_check"]["passed"] is (none_hi < s_lo and s_hi < e1_lo), k


def test_manipulation_check_failures_are_detected():
    # none as high as the subtle wording: the wording did not manipulate anything
    d = ph.decide(counts(none=60, S1=60, S2=60, S3=60, E1=600))
    assert d["selected"] == "S1" and d["manipulation_check"] == {
        "selected": "S1", "none_below_subtle": False, "subtle_below_explicit": True, "passed": False}
    assert not d["go"] and "manipulation check failed" in d["reasons"][0] and "`none`" in d["reasons"][0]
    # not ordered the other way: explicit below subtle -> nothing qualifies, no manipulation check possible
    d = ph.decide(counts(none=0, S1=120, S2=120, S3=120, E1=30))
    assert d["selected"] is None and d["manipulation_check"] is None and not d["go"]
    # ordered: passes
    assert ph.decide(counts(none=1, S1=60, S2=90, S3=45, E1=600))["manipulation_check"]["passed"]


def test_confound_flag_needs_a_big_drop_and_non_overlapping_cis():
    base = dict(none=0, S1=60, S2=90, S3=45, E1=600)
    assert not ph.decide(counts(**base, none_vis=1500, S1_vis=1400))["confound"]["any_flag"]  # -3.3 pp
    d = ph.decide(counts(**base, none_vis=1500, S1_vis=1300, S2_vis=1500, S3_vis=1500, E1_vis=1500))  # -6.7 pp, n=3000
    assert d["confound"]["any_flag"] and d["confound"]["by_wording"]["S1"]["flag"] and not d["confound"]["by_wording"]["S2"]["flag"]
    assert d["go"]  # non-blocking
    exactly_5 = ph.decide(counts(**base, none_vis=1500, S1_vis=1350))  # drop of exactly 5 pp is not "more than 5 pp"
    assert not exactly_5["confound"]["by_wording"]["S1"]["flag"]
    small_n = ph.decide(counts(n=100, none=0, S1=2, S2=3, S3=2, E1=20, none_vis=50, S1_vis=40))  # -10 pp but CIs overlap
    assert not small_n["confound"]["by_wording"]["S1"]["flag"] and small_n["confound"]["by_wording"]["S1"]["drop_pp"] == pytest.approx(10.0)
    higher = ph.decide(counts(**base, none_vis=1200, S1_vis=1800))  # hint RAISES honest pass: never a confound flag
    assert not higher["confound"]["any_flag"]


def test_decide_validates_input():
    good = counts(none=0, S1=60, S2=90, S3=45, E1=600)
    with pytest.raises(ValueError, match="missing"):
        ph.decide({k: v for k, v in good.items() if k != "E1"})
    bad = json.loads(json.dumps(good))
    bad["S1"]["k_attempt"] = 4000
    with pytest.raises(ValueError):
        ph.decide(bad)
    with pytest.raises(ValueError, match="z = 1.96"):
        ph.decide(good, z=2.58)


# ============================================================ planted runs
def probe(tmp_path, rates=None, honest_p=ph.MOCK_HONEST_P, name="p", n=None, min_samples=ph.MIN_SAMPLES, **kw):
    d = tiny_dir(tmp_path, name)
    out = tmp_path / name / "probe"
    kw.setdefault("selection_path", tmp_path / name / "hint_selection.json")
    kw.setdefault("behavior", ph.probe_behavior(rates, honest_p))
    res = ph.run_probe(cfg=CFG, processed_dir=d, out_dir=out, mock=True, n=n, min_samples=min_samples, **kw)
    return res, out


def binom_ok(k, n, p, sigmas=5.0):
    return abs(k - n * p) <= sigmas * math.sqrt(max(n * p * (1 - p), 1e-9)) + 1e-9


def test_planted_rates_are_recovered_and_default_n_gives_3000_samples(tmp_path):
    rates = {"none": 0.0, "S1": 0.02, "S2": 0.03, "S3": 0.015, "E1": 0.2}
    res, out = probe(tmp_path, rates, honest_p=0.4)  # defaults: 4 train problems -> n=750 -> 3000 per wording
    assert (res["n_problems"], res["n_per_problem"], res["n_samples_per_wording"], res["sample_size_ok"]) == (4, 750, 3000, True)
    w = res["decision"]["wordings"]
    for name, p in rates.items():
        assert w[name]["n"] == 3000 and binom_ok(w[name]["k_attempt"], 3000, p), (name, w[name]["k_attempt"])
        assert binom_ok(w[name]["k_visible"], 3000, 0.4)  # step-0 honest pass rate is recovered per wording
        assert w[name]["ci_attempt"] == list(wilson(w[name]["k_attempt"], 3000))
        assert w[name]["ci_visible"] == list(wilson(w[name]["k_visible"], 3000))
    assert w["none"]["k_attempt"] == 0
    assert res["decision"]["selected"] == "S1" and res["decision"]["go"] and res["exit_code"] == 0
    assert not res["decision"]["confound"]["any_flag"]
    probe_json = json.loads((out / "hint_probe.json").read_text(encoding="utf-8"))
    assert probe_json["decision"]["selected"] == "S1" and probe_json["wording_texts"]["S1"] == PROMPTS["hints"]["subtle"]["S1"]
    sel = json.loads(res["selection_path"].read_text(encoding="utf-8"))
    assert sel["selected_id"] == "S1" and sel["wording"] == PROMPTS["hints"]["subtle"]["S1"] and sel["level"] == "subtle"
    assert sel["rates_attempt_rt"]["S1"] == w["S1"]["rate_attempt"] and sel["ci95_attempt_rt"]["E1"] == w["E1"]["ci_attempt"]
    assert sel["rates_visible_pass"]["none"] == w["none"]["rate_visible"] and sel["sample_size_ok"] is True
    assert sel["probe_json_sha256"] and sel["mock"] is True and sel["prompts_hash"]
    assert res["seeds"] == {"none": 71000, "S1": 71001, "S2": 71002, "S3": 71003, "E1": 71004}


def test_selection_skips_a_candidate_that_is_out_of_range(tmp_path):
    rates = {"none": 0.0, "S1": 0.08, "S2": 0.02, "S3": 0.02, "E1": 0.25}
    res, _ = probe(tmp_path, rates, n=250, min_samples=1000)
    assert res["decision"]["selected"] == "S2" and res["decision"]["go"]
    assert json.loads(res["selection_path"].read_text(encoding="utf-8"))["selected_id"] == "S2"
    assert res["sample_size_ok"] is False and json.loads(res["selection_path"].read_text(encoding="utf-8"))["sample_size_ok"] is False


def test_no_qualifying_candidate_is_no_go_with_exit_code_3(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ph, "MOCK_ATTEMPT_RATES", {"none": 0.0, "S1": 0.0, "S2": 0.12, "S3": 0.0, "E1": 0.25})
    d = tiny_dir(tmp_path)
    sel = tmp_path / "sel.json"
    rc = ph.main(["--mock", "--processed-dir", str(d), "--out-dir", str(tmp_path / "out"), "--selection-path", str(sel),
                  "--n", "250", "--min-samples", "1000"])
    text = capsys.readouterr().out
    assert rc == 3 and not sel.exists()
    assert "NO-GO" in text and "no candidate" in text and "3 new subtle candidates" in text and "$0.3" in text
    assert (tmp_path / "out" / "hint_probe.json").exists()  # the evidence is still written
    assert json.loads((tmp_path / "out" / "hint_probe.json").read_text(encoding="utf-8"))["decision"]["go"] is False


def test_manipulation_check_failure_is_detected_when_rates_are_not_ordered(tmp_path):
    res, _ = probe(tmp_path, {"none": 0.02, "S1": 0.02, "S2": 0.02, "S3": 0.02, "E1": 0.25}, n=250, min_samples=1000)
    dec = res["decision"]
    assert dec["selected"] == "S1" and dec["manipulation_check"]["none_below_subtle"] is False
    assert not dec["go"] and res["exit_code"] == 3 and "manipulation check failed" in dec["reasons"][0]
    assert "selection_path" not in res and not (tmp_path / "p" / "hint_selection.json").exists()
    inverted, _ = probe(tmp_path, {"none": 0.0, "S1": 0.03, "S2": 0.03, "S3": 0.03, "E1": 0.005}, name="q", n=250, min_samples=1000)
    assert inverted["decision"]["selected"] is None and inverted["exit_code"] == 3


def test_confound_flag_is_reported_but_does_not_block(tmp_path, capsys):
    hp = {"none": 0.5, "S1": 0.3, "S2": 0.5, "S3": 0.5, "E1": 0.5}
    res, _ = probe(tmp_path, {"none": 0.0, "S1": 0.02, "S2": 0.03, "S3": 0.02, "E1": 0.25}, honest_p=hp, n=250, min_samples=1000)
    conf = res["decision"]["confound"]
    assert conf["any_flag"] and conf["by_wording"]["S1"]["flag"] and not conf["by_wording"]["S2"]["flag"]
    assert res["decision"]["go"] and res["exit_code"] == 0
    assert "CONFOUND" in ph.format_report(res)
    assert json.loads(res["selection_path"].read_text(encoding="utf-8"))["confound_flag"] is True


def test_existing_selection_is_never_overwritten_without_force(tmp_path, capsys):
    d = tiny_dir(tmp_path)
    sel = tmp_path / "hint_selection.json"
    sel.write_text('{"selected_id": "S3", "frozen": true}\n', encoding="utf-8")
    boom = gen.MockGenerator(ph.probe_behavior())
    with pytest.raises(ph.SelectionExists):
        ph.run_probe(cfg=CFG, processed_dir=d, out_dir=tmp_path / "o", selection_path=sel, mock=True, n=100, min_samples=400, generator=boom)
    assert boom.n_calls == 0 and not (tmp_path / "o").exists()  # refused before any generation
    args = ["--mock", "--processed-dir", str(d), "--out-dir", str(tmp_path / "o"), "--selection-path", str(sel), "--n", "250", "--min-samples", "1000"]
    assert ph.main(args) == 3 and "refused" in capsys.readouterr().err
    assert sel.read_text(encoding="utf-8") == '{"selected_id": "S3", "frozen": true}\n'
    assert ph.main([*args, "--force"]) == 0
    assert json.loads(sel.read_text(encoding="utf-8"))["selected_id"] == "S1"
    # generate-only never writes a selection, so it is not blocked by an existing one
    sel.write_text("keep\n", encoding="utf-8")
    assert ph.main([*args, "--generate-only"]) == 0 and sel.read_text(encoding="utf-8") == "keep\n"


def test_minimum_sample_size_is_enforced(tmp_path, capsys):
    d = tiny_dir(tmp_path)
    with pytest.raises(pl.PipelineError, match="below --min-samples"):
        ph.run_probe(cfg=CFG, processed_dir=d, out_dir=tmp_path / "o", selection_path=tmp_path / "s.json", mock=True, n=100)
    rc = ph.main(["--mock", "--processed-dir", str(d), "--out-dir", str(tmp_path / "o"), "--selection-path", str(tmp_path / "s.json"), "--n", "100"])
    assert rc == 2 and "min-samples" in capsys.readouterr().err
    assert ph.main(["--mock", "--processed-dir", str(d / "none-here"), "--out-dir", str(tmp_path / "o")]) == 2


# ============================================================ generation / grading split
def test_generate_only_then_grade_only_reproduces_the_default_path_byte_for_byte(tmp_path):
    def names(root):
        return {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}

    kw = dict(n=250, min_samples=1000)
    res_default, out_default = probe(tmp_path, name="default", **kw)
    assert res_default["decision"]["go"]

    d_gen = tiny_dir(tmp_path, "gen")
    ph.run_probe(cfg=CFG, processed_dir=d_gen, out_dir=tmp_path / "gen" / "probe", selection_path=tmp_path / "gen" / "sel.json",
                 mock=True, generate_only=True, behavior=ph.probe_behavior(), **kw)
    assert not (tmp_path / "gen" / "probe" / "hint_probe.json").exists()
    assert (tmp_path / "gen" / "probe" / "completions.jsonl.gz").read_bytes() == (out_default / "completions.jsonl.gz").read_bytes()

    d_cpu = tiny_dir(tmp_path, "cpu")
    (tmp_path / "cpu" / "probe").mkdir()
    (tmp_path / "cpu" / "probe" / "completions.jsonl.gz").write_bytes((tmp_path / "gen" / "probe" / "completions.jsonl.gz").read_bytes())
    res_cpu = ph.run_probe(cfg=CFG, processed_dir=d_cpu, out_dir=tmp_path / "cpu" / "probe", selection_path=tmp_path / "cpu" / "hint_selection.json",
                           mock=False, grade_only=True, workers=2, chunk_prompts=3)
    assert res_cpu["mock"] is True  # taken from the completions header, not from the flag
    expect = names(out_default)
    got = names(tmp_path / "cpu" / "probe")
    assert got == expect and set(expect) == {"completions.jsonl.gz", "hint_probe.json"}
    assert (tmp_path / "cpu" / "hint_selection.json").read_bytes() == (tmp_path / "default" / "hint_selection.json").read_bytes()
    with pytest.raises(pl.PipelineError, match="fixed by the completions"):
        ph.run_probe(cfg=CFG, processed_dir=d_cpu, out_dir=tmp_path / "cpu" / "probe", selection_path=tmp_path / "x.json", mock=True, grade_only=True, n=5)


# ============================================================ CLI, ledger, safety
def test_mock_cli_defaults_never_touch_real_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    tiny_dir(tmp_path, "data/fixture/processed")
    (tmp_path / "prereg").mkdir()
    assert ph.main(["--mock"]) == 0  # default n: 750 x 4 train problems = 3000 samples per wording
    out = capsys.readouterr().out
    assert "MOCK" in out and "GATE 1d: GO" in out and "subtle_selected: S1" in out
    assert (tmp_path / "results" / "probe_mock" / "hint_probe.json").exists()
    assert (tmp_path / "results" / "probe_mock" / "hint_selection.mock.json").exists()
    assert not (tmp_path / "prereg" / "hint_selection.json").exists() and not (tmp_path / "results" / "probe").exists()
    assert not (tmp_path / "results" / "ledger.jsonl").exists()


def test_real_path_records_ledger_kind_probe_with_generation_time_only(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    d = tiny_dir(tmp_path)
    res = ph.run_probe(cfg=CFG, processed_dir=d, out_dir=tmp_path / "o", selection_path=tmp_path / "s.json", mock=False, n=50, min_samples=200,
                       generator=gen.MockGenerator(ph.probe_behavior()), record_ledger=True, ledger=ledger)
    (entry,) = budget.read_entries(ledger)
    assert entry["kind"] == "probe" and entry["wall_s"] == pytest.approx(res["load_wall_s"] + res["gen_wall_s"])
    assert entry["usd"] == pytest.approx(entry["wall_s"] / 3600 * CFG.budget.usd_per_hour)
    ph.run_probe(cfg=CFG, processed_dir=d, out_dir=tmp_path / "o", selection_path=tmp_path / "s.json", mock=False, grade_only=True,
                 record_ledger=True, ledger=ledger, force=True)
    assert len(budget.read_entries(ledger)) == 1  # grade-only is CPU time: not billed


def test_probe_needs_train_problems_and_the_five_wordings(tmp_path):
    d = tiny_dir(tmp_path)
    rows = [dict(p, split="val") for p in tiny_problems()]
    (d / "problems.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(pl.PipelineError, match="no train problems"):
        ph.load_train_problems(d)
    cfg = {"template": PROMPTS["template"], "hints": {"none": "", "subtle": {"S1": "x"}, "explicit": {"E1": "y"}}}
    with pytest.raises(pl.PipelineError, match="S2"):
        ph.probe_groups(tiny_problems(), 5, cfg)
    groups = ph.probe_groups(tiny_problems(), 5, PROMPTS)
    assert [g.hint for g in groups] == list(ph.WORDINGS) and len({g.seed for g in groups}) == 5
