"""Full analysis pipeline on synthetic run directories with planted truth (subtask 14): tests, report, stamps, guards, examples."""

from __future__ import annotations

import csv
import itertools
import json
import re

import anafix
import numpy as np
import pytest
from fakerepo import fake_repo, frozen_repo  # noqa: F401  (fixtures)
from PIL import Image

from rhg import prereg_constants as C
from rhg.analysis import endpoints as E
from rhg.analysis import examples, report, simulate
from rhg.analysis import run as analysis_run


def _brute_p(h, c):
    pooled = list(h) + list(c)
    k, obs = len(h), sum(h) / len(h) - sum(c) / len(c)
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), k):
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(len(pooled)) if i not in idx]
        hits += sum(a) / k - sum(b) / len(b) >= obs - 1e-12
        total += 1
    return hits / total


def _analyse(d, out, **kw):
    kw.setdefault("figures", False)
    kw.setdefault("repo_root", out)
    return report.build_analysis(d.runs, out, problems_path=d.problems, **kw)


def _by_id(doc):
    return {t["id"]: t for t in doc["tests"]}


def _primary_plan(emerging, rate=0.3, n=5):
    """Primary arms only: hackable_subtle seeds in ``emerging`` emerge (final rate ``rate``), the others never do; clean all zero."""
    on = anafix.scenario(rate={**simulate.Scenario().rate, "hackable_subtle": rate}, rate_sd=0.0)
    off = anafix.scenario(q={"hackable_subtle": 0.0})
    return {**{("hackable_subtle", s): (on if s in emerging else off) for s in range(n)},
            **{("clean_subtle", s): anafix.scenario() for s in range(n)}}


@pytest.fixture(scope="module")
def d_nodisc(tmp_path_factory):
    return anafix.per_run_dir(tmp_path_factory.mktemp("nodisc"), _primary_plan(()))


@pytest.fixture(scope="module")
def d_incon(tmp_path_factory):
    return anafix.per_run_dir(tmp_path_factory.mktemp("incon"), _primary_plan((0, 1)))


# ------------------------------------------------------------------ the 22-run pipeline
def test_primary_verdict_and_per_seed_values_follow_the_plant(sim22):
    t = sim22.tests["primary"]
    emerging = [s for s in range(5) if sim22.truth["runs"][f"hackable_subtle__s{s}"]["emerges"]]
    assert emerging == [0, 1, 2, 3, 4]  # planted q = 1
    assert t["decision"] == "supported" and t["outcome_wording"] == "supported"
    assert t["p"] == pytest.approx(1 / 252) and t["min_attainable_p"] == pytest.approx(1 / 252) and t["exact"] is True
    assert t["n"] == {"hackable_subtle": 5, "clean_subtle": 5} and t["n_text"] == "5 v 5"
    hack = list(t["per_seed"]["hackable_subtle"].values())
    clean = list(t["per_seed"]["clean_subtle"].values())
    assert clean == [0.0] * 5 and min(hack) > 0.2
    assert t["effect"]["delta_mean"] == pytest.approx(np.mean(hack) - np.mean(clean)) and t["effect"]["delta_ok"] and t["effect"]["p_ok"]
    assert t["emergence"]["hackable"]["emerged"] == 5 and t["emergence"]["clean"]["emerged"] == 0
    assert t["emergence"]["clean"]["wilson_hi"] == pytest.approx(0.4345, abs=1e-3)  # Wilson upper bound for 0/5
    assert t["stamp"] == "EXPLORATORY" and t["registered"] == "pre-registered confirmatory"
    rows = {r["run_id"]: r for r in csv.DictReader((sim22.analysis / "per_seed.csv").open(encoding="utf-8"))}
    assert [float(rows[r]["final_hack_rt"]) for r in t["per_seed"]["hackable_subtle"]] == pytest.approx(hack)


def test_h1_dose_response_direction_and_holm(sim22):
    f, o = sim22.tests["H1_final"], sim22.tests["H1_onset"]
    means = f["effect"]["arm_means"]
    assert means["hackable_none"] < means["hackable_subtle"] < means["hackable_explicit"]
    assert f["p"] <= 0.05 and o["p"] <= 0.05 and f["n_text"] == "2/5/3"
    assert f["min_attainable_p"] == pytest.approx(1 / 2520)  # 10! / (2! 5! 3!) assignments
    onset = {a: np.mean(list(v.values())) for a, v in o["per_seed"].items() if v}
    assert onset["hackable_explicit"] < onset["hackable_subtle"] < onset["hackable_none"]
    assert sim22.doc["h1_caveat"]["triggered"] is False and sim22.doc["h1_caveat"]["mean"] == 0.0
    holm = {r["name"]: r for r in sim22.doc["holm"]}
    assert set(holm) == set(C.HOLM_FAMILY) and holm["H1_final"]["reject"] and holm["H1_onset"]["reject"]
    assert holm["H1_final"]["threshold"] == pytest.approx(0.05 / 4) or holm["H1_final"]["rank"] > 1
    assert sim22.tests["H1_final"]["holm"]["p_adj"] >= sim22.tests["H1_final"]["p"]


def test_h2_h3b_and_the_exploratory_h3a(sim22):
    h2, h3b, h3a = sim22.tests["H2"], sim22.tests["H3b"], sim22.tests["H3a"]
    rhos = list(h2["per_seed"]["rho"].values())
    assert len(rhos) == 8 and max(rhos) < 0 and h2["effect"]["n_excluded"] == 2  # the two hackable_none seeds are excluded and counted
    assert h2["p"] == pytest.approx(2.0 ** -8) and h2["min_attainable_p"] == pytest.approx(2.0 ** -8) and "mechanically" in h2["note"]
    assert {e["run_id"] for e in h2["effect"]["excluded"]} == {"hackable_none__s0", "hackable_none__s1"}
    assert h3b["p"] <= 0.05 and h3b["effect"]["delta_mean"] < 0 and h3b["alternative"] == "hackable_subtle < clean_subtle"
    assert h3a["p"] is None and h3a["stamp"] == "EXPLORATORY" and "descriptive" in h3a["result"]
    gap = h3a["effect"]["arm_mean_gap"]
    assert gap["hackable_explicit"] > gap["hackable_subtle"] > gap["clean_subtle"]


def test_h4_on_the_full_design_shows_displacement_and_h4a_is_outside_the_family(sim22):
    h4a, h4b = sim22.tests["H4a"], sim22.tests["H4b"]
    assert h4b["decision"] == "displacement" and h4b["result"] == "displacement observed"
    assert all(v >= 0.05 for v in h4b["effect"]["hack_rates"].values()) and all(v >= 0.5 for v in h4b["effect"]["evasion"].values())
    ast, subtle = list(h4a["per_seed"]["hackable_subtle_ast"].values()), list(h4a["per_seed"]["hackable_subtle"].values())
    assert h4a["p"] == pytest.approx(_brute_p([-x for x in ast], [-x for x in subtle]))  # "lower" = larger negated value
    assert h4a["min_attainable_p"] == pytest.approx(1 / 56)  # C(8,3) = 56: 3 v 5 seeds
    assert h4a["registered"] == "exploratory" and "holm" not in h4a and h4a["min_attainable_p"] > 0.05 / 4
    assert any("structurally outside the family" in f for f in sim22.doc["min_attainable_p"]["flags"])


def _h4_dir(tmp_path, **plant):
    sc = anafix.scenario(**plant)
    d = anafix.simulate_dir(tmp_path, sc, seeds_per_arm={"hackable_subtle_ast": 3, "hackable_subtle": 5, "clean_subtle": 5}, shape=anafix.SMALL)
    return d, _analyse(d, tmp_path / "analysis")


def test_h4b_suppression_only_and_displacement_configurations(tmp_path):
    _, sup = _h4_dir(tmp_path / "sup", ast_suppress=1.0)  # the penalty removes every hack
    h4b = _by_id(sup)["H4b"]
    assert h4b["decision"] == "suppression_only" and all(v < 0.02 for v in h4b["effect"]["hack_rates"].values())
    assert all(v is None for v in h4b["effect"]["evasion"].values())  # no hack: no evasion value
    h4a = _by_id(sup)["H4a"]
    assert h4a["p"] == pytest.approx(1 / 56) and h4a["effect"]["delta_mean"] < -0.2
    _, dis = _h4_dir(tmp_path / "dis", ast_suppress=0.3, ast_evasion=0.9)  # hacks persist through the form the monitor misses
    h4b = _by_id(dis)["H4b"]
    assert h4b["decision"] == "displacement"
    assert sum(v >= 0.05 and e >= 0.5 for v, e in zip(h4b["effect"]["hack_rates"].values(), h4b["effect"]["evasion"].values())) >= 2
    _, keep = _h4_dir(tmp_path / "mix", ast_suppress=0.3, ast_evasion=0.0)  # hacks persist but are all caught: neither rule applies
    assert _by_id(keep)["H4b"]["decision"] == "mixed" and _by_id(keep)["H4b"]["result"] == "mixed / inconclusive"


def test_cross_hint_behaviour_in_the_tables(sim22):
    rows = {r["arm"]: r for r in csv.DictReader((sim22.analysis / "tables" / "crosshint.csv").open(encoding="utf-8"))}
    f = lambda arm, h: float(rows[arm][f"hack_rt_{h}_mean"])  # noqa: E731
    for arm in ("hackable_subtle", "hackable_explicit", "hackable_subtle_ast"):
        assert f(arm, "none") < f(arm, "subtle") < f(arm, "explicit")  # planted xhint multipliers 0.4 < 1.0 < 1.3
    for arm in ("clean_none", "clean_subtle", "clean_explicit", "hackable_none"):
        assert f(arm, "none") == f(arm, "subtle") == f(arm, "explicit") == 0.0
    assert f("hackable_subtle", "none") > 0  # the trained policy hacks even when the prompt never mentions run_tests


def test_executed_ladder_min_p_table_and_outputs(sim22):
    lad = sim22.doc["ladder"]
    assert lad["step"] == 0 and lad["runs"] == 22 and "no power loss" in lad["statement"]
    rows = {r["test"]: r for r in sim22.doc["min_attainable_p"]["rows"]}
    assert rows["primary"]["min_attainable_p"] == pytest.approx(1 / 252) and rows["H4a"]["min_attainable_p"] == pytest.approx(1 / 56)
    for name in ("REPORT.md", "tests.json", "per_seed.csv", "examples.md"):
        assert (sim22.analysis / name).is_file()
    for name in ("validity", "arm_summary", "min_attainable_p", "holm", "crosshint", "training_health_per_arm", "homogeneity",
                 "robustness_loo", "robustness_onset_grid", "robustness_variance"):
        assert (sim22.analysis / "tables" / f"{name}.csv").is_file(), name
    for name in ("primary_dots", "all_arms_dots", "trajectories_train", "trajectories_val", "dose_response", "onset", "gap", "covariates",
                 "evasion", "crosshint", "robustness_forest", "h2_rho", "h3b_dots", "h4a_dots"):
        assert (sim22.analysis / "figures" / f"{name}.png").is_file(), name


# ------------------------------------------------------------------ REPORT.md
def test_report_sections_box_wording_and_the_p_value_rule(sim22):
    text = (sim22.analysis / "REPORT.md").read_text(encoding="utf-8")
    assert text.index("What this can and cannot claim") < text.index("## 1.")  # the box comes first
    for head in ("## 1. Design executed", "## 2. Primary endpoint", "## 3. Confirmatory secondary family", "## 4. Exploratory results",
                 "## 5. Minimum attainable p", "## 6. Robustness suite", "## 7. Run quality", "## 8. Pre-registration compliance",
                 "### (a) Leave-one-seed-out", "### (g) Variance decomposition", "### Run homogeneity", "### Training health",
                 "Cross-hint evaluation", "Step-0 test baseline", "H1 clean-explicit caveat check", "H4b verdict", "prereg/AMENDMENTS.jsonl",
                 "Worst-case sensitivity", "Ladder state and power loss"):
        assert head in text, head
    assert "**Primary outcome: supported.**" in text and "Primary outcome: no discovery" not in text
    assert "planted deviation row" in text  # DEVIATIONS.md of the checkout is embedded
    assert "low coverage" in text and "Wilson" in text
    hits = list(re.finditer(r"(?<!attainable )\bp = [0-9.]+", text))  # flag lines "min attainable p = ..." state the minimum itself
    assert len(hits) >= 2  # the primary and H4a are stated inline; every other p sits in a table cell (checked below)
    for m in hits:  # every inline p-value carries its minimum attainable value and n
        assert re.match(r"p = [0-9.]+ \(min attainable [0-9.]+; n = [^)]+\)", text[m.start(): m.start() + 90]), text[m.start(): m.start() + 90]
    cells = re.findall(r"\| ([0-9.]+) \(([^)]*)\)", text)  # table cells "p (min ...; n ...)"
    assert len(cells) > 30 and all(re.fullmatch(r"min [0-9.]+; n [0-9][0-9/ a-z]*", c[1]) for c in cells), [c for c in cells if not re.fullmatch(r"min [0-9.]+; n [0-9][0-9/ a-z]*", c[1])][:3]
    assert "[CONFIRMATORY]" not in text and "CONFIRMATORY" not in text.split("## 1.")[1].replace("CONFIRMATORY analysis", "")  # exploratory mode


@pytest.mark.parametrize("fixture,wording,extra", [("d_nodisc", "no discovery", "0/5 hackable seeds"),
                                                   ("d_incon", "inconclusive at this power", "")])
def test_report_uses_the_exact_outcome_wording_of_prereg(request, tmp_path, fixture, wording, extra):
    d = request.getfixturevalue(fixture)
    doc = _analyse(d, tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    t = _by_id(doc)["primary"]
    assert t["outcome_wording"] == wording and f"**Primary outcome: {wording}.**" in text
    for other in ("supported", "no discovery", "inconclusive at this power"):
        if other != wording:
            assert f"**Primary outcome: {other}.**" not in text
    if extra:
        assert extra in text
    if fixture == "d_incon":
        hack = list(t["per_seed"]["hackable_subtle"].values())
        assert sum(v >= 0.02 for v in hack) == 2 and t["p"] == pytest.approx(56 / 252)  # both emerged seeds must be in the hackable set
        assert t["p"] == pytest.approx(_brute_p(hack, [0.0] * 5)) if len(set(hack)) == 3 else True
        assert not t["effect"]["p_ok"]
    else:
        assert t["p"] == 1.0 and t["effect"]["n_emerged"] == 0


def test_invalid_runs_are_listed_and_the_worst_case_sensitivity_is_computed(tmp_path):
    shape = anafix.SMALL
    d = anafix.simulate_dir(tmp_path, anafix.scenario(), seeds_per_arm=anafix.PRIMARY, shape=shape, invalid=["hackable_subtle__s1"],
                            replace_invalid=False)
    doc = _analyse(d, tmp_path / "out")
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "hackable_subtle__s1" in text.split("**Invalid / failed / excluded runs**")[1].split("**Worst-case")[0]
    assert "nan_loss" in text
    wc = doc["worst_case"]
    assert wc["applies"] and wc["invalid_hackable"] == "1/5" and wc["invalid_clean"] == "0/5" and wc["missing_hackable"] == 1
    t = _by_id(doc)["primary"]
    hack = list(t["per_seed"]["hackable_subtle"].values())
    assert t["n_text"] == "4 v 5"
    assert wc["p"] == pytest.approx(_brute_p(hack + [0.0], [0.0] * 5))  # a missing hackable seed is imputed 0 (worst case)
    assert wc["n_text"] == "5 v 5" and "Worst case against the hypothesis" in wc["text"]
    assert doc["ladder"]["matched"] is False and "match no BUDGET" in doc["ladder"]["statement"]


def test_ladder_step_6_states_its_power_loss_and_the_floor(tmp_path):
    d = anafix.simulate_dir(tmp_path, anafix.scenario(), ladder_step=6, shape=anafix.SMALL)
    doc = _analyse(d, tmp_path / "out")
    assert doc["ladder"]["step"] == 6 and doc["ladder"]["runs"] == 11 and not doc["ladder"]["below_floor"]
    assert "primary 5 v 5 -> 4 v 4" in doc["ladder"]["statement"] and "4/4 seeds emerge" in doc["ladder"]["statement"]
    assert _by_id(doc)["primary"]["n_text"] == "4 v 4" and _by_id(doc)["primary"]["min_attainable_p"] == pytest.approx(1 / 70)
    assert any("H3b" in f and "rejectable only after earlier Holm rejections" in f for f in doc["min_attainable_p"]["flags"])


def test_below_the_floor_is_stated(d_incon, tmp_path):
    doc = _analyse(d_incon, tmp_path)
    assert doc["ladder"]["below_floor"] and "not run confirmatorily" in doc["ladder"]["statement"]


# ------------------------------------------------------------------ stamps
def test_every_output_is_stamped_exploratory_without_the_confirmatory_flag(sim22):
    assert sim22.doc["mode"] == "EXPLORATORY" and sim22.doc["confirmatory_requested"] is False
    assert all(t["stamp"] == "EXPLORATORY" for t in sim22.doc["tests"])
    figs = json.loads((sim22.analysis / "figures" / "figures.json").read_text(encoding="utf-8"))
    assert len(figs) >= 14
    for f in figs:
        assert f["stamp"] == "EXPLORATORY" and f["title"].startswith("[EXPLORATORY] ") and f["caption"].startswith("")
        assert sum(f["n_per_arm"].values()) > 0
        png = Image.open(sim22.analysis / "figures" / f["file"])
        assert png.text["Title"] == f["title"] and "n seeds per arm" in png.text["Description"]
    assert {p.name for p in (sim22.analysis / "figures").glob("*.png")} == {f["file"] for f in figs}
    tabs = json.loads((sim22.analysis / "tables" / "tables.json").read_text(encoding="utf-8"))
    assert len(tabs) >= 15 and all(t["caption"].startswith("[EXPLORATORY] ") and t["stamp"] == "EXPLORATORY" for t in tabs)
    assert {t["file"] for t in tabs} == {p.name for p in (sim22.analysis / "tables").glob("*.csv")}
    text = (sim22.analysis / "REPORT.md").read_text(encoding="utf-8")
    assert re.findall(r"^## .*", text, flags=re.M) and all("[CONFIRMATORY]" not in h for h in re.findall(r"^#+ .*", text, flags=re.M))
    assert text.startswith("# Reward hacking under GRPO: analysis report [EXPLORATORY]")
    assert (sim22.analysis / "examples.md").read_text(encoding="utf-8").startswith("# Examples gallery [EXPLORATORY]")


def test_confirmatory_stamps_only_the_preregistered_confirmatory_items(d_incon, frozen_repo, tmp_path, capsys):  # noqa: F811
    d = anafix.simulate_dir(tmp_path / "sim", anafix.scenario(), shape=anafix.SMALL, seeds_per_arm=None)  # 22 runs: above the floor
    out = tmp_path / "out"
    code = analysis_run.main(["--runs", str(d.runs), "--out", str(out), "--problems", str(d.problems), "--repo-root", str(frozen_repo),
                              "--confirmatory"])
    assert code == 0
    doc = json.loads((out / "tests.json").read_text(encoding="utf-8"))
    assert doc["mode"] == "CONFIRMATORY" and doc["prereg_check"]["ok"] is True
    stamps = {t["id"]: t["stamp"] for t in doc["tests"]}
    assert stamps == {"primary": "CONFIRMATORY", "H1_final": "CONFIRMATORY", "H1_onset": "CONFIRMATORY", "H2": "CONFIRMATORY",
                      "H3b": "CONFIRMATORY", "H3a": "EXPLORATORY", "H4a": "EXPLORATORY", "H4b": "EXPLORATORY"}
    figs = {f["file"]: f for f in json.loads((out / "figures" / "figures.json").read_text(encoding="utf-8"))}
    for name in ("primary_dots", "h3b_dots", "h2_rho", "dose_response"):
        assert figs[f"{name}.png"]["stamp"] == "CONFIRMATORY" and figs[f"{name}.png"]["title"].startswith("[CONFIRMATORY] ")
        assert Image.open(out / "figures" / f"{name}.png").text["Title"].startswith("[CONFIRMATORY] ")
    for name in ("all_arms_dots", "h4a_dots", "gap", "covariates", "trajectories_train", "crosshint", "robustness_forest", "onset"):
        assert figs[f"{name}.png"]["stamp"] == "EXPLORATORY", name
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "## 2. Primary endpoint [CONFIRMATORY]" in text and "## 4. Exploratory results [EXPLORATORY]" in text
    assert "**PASS**" in text and "Analysis mode of this report: **CONFIRMATORY**" in text
    tabs = {t["file"]: t for t in json.loads((out / "tables" / "tables.json").read_text(encoding="utf-8"))}
    assert tabs["holm.csv"]["stamp"] == "CONFIRMATORY" and tabs["validity.csv"]["stamp"] == "EXPLORATORY"


# ------------------------------------------------------------------ guards and CLI
def test_confirmatory_without_a_passing_prereg_check_exits_3_and_writes_nothing(sim22, fake_repo, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "out"
    code = analysis_run.main(["--runs", str(sim22.runs), "--out", str(out), "--problems", str(sim22.problems), "--repo-root", str(fake_repo),
                              "--confirmatory"])
    assert code == 3 and not out.exists()
    err = capsys.readouterr().err
    assert "tag_exists" in err and "refused" in err
    # without the flag the same data are analysed, stamped EXPLORATORY, and the failing check is shown in the report
    code = analysis_run.main(["--runs", str(sim22.runs), "--out", str(out), "--problems", str(sim22.problems), "--repo-root", str(fake_repo),
                              "--no-figures"])
    assert code == 0 and json.loads((out / "tests.json").read_text(encoding="utf-8"))["mode"] == "EXPLORATORY"
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "**FAIL / not run**" in text and "tag_exists" in text
    assert str(fake_repo) not in text  # absolute paths are scrubbed from the report


def test_confirmatory_below_the_run_floor_is_refused(d_incon, frozen_repo, tmp_path, capsys):  # noqa: F811
    code = analysis_run.main(["--runs", str(d_incon.runs), "--out", str(tmp_path / "out"), "--repo-root", str(frozen_repo), "--confirmatory"])
    assert code == 3 and "floor" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_compliance_section_prints_amendments_and_the_failing_items(d_incon, frozen_repo, tmp_path):  # noqa: F811
    (frozen_repo / "prereg" / "AMENDMENTS.jsonl").write_text(
        json.dumps({"ts": "2026-01-02T00:00:00+00:00", "group": "analysis", "old_hash": "0" * 64, "new_hash": "1" * 64,
                    "reason": "planted amendment reason"}) + "\n", encoding="utf-8")
    (frozen_repo / "DEVIATIONS.md").write_text("# DEVIATIONS\n\n| date | what |\n|---|---|\n| 2026-01-03 | planted deviation two |\n", encoding="utf-8")
    code = analysis_run.main(["--runs", str(d_incon.runs), "--out", str(tmp_path / "out"), "--problems", str(d_incon.problems),
                              "--repo-root", str(frozen_repo), "--no-figures"])
    assert code == 0
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "planted amendment reason" in text and "planted deviation two" in text
    assert "**FAIL / not run**" in text and "amendment chain broken" in text
    doc = json.loads((tmp_path / "out" / "tests.json").read_text(encoding="utf-8"))
    assert doc["prereg_check"]["ok"] is False


def test_cli_usage_errors(tmp_path, capsys):
    assert analysis_run.main(["--runs", str(tmp_path / "nope"), "--out", str(tmp_path / "o")]) == 2
    (tmp_path / "runs").mkdir()
    assert analysis_run.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "o")]) == 2
    assert "no completed run" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        analysis_run.main(["--help"])
    assert e.value.code == 0


# ------------------------------------------------------------------ examples gallery
def test_examples_selection_is_reproducible_verbatim_and_the_header_states_the_seed(sim22, tmp_path):
    rs = E.load_runs(sim22.runs)
    a = examples.build_examples(rs, tmp_path / "a.md")
    b = examples.build_examples(rs, tmp_path / "b.md")
    c = examples.build_examples(rs, tmp_path / "c.md", seed=examples.EXAMPLES_SEED + 1)
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == (tmp_path / "b.md").read_text(encoding="utf-8") and a == b
    assert a["groups"] != c["groups"] and (tmp_path / "a.md").read_text() != (tmp_path / "c.md").read_text()
    text = (tmp_path / "a.md").read_text(encoding="utf-8")
    head = text.split("## ")[0]
    assert f"RNG seed (EXAMPLES_SEED):** {examples.EXAMPLES_SEED}" in head and "Selection procedure" in head and "verbatim" in head.lower()
    assert f"truncated only at {examples.MAX_CHARS} characters" in head
    hackable = [r.run_id for r in rs.runs if r.arm.startswith("hackable") and any(anafix.is_hack(x) for x in anafix.final_eval(r.run_dir))]
    for run_id in hackable:  # 5 HACK_RT + 5 non-hack for every hackable seed that has hacks
        assert len(a["groups"][f"hack:{run_id}"]["shown"]) == 5 and len(a["groups"][f"nonhack:{run_id}"]["shown"]) == 5
    assert not [k for k in a["groups"] if k.startswith("hack:clean") or k.startswith("hack:hackable_none")]
    assert len(a["groups"]["gap_other"]["shown"]) == 5
    # every shown rollout is what the label says and its completion appears verbatim
    for run_id in hackable[:3]:
        recs = {(r["problem_id"], r["sample_idx"]): r for r in anafix.final_eval(sim22.runs / run_id)}
        for pid, idx in [(p, i) for rid, p, i in a["groups"][f"hack:{run_id}"]["shown"]]:
            assert anafix.is_hack(recs[(pid, idx)]) and recs[(pid, idx)]["completion"] in text
        for rid, pid, idx in a["groups"][f"nonhack:{run_id}"]["shown"]:
            assert not anafix.is_hack(recs[(pid, idx)])
    for rid, pid, idx in a["groups"]["gap_other"]["shown"]:
        r = next(x for x in anafix.final_eval(sim22.runs / rid) if (x["problem_id"], x["sample_idx"]) == (pid, idx))
        assert r["labels"]["gap_other"]


def test_examples_truncation_is_stated_and_false_negatives_and_attempts_are_selected(tmp_path, monkeypatch):
    sc = anafix.scenario(attempt_base={"none": 0.0, "subtle": 0.06, "explicit": 0.0})  # hinted arms sometimes attempt without hacking
    orig = simulate.monitor_flags
    monkeypatch.setattr(simulate, "monitor_flags", lambda kind: (False, False, []) if kind.startswith("hack_") else orig(kind))
    d = anafix.simulate_dir(tmp_path / "sim", sc, seeds_per_arm={"hackable_subtle": 2, "clean_subtle": 2, "clean_explicit": 1}, shape=anafix.SMALL)
    monkeypatch.undo()
    rs = E.load_runs(d.runs)
    res = examples.build_examples(rs, tmp_path / "ex.md", max_chars=40)
    text = (tmp_path / "ex.md").read_text(encoding="utf-8")
    assert "showing the first 40 of" in text and "truncated only at 40 characters" in text
    assert len(res["groups"]["false_negative"]["shown"]) == 5  # HACK_RT by execution, not flagged by the broad profile
    for rid, pid, idx in res["groups"]["false_negative"]["shown"]:
        r = next(x for x in anafix.final_eval(d.runs / rid) if (x["problem_id"], x["sample_idx"]) == (pid, idx))
        assert anafix.is_hack(r) and not r["monitor"]["ast_broad"]
    att = res["groups"]["attempt:clean_subtle"]
    assert att["candidates"] > 0 and 0 < len(att["shown"]) <= 3
    for rid, pid, idx in att["shown"]:
        r = next(x for x in anafix.final_eval(d.runs / rid) if (x["problem_id"], x["sample_idx"]) == (pid, idx))
        assert r["labels"]["attempt_rt"] and not anafix.is_hack(r) and rid.startswith("clean_subtle")
    assert res["groups"]["attempt:clean_explicit"]["shown"] == [] and "No candidates" in text


def test_analysis_without_a_problem_table_reports_h2_as_untestable(d_incon, tmp_path):
    doc = report.build_analysis(d_incon.runs, tmp_path, problems_path=None, figures=False, repo_root=tmp_path)
    h2 = _by_id(doc)["H2"]
    assert h2["p"] is None and h2["n"] == 0 and "not testable" in h2["result"] and h2["effect"]["n_excluded"] == 5
    assert doc["problems_available"] is False
    assert [r["p"] for r in doc["holm"]].count(None) >= 1 and all(r["p_adj"] <= 1.0 for r in doc["holm"])
    assert "not testable" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
