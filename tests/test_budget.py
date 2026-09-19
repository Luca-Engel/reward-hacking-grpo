import json
import subprocess
import sys
import threading

import pytest

from rhg import budget as bg

# Synthetic throughput file with hand-computed consequences (see comments).
SYNTH = {
    "gen_tokens_per_step": 51200,   # / 2000 tok/s      -> t_gen    = 25.6 s
    "gen_tok_per_s": 2000,
    "train_tokens_per_step": 60000,  # / 3000 tok/s      -> t_train  = 20.0 s
    "train_tok_per_s": 3000,
    "exec_s_per_rollout": 0.05,      # 128*0.05/8        -> t_reward = 0.8 s
    "cpu_workers": 8,
    "rollouts_per_step": 128,
    "sync_s_per_step": 2.0,          #                   -> t_step   = 48.4 s
    "t_eval_s": 120.0,               # 6 evals           -> 720 s
    "t_startup_s": 300.0,
    "n_steps_measured": 6,
    "step_times_s": [90.0, 60.0, 50.0, 52.0, 500.0, 51.0],  # median of steps >= 3 = (51+52)/2
}
T, USD_H = 100, 0.5
RUN_S = 100 * 48.4 + 6 * 120.0 + 300.0        # 5860 s
RUN_USD = RUN_S / 3600 * 0.5 * 1.08           # 0.879


@pytest.fixture
def bench(tmp_path):
    p = tmp_path / "throughput.json"
    p.write_text(json.dumps(SYNTH), encoding="utf-8")
    return p


# ------------------------------------------------------------------ ledger


def test_record_prices_wall_time_and_sums(tmp_path):
    led = tmp_path / "l.jsonl"
    e = bg.record("train", "a__s0", 3600, usd_per_hour=0.5, ledger=led)
    assert e["usd"] == 0.5 and e["kind"] == "train" and e["wall_s"] == 3600.0
    bg.record("judge", "", 0.0, usd=0.1, note="calibration", ledger=led)
    bg.record("bench", "b", 1800, usd_per_hour=0.4, ledger=led)  # 0.2
    assert bg.spent_by_kind(led) == pytest.approx({"train": 0.5, "judge": 0.1, "bench": 0.2})
    assert bg.spent_usd(led) == pytest.approx(0.8)
    entries = bg.read_entries(led)
    assert [set(x) for x in entries] == [{"ts", "kind", "run_id", "wall_s", "usd_per_hour", "usd", "note"}] * 3
    assert entries[1]["usd_per_hour"] is None and entries[1]["note"] == "calibration"


def test_missing_ledger_is_zero_spend(tmp_path):
    assert bg.spent_usd(tmp_path / "none.jsonl") == 0.0
    assert bg.spent_by_kind(tmp_path / "none.jsonl") == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "nope", "run_id": "x", "wall_s": 1, "usd": 1},
        {"kind": "train", "run_id": "x", "wall_s": -1, "usd": 1},
        {"kind": "train", "run_id": "x", "wall_s": 1},  # no price
        {"kind": "train", "run_id": "x", "wall_s": float("nan"), "usd": 1},
        {"kind": "judge", "run_id": "x", "wall_s": 0, "usd": -0.1},
    ],
)
def test_record_validation(tmp_path, kwargs):
    with pytest.raises(ValueError):
        bg.record(ledger=tmp_path / "l.jsonl", **kwargs)
    assert not (tmp_path / "l.jsonl").exists()


def test_corrupt_ledger_line_raises_not_skipped(tmp_path):
    led = tmp_path / "l.jsonl"
    bg.record("other", "x", 0, usd=1.0, ledger=led)
    with open(led, "a", encoding="utf-8") as f:
        f.write("{truncated\n")
    with pytest.raises(bg.LedgerCorruptError, match=":2:"):
        bg.spent_usd(led)


def test_concurrent_threaded_appends_do_not_corrupt(tmp_path):
    led = tmp_path / "l.jsonl"
    n_threads, per = 8, 40
    errs = []

    def work(i):
        try:
            for j in range(per):
                bg.record("other", f"t{i}", 1.0, usd=0.01, note="x" * 300 + f"{i}-{j}", ledger=led)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    lines = led.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * per
    parsed = [json.loads(x) for x in lines]  # every line is intact JSON
    assert len({p["note"] for p in parsed}) == n_threads * per
    assert bg.spent_usd(led) == pytest.approx(0.01 * n_threads * per)


def test_concurrent_process_appends_do_not_corrupt(tmp_path):
    led = tmp_path / "l.jsonl"
    code = (
        "import sys\nfrom rhg.budget import record\n"
        "for j in range(30):\n"
        "    record('other', sys.argv[2], 1.0, usd=0.01, note='y'*500+str(j), ledger=sys.argv[1])\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code, str(led), f"p{i}"]) for i in range(4)]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0, 0]
    lines = led.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 * 30
    assert all(json.loads(x)["kind"] == "other" for x in lines)
    assert bg.spent_usd(led) == pytest.approx(1.2)


# ------------------------------------------------------------------ launch guard


def test_guard_refuses_exactly_at_the_boundary(tmp_path):
    led = tmp_path / "l.jsonl"
    bg.record("train", "r0", 0, usd=27.0, ledger=led)
    assert bg.check_launch(1.0, 28.0, ledger=led).allowed          # 27 + 1 == 28 -> allowed
    assert not bg.check_launch(1.000001, 28.0, ledger=led).allowed  # a hair over -> refused
    assert bg.check_launch(0.0, 28.0, ledger=led).allowed
    assert bg.main(["--ledger", str(led), "check", "--next-run-usd", "1.0"]) == 0
    assert bg.main(["--ledger", str(led), "check", "--next-run-usd", "1.01"]) == 3


def test_guard_boundary_with_float_dust(tmp_path):
    led = tmp_path / "l.jsonl"
    for _ in range(3):
        bg.record("other", "x", 0, usd=0.1, ledger=led)  # 0.30000000000000004 in naive float sums
    assert bg.check_launch(27.7, 28.0, ledger=led).allowed


def test_guard_default_stop_at_and_empty_ledger(tmp_path):
    led = tmp_path / "none.jsonl"
    assert bg.check_launch(28.0, ledger=led).allowed
    assert not bg.check_launch(28.01, ledger=led).allowed
    assert bg.DEFAULT_STOP_AT_USD == 28.0


def test_guard_message_recommends_ladder_step(tmp_path, capsys):
    led = tmp_path / "l.jsonl"
    for i in range(20):  # 20 train runs at 1.38 = 27.6
        bg.record("train", f"a__s{i}", 0, usd=1.38, ledger=led)
    dec = bg.check_launch(0.5, 28.0, ledger=led)  # 28.1 > 28; $0.4 left -> 0 more runs; 20 done
    assert not dec.allowed
    assert dec.recommended.step == 2 and dec.recommended.runs == 19  # 22, 21 > 20 >= 19
    assert "ladder step 2" in dec.message and "drop clean_none" in dec.message
    assert bg.main(["--ledger", str(led), "check", "--next-run-usd", "0.5"]) == 3
    assert "ladder step 2" in capsys.readouterr().err


def test_guard_below_floor_message(tmp_path):
    led = tmp_path / "l.jsonl"
    bg.record("train", "a__s0", 0, usd=27.5, ledger=led)
    dec = bg.check_launch(1.0, 28.0, ledger=led)
    assert not dec.allowed and dec.recommended is None
    assert "no confirmatory study" in dec.message.lower()


def test_recommend_step_hand_cases():
    # (runs affordable in total) -> first ladder step with runs <= that number
    expect = {30: 0, 22: 0, 21: 1, 20: 2, 19: 2, 18: 3, 16: 3, 15: 4, 14: 4, 13: 5, 12: 6, 11: 6}
    for n, step in expect.items():
        assert bg.recommend_step(0, n).step == step, n
    assert bg.recommend_step(0, 10) is None
    assert bg.recommend_step(4, 7).step == 6  # done + affordable = 11
    assert bg.recommend_step(10, 0) is None


# ------------------------------------------------------------------ LADDER


def test_ladder_matches_budget_md_table():
    assert [s.runs for s in bg.LADDER] == [22, 21, 19, 16, 14, 13, 11]
    assert [s.step for s in bg.LADDER] == list(range(7))
    s0, s1, s2, s3, s4, s5, s6 = bg.LADDER
    assert s0.seeds == {
        "clean_none": 2, "clean_subtle": 5, "clean_explicit": 2, "hackable_none": 2,
        "hackable_subtle": 5, "hackable_explicit": 3, "hackable_subtle_ast": 3,
    }
    assert s1.seeds["hackable_none"] == 1
    assert s2.seeds["clean_none"] == 0
    assert s3.seeds["hackable_subtle_ast"] == 0
    assert s4.seeds["clean_explicit"] == 0
    assert s5.seeds["hackable_explicit"] == 2
    assert (s6.seeds["clean_subtle"], s6.seeds["hackable_subtle"]) == (4, 4)
    assert s6.seeds == {
        "clean_none": 0, "clean_subtle": 4, "clean_explicit": 0, "hackable_none": 1,
        "hackable_subtle": 4, "hackable_explicit": 2, "hackable_subtle_ast": 0,
    }
    assert bg.FLOOR_RUNS == 11 and all(s.consequence for s in bg.LADDER)


# ------------------------------------------------------------------ cost model


def test_throughput_median_is_robust_and_from_step_3(bench):
    tp = bg.load_throughput(bench)
    assert tp.t_step_measured_s == 51.5  # median([50, 52, 500, 51]); the 500 s outlier is ignored


def test_run_cost_matches_hand_computation(bench):
    c = bg.run_cost(bg.load_throughput(bench), T, USD_H)
    assert c.t_gen == pytest.approx(25.6, abs=1e-12)
    assert c.t_reward == pytest.approx(0.8, abs=1e-12)
    assert c.t_train == pytest.approx(20.0, abs=1e-12)
    assert c.t_sync == 2.0
    assert c.t_step == pytest.approx(48.4, abs=1e-12)
    assert c.run_s == pytest.approx(5860.0, abs=1e-9)
    assert c.run_usd == pytest.approx(0.879, abs=1e-12)  # 5860/3600 * 0.5 * 1.08


def test_t_step_source_options(bench):
    tp = bg.load_throughput(bench)
    assert bg.run_cost(tp, T, USD_H, t_step_source="measured").t_step == 51.5
    assert bg.run_cost(tp, T, USD_H, t_step_source="max").t_step == 51.5
    with pytest.raises(ValueError):
        bg.run_cost(tp, T, USD_H, t_step_source="bogus")


@pytest.mark.parametrize(
    "n_boxes,expected_step",
    [
        (2, 4),     # step 3: cost 14.064 ok but wall 16*5860/3600/2 = 13.02 h > 12; step 4: 12.306 USD, 11.39 h
        (3, 3),     # step 3: 14.064 USD <= 16 (19 runs = 16.70 no), wall 8.68 h
        (1, None),  # 11 runs alone need 17.9 h > 12 h: below the floor
    ],
)
def test_ladder_selection_hand_computed(bench, n_boxes, expected_step):
    cost = bg.run_cost(bg.load_throughput(bench), T, USD_H)
    rows = bg.evaluate_ladder(cost, n_boxes, main_cap=16.0, max_wall_h=12.0)
    chosen = bg.select_step(rows)
    assert (None if chosen is None else chosen.step.step) == expected_step
    for r in rows:  # independent per-row arithmetic
        assert r.main_usd == pytest.approx(r.step.runs * 0.879)
        assert r.wall_h == pytest.approx(r.step.runs * 5860.0 / 3600.0 / n_boxes)


def test_ladder_selection_full_design_when_everything_fits(bench):
    cost = bg.run_cost(bg.load_throughput(bench), T, USD_H)
    chosen = bg.select_step(bg.evaluate_ladder(cost, 4, main_cap=20.0, max_wall_h=12.0))
    assert chosen.step.step == 0  # 22*0.879 = 19.338 <= 20; 22*5860/3600/4 = 8.95 h


def test_ladder_cost_boundary_is_inclusive():
    cost = bg.RunCost(0, 0, 0, 0, 1.0, "formula", 3600.0, 1.0)
    rows = bg.evaluate_ladder(cost, n_boxes=100, main_cap=22.0)
    assert bg.select_step(rows).step.step == 0  # 22 runs * $1 == cap exactly
    assert bg.select_step(bg.evaluate_ladder(cost, 100, main_cap=21.99)).step.step == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("gen_tok_per_s"),
        lambda d: d.update(cpu_workers=0),
        lambda d: d.update(train_tok_per_s="fast"),
        lambda d: d.update(step_times_s=[50.0, 51.0]),
        lambda d: d.update(step_times_s=[50.0, -1.0, 3.0]),
        lambda d: d.update(rollouts_per_step=12.5),
    ],
)
def test_malformed_bench_is_rejected(tmp_path, mutate):
    d = dict(SYNTH)
    mutate(d)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(bg.BenchError):
        bg.load_throughput(p)
    argv = ["cost_model", "--bench", str(p), "--usd-per-hour", "0.5", "--T", "100",
            "--out-md", str(tmp_path / "m.md"), "--decision", str(tmp_path / "d.md")]
    assert bg.main(argv) == 2
    assert not (tmp_path / "d.md").exists()


# ------------------------------------------------------------------ cost_model CLI


def _argv(tmp_path, bench, *extra):
    return ["cost_model", "--bench", str(bench), "--usd-per-hour", "0.5", "--T", "100",
            "--out-md", str(tmp_path / "BUDGET_MEASURED.md"), "--decision", str(tmp_path / "prereg" / "budget_decision.md"),
            *extra]


def test_cost_model_missing_bench_exits_2_with_instructions(tmp_path, capsys):
    assert bg.main(_argv(tmp_path, tmp_path / "absent.json")) == 2
    err = capsys.readouterr().err
    assert "throughput" in err and "bench" in err
    assert not (tmp_path / "BUDGET_MEASURED.md").exists()


def test_cost_model_writes_reports(tmp_path, bench, capsys):
    assert bg.main(_argv(tmp_path, bench, "--n-boxes", "2")) == 0
    out = capsys.readouterr().out
    assert "first fitting ladder step: 4" in out
    measured = (tmp_path / "BUDGET_MEASURED.md").read_text(encoding="utf-8")
    assert "0.8790" in measured and "5860.0 s" in measured and "48.400 s" in measured
    assert "**Selected: ladder step 4" in measured
    decision = (tmp_path / "prereg" / "budget_decision.md").read_text(encoding="utf-8")
    assert "DRAFT" in decision and "step 4: drop clean_explicit" in decision and "14 runs" in decision
    assert "| hackable_subtle | 5 |" in decision and "| clean_explicit | 0 |" in decision


def test_cost_model_refuses_to_overwrite_decision_without_force(tmp_path, bench, capsys):
    argv = _argv(tmp_path, bench, "--n-boxes", "2")
    assert bg.main(argv) == 0
    decision = tmp_path / "prereg" / "budget_decision.md"
    decision.write_text("HUMAN EDITED", encoding="utf-8")
    assert bg.main(argv) == 3
    assert "--force" in capsys.readouterr().err
    assert decision.read_text(encoding="utf-8") == "HUMAN EDITED"
    assert bg.main(argv + ["--force"]) == 0
    assert "DRAFT" in decision.read_text(encoding="utf-8")


def test_cost_model_below_floor_is_reported(tmp_path, bench, capsys):
    assert bg.main(_argv(tmp_path, bench, "--n-boxes", "1")) == 0
    assert "no confirmatory study" in capsys.readouterr().out.lower()
    assert "no confirmatory study" in (tmp_path / "prereg" / "budget_decision.md").read_text(encoding="utf-8").lower()


def test_cost_model_warns_when_step_time_disagrees_with_formula(tmp_path):
    d = dict(SYNTH, step_times_s=[90.0, 60.0, 90.0, 92.0, 91.0])  # ~91 s measured vs 48.4 s formula
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert bg.main(_argv(tmp_path, p)) == 0
    assert "WARNING" in (tmp_path / "BUDGET_MEASURED.md").read_text(encoding="utf-8")


def test_record_and_status_cli(tmp_path, capsys):
    led = str(tmp_path / "l.jsonl")
    assert bg.main(["--ledger", led, "record", "--kind", "train", "--run-id", "x__s0", "--wall-s", "3600",
                    "--usd-per-hour", "0.45"]) == 0
    assert bg.main(["record", "--ledger", led, "--kind", "judge", "--run-id", "j", "--wall-s", "0", "--usd", "0.05"]) == 0
    assert bg.main(["status", "--ledger", led]) == 0
    out = capsys.readouterr().out
    assert "train" in out and "$0.4500" in out and "total" in out and "$0.5000" in out


def test_module_runs_as_script(tmp_path):
    led = tmp_path / "l.jsonl"
    r = subprocess.run([sys.executable, "-m", "rhg.budget", "--ledger", str(led), "check", "--next-run-usd", "29"],
                       capture_output=True, text=True)
    assert r.returncode == 3 and "REFUSED" in r.stderr
    r = subprocess.run([sys.executable, "-m", "rhg.budget", "check", "--ledger", str(led), "--next-run-usd", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0
