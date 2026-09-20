"""Throughput bench: mock path -> throughput.json -> ``rhg.budget cost_model``; aggregation by hand; stub-GPU path."""

from __future__ import annotations

import json
import statistics

import datasets  # noqa: F401  (imported before the torch stub is installed)
import pytest
import stubs
from evalfix import full_candidates_dir, tiny_problems, write_jsonl
from stubs.helpers import script

from rhg import budget, runlog
from rhg.config import load_config
from rhg.data.fixture import fixture_candidates
from rhg.env import cache as grade_cache
from rhg.eval import bench
from rhg.eval.generate import honest_completion
from rhg.train import run as run_mod


def step(i, **kw):
    base = dict(step=i, reward_mean=0.0, loss=0.0, grad_norm=0.0, completion_len_mean=1.0, truncation_rate=0.0,
                frac_zero_adv_groups=0.0, hack_rt_rate_train=0.0, attempt_rt_rate_train=0.0, correct_rate_train=0.0,
                t_gen=1.0, t_reward=1.0, t_train=1.0, t_sync=0.0, t_step=3.0, tokens_gen=100, tokens_train=200)
    return runlog.StepRecord(**{**base, **kw})


@pytest.fixture(autouse=True)
def _restore_cache():
    yield
    grade_cache.configure_cache(None)


# ------------------------------------------------------------------ pure helpers, checked by hand
def test_aggregate_uses_steps_from_three_and_ratio_of_sums():
    rows = [
        step(1, t_step=50.0, t_gen=30.0, t_reward=9.0, t_train=9.0, tokens_gen=999, tokens_train=999),  # warm-up: ignored
        step(2, t_step=20.0, t_gen=9.0, t_reward=9.0, t_train=9.0, tokens_gen=999, tokens_train=999),  # warm-up: ignored
        step(3, t_step=10.0, t_gen=4.0, t_reward=2.0, t_train=3.0, t_sync=1.0, tokens_gen=400, tokens_train=900),
        step(4, t_step=12.0, t_gen=6.0, t_reward=4.0, t_train=1.0, t_sync=3.0, tokens_gen=600, tokens_train=700),
    ]
    deltas = {1: {"hits": 0, "misses": 100, "dedup": 0}, 2: {"hits": 0, "misses": 100, "dedup": 0},
              3: {"hits": 30, "misses": 60, "dedup": 10}, 4: {"hits": 60, "misses": 20, "dedup": 20}}
    a = bench.aggregate(rows, deltas, cpu_workers=4, rollouts_per_step=100)
    assert a["gen_tokens_per_step"] == 500 and a["train_tokens_per_step"] == 800  # mean of (400, 600) / (900, 700)
    assert a["gen_tok_per_s"] == pytest.approx(1000 / 10)  # sum tokens / sum t_gen = 1000 / (4 + 6)
    assert a["train_tok_per_s"] == pytest.approx(1600 / 4)  # (900 + 700) / (3 + 1)
    assert a["exec_s_per_rollout"] == pytest.approx((2 + 4) * 4 / (2 * 100))  # sum t_reward * workers / rollouts
    assert a["sync_s_per_step"] == 2.0
    assert a["cache_hit_rate"] == pytest.approx((30 + 10 + 60 + 20) / 200)  # served / lookups over steps 3-4
    assert a["cache_hit_rate_all_steps"] == pytest.approx(120 / 400)
    assert a["step_times_s"] == [50.0, 20.0, 10.0, 12.0] and a["t_step_median_s"] == 11.0
    with pytest.raises(bench.BenchError, match="at least 3"):
        bench.aggregate(rows[:2], deltas, cpu_workers=4, rollouts_per_step=100)
    with pytest.raises(bench.BenchError, match="zero generation"):
        bench.aggregate(rows[:2] + [step(3, t_gen=0.0)], deltas, cpu_workers=4, rollouts_per_step=100)


def test_aggregate_output_satisfies_the_cost_model_schema_and_formula():
    """t_reward of BUDGET §2 (rollouts * exec / workers) reproduces the measured reward time by construction."""
    rows = [step(i, t_reward=0.5 * i) for i in range(1, 6)]
    a = bench.aggregate(rows, {}, cpu_workers=7, rollouts_per_step=128)
    tp = budget.parse_throughput({
        **{k: a[k] for k in ("gen_tokens_per_step", "gen_tok_per_s", "train_tokens_per_step", "train_tok_per_s",
                             "exec_s_per_rollout", "sync_s_per_step", "step_times_s")},
        "cpu_workers": 7, "rollouts_per_step": 128, "t_eval_s": 1.0, "t_startup_s": 1.0, "n_steps_measured": 5,
    })
    cost = budget.run_cost(tp, T=100, usd_per_hour=0.45)
    assert cost.t_reward == pytest.approx(statistics.fmean(r.t_reward for r in rows[2:]))  # mean of steps 3..5


def test_eval_pass_size_is_the_average_of_five_val_and_one_test_eval():
    cfg = load_config("hackable_subtle")
    # hand-computed: val 40x4 = 160, test 60x8 = 480 rollouts; (5*160 + 480)/6 = 213.33 rollouts = 53.3 problems of 4 samples
    assert (bench.PLANNED_VAL_PROBLEMS, bench.PLANNED_TEST_PROBLEMS) == (40, 60)
    assert bench.eval_pass_problems(cfg) == 54


def test_select_problems_is_seeded_and_sorted_and_marks_train():
    rows = [{"problem_id": f"p{i:02d}", "reward_tests": [1], "heldout_tests": [1]} for i in range(30)]
    a, b, c = bench.select_problems(rows, 10, 9000), bench.select_problems(rows, 10, 9000), bench.select_problems(rows, 10, 9001)
    assert list(a) == list(b) == sorted(a) and len(a) == 10 and set(a) != set(c)
    assert all(p["split"] == "train" for p in a.values())
    with pytest.raises(bench.BenchError, match="duplicate"):
        bench.select_problems(rows + [rows[0]], 5, 1)


def test_load_candidates_keeps_only_problems_with_tests_and_survives_unicode_line_separators(tmp_path):
    rows = [{"problem_id": "a", "description": "x y\u0085z\x0cw", "reward_tests": [1], "heldout_tests": [1]},
            {"problem_id": "b", "reward_tests": [], "heldout_tests": [1]}, {"problem_id": "c", "reward_tests": [1]}]
    d = tmp_path / "proc"
    d.mkdir()
    (d / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    got = bench.load_candidates(d)  # str.splitlines() would have cut the first record at U+2028
    assert [r["problem_id"] for r in got] == ["a"] and got[0]["description"] == "x y\u0085z\x0cw"
    with pytest.raises(bench.BenchError, match="not found"):
        bench.load_candidates(tmp_path / "missing")


def test_uncached_timing_bypasses_the_cache_and_samples_64(tmp_path):
    problems = {p["problem_id"]: p for p in tiny_problems()}
    pids = sorted(problems) * 8  # 64 rollouts, 8 distinct (code, problem) pairs
    comps = [honest_completion(problems[p]) for p in pids]
    cfg_nc = load_config("hackable_subtle", ["sandbox.cache=false", "run.mode=mock"])
    grade_cache.configure_cache(None)
    before = grade_cache.cache_stats()
    per_rollout, n, dt = bench.time_uncached(cfg_nc, problems, pids, comps, None, 64, seed=9000, workers=4)
    assert n == 64 and per_rollout == pytest.approx(dt * 4 / 64) and per_rollout > 0
    assert grade_cache.cache_stats() == before  # the cache was neither read nor written nor deduplicated
    assert bench.time_uncached(cfg_nc, problems, pids, comps, None, 10, seed=9000, workers=4)[1] == 10


# ------------------------------------------------------------------ CLI guards
def test_help_and_usage_errors(capsys, tmp_path):
    with pytest.raises(SystemExit) as ei:
        bench.main(["--help"])
    assert ei.value.code == 0 and "--mock" in capsys.readouterr().out
    assert bench.main(["--mock", "--steps", "2", "--out", str(tmp_path / "t.json")]) == 2
    (tmp_path / "exists.json").write_text("{}")
    assert bench.main(["--mock", "--out", str(tmp_path / "exists.json")]) == 3  # refuses to overwrite without --force


def test_mock_output_never_defaults_to_the_real_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_run(args):
        seen["out"] = args.out
        raise bench.BenchError("stop here")

    monkeypatch.setattr(bench, "run_bench", fake_run)
    assert bench.main(["--mock"]) == 1
    assert seen["out"] == bench.MOCK_OUT and bench.MOCK_OUT != bench.REAL_OUT
    assert bench.main(["--backend", "trl"]) == 1 and seen["out"] == bench.REAL_OUT


# ------------------------------------------------------------------ bench --mock -> throughput.json -> cost_model
def test_bench_mock_writes_throughput_and_cost_model_prices_it(tmp_path, capsys, monkeypatch):
    proc = full_candidates_dir(tmp_path)
    out = tmp_path / "bench" / "throughput.json"
    ledger = tmp_path / "ledger.jsonl"
    code = bench.main(["--mock", "--steps", "3", "--processed-dir", str(proc), "--out", str(out), "--set", f"budget.ledger={ledger}"])
    printed = capsys.readouterr().out
    assert code == 0 and out.is_file()
    assert "UNCACHED" in printed and "CACHED" in printed and "hit rate" in printed and "conservative" in printed
    assert not ledger.exists()  # a mock bench is never billed
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["mock"] is True and raw["backend"] == "mock" and raw["n_problems"] == 40
    tp = budget.load_throughput(out)  # exactly the documented schema
    assert tp.mock and tp.n_steps_measured == 3 and len(tp.step_times_s) == 3 and tp.rollouts_per_step == 128
    assert tp.cpu_workers >= 1 and tp.cache_hit_rate is not None and 0.0 <= tp.cache_hit_rate <= 1.0
    assert tp.exec_s_per_rollout > 0 and tp.exec_s_per_rollout_uncached is not None and tp.exec_s_per_rollout_uncached > 0
    assert raw["n_uncached_rollouts"] == 64 and raw["eval_rollouts"] == 40 * 4 and 0.0 <= raw["cache_hit_rate_all_steps"] <= 1.0
    # fields are the aggregation of the logged steps (independent recomputation from steps.jsonl)
    rows = runlog.read_steps(out.parent / "run")
    warm = rows[2:]
    assert tp.gen_tokens_per_step == pytest.approx(sum(r.tokens_gen for r in warm) / len(warm))
    assert tp.exec_s_per_rollout == pytest.approx(sum(r.t_reward for r in warm) * tp.cpu_workers / (len(warm) * 128))
    assert list(tp.step_times_s) == pytest.approx([r.t_step for r in rows])
    # the cost model runs on it end to end (explicit output paths: never the default planning artifacts)
    monkeypatch.chdir(tmp_path)
    md, decision = tmp_path / "BUDGET_MEASURED.test.md", tmp_path / "budget_decision.test.md"
    assert budget.main(["cost_model", "--bench", str(out), "--out-md", str(md), "--decision", str(decision)]) == 0
    text = capsys.readouterr()
    assert "first fitting ladder step" in text.out and "NOT measurements" in text.err
    assert md.is_file() and decision.is_file()
    # ... and refuses to turn a mock bench into the default BUDGET_MEASURED.md / prereg/budget_decision.md
    assert budget.main(["cost_model", "--bench", str(out)]) == 3
    assert not (tmp_path / "BUDGET_MEASURED.md").exists() and not (tmp_path / "prereg").exists()
    # the launch guard of the run driver ignores nothing silently: a mock file at the real path is an error
    monkeypatch.setattr(run_mod, "BENCH_PATH", out)
    with pytest.raises(budget.BenchError, match="--mock"):
        run_mod.estimate_run_usd(load_config("hackable_subtle"), mock=False)


def test_mock_bench_needs_enough_problems_for_a_step(tmp_path, capsys):
    proc = tmp_path / "proc"
    write_jsonl(proc / "candidates.jsonl", fixture_candidates()[:8])
    assert bench.main(["--mock", "--processed-dir", str(proc), "--out", str(tmp_path / "o.json"), "--no-ledger"]) == 1
    assert "need >= 16 usable problems" in capsys.readouterr().err


# ------------------------------------------------------------------ real path against the stub GPU stack
def cloned_candidates(n_copies=2):
    out = []
    for k in range(n_copies):
        for p in fixture_candidates():
            out.append({**p, "problem_id": f"{p['problem_id']}-c{k}", "description": f"{p['description']}\n(variant {k})"})
    return out


@pytest.fixture
def stub_world(monkeypatch, tmp_path):
    w = stubs.install(monkeypatch)
    monkeypatch.setattr(grade_cache, "DEFAULT_DISK_DIR", tmp_path / "gradecache")
    return w


def test_real_backend_path_with_stubs_records_gpu_timings_and_bills_the_ledger(stub_world, tmp_path, capsys):
    cands = cloned_candidates()  # 80 problems >= the 64 floor
    proc = tmp_path / "proc"
    write_jsonl(proc / "candidates.jsonl", cands)
    script(stub_world, {p["problem_id"]: p for p in cands})
    out, ledger = tmp_path / "bench" / "throughput.json", tmp_path / "ledger.jsonl"
    code = bench.main(["--steps", "3", "--processed-dir", str(proc), "--out", str(out), "--set", f"budget.ledger={ledger}",
                       "--set", "grpo.prompts_per_step=4"])  # 32 rollouts per step keeps the test fast; shape is tested in the mock test
    captured = capsys.readouterr()
    assert code == 0, captured.err
    raw = json.loads(out.read_text(encoding="utf-8"))
    tp = budget.load_throughput(out)
    assert raw["mock"] is False and raw["backend"] == "trl" and raw["n_problems"] == 64 and not tp.mock
    assert raw["timers_measured"] == {"gen": True, "sync": True}
    assert tp.sync_s_per_step >= stub_world.sleep_sync * 0.9 and raw["phase_median_s"]["gen"] >= stub_world.sleep_gen * 0.9
    assert tp.gen_tok_per_s > 0 and tp.train_tok_per_s > 0 and tp.t_eval_s > raw["t_eval_load_s"] >= 0
    cfg = load_config("hackable_subtle")  # training-phase peak: CUDA context + trainer + colocated vLLM budget (fraction of the card)
    expected_peak = stub_world.context_gib + stub_world.trainer_gib + cfg.grpo.vllm_gpu_mem_util * stub_world.gpu_total_gib
    assert raw["peak_gpu_mem_gib"] == pytest.approx(expected_peak) and raw["peak_torch_reserved_gib"] > 0
    assert tp.t_startup_s >= 0 and tp.n_steps_measured == 3 and raw["n_uncached_rollouts"] == 32 == tp.rollouts_per_step
    assert raw["eval_rollouts"] == 54 * 4  # the timed eval pass has the planned average size when 54 problems exist
    assert tp.exec_s_per_rollout_uncached is not None and tp.cache_hit_rate is not None
    # billed once as a bench, nothing else
    entries = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [e["kind"] for e in entries] == ["bench"] and entries[0]["usd"] > 0 and entries[0]["note"] == "ok"
    # the eval pass used a fresh engine with the adapter of the last step, after the trainer was torn down
    assert len(stub_world.llm_inits) == 1 and stub_world.llm_inits[0]["enable_lora"] is True
    assert stub_world.events.index("destroy_model_parallel") < stub_world.events.index("LLM.__init__")
    assert "GPU" in captured.out and "CACHED" in captured.out
    # and the cost model accepts it as a measurement (no mock warning)
    assert budget.main(["cost_model", "--bench", str(out), "--out-md", str(tmp_path / "m.md"), "--decision", str(tmp_path / "d.md")]) == 0
    assert "NOT measurements" not in capsys.readouterr().err


def test_real_bench_refuses_fewer_than_64_problems_and_records_the_failed_attempt(stub_world, tmp_path, capsys):
    proc = tmp_path / "proc"
    write_jsonl(proc / "candidates.jsonl", fixture_candidates())  # 40 < 64
    script(stub_world, {p["problem_id"]: p for p in fixture_candidates()})
    ledger = tmp_path / "ledger.jsonl"
    code = bench.main(["--steps", "3", "--processed-dir", str(proc), "--out", str(tmp_path / "o.json"), "--set", f"budget.ledger={ledger}"])
    assert code == 1 and "need >= 64 usable problems" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()
    assert json.loads(ledger.read_text().splitlines()[0])["note"].startswith("failed")
    # the test-only escape hatch works and is explicit
    code = bench.main(["--steps", "3", "--allow-few-problems", "--n-problems", "8", "--processed-dir", str(proc),
                       "--out", str(tmp_path / "o2.json"), "--no-ledger", "--set", "grpo.prompts_per_step=4"])
    assert code == 0 and json.loads((tmp_path / "o2.json").read_text())["n_problems"] == 8


def test_prompt_length_filter_drops_over_long_prompts(stub_world):
    cands = cloned_candidates()
    stub_world.tokens_per_word = 1
    cfg = load_config("hackable_subtle")
    book = run_mod.PromptBook({p["problem_id"]: p for p in cands})
    lengths = sorted(len(book(p["problem_id"], cfg.arm.hint).split()) for p in cands)
    limit = lengths[len(lengths) // 2]  # about half the candidates are longer than this
    kept, dropped = bench.filter_prompt_length(cands, load_config("hackable_subtle", [f"grpo.max_prompt_tokens={limit}"]))
    assert dropped == sum(1 for n in lengths if n > limit) > 0 and len(kept) + dropped == len(cands)
    assert all(len(book(p["problem_id"], cfg.arm.hint).split()) <= limit for p in kept)
