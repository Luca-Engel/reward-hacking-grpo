"""Pass-rate stages A/B on the synthetic fixture with mock generators. Mock/fixture only."""

from __future__ import annotations

import gzip
import json
import math
import threading
import time
from pathlib import Path

import pytest
from evalfix import full_candidates_dir, tiny_dir, tiny_problems, visible_only_completion, write_jsonl

from rhg import budget
from rhg.config import load_config
from rhg.data import build
from rhg.eval import generate as gen
from rhg.eval import pass_rate as pr
from rhg.eval import pipeline as pl
from rhg.eval.generate import MockGenerator

CFG = load_config("clean_none")
N = 16


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(stage, d, **kw):
    kw.setdefault("mock", True)
    kw.setdefault("n", N)
    return pr.run_stage(stage, cfg=CFG, processed_dir=d, **kw)


def binom_ok(k, n, p, sigmas=5.0):
    return abs(k - n * p) <= sigmas * math.sqrt(max(n * p * (1 - p), 1e-9)) + (1e-9 if p in (0, 1) else 0)


# ------------------------------------------------------------------ planted rates are recovered
def test_stage_a_recovers_planted_per_problem_pass_rates(tmp_path):
    d = full_candidates_dir(tmp_path)
    plant = {"count-vowels": 0.0, "digit-sum": 1.0}  # exact extremes; everything else follows mock_success_prob
    prob = lambda pid: plant.get(pid, gen.mock_success_prob(pid))  # noqa: E731
    res = run("A", d, behavior=gen.planted_pass_behavior(prob))
    rows = {r["problem_id"]: r for r in res["rates"]}
    assert len(rows) == 40 and all(r["n"] == N for r in rows.values())
    assert rows["count-vowels"]["k_visible"] == 0 and rows["digit-sum"]["k_visible"] == N == rows["digit-sum"]["k_full"]
    for pid, r in rows.items():
        assert binom_ok(r["k_visible"], N, prob(pid)), (pid, r, prob(pid))
        assert r["k_full"] == r["k_visible"]  # honest reference solutions pass everything they pass visibly
    tot = sum(r["k_visible"] for r in rows.values())
    p_tot = sum(prob(p) for p in rows) * N
    assert abs(tot - p_tot) < 5 * math.sqrt(sum(N * prob(p) * (1 - prob(p)) for p in rows))
    assert (d / "passrate_A.jsonl").read_text(encoding="utf-8").count("\n") == 40


def test_k_full_is_below_k_visible_for_visible_only_solutions(tmp_path):
    d = tiny_dir(tmp_path)
    by_id = {p["problem_id"]: p for p in tiny_problems()}
    target = by_id["digit-sum"]

    def behavior(meta, rng):
        problem = meta["problem"]
        u = rng.random()
        if problem["problem_id"] == target["problem_id"]:
            if u < 0.25:
                return gen.honest_completion(problem)
            return visible_only_completion(problem) if u < 0.75 else gen.wrong_completion(problem)
        return gen.honest_completion(problem)

    res = run("A", d, behavior=behavior, n=40)
    row = next(r for r in res["rates"] if r["problem_id"] == "digit-sum")
    assert 0 < row["k_full"] < row["k_visible"] < 40
    assert binom_ok(row["k_full"], 40, 0.25) and binom_ok(row["k_visible"], 40, 0.75)
    others = [r for r in res["rates"] if r["problem_id"] != "digit-sum"]
    assert all(r["k_visible"] == r["k_full"] == 40 for r in others)


def test_side_file_counts_lengths_truncation_and_extraction_failures(tmp_path):
    d = tiny_dir(tmp_path)

    def behavior(meta, rng):
        problem, j = meta["problem"], meta["sample_idx"]
        if j % 4 == 0:
            return gen.no_code_completion(problem)
        if j % 4 == 1:
            return gen.honest_completion(problem) + "\n" + "z" * 5000  # code first, then rambling: cut by max_tokens
        return gen.honest_completion(problem) if j % 4 == 2 else gen.hack_completion(problem)

    res = run("A", d, behavior=behavior, n=8, limit=3)
    stats = read(d / "passrate_A_stats.jsonl")
    assert [s["problem_id"] for s in stats] == [p["problem_id"] for p in tiny_problems()[:3]]
    for s in stats:
        assert (s["n"], s["stage"], s["seed"]) == (8, "A", pr.STAGE_SEEDS["A"])
        assert s["n_truncated"] == 2 and s["n_extract_fail"] == 2 and s["n_defines_rt"] == 2
        assert s["n_timeout"] == s["n_crash"] == 0 and s["n_tokens_max"] == 1024
        assert s["n_tokens_mean"] > 0
    rates = read(d / "passrate_A.jsonl")
    assert all(r["k_visible"] == 4 for r in rates)  # j%4 in {1, 2}: honest; hack (3) and no-code (0) fail visibly
    s = res["summary"]
    assert s["truncation_rate"] == 0.25 and s["extract_fail_rate"] == 0.25


def test_output_schema_is_exactly_the_documented_one(tmp_path):
    d = tiny_dir(tmp_path)
    run("A", d)
    for r in read(d / "passrate_A.jsonl"):
        assert set(r) == {"problem_id", "n", "k_visible", "k_full"}
        assert 0 <= r["k_full"] <= r["k_visible"] <= r["n"] and r["n"] > 0
    assert build.read_passrates(d / "passrate_A.jsonl", {p["problem_id"] for p in tiny_problems()})


# ------------------------------------------------------------------ stage seeds
def test_stage_seeds_differ_and_are_enforced(tmp_path):
    assert pr.STAGE_SEEDS["A"] != pr.STAGE_SEEDS["B"]
    d = tiny_dir(tmp_path)
    run("A", d)
    assert pr.recorded_seed("A", d) == pr.STAGE_SEEDS["A"]
    with pytest.raises(ValueError, match="independent"):
        pr.check_seed_independence("B", pr.STAGE_SEEDS["A"], d)  # the recorded stage-A seed
    with pytest.raises(ValueError, match="reserved"):
        pr.check_seed_independence("B", pr.STAGE_SEEDS["A"], tmp_path / "empty")
    with pytest.raises(ValueError):
        pr.stage_seed("C")
    write_stage_b_selection(d, [p["problem_id"] for p in tiny_problems()[:3]])
    with pytest.raises(ValueError, match="independent|reserved"):
        run("B", d, seed=pr.STAGE_SEEDS["A"])
    res = run("B", d)
    header, _ = pl.read_completions(d / "completions_B.jsonl.gz")
    assert header["groups"][0]["seed"] == pr.STAGE_SEEDS["B"] != pr.recorded_seed("A", d)
    assert res["summary"]["n_problems"] == 3


def write_stage_b_selection(d: Path, ids: list[str]) -> None:
    (d / "selected_A.json").write_text(json.dumps({"band": {}, "problem_ids": ids}), encoding="utf-8")


def test_stage_b_samples_are_independent_draws_of_the_same_prompts(tmp_path):
    d = tiny_dir(tmp_path)
    ids = [p["problem_id"] for p in tiny_problems()[:4]]
    beh = gen.planted_pass_behavior(lambda pid: 0.5)  # success and failure completions differ textually
    run("A", d, limit=4, n=8, behavior=beh)
    write_stage_b_selection(d, ids)
    run("B", d, n=8, behavior=beh)
    _, ra = pl.read_completions(d / "completions_A.jsonl.gz")
    _, rb = pl.read_completions(d / "completions_B.jsonl.gz")
    assert {r["prompt_sha"] for r in ra} == {r["prompt_sha"] for r in rb}  # identical prompts ...
    assert {r["seed"] for r in ra} == {pr.STAGE_SEEDS["A"]} and {r["seed"] for r in rb} == {pr.STAGE_SEEDS["B"]}
    texts_a = {(r["problem_id"], r["sample_idx"]): r["completion"] for r in ra}
    texts_b = {(r["problem_id"], r["sample_idx"]): r["completion"] for r in rb}
    assert texts_a != texts_b  # ... but different draws (32 fair coin flips agreeing by chance: 2**-32)
    # control: the rows are exactly what the mock produces for the recorded seed, and that seed alone explains the difference
    problem = tiny_problems()[0]
    prompt = [pl.render_prompt(problem, "none", pr.load_prompts_cfg(), False)]
    meta = [{"problem_id": problem["problem_id"], "hint": "none", "problem": problem}]
    mock = MockGenerator(beh)
    same_a = mock.generate(prompt, 8, gen.SamplingParams(), seed=pr.STAGE_SEEDS["A"], prompt_meta=meta)[0]
    assert [c.text for c in same_a] == [texts_a[(problem["problem_id"], j)] for j in range(8)]


def test_stage_b_needs_the_band_selection_file(tmp_path, capsys):
    d = tiny_dir(tmp_path)
    assert pr.main(["--stage", "B", "--mock", "--processed-dir", str(d), "--n", "4"]) == 2
    assert "selected_A.json" in capsys.readouterr().err


# ------------------------------------------------------------------ end to end into the split stage
def test_stage_a_select_stage_b_then_split_end_to_end(tmp_path):
    d = full_candidates_dir(tmp_path)
    cfg = CFG
    run("A", d, n=N)
    sel = build.stage_split(d, cfg, select_only=True)
    assert sel["n_selected"] >= 8
    selected = json.loads((d / "selected_A.json").read_text(encoding="utf-8"))["problem_ids"]
    a_rows = {r["problem_id"]: r for r in read(d / "passrate_A.jsonl")}
    assert selected == sorted(pid for pid, r in a_rows.items() if build.in_band(r["k_visible"], r["n"], 0.10, 0.40))
    run("B", d, n=N)
    b_rows = {r["problem_id"]: r for r in read(d / "passrate_B.jsonl")}
    assert sorted(b_rows) == selected  # stage B covers exactly the selected problems
    splits = build.stage_split(d, cfg)
    problems = {p["problem_id"]: p for p in build.read_jsonl(d / "problems.jsonl")}
    assert sorted(problems) == selected and splits["n_selected"] == len(selected)
    for pid, p in problems.items():
        assert p["p_A"] == a_rows[pid]["k_visible"] / N
        assert p["p_B_full"] == b_rows[pid]["k_full"] / N and p["p_B_visible"] == b_rows[pid]["k_visible"] / N
        assert p["split"] in ("train", "val", "test")
    assert build.main(["--stage", "split", "--processed-dir", str(d)]) == 0  # the CLI consumer accepts the files too


# ------------------------------------------------------------------ generation / grading split
def snapshot(d: Path, stage: str) -> dict[str, bytes]:
    names = [f"passrate_{stage}.jsonl", f"passrate_{stage}_stats.jsonl", f"completions_{stage}.jsonl.gz"]
    return {n: (d / n).read_bytes() for n in names if (d / n).exists()}


def test_generate_only_then_grade_only_reproduces_the_default_path_byte_for_byte(tmp_path):
    ref = tiny_dir(tmp_path, "ref")
    run("A", ref, n=8)
    default = snapshot(ref, "A")
    assert len(default) == 3

    gen_dir = tiny_dir(tmp_path, "gen")
    res = run("A", gen_dir, n=8, generate_only=True)
    assert "summary" not in res and not (gen_dir / "passrate_A.jsonl").exists()
    assert (gen_dir / "completions_A.jsonl.gz").read_bytes() == default["completions_A.jsonl.gz"]  # no grading involved

    # grade on a "different machine": only the completions file and the problem files are needed
    cpu = tiny_dir(tmp_path, "cpu")
    (cpu / "completions_A.jsonl.gz").write_bytes((gen_dir / "completions_A.jsonl.gz").read_bytes())
    run("A", cpu, grade_only=True, mock=False, n=None)
    assert snapshot(cpu, "A") == default

    # grading is deterministic given the completions (fresh in-process cache, different chunking and worker count)
    from rhg.env.cache import configure_cache

    configure_cache(None)
    again = tiny_dir(tmp_path, "again")
    (again / "completions_A.jsonl.gz").write_bytes((gen_dir / "completions_A.jsonl.gz").read_bytes())
    run("A", again, grade_only=True, mock=False, n=None, chunk_prompts=3, workers=2)
    assert snapshot(again, "A") == default


def test_grade_only_rejects_mismatched_or_incomplete_inputs(tmp_path):
    d = tiny_dir(tmp_path)
    run("A", d, n=4, generate_only=True)
    path = d / "completions_A.jsonl.gz"
    lines = gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()

    def write(ls):
        path.write_bytes(gzip.compress(("\n".join(ls) + "\n").encode("utf-8"), mtime=0))

    write(lines[:-1])  # one sample missing
    with pytest.raises(pl.PipelineError, match="incomplete"):
        run("A", d, grade_only=True, mock=False, n=None)
    write(lines[:-1] + [lines[1]])  # duplicate instead of the missing sample
    with pytest.raises(pl.PipelineError, match="duplicate"):
        run("A", d, grade_only=True, mock=False, n=None)
    write(lines)
    tampered = tiny_problems()
    tampered[0]["description"] += " (changed)"
    write_jsonl(d / "candidates.jsonl", tampered)
    with pytest.raises(pl.PipelineError, match="differs from the one used"):
        run("A", d, grade_only=True, mock=False, n=None)
    write_jsonl(d / "candidates.jsonl", tiny_problems())
    with pytest.raises(pl.PipelineError, match="fixed by the completions"):
        run("A", d, grade_only=True, mock=False, n=8)
    (d / "completions_B.jsonl.gz").write_bytes(path.read_bytes())  # a stage-A file under the stage-B name
    with pytest.raises(pl.PipelineError, match="not generated for stage B"):
        run("B", d, grade_only=True, mock=False, n=None)
    with pytest.raises(pl.PipelineError, match="mutually exclusive"):
        run("A", d, grade_only=True, generate_only=True)
    with pytest.raises(pl.PipelineError, match="not found"):
        run("A", tmp_path / "nowhere", grade_only=True, n=None)


def test_completions_file_layout(tmp_path):
    d = tiny_dir(tmp_path)
    run("A", d, n=3, limit=2)
    header, rows = pl.read_completions(d / "completions_A.jsonl.gz")
    assert header["purpose"] == "passrate_A" and header["stage"] == "A" and header["generator"]["kind"] == "mock"
    assert header["sampling"] == {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1024}
    assert header["groups"] == [{"hint": "none", "seed": pr.STAGE_SEEDS["A"], "n": 3,
                                 "problem_ids": [p["problem_id"] for p in tiny_problems()[:2]]}]
    assert len(rows) == 6 and {r["sample_idx"] for r in rows} == {0, 1, 2}
    assert set(rows[0]) == {"kind", "problem_id", "hint", "sample_idx", "seed", "prompt_sha", "completion", "n_tokens", "truncated"}
    assert not (d / "completions_A.jsonl.gz.part").exists()


def test_failed_generation_leaves_only_a_part_file(tmp_path):
    d = tiny_dir(tmp_path)

    class Boom(MockGenerator):
        def generate(self, *a, **k):
            raise RuntimeError("gpu fell over")

    with pytest.raises(RuntimeError, match="gpu fell over"):
        run("A", d, generator=Boom())
    assert not (d / "completions_A.jsonl.gz").exists() and (d / "completions_A.jsonl.gz.part").exists()
    assert not (d / "passrate_A.jsonl").exists()


# ------------------------------------------------------------------ overlap of generation and grading
def test_grading_overlaps_generation_and_fails_fast(tmp_path, monkeypatch):
    d = tiny_dir(tmp_path)
    events: list[tuple[str, float]] = []
    lock = threading.Lock()

    def log(name):
        with lock:
            events.append((name, time.monotonic()))

    class Slowish(MockGenerator):
        def generate(self, *a, **k):
            log("gen_start")
            time.sleep(0.05)
            out = super().generate(*a, **k)
            log("gen_end")
            return out

    real = pl.grade_rows

    def slow_grade(*a, **k):
        log("grade_start")
        time.sleep(0.4)
        out = real(*a, **k)
        log("grade_end")
        return out

    monkeypatch.setattr(pl, "grade_rows", slow_grade)
    run("A", d, generator=Slowish(gen.planted_pass_behavior()), n=2, chunk_prompts=2)  # 4 chunks
    names = [e[0] for e in events]
    assert names.count("gen_start") == 4 and names.count("grade_start") == 4
    first_grade_end = names.index("grade_end")
    assert names[:first_grade_end].count("gen_start") >= 2  # the next chunk was already generating while chunk 0 graded

    events.clear()

    def bad_grade(*a, **k):
        raise RuntimeError("sandbox exploded")

    monkeypatch.setattr(pl, "grade_rows", bad_grade)
    g = Slowish(gen.planted_pass_behavior())
    with pytest.raises(RuntimeError, match="sandbox exploded"):
        run("A", d, generator=g, n=2, chunk_prompts=1)  # 8 chunks
    assert [e[0] for e in events].count("gen_start") < 8  # stopped renting the GPU early


# ------------------------------------------------------------------ CLI, ledger, safety
def test_cli_mock_writes_files_and_no_ledger(tmp_path, capsys):
    d = tiny_dir(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    rc = pr.main(["--stage", "A", "--mock", "--processed-dir", str(d), "--n", "4", "--set", f"budget.ledger={ledger.as_posix()}"])
    assert rc == 0 and not ledger.exists()
    out = capsys.readouterr().out
    assert "MOCK" in out and "stage A" in out and (d / "passrate_A.jsonl").exists()
    assert pr.main(["--stage", "A", "--mock", "--processed-dir", str(d), "--generate-only", "--grade-only"]) == 2
    assert pr.main(["--stage", "A", "--mock", "--processed-dir", str(d / "nope")]) == 2
    assert pr.main(["--stage", "A", "--mock", "--processed-dir", str(d), "--set", "no.such.key=1"]) == 2
    assert pr.main(["--stage", "A", "--mock", "--processed-dir", str(d), "--limit", "0"]) == 2


def test_mock_without_a_directory_never_touches_real_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fx = tiny_dir(tmp_path, "data/fixture/processed")
    real = tmp_path / "data" / "processed"
    write_jsonl(real / "candidates.jsonl", tiny_problems())
    assert pr.main(["--stage", "A", "--mock", "--n", "2", "--limit", "2"]) == 0
    assert (fx / "passrate_A.jsonl").exists() and not (real / "passrate_A.jsonl").exists()
    assert pr.main(["--stage", "A", "--mock", "--n", "2", "--limit", "2", "--generate-only"]) == 0


def test_real_path_records_the_ledger_with_generation_time_only(tmp_path):
    d = tiny_dir(tmp_path)
    ledger = tmp_path / "led" / "ledger.jsonl"

    class Sleepy(MockGenerator):
        def generate(self, *a, **k):
            time.sleep(0.2)
            return super().generate(*a, **k)

    res = run("A", d, n=2, limit=2, generator=Sleepy(gen.planted_pass_behavior()), mock=False, record_ledger=True, ledger=ledger)
    (entry,) = budget.read_entries(ledger)
    assert entry["kind"] == "passrate" and entry["run_id"] == "passrate_A"
    assert entry["wall_s"] == pytest.approx(res["load_wall_s"] + res["gen_wall_s"]) and res["gen_wall_s"] >= 0.2
    assert entry["usd_per_hour"] == CFG.budget.usd_per_hour
    assert entry["usd"] == pytest.approx(entry["wall_s"] / 3600 * CFG.budget.usd_per_hour)
    # grading is not part of the recorded time: a grade-only run adds nothing
    run("A", d, grade_only=True, mock=False, n=None, record_ledger=True, ledger=ledger)
    assert len(budget.read_entries(ledger)) == 1
    # generate-only on the box is billed as well
    run("A", d, n=2, limit=2, generate_only=True, generator=Sleepy(gen.planted_pass_behavior()), mock=False, record_ledger=True, ledger=ledger)
    assert len(budget.read_entries(ledger)) == 2


def test_vllm_backend_is_required_for_a_real_run(tmp_path, monkeypatch):
    d = tiny_dir(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "vllm", None)  # simulate a machine without vllm
    assert pr.main(["--stage", "A", "--processed-dir", str(d), "--n", "2", "--limit", "1"]) == 2
