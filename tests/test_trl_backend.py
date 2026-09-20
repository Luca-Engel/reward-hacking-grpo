"""TrlBackend against the stub GPU stack (tests/stubs): kwarg names, call order, callbacks, adapters, post-hoc evals.

No real torch/trl/peft/vllm/transformers is involved. What is checked is the *contract* of our code with the pinned
TRL 1.13.0 API as recorded in ``docs/trl_grpoconfig_fields.json`` and read from TRL's source, not TRL's behaviour.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path

import datasets  # noqa: F401  (imported before the torch stub is installed so datasets' own probing sees no torch)
import pytest
import stubs
from stubs.helpers import by_prompt, script
from evalfix import tiny_dir, tiny_problems

from rhg import runlog
from rhg.config import load_config
from rhg.data.fixture import fixture_candidates
from rhg.env import cache as grade_cache
from rhg.env.monitor import make_monitor
from rhg.eval.generate import SamplingParams, VLLMGenerator
from rhg.train import run as run_mod
from rhg.train import trl_config, trl_trainer as tt
from rhg.train.backend import InvalidRunError, RunFailedError, TrainContext
from rhg.train.rollout_io import RolloutLogger, StepCounter, make_reward_fn

REPO = Path(__file__).resolve().parents[1]
PROBLEMS = {p["problem_id"]: p for p in tiny_problems()}
ARM = "hackable_subtle"


# ------------------------------------------------------------------ fixtures / helpers
@pytest.fixture
def world(monkeypatch, tmp_path):
    w = stubs.install(monkeypatch)
    monkeypatch.setattr(grade_cache, "DEFAULT_DISK_DIR", tmp_path / "gradecache")  # the driver enables the disk cache
    yield w
    grade_cache.configure_cache(None)


def make_cfg(tmp_path, steps=3, ppS=4, *extra, arm=ARM, seed=0):
    return load_config(
        arm,
        [f"grpo.max_steps={steps}", f"grpo.prompts_per_step={ppS}", f"run.output_root={tmp_path / 'runs'}",
         f"budget.ledger={tmp_path / 'ledger.jsonl'}", "eval.val_every=5", *extra],
        seed=seed,
    )


def make_ctx(cfg, problems, run_dir, backend, snapshot_steps=frozenset(), heartbeats=None):
    train_ids = run_mod.split_ids(problems, "train")
    schedule = run_mod.make_schedule(train_ids, cfg.grpo.max_steps, cfg.grpo.prompts_per_step, cfg.run.seed)
    book = run_mod.PromptBook(problems)
    counter = StepCounter(0)
    logger = RolloutLogger(cfg.run_id, None, counter)
    reward = make_reward_fn(cfg, problems, make_monitor(cfg.arm.monitor), logger)
    beats = heartbeats if heartbeats is not None else []
    ctx = TrainContext(
        cfg=cfg, problems=problems, schedule=schedule, prompts={pid: book(pid, cfg.arm.hint) for pid in train_ids},
        sampling=SamplingParams.from_config(cfg), reward_fn=reward, logger=logger, counter=counter,
        step_writer=runlog.StepWriter(run_dir), snapshot_steps=frozenset(snapshot_steps), snapshot_fn=backend.snapshot,
        heartbeat=beats.append,
    )
    return ctx


def steps_of(run_dir):
    return runlog.read_steps(run_dir)


def read_status(run_dir):
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def driver(tmp_path, proc, *extra, steps=10, arm=ARM, seed=0, backend="trl"):
    argv = ["--arm", arm, "--seed", str(seed), "--backend", backend, "--steps", str(steps),
            "--set", f"run.output_root={tmp_path / 'runs'}", "--set", f"budget.ledger={tmp_path / 'ledger.jsonl'}",
            "--set", f"data.processed_dir={proc}", "--set", "grpo.prompts_per_step=4", "--set", "eval.val_every=5",
            "--force", *extra]
    code = run_mod.main(argv)
    return code, tmp_path / "runs" / f"{arm}__s{seed}"


# ------------------------------------------------------------------ import / construction without and with the stack
def test_import_without_gpu_stack_does_not_import_it():
    code = (
        "import sys, rhg.train.trl_trainer, rhg.eval.bench; "
        "bad = [m for m in ('torch','transformers','trl','peft','vllm') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_construction_without_stack_raises_clear_error(monkeypatch, tmp_path):
    for name in tt.REQUIRED_MODULES:  # None in sys.modules = "import blocked", also when the package is installed
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(tt.GpuStackMissingError, match="requirements-gpu.txt") as ei:
        tt.TrlBackend(make_cfg(tmp_path), problems=PROBLEMS, run_dir=tmp_path)
    assert isinstance(ei.value, ImportError) and "torch" in str(ei.value)
    with pytest.raises(ImportError, match="requirements-gpu.txt"):
        tt.create_backend(make_cfg(tmp_path), problems=PROBLEMS, run_dir=tmp_path)


def test_construction_builds_configs_from_the_mapping(world, tmp_path):
    cfg = make_cfg(tmp_path, ppS=16)
    b = tt.TrlBackend(cfg, problems=PROBLEMS, run_dir=tmp_path / "run")
    seen = world.kwargs_seen["GRPOConfig"]  # after the stub's real post_init arithmetic
    assert seen["generation_batch_size"] == cfg.rollouts_per_step == 128
    assert seen["generation_batch_size"] // seen["num_generations"] == cfg.grpo.prompts_per_step
    assert seen["per_device_train_batch_size"] * seen["gradient_accumulation_steps"] == cfg.rollouts_per_step
    assert seen["steps_per_generation"] == seen["gradient_accumulation_steps"]
    assert (seen["use_vllm"], seen["vllm_mode"], seen["beta"], seen["loss_type"]) == (True, "colocate", 0.0, "dr_grpo")
    assert seen["max_completion_length"] == cfg.grpo.max_completion_tokens and seen["shuffle_dataset"] is False
    assert b.lora_kwargs == trl_config.build_lora_config(cfg) and b.lora_config.r == cfg.lora.r
    assert b.name == "trl" and str(b.adapters_dir).endswith("adapters")


def test_invalid_kwargs_are_rejected(world, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    with pytest.raises(ValueError, match="not in the pinned signature"):
        tt.TrlBackend(cfg, problems=PROBLEMS, run_dir=tmp_path, extra_grpo={"no_such_field": 1})
    with pytest.raises(ValueError, match="batch keys"):
        tt.TrlBackend(cfg, problems=PROBLEMS, run_dir=tmp_path, extra_grpo={"num_generations": 4})
    # a key that slipped past the recorded-signature check is still caught against the (stub) installed class
    real = trl_config.build_grpo_config
    monkeypatch.setattr(tt.trl_config, "build_grpo_config", lambda *a, **k: {**real(*a, **{**k, "validate": False}), "typo_key": 1})
    with pytest.raises(ValueError, match="typo_key"):
        tt.TrlBackend(cfg, problems=PROBLEMS, run_dir=tmp_path)
    # and the stub classes themselves reject unknown kwargs like the real dataclasses
    import peft
    import trl

    with pytest.raises(TypeError):
        trl.GRPOConfig(typo_key=1)
    with pytest.raises(TypeError):
        peft.LoraConfig(rank=8)
    with pytest.raises(ValueError, match="thinking off"):
        tt.TrlBackend(make_cfg(tmp_path, 3, 4, "model.enable_thinking=true"), problems=PROBLEMS, run_dir=tmp_path)


# ------------------------------------------------------------------ one fake training run through the backend
def run_backend(world, tmp_path, *, steps=3, ppS=4, snapshot_steps=(), problems=None, cfg_extra=(), beats=None):
    problems = problems or PROBLEMS
    script(world, problems)
    cfg = make_cfg(tmp_path, steps, ppS, *cfg_extra)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    backend = tt.TrlBackend(cfg, problems=problems, run_dir=run_dir)
    ctx = make_ctx(cfg, problems, run_dir, backend, snapshot_steps, beats)
    seen: list[tuple[int, int]] = []
    inner = ctx.reward_fn

    def spy(**kw):
        seen.append((ctx.counter.value, len(kw["completions"])))
        return inner(**kw)

    ctx.reward_fn = spy
    try:
        backend.train(ctx)
    finally:
        ctx._writer.close()
    return cfg, backend, ctx, run_dir, seen


def test_reward_fn_gets_exact_batches_and_counter_increments(world, tmp_path):
    problems = {p["problem_id"]: {**p, "split": "train" if i < 20 else "val"} for i, p in enumerate(fixture_candidates())}
    cfg, backend, ctx, run_dir, seen = run_backend(world, tmp_path, steps=2, ppS=16, problems=problems)
    assert cfg.grpo.prompts_per_step * cfg.grpo.gens_per_prompt == 128
    assert seen == [(1, 128), (2, 128)]  # exactly ppS x gens completions per call, counter 1, 2
    assert [r.step for r in steps_of(run_dir)] == [1, 2]
    # the dataset order is the driver's schedule, one step at a time
    assert len(ctx.schedule) == 2 and all(len(set(s)) == 16 for s in ctx.schedule)


def test_step_log_is_schema_valid_and_token_counts_match_an_independent_count(world, tmp_path):
    world.tokens_per_word = 3
    cfg, backend, ctx, run_dir, seen = run_backend(world, tmp_path, steps=4)
    rows = steps_of(run_dir)  # StepRecord.model_validate_json: schema-valid, finite
    assert [r.step for r in rows] == [1, 2, 3, 4]
    table = by_prompt(PROBLEMS)
    prompt_of = {pid: pr for pr, p in table.items() for pid in [p["problem_id"]]}
    G = cfg.grpo.gens_per_prompt
    for r in rows:
        pids = ctx.schedule[r.step - 1]
        gen = sum(len(world.completion_fn(prompt_of[p], j)[1]) for p in pids for j in range(G))
        prompt_toks = sum(len(prompt_of[p].split()) * 3 * G for p in pids)
        assert r.tokens_gen == gen and r.tokens_train == gen + prompt_toks
        # truncation: samples whose ids do not end in EOS (j % 4 == 3 -> 2 of 8), TRL's own rule
        assert r.truncation_rate == pytest.approx(2 / 8)
        assert r.completion_len_mean == pytest.approx(gen / (len(pids) * G))
        assert r.loss == pytest.approx(0.01 * r.step) and r.grad_norm == 0.5
        # measured phases (stub sleeps: gen 2 ms, sync 1 ms, train 3 ms) and their consistency with t_step
        assert r.t_gen >= world.sleep_gen * 0.9 and r.t_sync >= world.sleep_sync * 0.9 and r.t_train >= world.sleep_train * 0.9
        assert r.t_reward > 0 and r.t_gen + r.t_reward + r.t_train + r.t_sync <= r.t_step + 1e-3
        assert 0.0 <= r.frac_zero_adv_groups <= 1.0 and 0.0 <= r.hack_rt_rate_train <= 1.0
    timing = [json.loads(x) for x in (run_dir / tt.TIMING_FILE).read_text().splitlines()]
    assert len(timing) == 4 and all(t["gen_measured"] and t["sync_measured"] for t in timing)
    assert backend.monitor.measured == {"gen": True, "sync": True}


def test_adapters_are_saved_at_the_planned_steps(world, tmp_path):
    cfg0 = load_config(ARM)
    plan = run_mod.eval_plan(cfg0)
    assert plan["snapshots"] == [0, 20, 40, 60, 80, 100]  # the deliverable's step set, from the driver's plan
    cfg, backend, ctx, run_dir, _ = run_backend(world, tmp_path, steps=100, snapshot_steps=plan["snapshots"], cfg_extra=("eval.val_every=20",))
    # driver behaviour: snapshot 0 is requested before train (no trainer yet) -> written from on_train_begin
    assert sorted(ctx.snapshots) == [20, 40, 60, 80, 100]
    saves = {Path(p).name: gs for p, gs in world.adapter_saves}
    assert saves == {"step_20": 20, "step_40": 40, "step_60": 60, "step_80": 80, "step_100": 100}
    for step in (20, 40, 60, 80, 100):
        d = backend.adapters_dir / f"step_{step}"
        assert (d / "adapter_config.json").is_file() and (d / "adapter_model.safetensors").is_file()


def test_step_zero_snapshot_is_written_from_on_train_begin(world, tmp_path):
    script(world, PROBLEMS)
    cfg = make_cfg(tmp_path, 2, 4)
    run_dir = tmp_path / "run"
    backend = tt.TrlBackend(cfg, problems=PROBLEMS, run_dir=run_dir)
    snap = backend.snapshot(0)  # what the driver does before train(): remembered, not written
    assert snap.step == 0 and not Path(snap.path).exists()
    with pytest.raises(RuntimeError, match="no live trainer"):
        backend.snapshot(7)
    ctx = make_ctx(cfg, PROBLEMS, run_dir, backend, {2})
    backend.train(ctx)
    ctx._writer.close()
    assert (Path(snap.path) / "adapter_config.json").is_file()
    order = [e for e in world.events if e.startswith(("GRPOTrainer", "save_pretrained", "sync_weights"))]
    assert order[:3] == ["GRPOTrainer.__init__", "save_pretrained step_0", "sync_weights"]  # untrained adapter first
    assert [Path(p).name for p, gs in world.adapter_saves] == ["step_0", "step_2"] and world.adapter_saves[0][1] == 0


def test_heartbeat_every_step_and_keepalive_during_setup(world, tmp_path):
    beats: list[str] = []
    run_backend(world, tmp_path, steps=3, beats=beats)
    assert [b for b in beats if b.startswith("step ")] == ["step 1", "step 2", "step 3"]
    assert "trainer ready" in beats
    got: list[str] = []
    with tt.keepalive(got.append, "loading", interval_s=0.01, grace_s=5):
        time.sleep(0.15)
    assert len(got) >= 3 and set(got) == {"loading"}
    got.clear()
    with tt.keepalive(got.append, "hung", interval_s=0.005, grace_s=0.1):  # beats stop after the grace period
        time.sleep(0.4)
        n_after_grace = len(got)
        time.sleep(0.2)
        assert n_after_grace >= 1 and len(got) == n_after_grace


def test_reward_call_stamps_the_step_counter_before_grading(world, tmp_path):
    world.global_step_offset = 1  # trainer_state.global_step disagrees with the callback-driven counter
    with pytest.raises(tt.RolloutBatchError, match="callback order"):
        run_backend(world, tmp_path, steps=2)


# ------------------------------------------------------------------ the reward wrapper on its own
class FakeCtx:
    def __init__(self, schedule, step=1, ppS=2, gens=2):
        self.cfg = types.SimpleNamespace(grpo=types.SimpleNamespace(prompts_per_step=ppS, gens_per_prompt=gens, max_completion_tokens=5))
        self.schedule, self.counter = schedule, types.SimpleNamespace(value=step)
        self.calls = []
        self.reward_fn = self._reward

    def _reward(self, **kw):
        self.calls.append(kw)
        return [1.0] * len(kw["completions"])


def wrapper(ctx, **kw):
    mon = types.SimpleNamespace(clock=time.perf_counter, note_reward=lambda *a: None)
    return tt.make_trl_reward(ctx, mon, {"P1": "p1", "P2": "p2", "P3": "p3"}, {"p1": 4, "p2": 6, "p3": 8}, lambda: [9, 0])


def test_reward_wrapper_checks_the_batch_and_derives_n_tokens_and_truncation():
    ctx = FakeCtx([["p1", "p2"]])
    fn = wrapper(ctx)
    ids = [[1, 2, 9], [1, 2, 3, 4, 5], [7, 9], [1, 2, 3, 4, 5]]  # lengths 3,5,2,5; ends in EOS: T,F,T,F
    out = fn(prompts=["P1", "P1", "P2", "P2"], completions=list("abcd"), completion_ids=ids, problem_id=["p1", "p1", "p2", "p2"])
    assert out == [1.0] * 4
    kw = ctx.calls[0]
    assert kw["problem_id"] == ["p1", "p1", "p2", "p2"] and kw["n_tokens"] == [3, 5, 2, 5]
    assert kw["truncated"] == [False, True, False, True]
    good = dict(prompts=["P1", "P1", "P2", "P2"], completions=list("abcd"), completion_ids=ids)
    for bad, match in [
        ({"prompts": ["P1", "P1", "P2", "P3"]}, "scheduled|exactly"),  # P3 is a train prompt but not in this step
        ({"prompts": ["P1", "P1", "P1", "P1"]}, "distinct prompts"),  # sampler repeated one prompt
        ({"prompts": ["P1", "P1", "P2", "nope"]}, "not one of the scheduled"),
        ({"completions": list("abc"), "completion_ids": ids[:3], "prompts": ["P1", "P1", "P2"]}, "expected"),
        ({"completions": [["chat"], "b", "c", "d"]}, "strings"),
        ({"problem_id": ["p2", "p1", "p2", "p2"]}, "disagrees"),
        ({"completion_ids": None}, "equal length"),
    ]:
        with pytest.raises(tt.RolloutBatchError, match=match):
            fn(**{**good, **bad})
    ctx.counter.value = 0
    with pytest.raises(tt.RolloutBatchError, match="outside a training step"):
        fn(**good)


def test_reward_wrapper_rejects_non_finite_rewards_as_invalid():
    ctx = FakeCtx([["p1", "p2"]])
    ctx.reward_fn = lambda **kw: [float("nan")] + [1.0] * (len(kw["completions"]) - 1)
    with pytest.raises(InvalidRunError, match="non-finite reward"):
        wrapper(ctx)(prompts=["P1", "P1", "P2", "P2"], completions=list("abcd"), completion_ids=[[9]] * 4)


# ------------------------------------------------------------------ StepMonitor arithmetic with a scripted clock
class Clock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


class RecordingCtx:
    def __init__(self):
        self.calls, self.beats = [], []
        self.heartbeat = self.beats.append

    def begin_step(self, step):
        self.calls.append(("begin", step))

    def end_step(self, step, **kw):
        self.calls.append(("end", step, kw))
        return types.SimpleNamespace(step=step, t_step=9.0, t_reward=0.5)


def test_step_monitor_phase_arithmetic_measured_and_approximated():
    ctx = RecordingCtx()
    # clock reads: step_begin=100 | reward 110.0..110.5 (t_reward) | step_end 116 | (no gpu sampler)
    mon = tt.StepMonitor(ctx, clock=Clock(100.0, 116.0))
    mon.step_begin(0)
    mon.add_sync(2.0)
    mon.note_reward(110.0, 110.5, [1.0, 0.0], 10, 20)
    mon.step_end(1)
    mon.log(1, {"loss": 0.25, "grad_norm": 1.5, "reward": 0.5})
    kind, step, kw = ctx.calls[1]
    assert (kind, step) == ("end", 1)
    # not measured (no wrapped generate): t_gen = (110 - 100) - t_sync = 8; t_train = 116 - 110.5 = 5.5
    assert kw["t_gen"] == pytest.approx(8.0) and kw["t_sync"] == 2.0 and kw["t_train"] == pytest.approx(5.5)
    assert kw["tokens_train"] == 30 and kw["loss"] == 0.25 and kw["grad_norm"] == 1.5 and kw["rewards"] == [1.0, 0.0]
    # measured gen wins over the approximation
    mon2 = tt.StepMonitor(ctx, clock=Clock(0.0, 3.0))
    mon2.measured["gen"] = True
    mon2.step_begin(0)
    mon2.add_gen(1.25)
    mon2.note_reward(2.0, 2.1, [1.0], 1, 1)
    mon2.step_end(1)
    mon2.log(1, {"loss": 0.0, "grad_norm": 0.0})
    assert ctx.calls[-1][2]["t_gen"] == 1.25 and ctx.calls[-1][2]["t_train"] == pytest.approx(0.9)


def test_step_monitor_protocol_errors():
    ctx = RecordingCtx()
    mon = tt.StepMonitor(ctx)
    with pytest.raises(RuntimeError, match="complete"):
        mon.step_begin(3)
    mon.step_begin(0)
    with pytest.raises(tt.RolloutBatchError, match="without a reward-function call"):
        mon.step_end(1)
    mon.note_reward(0.0, 0.1, [1.0], 1, 1)
    with pytest.raises(tt.RolloutBatchError, match="twice"):
        mon.note_reward(0.0, 0.1, [1.0], 1, 1)
    mon.step_end(1)
    mon.log(1, {"train_runtime": 5.0})  # summary log: ignored, the step stays pending
    with pytest.raises(RuntimeError, match="never logged its loss"):
        mon.step_begin(1)
    with pytest.raises(RuntimeError, match="no grad_norm"):
        mon.log(1, {"loss": 1.0})
    with pytest.raises(RuntimeError, match="never finalised"):
        mon.train_end()


def test_install_timers_is_optional_and_transparent():
    mon = tt.StepMonitor(RecordingCtx())
    tt.install_timers(types.SimpleNamespace(), mon)  # no vllm_generation at all
    tt.install_timers(types.SimpleNamespace(vllm_generation=types.SimpleNamespace(generate=None)), mon)
    assert mon.measured == {"gen": False, "sync": False}
    vg = types.SimpleNamespace(generate=lambda *a, **k: ("out", a, k), sync_weights=lambda: time.sleep(0.01))
    tt.install_timers(types.SimpleNamespace(vllm_generation=vg), mon)
    assert vg.generate(1, x=2) == ("out", (1,), {"x": 2}) and mon.measured == {"gen": True, "sync": True}
    vg.sync_weights()
    assert mon._sync_s >= 0.009 and mon._gen_s >= 0.0


# ------------------------------------------------------------------ failure handling through the driver
@pytest.fixture
def proc(tmp_path):
    return tiny_dir(tmp_path, "proc")


def test_oom_is_status_failed_reason_oom(world, tmp_path, proc):
    script(world, PROBLEMS)
    world.oom_at_step = 2
    code, d = driver(tmp_path, proc, steps=4)
    st = read_status(d)
    assert code == 1 and st["status"] == "failed" and st["reason"] == "oom" and st["step"] >= 1
    assert json.loads((d / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("attr,step", [("nan_loss_at_step", 2), ("inf_grad_at_step", 3)])
def test_non_finite_loss_or_grad_is_invalid(world, tmp_path, proc, attr, step):
    script(world, PROBLEMS)
    setattr(world, attr, step)
    code, d = driver(tmp_path, proc, steps=4)
    st = read_status(d)
    assert code == 1 and st["status"] == "invalid" and "non-finite" in st["reason"] and f"step {step}" in st["reason"]
    assert len(steps_of(d)) == step  # the offending step is still logged for diagnosis


def test_nan_reward_from_the_grader_is_invalid(world, tmp_path, proc, monkeypatch):
    script(world, PROBLEMS)
    monkeypatch.setattr("rhg.train.rollout_io.grade_batch", lambda items, **kw: [types.SimpleNamespace(
        reward=float("nan"), labels={k: False for k in runlog.Labels.model_fields}, raw={"code_extracted": False},
        monitor={}) for _ in items])
    code, d = driver(tmp_path, proc, steps=3)
    assert code == 1 and read_status(d)["status"] == "invalid" and "non-finite reward" in read_status(d)["reason"]


def test_missing_grad_norm_or_loss_log_fails_loudly(world, tmp_path, proc):
    script(world, PROBLEMS)
    world.missing_grad_norm = True
    code, d = driver(tmp_path, proc, steps=3)
    assert code == 1 and read_status(d)["status"] == "failed" and "grad_norm" in read_status(d)["reason"]
    world.missing_grad_norm, world.skip_loss_log = False, True
    code, d = driver(tmp_path, proc, steps=3)
    assert code == 1 and read_status(d)["status"] == "failed" and "never logged its loss" in read_status(d)["reason"]


def test_sampler_misconfiguration_fails_at_the_first_step(world, tmp_path, proc):
    script(world, PROBLEMS)
    world.wrong_batch_at_step = 1
    code, d = driver(tmp_path, proc, steps=3)
    assert code == 1 and read_status(d)["status"] == "failed" and "RolloutBatchError" in read_status(d)["reason"]
    assert steps_of(d) == [] or len(steps_of(d)) == 0


def test_over_long_prompt_fails_before_any_model_is_loaded(world, tmp_path, proc):
    script(world, PROBLEMS)
    world.tokens_per_word = 500  # every prompt now has hundreds of "tokens" vs max_prompt_tokens 768
    code, d = driver(tmp_path, proc, steps=3)
    st = read_status(d)
    assert code == 1 and st["status"] == "failed" and "PromptTooLongError" in st["reason"]
    assert "GRPOTrainer.__init__" not in world.events and not world.llm_inits


def test_prompt_length_helper():
    class T:
        def __call__(self, text):
            return {"input_ids": [list(range(len(t.split()))) for t in text]}

    assert tt.check_prompt_lengths({"a": "x y z", "b": "x"}, T(), 3) == {"a": 3, "b": 1}
    with pytest.raises(tt.PromptTooLongError, match="a \\(3\\)"):
        tt.check_prompt_lengths({"a": "x y z", "b": "x"}, T(), 2)
    with pytest.raises(ValueError, match="same prompt"):
        tt.build_prompt_table({"a": PROBLEMS["digit-sum"], "b": PROBLEMS["digit-sum"]}, ["a", "b"], "none")


def test_is_oom_walks_the_exception_chain():
    class OutOfMemoryError(RuntimeError):
        pass

    assert tt.is_oom(OutOfMemoryError("x")) and tt.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    try:
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as inner:
            raise ValueError("engine died") from inner
    except ValueError as e:
        assert tt.is_oom(e)
    assert not tt.is_oom(ValueError("boom")) and not tt.is_oom(None)


# ------------------------------------------------------------------ post-hoc evals, memory, adapter lifecycle (driver end to end)
def test_driver_end_to_end_evals_every_adapter_with_a_fresh_engine_and_matches_the_mock_schema(world, tmp_path, proc):
    script(world, PROBLEMS)
    code, d = driver(tmp_path, proc, steps=10)
    assert code == 0, read_status(d)
    assert read_status(d)["status"] == "completed" and runlog.validate_run(d) == []
    # one fresh engine per snapshot {0, 5, 10}, each with LoRA enabled and the adapter of that step, closed before the next
    assert [k["enable_lora"] for k in world.llm_inits] == [True] * 3 and all(k["max_lora_rank"] == 32 for k in world.llm_inits)
    used_paths = []
    for g in world.llm_generates:
        if g["lora_path"] not in used_paths:
            used_paths.append(g["lora_path"])
    assert [Path(p).name for p in used_paths] == ["step_0", "step_5", "step_10"]
    assert world.peak_used_gib < world.gpu_total_gib  # never two engines (or trainer + engine) at once
    ev = world.events
    first_eval_engine = ev.index("LLM.__init__")
    assert ev.index("destroy_model_parallel") < first_eval_engine and ev.index("cuda.empty_cache") < first_eval_engine
    assert ev.index("GRPOTrainer.__init__") < ev.index("save_pretrained step_0") < ev.index("sync_weights")
    # same record schema and evaluation points as the mock backend
    code_m, dm = driver(tmp_path / "mock", proc, steps=10, backend="mock")
    assert code_m == 0
    ev_t, ev_m = runlog.read_evals(d), runlog.read_evals(dm)
    assert [p.key for p in ev_t.points] == [p.key for p in ev_m.points]
    assert [sorted(p.totals) for p in ev_t.points] == [sorted(p.totals) for p in ev_m.points]
    assert [p.n_rollouts for p in ev_t.points] == [p.n_rollouts for p in ev_m.points]
    rt = next(runlog.iter_rollouts(d, phase="eval_val"))
    rm = next(runlog.iter_rollouts(dm, phase="eval_val"))
    assert rt.model_dump().keys() == rm.model_dump().keys() and rt.labels.model_dump().keys() == rm.labels.model_dump().keys()
    assert {r.phase for r in runlog.iter_rollouts(d)} == {r.phase for r in runlog.iter_rollouts(dm)}
    assert {s.step for s in steps_of(d)} == {s.step for s in steps_of(dm)} == set(range(1, 11))
    # vLLM per-request seeds come from the eval seed (same sampler seeding path as the mock)
    assert all(s is not None for g in world.llm_generates for s in g["seeds"])
    # adapters are deleted after a successful eval
    assert not (d / "adapters").exists()
    # sampler passed to the engine is the training sampler
    assert world.llm_inits[0]["gpu_memory_utilization"] == tt.EVAL_GPU_MEM_UTIL


def test_keep_adapters_flag_and_adapters_survive_a_failed_run(world, tmp_path, proc):
    script(world, PROBLEMS)
    code, d = driver(tmp_path, proc, "--keep-adapters", steps=10)
    assert code == 0 and sorted(p.name for p in (d / "adapters").iterdir()) == ["step_0", "step_10", "step_5"]
    world.leak_gib = 12.0  # the trainer does not give its memory back
    code, d = driver(tmp_path / "leak", proc, steps=10)
    st = read_status(d)
    assert code == 1 and st["status"] == "failed" and st["reason"].startswith("gpu_memory_not_released")
    assert (d / "adapters" / "step_10" / "adapter_config.json").is_file()  # kept for a re-run of the evals
    assert not world.llm_inits[3:]  # no eval engine was started on the polluted card


def test_engines_that_are_not_closed_would_be_caught_by_the_memory_check(world, tmp_path):
    script(world, PROBLEMS)
    cfg, backend, ctx, run_dir, _ = run_backend(world, tmp_path, steps=2, snapshot_steps={2})
    backend.end_training()
    assert world.used_gib == pytest.approx(world.context_gib)  # trainer + colocated engine budget fully released
    a = backend.eval_generator(ctx.snapshots[2])
    with pytest.raises(RunFailedError, match="gpu_memory_not_released"):
        backend.eval_generator(ctx.snapshots[2])  # a second engine while the first is open does not fit
    a.close()
    b = backend.eval_generator(ctx.snapshots[2])  # ... but does once the first was closed
    b.close()
    with pytest.raises(FileNotFoundError, match="no saved adapter"):
        backend.eval_generator(tt.AdapterSnapshot(9, str(tmp_path / "nope")))


def test_eval_generator_is_a_vllm_generator_with_the_adapter(world, tmp_path):
    cfg, backend, ctx, run_dir, _ = run_backend(world, tmp_path, steps=2, snapshot_steps={2})
    backend.end_training()
    gen = backend.eval_generator(ctx.snapshots[2])
    assert isinstance(gen, VLLMGenerator) and gen.adapter_path == ctx.snapshots[2].path
    out = gen.generate(["hello world"], 2, SamplingParams.from_config(cfg), seed=1)
    assert world.llm_generates[-1]["lora_path"] == ctx.snapshots[2].path and len(out[0]) == 2
    gen.close()


def test_delete_snapshots_only_touches_the_adapters_dir(world, tmp_path):
    cfg, backend, ctx, run_dir, _ = run_backend(world, tmp_path, steps=2, snapshot_steps={2})
    keep = run_dir / "steps.jsonl"
    assert keep.is_file() and backend.adapters_dir.is_dir()
    backend.delete_snapshots()
    assert not backend.adapters_dir.exists() and keep.is_file()
    backend.adapters_dir = tmp_path  # not inside run_dir: refuse to delete
    (tmp_path / "precious.txt").write_text("x")
    backend.delete_snapshots()
    assert (tmp_path / "precious.txt").is_file()


def test_close_releases_a_live_trainer(world, tmp_path):
    script(world, PROBLEMS)
    cfg = make_cfg(tmp_path, 2, 4)
    backend = tt.TrlBackend(cfg, problems=PROBLEMS, run_dir=tmp_path / "run")
    backend.close()  # nothing built yet: fine, idempotent
    backend.close()
    cfg, backend, ctx, run_dir, _ = run_backend(world, tmp_path / "second", steps=2)
    assert world.used_gib > world.context_gib
    backend.close()
    assert world.used_gib == pytest.approx(world.context_gib)
