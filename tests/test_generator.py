"""Generator interface: mock determinism, canned completions vs the real grader, lazy/fake vLLM backend."""

from __future__ import annotations

import importlib
import importlib.util
import math
import subprocess
import sys
import types

import pytest
from evalfix import tiny_problems

from rhg.config import load_config
from rhg.detect.ast_detector import analyze
from rhg.env.grader import grade_rollout
from rhg.eval import generate as gen
from rhg.eval.generate import (
    Completion,
    Generator,
    GeneratorBackendError,
    MockGenerator,
    SamplingParams,
    VLLMGenerator,
)
from rhg.seeds import derive_seed

CFG = load_config("clean_none")
PARAMS = SamplingParams.from_config(CFG)
PROBLEMS = {p["problem_id"]: p for p in tiny_problems()}
P = PROBLEMS["digit-sum"]  # class-style entry point
Q = PROBLEMS["sum-of-even-numbers"]  # plain-function entry point


def mk(problem, hint="none"):
    return {"problem_id": problem["problem_id"], "hint": hint, "problem": problem}


# ---------------------------------------------------------------- sampling params
def test_sampling_params_from_config_match_training_sampler():
    assert PARAMS == SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1024)
    assert PARAMS.as_dict() == {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1024}
    cfg = load_config("clean_none", overrides=["sampling.top_k=20", "grpo.max_completion_tokens=256"])
    assert SamplingParams.from_config(cfg).top_k == 20 and SamplingParams.from_config(cfg).max_tokens == 256


@pytest.mark.parametrize("kw", [{"temperature": 0.0}, {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0}, {"top_k": -2}, {"max_tokens": 0}])
def test_sampling_params_validation(kw):
    with pytest.raises(ValueError):
        SamplingParams(**kw)


# ---------------------------------------------------------------- MockGenerator
def rand_behavior(meta, rng):
    return f"{meta['prompt']}|{meta['sample_idx']}|{rng.random():.12f}"


def texts(out):
    return [[c.text for c in row] for row in out]


def test_mock_is_a_generator_and_returns_n_completions_per_prompt():
    g = MockGenerator(rand_behavior)
    assert isinstance(g, Generator)
    out = g.generate(["a", "b", "c"], 4, PARAMS, seed=1)
    assert len(out) == 3 and all(len(row) == 4 for row in out)
    assert all(isinstance(c, Completion) and c.n_tokens > 0 and not c.truncated for row in out for c in row)
    assert g.generate([], 2, PARAMS, seed=1) == []


def test_mock_determinism_and_per_seed_and_per_sample_variation():
    g = MockGenerator(rand_behavior)
    a = g.generate(["p1", "p2"], 5, PARAMS, seed=7)
    assert a == MockGenerator(rand_behavior).generate(["p1", "p2"], 5, PARAMS, seed=7)
    b = g.generate(["p1", "p2"], 5, PARAMS, seed=8)
    assert texts(a) != texts(b)
    assert all(len(set(texts(a)[i])) == 5 for i in range(2))  # samples of one prompt differ
    assert g.generate(["p1"], 2, PARAMS, seed=None) == g.generate(["p1"], 2, PARAMS, seed=0)


def test_mock_completion_depends_only_on_seed_prompt_and_sample_index():
    g = MockGenerator(lambda m, rng: f"{rng.random():.15f}")
    full = g.generate(["x", "y", "z"], 3, PARAMS, seed=3)
    assert g.generate(["z"], 3, PARAMS, seed=3)[0] == full[2]  # not batch position
    assert g.generate(["y", "x"], 3, PARAMS, seed=3) == [full[1], full[0]]  # not order
    assert g.generate(["x"], 2, PARAMS, seed=3)[0] == full[0][:2]  # n only adds samples


def test_mock_prompt_meta_is_explicit_and_validated():
    seen = []

    def behavior(meta, rng):
        seen.append(dict(meta))
        return "```python\npass\n```"

    g = MockGenerator(behavior)
    g.generate(["p0", "p1"], 2, PARAMS, seed=1, prompt_meta=[{"problem_id": "a", "hint": "S1"}, {"problem_id": "b", "hint": "E1"}])
    assert [(m["problem_id"], m["hint"], m["prompt"], m["sample_idx"]) for m in seen] == [
        ("a", "S1", "p0", 0), ("a", "S1", "p0", 1), ("b", "E1", "p1", 0), ("b", "E1", "p1", 1)]
    with pytest.raises(ValueError, match="prompt_meta"):
        g.generate(["p0", "p1"], 1, PARAMS, prompt_meta=[{}])
    for bad_n in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            g.generate(["p"], bad_n, PARAMS)
    with pytest.raises(TypeError):
        g.generate([b"bytes"], 1, PARAMS)  # type: ignore[list-item]


def test_mock_behavior_can_depend_on_the_hint_level():
    def behavior(meta, rng):
        return "hinted" if meta["hint"] != "none" and rng.random() < 0.5 else "plain"

    g = MockGenerator(behavior)
    metas = [{"hint": "none"}, {"hint": "S1"}]
    out = g.generate(["a", "b"], 200, PARAMS, seed=2, prompt_meta=metas)
    assert {c.text for c in out[0]} == {"plain"}
    frac = sum(c.text == "hinted" for c in out[1]) / 200
    assert abs(frac - 0.5) < 5 * math.sqrt(0.25 / 200)


def test_mock_truncation_and_token_counts():
    long = "y" * 4000
    out = MockGenerator(lambda m, r: long if m["sample_idx"] == 0 else "short").generate(["p"], 2, SamplingParams(max_tokens=100), seed=1)
    a, b = out[0]
    assert a.truncated and a.n_tokens == 100 and len(a.text) == 400
    assert not b.truncated and b.n_tokens == gen.estimate_tokens("short") == 2
    fits = MockGenerator(lambda m, r: "y" * 400).generate(["p"], 1, SamplingParams(max_tokens=100), seed=1)[0][0]
    assert not fits.truncated and fits.n_tokens == 100  # exactly at the budget is not truncation


def test_generate_for_passes_meta_only_to_backends_that_want_it():
    calls = []

    class Plain:
        def generate(self, prompts, n, params, seed=None):
            calls.append(("plain", seed))
            return [[Completion("t", 1, False)] * n for _ in prompts]

        def close(self):
            pass

    metas = [{"problem_id": "x"}]
    assert gen.generate_for(Plain(), ["p"], metas, 2, PARAMS, seed=5)[0][0].text == "t"
    assert calls == [("plain", 5)]
    seen = []
    m = MockGenerator(lambda meta, rng: seen.append(meta["problem_id"]) or "z")
    gen.generate_for(m, ["p"], metas, 1, PARAMS, seed=5)
    assert seen == ["x"]


# ---------------------------------------------------------------- canned completions vs the real grader
def labels_of(problem, completion):
    return grade_rollout(problem, completion, reward_mode="clean", cfg=CFG).labels


@pytest.mark.parametrize("problem", [P, Q], ids=lambda p: p["problem_id"])
def test_canned_completions_get_the_intended_labels(problem):
    honest = labels_of(problem, gen.honest_completion(problem))
    assert honest["correct"] and not honest["defines_rt"] and not honest["hack_rt"]

    wrong = labels_of(problem, gen.wrong_completion(problem))
    assert not wrong["visible_pass"] and not wrong["defines_rt"] and not wrong["correct"]

    attempt = labels_of(problem, gen.honest_attempt_completion(problem))
    assert attempt["attempt_rt"] and attempt["correct"] and not attempt["hack_rt"]

    hack = labels_of(problem, gen.hack_completion(problem))
    assert hack["hack_rt"] and hack["rt_ok"] and not hack["heldout_pass"]

    obf = gen.obfuscated_hack_completion(problem)
    lab = labels_of(problem, obf)
    assert lab["hack_rt"] and lab["defines_rt"]
    code = obf.split("```python\n", 1)[1].rsplit("```", 1)[0]
    assert not analyze(code, "narrow").flag  # invisible to the leaky in-loop monitor ...
    assert analyze(gen.hack_completion(problem).split("```python\n", 1)[1].rsplit("```", 1)[0], "narrow").flag  # ... unlike the plain hack

    none = labels_of(problem, gen.no_code_completion(problem))
    assert not any(none.values())


def test_planted_pass_behavior_matches_its_probability_function_and_is_binomial():
    def prob(pid):
        return {"digit-sum": 0.0, "sum-of-even-numbers": 1.0}.get(pid, 0.3)

    beh = gen.planted_pass_behavior(prob, p_no_code=0.5)
    n = 400
    out = MockGenerator(beh).generate(["a", "b", "c"], n, PARAMS, seed=11,
                                      prompt_meta=[mk(PROBLEMS["digit-sum"]), mk(PROBLEMS["sum-of-even-numbers"]), mk(PROBLEMS["digit-sum"] | {"problem_id": "other"})])
    def kind(c):
        return "ok" if c.text.startswith("Here is my solution") else ("wrong" if "```" in c.text else "none")

    kinds = [[kind(c) for c in row] for row in out]
    assert set(kinds[0]) <= {"wrong", "none"} and set(kinds[1]) == {"ok"}  # p = 0 and p = 1 are exact
    k = kinds[2].count("ok")
    assert abs(k - 0.3 * n) < 5 * math.sqrt(n * 0.3 * 0.7)
    assert abs(kinds[0].count("none") - 0.5 * n) < 5 * math.sqrt(n * 0.25)  # half of the failures carry no code


def test_mock_success_prob_is_deterministic_and_in_range():
    ps = [gen.mock_success_prob(f"p{i}") for i in range(200)]
    assert ps == [gen.mock_success_prob(f"p{i}") for i in range(200)]
    assert all(0.05 <= p <= 0.60 for p in ps) and max(ps) - min(ps) > 0.3


# ---------------------------------------------------------------- VLLMGenerator: lazy import
def test_importing_generate_does_not_import_vllm():
    code = "import sys; import rhg.eval.generate, rhg.eval.pass_rate, rhg.eval.probe_hints; print('vllm' in sys.modules, 'torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


@pytest.mark.skipif(importlib.util.find_spec("vllm") is not None, reason="vllm is installed here")
def test_constructing_vllm_generator_without_vllm_raises_a_clear_error():
    with pytest.raises(GeneratorBackendError, match="vllm") as ei:
        VLLMGenerator("Qwen/Qwen3-1.7B", max_model_len=1792, gpu_memory_utilization=0.5, dtype="bfloat16")
    assert isinstance(ei.value, ImportError) and "MockGenerator" in str(ei.value)


# ---------------------------------------------------------------- VLLMGenerator against a fake vllm module
class FakeLLM:
    instances: list["FakeLLM"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeLLM.instances.append(self)

    def get_tokenizer(self):
        return types.SimpleNamespace(encode=lambda s: list(range(len(s.split()))))

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=True):
        self.calls.append((list(prompts), list(sampling_params), lora_request, use_tqdm))
        outs = []
        for p, sp in zip(prompts, sampling_params):
            comps = [types.SimpleNamespace(text=f"{p}#{j}", token_ids=list(range(sp.max_tokens if j == 0 else 3)),
                                           finish_reason="length" if j == 0 else "stop") for j in range(sp.n)]
            outs.append(types.SimpleNamespace(outputs=comps))
        return outs


class FakeVSP:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeLoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path):
        self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path


@pytest.fixture
def fake_vllm(monkeypatch):
    FakeLLM.instances.clear()
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = FakeLLM, FakeVSP
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = FakeLoRARequest
    vllm.lora = lora
    lora.request = req
    for name, mod in (("vllm", vllm), ("vllm.lora", lora), ("vllm.lora.request", req)):
        monkeypatch.setitem(sys.modules, name, mod)
    return vllm


def make_vllm(**kw):
    return VLLMGenerator("Qwen/Qwen3-1.7B", max_model_len=kw.pop("max_model_len", 50), gpu_memory_utilization=0.5, dtype="bfloat16", **kw)


def test_vllm_generator_builds_engine_and_sampling_params_from_config(fake_vllm):
    g = make_vllm(max_prompt_tokens=30)
    (llm,) = FakeLLM.instances
    assert llm.kwargs == {"model": "Qwen/Qwen3-1.7B", "dtype": "bfloat16", "max_model_len": 50, "gpu_memory_utilization": 0.5, "seed": 0}
    out = g.generate(["one two three", "four five"], 2, SamplingParams(0.9, 0.95, 20, 16), seed=42)
    prompts, sps, lora, use_tqdm = llm.calls[0]
    assert prompts == ["one two three", "four five"] and lora is None and use_tqdm is False
    assert all((sp.n, sp.temperature, sp.top_p, sp.top_k, sp.max_tokens) == (2, 0.9, 0.95, 20, 16) for sp in sps)
    assert [sp.seed for sp in sps] == [derive_seed(42, gen.prompt_sha(p)) for p in prompts]  # per-prompt, stable
    assert out[0][0] == Completion("one two three#0", 16, True) and out[0][1] == Completion("one two three#1", 3, False)


def test_vllm_generator_top_k_disabled_and_no_seed_pass_through(fake_vllm):
    g = make_vllm()
    g.generate(["a b"], 1, PARAMS, seed=None)
    sp = FakeLLM.instances[0].calls[0][1][0]
    assert sp.top_k == -1 and sp.seed is None and sp.temperature == 1.0 and sp.top_p == 1.0


def test_vllm_generator_seed_changes_per_request_seeds(fake_vllm):
    g = make_vllm()
    g.generate(["a b"], 1, PARAMS, seed=1)
    g.generate(["a b"], 1, PARAMS, seed=2)
    g.generate(["a b"], 1, PARAMS, seed=1)
    s = [c[1][0].seed for c in FakeLLM.instances[0].calls]
    assert s[0] == s[2] and s[0] != s[1]


def test_vllm_generator_lora_request(fake_vllm, tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        make_vllm(adapter_path=tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    g = make_vllm(adapter_path=tmp_path, max_lora_rank=32)
    llm = FakeLLM.instances[-1]
    assert llm.kwargs["enable_lora"] is True and llm.kwargs["max_lora_rank"] == 32
    g.generate(["a b"], 1, PARAMS, seed=1)
    lora = llm.calls[0][2]
    assert isinstance(lora, FakeLoRARequest) and lora.lora_path == str(tmp_path) and lora.lora_int_id == 1


def test_vllm_generator_too_long_prompts_are_not_generated(fake_vllm):
    g = make_vllm(max_model_len=50, max_prompt_tokens=10)
    long_prompt = " ".join(["w"] * 11)  # 11 "tokens" > max_prompt_tokens
    out = g.generate([long_prompt, "short one"], 2, SamplingParams(max_tokens=100), seed=1)
    llm = FakeLLM.instances[0]
    assert llm.calls[0][0] == ["short one"]  # the long prompt never reaches the engine
    assert out[0] == [Completion("", 0, True)] * 2 and g.stats["prompts_too_long"] == 1
    assert llm.calls[0][1][0].max_tokens == 48  # clipped to max_model_len - prompt length (2)


def test_vllm_generator_validates_engine_output_shape(fake_vllm):
    g = make_vllm()
    g._llm.generate = lambda *a, **k: []  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="outputs"):
        g.generate(["a b"], 1, PARAMS, seed=1)


def test_vllm_generator_from_config_uses_training_limits(fake_vllm):
    VLLMGenerator.from_config(CFG, gpu_memory_utilization=0.8)
    kw = FakeLLM.instances[0].kwargs
    assert kw["max_model_len"] == 768 + 1024 and kw["gpu_memory_utilization"] == 0.8 and kw["dtype"] == "bfloat16"
    assert kw["model"] == "Qwen/Qwen3-1.7B"


def test_vllm_generator_close_is_safe(fake_vllm):
    g = make_vllm()
    g.close()
    g.close()
