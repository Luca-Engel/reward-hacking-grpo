"""GPU pins, recorded TRL signature, config -> GRPOConfig/LoraConfig mapping, memory estimator.

CPU only: nothing here imports trl/torch/vllm/peft (one test is skipped unless ``trl`` happens to be installed).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rhg.config import load_config
from rhg.train import gpu_memory as gm
from rhg.train import trl_config as tc

REPO = Path(__file__).resolve().parents[1]
FIELDS = json.loads((REPO / "docs" / "trl_grpoconfig_fields.json").read_text(encoding="utf-8"))
GRPO_FIELDS = set(FIELDS["GRPOConfig"])
LORA_FIELDS = set(FIELDS["LoraConfig"])


def _cfg(*overrides: str, seed: int = 0, arm: str = "hackable_subtle"):
    return load_config(arm, list(overrides), seed=seed)


def _pairs(p: int, g: int, *extra: str):
    return _cfg(f"grpo.prompts_per_step={p}", f"grpo.gens_per_prompt={g}", *extra)


# --------------------------------------------------------------------------------------------- batch arithmetic


@pytest.mark.parametrize("p,g", [(16, 8), (16, 4), (3, 5), (4, 4), (7, 2), (1, 2), (5, 3), (13, 8), (2, 16)])
@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_split_rollouts_matches_brute_force(p, g, cap):
    r = p * g
    micro, accum = tc.split_rollouts(p, g, cap)
    best = max(d for d in range(1, r + 1) if r % d == 0 and d <= cap)  # brute force over all divisors
    assert micro == best and micro * accum == r and micro <= cap


@pytest.mark.parametrize("p,g", [(16, 8), (3, 5), (4, 4), (7, 2), (1, 2), (13, 8)])
def test_one_optimizer_step_sees_exactly_prompts_per_step_distinct_prompts(p, g):
    """Re-derive TRL's batch fields from the documented post-init rule and simulate its sampler.

    Rule (recorded by CPU probes in the JSON): steps_per_generation = gradient_accumulation_steps,
    generation_batch_size = per_device_train_batch_size * steps_per_generation. The sampler yields
    ``generation_batch_size // num_generations`` prompts, each repeated ``num_generations`` times, the whole chunk
    ``steps_per_generation`` times; the trainer generates from the first ``generation_batch_size`` rows only.
    """
    kw = tc.build_grpo_config(_pairs(p, g))
    bs, ga, spg, ng = (kw[k] for k in (
        "per_device_train_batch_size", "gradient_accumulation_steps", "steps_per_generation", "num_generations"))
    assert spg == ga  # accumulation steps == micro-batches per generation: one optimizer step per generation batch
    gen_batch = bs * spg
    assert gen_batch == p * g and gen_batch % ng == 0
    n_prompts_needed = 3 * p
    chunk_size = gen_batch // ng
    rows = []
    for chunk_start in range(0, n_prompts_needed, chunk_size):
        chunk = list(range(chunk_start, chunk_start + chunk_size))
        for _ in range(spg):  # repeat_count = num_iterations * steps_per_generation with num_iterations = 1
            rows.extend(i for i in chunk for _ in range(ng))
    for step in range(3):
        # the trainer takes one dataloader item (gen_batch rows) per micro-step and only generates every spg-th one
        generation_rows = rows[step * spg * gen_batch : step * spg * gen_batch + gen_batch]
        assert len(set(generation_rows)) == p
        assert all(generation_rows.count(x) == g for x in set(generation_rows))
        assert generation_rows == sorted(generation_rows)  # shuffle_dataset=False: schedule order is preserved


def test_batch_fields_agree_with_recorded_trl_probes():
    """The rule used above reproduces what the real pinned GRPOConfig computed on CPU."""
    for probe in FIELDS["postinit_probe_defaults"]:
        assert probe["steps_per_generation"] == probe["gradient_accumulation_steps"]
        assert probe["generation_batch_size"] == (
            probe["per_device_train_batch_size"] * probe["gradient_accumulation_steps"]
        )
    explicit = FIELDS["postinit_probe_explicit_steps_per_generation"]
    # why we set gradient_accumulation_steps ourselves: steps_per_generation alone leaves accumulation at 1
    assert explicit["steps_per_generation"] == 32 and explicit["gradient_accumulation_steps"] == 1


def test_default_config_batch_split():
    kw = tc.build_grpo_config(_cfg())
    assert (kw["per_device_train_batch_size"], kw["gradient_accumulation_steps"]) == (4, 32)
    assert kw["num_generations"] == 8 and kw["num_iterations"] == 1
    assert tc.build_grpo_config(_cfg(), micro_batch_cap=2)["gradient_accumulation_steps"] == 64
    assert "generation_batch_size" not in kw


def test_split_rollouts_rejects_bad_input():
    with pytest.raises(ValueError):
        tc.split_rollouts(16, 1)  # TRL needs num_generations >= 2
    with pytest.raises(ValueError):
        tc.split_rollouts(0, 8)
    with pytest.raises(ValueError):
        tc.split_rollouts(16, 8, 0)


# ------------------------------------------------------------------------------------------ keys and values


def test_every_emitted_key_is_in_the_recorded_signatures():
    for p, g in [(16, 8), (3, 5), (4, 4)]:
        for arm in ("clean_none", "hackable_subtle_ast"):
            cfg = _cfg(f"grpo.prompts_per_step={p}", f"grpo.gens_per_prompt={g}", arm=arm)
            assert set(tc.build_grpo_config(cfg)) <= GRPO_FIELDS
            assert set(tc.build_lora_config(cfg)) <= LORA_FIELDS


def test_no_key_of_an_unpinned_version_leaks_in():
    kw = tc.build_grpo_config(_cfg())
    # names from other TRL/transformers versions or from other trainers
    stale = {"max_prompt_length", "torch_dtype", "use_liger_loss", "vllm_server_host_ip", "vllm_gpu_memory_util",
             "vllm_guided_decoding_regex", "warmup_ratio", "num_generations_per_prompt", "epochs"}
    assert not stale & set(kw) and not stale & GRPO_FIELDS
    with pytest.raises(ValueError, match="max_prompt_length"):
        tc.check_keys({**kw, "max_prompt_length": 768})
    with pytest.raises(ValueError, match="not_a_field"):
        tc.build_grpo_config(_cfg(), extra={"not_a_field": 1})
    with pytest.raises(ValueError, match="not_a_lora_field"):
        tc.check_keys({"r": 8, "not_a_lora_field": 1}, "LoraConfig")


def test_recorded_versions_equal_the_pins():
    pins = _requirements()
    for pkg in ("trl", "peft", "transformers", "torch", "accelerate", "datasets"):
        assert FIELDS["versions"][pkg] == pins[pkg], pkg
    assert FIELDS["python"].startswith("3.12")


@pytest.mark.parametrize("top_k,expected", [(-1, 0), (20, 20), (1, 1)])
def test_top_k_handling(top_k, expected):
    kw = tc.build_grpo_config(_cfg(f"sampling.top_k={top_k}"))
    assert kw["top_k"] == expected
    assert FIELDS["GRPOConfig"]["top_k"]["default"] == 0  # 0 = disabled in the pinned signature
    assert 0 <= kw["top_k"]


def test_values_flow_from_config():
    cfg = _cfg(
        "grpo.max_steps=7", "grpo.lr=1e-4", "grpo.vllm_gpu_mem_util=0.25", "grpo.loss_type=bnpo",
        "grpo.max_prompt_tokens=100", "grpo.max_completion_tokens=200", "sampling.temperature=0.7",
        "sampling.top_p=0.9", "grpo.grad_checkpointing=false", "lora.r=16", "lora.alpha=8", "lora.dropout=0.05",
        seed=42,
    )
    kw, lora = tc.build_grpo_config(cfg), tc.build_lora_config(cfg)
    assert (kw["seed"], kw["data_seed"]) == (42, 42)
    assert kw["max_steps"] == 7 and kw["learning_rate"] == 1e-4
    assert kw["vllm_gpu_memory_utilization"] == 0.25
    assert kw["loss_type"] == "bnpo" and kw["beta"] == 0.0
    assert kw["max_completion_length"] == 200 and kw["vllm_max_model_length"] == 300
    assert kw["temperature"] == 0.7 and kw["top_p"] == 0.9
    assert kw["gradient_checkpointing"] is False
    assert (lora["r"], lora["lora_alpha"], lora["lora_dropout"]) == (16, 8, 0.05)


def test_fixed_settings_of_the_recipe():
    kw = tc.build_grpo_config(_cfg())
    assert kw["use_vllm"] is True and kw["vllm_mode"] == "colocate" and kw["vllm_tensor_parallel_size"] == 1
    assert kw["vllm_gpu_memory_utilization"] == 0.35 and kw["vllm_max_model_length"] == 768 + 1024
    assert kw["bf16"] is True and kw["fp16"] is False and kw["model_init_kwargs"] == {"dtype": "bfloat16"}
    assert kw["report_to"] == "none" and kw["save_strategy"] == "no"
    assert kw["beta"] == 0.0 and kw["loss_type"] == "dr_grpo" and kw["max_completion_length"] == 1024
    assert kw["temperature"] == 1.0 and kw["top_p"] == 1.0
    assert kw["learning_rate"] == 7e-5 and kw["max_steps"] == 100 and kw["gradient_checkpointing"] is True
    assert kw["output_dir"] == "results/runs/hackable_subtle__s0/trl"
    lora = tc.build_lora_config(_cfg())
    assert lora == {"r": 32, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": "all-linear",
                    "bias": "none", "task_type": "CAUSAL_LM"}


def test_dtype_flags():
    assert tc.build_grpo_config(_cfg("model.dtype=float16"))["fp16"] is True
    fp32 = tc.build_grpo_config(_cfg("model.dtype=float32"))
    assert fp32["bf16"] is False and fp32["fp16"] is False and fp32["model_init_kwargs"] == {"dtype": "float32"}


def test_extra_overrides_but_not_batch_keys():
    kw = tc.build_grpo_config(_cfg(), extra={"vllm_enable_sleep_mode": True, "vllm_importance_sampling_correction": True})
    assert kw["vllm_enable_sleep_mode"] is True and kw["vllm_importance_sampling_correction"] is True
    for k in ("per_device_train_batch_size", "steps_per_generation", "generation_batch_size", "num_generations"):
        with pytest.raises(ValueError, match="batch keys"):
            tc.build_grpo_config(_cfg(), extra={k: 1})


def test_estimand_notes_match_recorded_defaults_and_emitted_values():
    kw = tc.build_grpo_config(_cfg())
    seen = set()
    for note in tc.ESTIMAND_NOTES:
        seen.add(note.key)
        assert note.why.strip()
        if note.key in GRPO_FIELDS:
            assert FIELDS["GRPOConfig"][note.key]["default"] == note.trl_default, note.key  # from the CPU dump
            assert kw[note.key] == note.ours, note.key
    # the silent estimand changers named in the design docs are all covered
    assert {"scale_rewards", "mask_truncated_completions", "loss_type", "vllm_importance_sampling_correction",
            "lr_scheduler_type"} <= seen
    # and the recorded defaults really are the risky ones (guards the reasoning in the notes)
    assert FIELDS["GRPOConfig"]["loss_type"]["default"] == "dapo"
    assert FIELDS["GRPOConfig"]["vllm_importance_sampling_correction"]["default"] is True
    assert FIELDS["GRPOConfig"]["lr_scheduler_type"]["default"] == "linear"
    assert FIELDS["GRPOConfig"]["model_init_kwargs"]["default"] is None


def test_config_loss_types_are_supported_by_the_pinned_trl():
    # loss_type values listed in the help text of the pinned GRPOConfig (read from trl 1.13.0's grpo_config.py)
    trl_loss_types = {"grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo", "luspo", "vespo"}
    from rhg import config as rc

    assert rc._LOSS_TYPES <= trl_loss_types
    assert FIELDS["GRPOConfig"]["loss_type"]["default"] in trl_loss_types


@pytest.mark.skipif(importlib.util.find_spec("trl") is None, reason="trl not installed (CPU dev box)")
def test_keys_exist_in_installed_trl():  # pragma: no cover - only on a box with trl
    import trl
    from peft import LoraConfig

    cfg = _cfg()
    tc.check_against_installed(tc.build_grpo_config(cfg), tc.build_lora_config(cfg))
    assert {f.name for f in dataclasses.fields(trl.GRPOConfig)} == GRPO_FIELDS
    assert {f.name for f in dataclasses.fields(LoraConfig)} == LORA_FIELDS


# ------------------------------------------------------------------------------------ requirements-gpu.txt


def _requirements() -> dict[str, str]:
    out = {}
    for raw in (REPO / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.!+\-]+", line), f"not an exact pin: {line!r}"
        name, ver = line.split("==")
        assert name.lower() not in out, f"duplicate pin {name}"
        out[name.lower()] = ver
    return out


def test_requirements_gpu_every_line_pinned_and_covers_the_compat_doc():
    pins = _requirements()
    assert pins, "requirements-gpu.txt has no pins"
    doc = (REPO / "docs" / "GPU_COMPAT.md").read_text(encoding="utf-8")
    section = doc.split("## 1. Pins", 1)[1].split("\n## ", 1)[0]
    documented = dict(re.findall(r"^\| `([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.!+\-]+)`", section, flags=re.M))
    assert documented == pins  # same packages and same versions, in both directions
    for pkg in ("torch", "transformers", "trl", "peft", "vllm", "accelerate", "datasets"):
        assert pkg in pins
    major, minor, *_ = (int(x) for x in pins["transformers"].split("."))
    assert (major, minor) >= (4, 51)  # DESIGN §2.1 floors
    assert tuple(int(x) for x in pins["vllm"].split(".")) >= (0, 8, 5)


def test_gpu_stack_is_not_a_project_dependency():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8") + (REPO / "uv.lock").read_text(encoding="utf-8")
    for pkg in ("vllm", "trl", "peft", "torch", "accelerate"):
        assert not re.search(rf'^name = "{pkg}"$', text, flags=re.M), pkg
        assert not re.search(rf'^\s*"{pkg}[<>=~! ]', text, flags=re.M), pkg


# ---------------------------------------------------------------------------------- imports without a GPU stack


def test_modules_import_without_the_gpu_stack():
    code = (
        "import sys; import rhg.train.trl_config, rhg.train.gpu_memory; "
        "bad = {'torch','trl','vllm','peft','transformers','accelerate'} & set(sys.modules); "
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------------------- memory estimator

# Toy architecture, all numbers below computed by hand (see comments), not by the code under test.
TOY = gm.ArchSpec(hidden=8, layers=2, heads=2, kv_heads=1, head_dim=4, intermediate=16, vocab=32)


def test_param_counts_toy_by_hand():
    # embeddings 32*8 = 256 (tied). per layer: q 8*8=64, k 8*4=32, v 32, o 8*8=64 -> 192; mlp 3*8*16 = 384;
    # 2 RMSNorm 16; q/k norm 2*4 = 8 -> 600; x2 layers = 1200; final norm 8. Total 256 + 1200 + 8.
    assert gm.param_count(TOY) == 1464
    # untied adds a separate lm_head of 32*8 = 256
    assert gm.param_count(dataclasses.replace(TOY, tie_embeddings=False)) == 1464 + 256
    assert gm.param_count(dataclasses.replace(TOY, qk_norm=False)) == 1464 - 2 * 8
    # LoRA r=2: q 2*(8+8)=32, k 2*(8+4)=24, v 24, o 2*(8+8)=32 -> 112; gate/up/down 3*2*(8+16)=144; 256/layer x 2
    assert gm.lora_param_count(TOY, 2) == 512
    # KV per token: 2 (k,v) * 2 layers * 1 kv-head * 4 dims * 2 bytes
    assert gm.kv_bytes_per_token(TOY) == 32


def test_param_counts_qwen3_1_7b_by_hand():
    # embeddings 151936*2048 = 311,164,928; per layer q 2048*2048=4,194,304 + k 2048*1024=2,097,152 + v 2,097,152
    # + o 4,194,304 = 12,582,912; mlp 3*2048*6144 = 37,748,736; norms 2*2048 + 2*128 = 4,352 -> 50,336,000;
    # x28 = 1,409,408,000 (model card: 1.4B non-embedding); + final norm 2048.
    assert gm.param_count(gm.QWEN3_1_7B) == 311_164_928 + 1_409_408_000 + 2_048 == 1_720_574_976
    # LoRA r=32: q 32*4096 = 131,072; k 32*3072 = 98,304; v 98,304; o 131,072; gate/up/down 3*32*8192 = 786,432
    # -> 1,245,184 per layer, x28
    assert gm.lora_param_count(gm.QWEN3_1_7B, 32) == 34_865_152
    assert gm.kv_bytes_per_token(gm.QWEN3_1_7B) == 2 * 28 * 8 * 128 * 2 == 114_688


def test_estimate_toy_by_hand():
    a = gm.Assumptions(vllm_overhead_bytes=1000, context_bytes=500, fragmentation=0.25)
    e = gm.estimate(TOY, lora_r=2, micro_batch=1, prompt_tokens=3, completion_tokens=5, vllm_util=0.5,
                    gpu_total_bytes=1_000_000, assumptions=a)
    # trainer weights 1464*2 = 2928; LoRA 512*16 = 8192
    # activations: checkpoints 1*8*8*2*2 layers = 256; live layer 1*8*(10*8+3*16)*2 = 2048
    # logits: 1*5*32*2*2 = 640 + 5*32*4*2 = 1280 -> 1920
    # trainer = 2928 + 8192 + 256 + 2048 + 1920 = 15344; with 25% fragmentation 19180
    assert dict(e.rows)["trainer weights (bf16)"] == 2928
    assert dict(e.rows)["LoRA params+grads+Adam (fp32)"] == 8192
    assert dict(e.rows)["activations (checkpoints + live layer)"] == 256 + 2048
    assert dict(e.rows)["logits (completion tokens)"] == 1920
    assert e.trainer_bytes == 19180
    # vLLM: budget 0.5 * 1e6 = 500000; weights 2928; overhead 1000 -> KV 496072 B = 15502 tokens of 32 B (floor)
    assert e.vllm_budget_bytes == 500_000 and e.kv_cache_bytes == 496_072 and e.kv_tokens == 15_502
    assert e.full_length_sequences == 15_502 // 8  # sequences of prompt + completion = 8 tokens
    assert e.peak_bytes == 500_000 + 19_180 + 500 and e.headroom_bytes == 1_000_000 - 519_680


def test_estimate_without_checkpointing_and_kv_floor():
    a = gm.Assumptions(vllm_overhead_bytes=1000, context_bytes=0, fragmentation=0.0, grad_checkpointing=False)
    e = gm.estimate(TOY, lora_r=2, micro_batch=2, prompt_tokens=3, completion_tokens=5, vllm_util=0.001,
                    gpu_total_bytes=1_000_000, assumptions=a)
    # no checkpoints; the live working set is kept for both layers: 2*8*(80+48)*2 = 4096, x2 layers
    assert dict(e.rows)["activations (no checkpointing)"] == 2 * 4096
    # vLLM budget 1000 B is below weights + overhead: KV cache floors at 0 instead of going negative
    assert e.kv_cache_bytes == 0 and e.kv_tokens == 0 and e.full_length_sequences == 0


def test_estimate_scales_as_expected():
    base = dict(lora_r=32, micro_batch=4, prompt_tokens=768, completion_tokens=1024, gpu_total_bytes=24 * gm.GIB)
    lo = gm.estimate(gm.QWEN3_1_7B, vllm_util=0.25, **base)
    hi = gm.estimate(gm.QWEN3_1_7B, vllm_util=0.35, **base)
    assert hi.kv_cache_bytes > lo.kv_cache_bytes and hi.peak_bytes > lo.peak_bytes
    # only the vLLM budget moves (peak is truncated to int, hence the tolerance of one byte)
    assert abs((hi.peak_bytes - lo.peak_bytes) - (hi.vllm_budget_bytes - lo.vllm_budget_bytes)) <= 1
    small = gm.estimate(gm.QWEN3_1_7B, vllm_util=0.35, **{**base, "micro_batch": 2})
    assert small.trainer_bytes < hi.trainer_bytes


def test_estimate_from_config_and_cli(capsys):
    e = gm.estimate_from_config(_cfg())
    assert e.n_params == 1_720_574_976 and e.n_lora_params == 34_865_152
    assert e.vllm_budget_bytes == int(0.35 * 24 * gm.GIB)
    assert 0.5 * 24 * gm.GIB < e.peak_bytes < 24 * gm.GIB  # plausible and inside the card at the default settings
    assert gm.main(["--arm", "hackable_subtle", "--measured-peak-gib", f"{e.peak_bytes / gm.GIB:.4f}"]) == 0
    out = capsys.readouterr().out
    assert "ESTIMATED" in out and "NOT a measurement" in out and "EXPECTED PEAK (estimate)" in out
    assert "OK (within 25%)" in out
    assert gm.main(["--measured-peak-gib", "40"]) == 0
    assert "WARN" in capsys.readouterr().out
    assert gm.main(["--set", "model.name=Some/Other-Model"]) == 2  # unknown architecture: honest error, no guess
    assert gm.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 0)[0])["n_params"] == 1_720_574_976


def test_compare_measured_thresholds():
    e = gm.estimate_from_config(_cfg())
    est_gib = e.peak_bytes / gm.GIB
    assert "OK" in gm.compare_measured(e, est_gib * 1.2)
    assert "WARN" in gm.compare_measured(e, est_gib * 1.3)
    assert "WARN" in gm.compare_measured(e, est_gib * 0.7)
    assert math.isclose(float(re.search(r"= ([0-9.]+) ", gm.compare_measured(e, est_gib)).group(1)), 1.0, abs_tol=0.01)
