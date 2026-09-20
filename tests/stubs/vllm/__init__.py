"""Fake ``vllm``: ``LLM`` / ``SamplingParams`` / ``LoRARequest`` validate their keyword arguments against the argument
names ``rhg.eval.generate.VLLMGenerator`` and TRL's colocate engine use (names from vLLM's ``EngineArgs``/``SamplingParams``
documentation -- UNVERIFIED, no vllm signature was recorded), simulate the GPU-memory start-up check ("free memory is
less than desired GPU memory utilization") and release the memory when the engine object is garbage collected."""

from __future__ import annotations

import os
import types

from stubs._world import WORLD

_LLM_KWARGS = {
    "model", "dtype", "max_model_len", "gpu_memory_utilization", "seed", "enable_lora", "max_lora_rank", "tensor_parallel_size",
    "distributed_executor_backend", "max_num_seqs", "max_num_batched_tokens", "logprobs_mode", "enable_sleep_mode",
    "trust_remote_code", "enforce_eager", "max_loras",
}
_SP_KWARGS = {"n", "temperature", "top_p", "top_k", "max_tokens", "seed", "min_p", "repetition_penalty", "logprobs", "stop"}


class SamplingParams:
    def __init__(self, **kwargs):
        bad = sorted(set(kwargs) - _SP_KWARGS)
        if bad:
            raise TypeError(f"SamplingParams got unexpected kwargs {bad}")
        self.__dict__.update({"n": 1, "seed": None, **kwargs})


class LoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path, **kwargs):
        if kwargs:
            raise TypeError(f"LoRARequest got unexpected kwargs {sorted(kwargs)}")
        self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path


class LLM:
    def __init__(self, **kwargs):
        bad = sorted(set(kwargs) - _LLM_KWARGS)
        if bad:
            raise TypeError(f"LLM got unexpected kwargs {bad}")
        request = kwargs["gpu_memory_utilization"] * WORLD.gpu_total_gib
        free = WORLD.gpu_total_gib - WORLD.used_gib
        if free < request:
            raise ValueError(
                f"Free memory on device ({free:.2f}/{WORLD.gpu_total_gib:.2f} GiB) on startup is less than desired GPU memory "
                f"utilization ({kwargs['gpu_memory_utilization']}, {request:.2f} GiB)."
            )
        self.kwargs = kwargs
        WORLD.llm_inits.append(kwargs)
        WORLD.log("LLM.__init__")
        WORLD.owner(self, request)

    def get_tokenizer(self):
        return types.SimpleNamespace(encode=lambda s: list(range(len(s.split()) * WORLD.tokens_per_word)))

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=True):
        if lora_request is not None and not self.kwargs.get("enable_lora"):
            raise ValueError("LoRA request passed to an engine without enable_lora=True")
        if len(sampling_params) != len(prompts):
            raise ValueError("need one SamplingParams per prompt")
        path = None
        if lora_request is not None:
            path = lora_request.lora_path
            if not os.path.isfile(os.path.join(path, "adapter_config.json")):
                raise ValueError(f"LoRA path {path} has no adapter_config.json")
        WORLD.llm_generates.append({"n_prompts": len(prompts), "lora_path": path, "seeds": [sp.seed for sp in sampling_params]})
        outs = []
        for p, sp in zip(prompts, sampling_params):
            comps = []
            for j in range(sp.n):
                text = WORLD.eval_completion_fn(p, j, path)
                comps.append(types.SimpleNamespace(text=text, token_ids=list(range(max(1, len(text) // 4))), finish_reason="stop"))
            outs.append(types.SimpleNamespace(outputs=comps))
        return outs


def _destroy(name):
    def fn():
        WORLD.log(name)

    return fn


parallel_state = types.ModuleType("vllm.distributed.parallel_state")
parallel_state.destroy_model_parallel = _destroy("destroy_model_parallel")
parallel_state.destroy_distributed_environment = _destroy("destroy_distributed_environment")
distributed = types.ModuleType("vllm.distributed")
distributed.parallel_state = parallel_state
lora = types.ModuleType("vllm.lora")
lora_request = types.ModuleType("vllm.lora.request")
lora_request.LoRARequest = LoRARequest
lora.request = lora_request
SUBMODULES = {
    "vllm.distributed": distributed, "vllm.distributed.parallel_state": parallel_state,
    "vllm.lora": lora, "vllm.lora.request": lora_request,
}
