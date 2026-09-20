"""Fake ``trl``. ``GRPOConfig`` is a dataclass over the 189 recorded pinned fields (an unknown kwarg raises TypeError)
with the real ``__post_init__`` batch arithmetic (verified from the source of trl 1.13.0 and the recorded probes).
``GRPOTrainer`` has the real constructor signature (read from trl 1.13.0's ``grpo_trainer.py``) and drives the callbacks in
the order of ``transformers.Trainer`` (train_begin, per step: step_begin -> [sync, generate] -> reward functions -> fake
fwd/bwd -> step_end -> log; then a summary log and train_end), calling reward functions exactly like the real
``_calculate_rewards`` (``prompts, completions, completion_ids`` + dataset columns + ``trainer_state``, ``log_extra``,
``log_metric``)."""

from __future__ import annotations

import dataclasses
import json
import time
import types
from pathlib import Path
from typing import Any

import peft as _peft
import torch as _torch

from stubs._world import WORLD

_FIELDS = json.loads((Path(__file__).resolve().parents[3] / "docs" / "trl_grpoconfig_fields.json").read_text(encoding="utf-8"))["GRPOConfig"]
_DEFAULTS = {
    "per_device_train_batch_size": 8, "gradient_accumulation_steps": 1, "num_generations": 8, "max_steps": -1,
    "loss_type": "dapo", "scale_rewards": "group", "vllm_mode": "colocate", "beta": 0.0, "temperature": 1.0,
    "bf16": False, "fp16": False, "use_vllm": False, "logging_steps": 500, "num_iterations": 1,
}
_LOSS_TYPES = {"grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo", "luspo", "vespo"}


def _post_init(self):
    if self.generation_batch_size is None and self.steps_per_generation is None:
        self.steps_per_generation = self.gradient_accumulation_steps
        self.generation_batch_size = self.per_device_train_batch_size * self.steps_per_generation
    elif self.generation_batch_size is not None and self.steps_per_generation is None:
        if self.generation_batch_size % self.per_device_train_batch_size != 0:
            raise ValueError("generation_batch_size must be divisible by the global batch size")
        self.steps_per_generation = self.generation_batch_size // self.per_device_train_batch_size
    elif self.generation_batch_size is None:
        self.generation_batch_size = self.per_device_train_batch_size * self.steps_per_generation
    else:
        raise ValueError("'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time")
    if self.generation_batch_size % self.num_generations != 0:
        raise ValueError("generation_batch_size must be divisible by num_generations")
    if self.num_generations < 2:
        raise ValueError("GRPO requires at least 2 generations per prompt")
    if self.loss_type not in _LOSS_TYPES:
        raise ValueError(f"unknown loss_type {self.loss_type!r}")
    if self.vllm_mode not in ("colocate", "server"):
        raise ValueError(f"unknown vllm_mode {self.vllm_mode!r}")
    WORLD.kwargs_seen["GRPOConfig"] = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


GRPOConfig = dataclasses.make_dataclass(
    "GRPOConfig",
    [(name, Any, _DEFAULTS.get(name)) for name in _FIELDS],
    namespace={"__post_init__": _post_init, "__module__": __name__},
)


class FakeVLLMGeneration:
    """What ``trainer.vllm_generation`` exposes in trl 1.13.0 (``sync_weights`` and ``generate``)."""

    def sync_weights(self):
        WORLD.log("sync_weights")
        time.sleep(WORLD.sleep_sync)

    def generate(self, prompts, images=None, num_generations=1, profiler=None):
        """``prompts`` are already repeated ``num_generations`` times each (the stub trainer does the repeating)."""
        WORLD.log("vllm.generate")
        time.sleep(WORLD.sleep_gen)
        return prompts, [WORLD.completion_fn(p, i % num_generations)[1] for i, p in enumerate(prompts)], [], None


class GRPOTrainer:
    def __init__(
        self,
        model,
        reward_funcs=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        quantization_config=None,
        peft_config=None,
        tools=None,
        rollout_func=None,
        environment_factory=None,
    ):
        if not isinstance(args, GRPOConfig):
            raise TypeError("args must be a GRPOConfig")
        if not isinstance(model, str):
            raise TypeError("the fake trainer only takes a model name")
        if peft_config is not None and not isinstance(peft_config, _peft.PeftConfig):
            raise TypeError("`peft_config` must be a `peft.PeftConfig` instance")
        funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        if not funcs or not all(callable(f) for f in funcs):
            raise TypeError("reward_funcs must be callables")
        if train_dataset is None or "prompt" not in train_dataset.column_names:
            raise ValueError("train_dataset needs a 'prompt' column")
        if not (args.use_vllm and args.vllm_mode == "colocate"):
            raise ValueError("this project only uses vLLM colocate")
        WORLD.log("GRPOTrainer.__init__")
        WORLD.kwargs_seen["GRPOTrainer"] = {"model": model, "n_reward_funcs": len(funcs), "callbacks": len(callbacks or [])}
        self.args, self.reward_funcs, self.train_dataset = args, funcs, train_dataset
        self.processing_class, self.callbacks = processing_class, list(callbacks or [])
        self.model = _peft.PeftModel(model, peft_config)
        self.model._step_ref = lambda: self.state.global_step
        self.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m)
        self.state = types.SimpleNamespace(global_step=0)
        self.vllm_generation = FakeVLLMGeneration()
        # memory: trainer + colocated engine budget (fraction of the whole card); a leak is kept after teardown
        WORLD.owner(self, WORLD.trainer_gib + args.vllm_gpu_memory_utilization * WORLD.gpu_total_gib, leak=WORLD.leak_gib)
        if WORLD.used_gib > WORLD.gpu_total_gib:
            raise _torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate (stub: building the trainer)")

    def _fire(self, event, **extra):
        for cb in self.callbacks:
            getattr(cb, event)(self.args, self.state, types.SimpleNamespace(), **extra)

    def train(self):
        a, W = self.args, WORLD
        per_step = a.generation_batch_size // a.num_generations
        rows = self.train_dataset
        self._fire("on_train_begin", model=self.model)
        idx, last_loaded = 0, -1
        for k in range(1, a.max_steps + 1):
            self._fire("on_step_begin")
            if self.state.global_step != last_loaded:
                self.vllm_generation.sync_weights()
                last_loaded = self.state.global_step
            chunk = rows[idx : idx + per_step]
            idx += per_step
            prompts = [p for p in chunk["prompt"] for _ in range(a.num_generations)]
            if W.wrong_batch_at_step == k:
                prompts = [chunk["prompt"][0]] * len(prompts)
            columns = {c: [v for v in chunk[c] for _ in range(a.num_generations)] for c in chunk if c != "prompt"}
            _, completion_ids, _, _ = self.vllm_generation.generate(prompts, num_generations=a.num_generations)
            completions = [W.completion_fn(p, i % a.num_generations)[0] for i, p in enumerate(prompts)]
            state = types.SimpleNamespace(global_step=self.state.global_step + W.global_step_offset)
            for fn in self.reward_funcs:
                out = fn(
                    prompts=prompts, completions=completions, completion_ids=completion_ids, **columns,
                    trainer_state=state, log_extra=lambda *a_, **k_: None, log_metric=lambda *a_, **k_: None,
                )
                if len(out) != len(prompts):
                    raise ValueError("reward function returned the wrong number of rewards")
            time.sleep(W.sleep_train)
            if W.oom_at_step == k:
                raise _torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (stub)")
            self.state.global_step += 1
            self._fire("on_step_end")
            logs = {"loss": float("nan") if W.nan_loss_at_step == k else 0.01 * k, "learning_rate": a.learning_rate,
                    "reward": sum(out) / len(out)}
            if not W.missing_grad_norm:
                logs["grad_norm"] = float("inf") if W.inf_grad_at_step == k else 0.5
            if not (W.skip_loss_log and k == 1):
                self._fire("on_log", logs=logs)
        self._fire("on_log", logs={"train_runtime": 1.0, "train_loss": 0.0})
        self._fire("on_train_end")
