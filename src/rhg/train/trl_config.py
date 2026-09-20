"""Pure mapping from our config to the pinned TRL ``GRPOConfig`` / PEFT ``LoraConfig`` kwargs.

No heavy imports: ``trl``/``peft``/``torch`` are never imported here (``check_against_installed`` imports ``trl``
lazily and only on a machine that has it). The recorded field lists in ``docs/trl_grpoconfig_fields.json`` were
dumped from ``trl==1.13.0`` / ``peft==0.20.0`` (``requirements-gpu.txt``); every emitted key is checked against them,
so a typo or a key that only exists in another TRL version fails on the CPU box, not on the GPU box.

Batch arithmetic in the pinned TRL (verified from source and from ``GRPOConfig.__post_init__`` on CPU, see
``postinit_probe_*`` in the JSON): the trainer draws ``generation_batch_size / num_generations`` prompts, repeats each
``num_generations`` times, generates the whole ``generation_batch_size`` completions at once, splits them into
``steps_per_generation`` micro-batches of ``per_device_train_batch_size`` rows and then makes one optimizer step per
``gradient_accumulation_steps`` micro-batches. With ``steps_per_generation == gradient_accumulation_steps`` (what we
emit, explicitly, and never ``generation_batch_size``) one optimizer step therefore sees exactly
``per_device_train_batch_size * gradient_accumulation_steps = prompts_per_step * gens_per_prompt`` rollouts of
``prompts_per_step`` distinct prompts. If only ``steps_per_generation`` were set, ``gradient_accumulation_steps`` would
stay 1 and TRL would take one optimizer step per micro-batch on a stale generation batch (observed in the probe).
Advantages are computed over the whole generation batch, so the micro-batch split does not change them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIELDS_PATH = Path(__file__).resolve().parents[3] / "docs" / "trl_grpoconfig_fields.json"
# Rows per micro-batch. Bounded by the logits tensor (rows x completion tokens x 151936 vocab) next to the
# colocated vLLM engine; see rhg.train.gpu_memory. The 5-minute smoke run (scripts/smoke.sh) is where it is tuned.
DEFAULT_MICRO_BATCH_CAP = 4
_BATCH_KEYS = frozenset(
    {"per_device_train_batch_size", "gradient_accumulation_steps", "steps_per_generation", "generation_batch_size",
     "num_generations"}
)
_DTYPE_FLAGS = {"bfloat16": (True, False), "float16": (False, True), "float32": (False, False)}  # (bf16, fp16)


@dataclass(frozen=True)
class EstimandNote:
    """A TRL setting that changes what is being estimated if left at its default (or that we pin explicitly)."""

    key: str
    ours: Any  # the value we emit for the default config (see tests for the derivation)
    trl_default: Any  # the pinned TRL default; tests compare it with docs/trl_grpoconfig_fields.json
    why: str


ESTIMAND_NOTES: tuple[EstimandNote, ...] = (
    EstimandNote(
        "loss_type", "dr_grpo", "dapo",
        "TRL's default 'dapo' normalises by the number of active tokens of the whole generation batch, so a token's "
        "weight depends on the other completions' lengths. 'dr_grpo' divides by rows x max_completion_length, a "
        "constant, so it has no length bias and does not depend on the micro-batch split (configs/base.yaml).",
    ),
    EstimandNote(
        "scale_rewards", "group", "group",
        "KEPT at TRL's default (advantage = (r - mean) / (std + 1e-4) within each prompt group, i.e. original GRPO "
        "advantages). Dr. GRPO would drop the std ('none'), but the std scaling up-weights groups with a rare success, "
        "and non-emergence of the hack is the dominant risk (DESIGN §10). Pinned explicitly so a TRL default change "
        "cannot move it; flagged for Day-0 review in docs/SPEC_DEVIATIONS.md.",
    ),
    EstimandNote(
        "vllm_importance_sampling_correction", False, True,
        "Default 'sequence_mask' (C_max=3) drops whole completions whose vLLM-vs-trainer log-ratio summed over tokens "
        "exceeds ln 3: a length-dependent, silent filter on exactly the rollouts under study, and it costs an extra "
        "forward pass. Off = plain on-policy GRPO (DESIGN §2.1). With it off TRL logs no importance-sampling ratio, so "
        "the vLLM/trainer mismatch is not monitored: the smoke run should do one short diagnostic run with it on "
        "(extra={...: True}), see docs/GPU_COMPAT.md §4.",
    ),
    EstimandNote(
        "mask_truncated_completions", False, False,
        "Truncated completions stay in the loss and get the reward of whatever code was extracted (truncation is a "
        "logged covariate, DESIGN §4). Pinned explicitly.",
    ),
    EstimandNote(
        "lr_scheduler_type", "constant", "linear",
        "TRL/transformers default is a linear decay to 0 over max_steps, which would silently make the late steps "
        "(the endpoint of every estimand) learn less; lr 7e-5 is meant as a constant rate. warmup_steps=0.",
    ),
    EstimandNote(
        "num_iterations", 1, 1,
        "One optimizer step per generation batch: exactly on-policy, so the PPO clip (epsilon=0.2) never binds.",
    ),
    EstimandNote("beta", 0.0, 0.0, "No KL term and no reference model (DESIGN §2.1)."),
    EstimandNote(
        "top_k", 0, 0,
        "In the pinned TRL and vLLM 0 disables top-k (TRL's default); our config's -1 maps to 0.",
    ),
    EstimandNote(
        "shuffle_dataset", False, True,
        "The prompt order is the driver's seeded schedule (rhg.train.run); the backend feeds it in order, so TRL must "
        "not reshuffle it.",
    ),
    EstimandNote(
        "model_init_kwargs", {"dtype": "bfloat16"}, None,
        "For a model given by name TRL loads float32 unless dtype is set (memory trap: 6.9 GB weights + fp32 LoRA).",
    ),
    EstimandNote(
        "vllm_gpu_memory_utilization", 0.35, 0.3,
        "From grpo.vllm_gpu_mem_util. Fraction of the WHOLE GPU the colocated vLLM engine may use.",
    ),
    EstimandNote(
        "max_grad_norm", 1.0, 1.0, "Gradient clipping at 1.0 (transformers default), pinned explicitly."
    ),
    EstimandNote(
        "max_prompt_tokens", None, None,
        "No such field in the pinned GRPOConfig (max_prompt_length was removed) and TRL does not truncate prompts: the "
        "backend must drop/verify prompts longer than grpo.max_prompt_tokens; the value only sets vllm_max_model_length.",
    ),
)


def load_recorded_fields(path: Path | str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or FIELDS_PATH).read_text(encoding="utf-8"))


def check_keys(kwargs: dict[str, Any], section: str = "GRPOConfig", *, path: Path | str | None = None) -> None:
    """Raise if any key is not a field of the recorded (pinned) ``GRPOConfig``/``LoraConfig``."""
    known = load_recorded_fields(path)[section]
    unknown = sorted(set(kwargs) - set(known))
    if unknown:
        raise ValueError(f"{section} kwargs not in the pinned signature (docs/trl_grpoconfig_fields.json): {unknown}")


def check_against_installed(grpo_kwargs: dict[str, Any], lora_kwargs: dict[str, Any] | None = None) -> None:
    """Same check against the *installed* ``trl``/``peft`` (GPU box; imports them lazily)."""
    import dataclasses

    from trl import GRPOConfig

    unknown = sorted(set(grpo_kwargs) - {f.name for f in dataclasses.fields(GRPOConfig)})
    if unknown:
        raise ValueError(f"GRPOConfig kwargs not accepted by the installed trl: {unknown}")
    if lora_kwargs is not None:
        from peft import LoraConfig

        unknown = sorted(set(lora_kwargs) - {f.name for f in dataclasses.fields(LoraConfig)})
        if unknown:
            raise ValueError(f"LoraConfig kwargs not accepted by the installed peft: {unknown}")


def split_rollouts(
    prompts_per_step: int, gens_per_prompt: int, micro_batch_cap: int = DEFAULT_MICRO_BATCH_CAP
) -> tuple[int, int]:
    """``(per_device_train_batch_size, gradient_accumulation_steps)`` with product ``prompts_per_step * gens_per_prompt``.

    The micro-batch is the largest divisor of the rollouts per step that is <= ``micro_batch_cap``.
    """
    if min(prompts_per_step, gens_per_prompt, micro_batch_cap) < 1:
        raise ValueError("prompts_per_step, gens_per_prompt and micro_batch_cap must be >= 1")
    if gens_per_prompt < 2:
        raise ValueError("GRPO needs gens_per_prompt >= 2 (TRL raises for num_generations < 2)")
    rollouts = prompts_per_step * gens_per_prompt
    micro = max(d for d in range(1, min(micro_batch_cap, rollouts) + 1) if rollouts % d == 0)
    return micro, rollouts // micro


def build_grpo_config(
    cfg,
    *,
    output_dir: str | None = None,
    micro_batch_cap: int = DEFAULT_MICRO_BATCH_CAP,
    extra: dict[str, Any] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Kwargs for ``trl.GRPOConfig`` from a resolved ``rhg.config.Config`` (see the module docstring).

    ``extra`` overrides emitted keys (fallback levers of docs/GPU_COMPAT.md such as
    ``{"vllm_enable_sleep_mode": True}`` or a diagnostic ``{"vllm_importance_sampling_correction": True}``); it can
    only set fields of the pinned signature, and never the batch keys (they would break rollouts_per_step).
    """
    g, s, m = cfg.grpo, cfg.sampling, cfg.model
    micro, accum = split_rollouts(g.prompts_per_step, g.gens_per_prompt, micro_batch_cap)
    bf16, fp16 = _DTYPE_FLAGS[m.dtype]
    kwargs: dict[str, Any] = {
        "output_dir": output_dir or f"{cfg.run.output_root}/{cfg.run_id}/trl",
        # batch structure
        "per_device_train_batch_size": micro,
        "gradient_accumulation_steps": accum,
        "steps_per_generation": accum,
        "num_generations": g.gens_per_prompt,
        "num_iterations": 1,
        "max_steps": g.max_steps,
        "shuffle_dataset": False,
        # objective
        "beta": g.beta,
        "loss_type": g.loss_type,
        "scale_rewards": "group",
        "mask_truncated_completions": False,
        "vllm_importance_sampling_correction": False,
        # sampling: on-policy sampler, identical for training and evaluation
        "temperature": s.temperature,
        "top_p": s.top_p,
        "top_k": s.top_k if s.top_k_enabled else 0,
        "repetition_penalty": 1.0,
        "max_completion_length": g.max_completion_tokens,
        # generation engine
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": g.vllm_gpu_mem_util,
        "vllm_max_model_length": g.max_prompt_tokens + g.max_completion_tokens,
        "vllm_tensor_parallel_size": 1,
        "vllm_enable_sleep_mode": False,
        "vllm_model_impl": "vllm",
        # optimisation
        "learning_rate": g.lr,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": g.grad_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "model_init_kwargs": {"dtype": m.dtype},
        # bookkeeping
        "seed": cfg.run.seed,
        "data_seed": cfg.run.seed,
        "report_to": "none",
        "save_strategy": "no",
        "logging_steps": 1,
        "log_completions": False,
    }
    assert kwargs["per_device_train_batch_size"] * kwargs["gradient_accumulation_steps"] == cfg.rollouts_per_step
    if extra:
        locked = sorted(set(extra) & _BATCH_KEYS)
        if locked:
            raise ValueError(f"extra may not override the batch keys {locked}; use grpo.* and the micro-batch cap")
        kwargs.update(extra)
    if validate:
        check_keys(kwargs, "GRPOConfig")
    return kwargs


def build_lora_config(cfg, *, validate: bool = True) -> dict[str, Any]:
    """Kwargs for ``peft.LoraConfig`` (``lora`` block; ``target: all-linear`` excludes the LM head)."""
    kwargs: dict[str, Any] = {
        "r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "target_modules": cfg.lora.target,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if validate:
        check_keys(kwargs, "LoraConfig")
    return kwargs
