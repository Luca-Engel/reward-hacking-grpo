"""Expected peak GPU memory of colocated GRPO+LoRA on one card (subtask 10). **An estimate, not a measurement.**

Pure arithmetic, no torch. Purpose: (1) sanity-check ``grpo.vllm_gpu_mem_util`` and the micro-batch before
renting a GPU, (2) give ``scripts/smoke.sh`` a number to compare the measured peak against
(``--measured-peak-gib``). Every modelling choice below is an assumption unless marked *verified*; the table is
printed with its assumptions so a big gap to the measurement points at the assumption to fix.

Model (additive; colocate = trainer and vLLM share one process and one device):

* trainer weights: ``n_params * 2`` bytes (bf16, ``model_init_kwargs.dtype``).
* LoRA: ``n_lora * 16`` bytes = fp32 param 4 + fp32 grad 4 + AdamW two fp32 states 8 (PEFT keeps adapters in fp32
  on a single GPU: verified from the comment in TRL's ``grpo_trainer.py`` about ``autocast_adapter_dtype``).
* activations with gradient checkpointing: one bf16 checkpoint of ``[micro, seq, hidden]`` per layer plus the live
  working set of one layer, ``micro * seq * (10 * hidden + 3 * intermediate) * 2`` bytes (10 hidden-sized and 3
  MLP-sized tensors: an assumption). Without checkpointing the working set is kept for every layer. Attention is
  assumed memory-efficient (SDPA/flash): no ``seq^2`` term.
* logits (completion tokens only, TRL passes ``logits_to_keep``): bf16 logits + their grad
  ``micro * C * vocab * 2 * 2`` plus one fp32 row-wise working copy ``C * vocab * 4 * 2`` (TRL's
  ``selective_log_softmax`` loops over rows; the exact peak depends on the version: assumption).
* vLLM: it may use ``vllm_gpu_memory_utilization * total`` (*verified from docs*: fraction of the whole GPU); in it
  bf16 weights, a fixed activation/CUDA-graph overhead (assumption) and the rest is KV cache. Assumed to be
  charged on top of everything the trainer holds (conservative; vLLM's profiler may instead net out memory that
  was already allocated in the process; unverified).
* CUDA context/workspaces: a constant; allocator fragmentation: a fraction of the trainer's own allocations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

GIB = 1024**3


@dataclass(frozen=True)
class ArchSpec:
    """Decoder-only transformer dimensions (Qwen3-style: GQA, gated MLP, per-head q/k RMSNorm, RMSNorm layers)."""

    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tie_embeddings: bool = True
    qk_norm: bool = True


# Dimensions from the model's config.json (Qwen/Qwen3-1.7B, fetched 2026-09-20; the model card lists 1.7B
# parameters, 1.4B non-embedding, 28 layers, 16 q / 8 kv heads, which the derived counts below reproduce).
QWEN3_1_7B = ArchSpec(
    hidden=2048, layers=28, heads=16, kv_heads=8, head_dim=128, intermediate=6144, vocab=151936, tie_embeddings=True
)
KNOWN_ARCHS = {"Qwen/Qwen3-1.7B": QWEN3_1_7B}


def _attn_dims(a: ArchSpec) -> tuple[int, int]:
    return a.heads * a.head_dim, a.kv_heads * a.head_dim  # q/o width, k/v width


def param_count(a: ArchSpec) -> int:
    q, kv = _attn_dims(a)
    attn = a.hidden * q + 2 * a.hidden * kv + q * a.hidden
    mlp = 3 * a.hidden * a.intermediate
    norms = 2 * a.hidden + (2 * a.head_dim if a.qk_norm else 0)
    total = a.vocab * a.hidden + a.layers * (attn + mlp + norms) + a.hidden
    return total if a.tie_embeddings else total + a.vocab * a.hidden


def lora_param_count(a: ArchSpec, r: int) -> int:
    """LoRA on q,k,v,o,gate,up,down of every layer (``target: all-linear`` without the LM head): ``r*(in+out)`` each."""
    q, kv = _attn_dims(a)
    per_layer = (
        r * (a.hidden + q) + 2 * r * (a.hidden + kv) + r * (q + a.hidden) + 3 * r * (a.hidden + a.intermediate)
    )
    return a.layers * per_layer


def kv_bytes_per_token(a: ArchSpec, kv_dtype_bytes: int = 2) -> int:
    return 2 * a.layers * a.kv_heads * a.head_dim * kv_dtype_bytes


@dataclass(frozen=True)
class Assumptions:
    weight_dtype_bytes: int = 2
    lora_bytes_per_param: int = 16
    kv_dtype_bytes: int = 2
    vllm_overhead_bytes: int = int(1.5 * GIB)  # activation profile + CUDA graphs of the engine (UNVERIFIED)
    context_bytes: int = int(1.0 * GIB)  # CUDA context, cuBLAS/NCCL workspaces (UNVERIFIED)
    fragmentation: float = 0.10  # of the trainer's own allocations (UNVERIFIED)
    grad_checkpointing: bool = True


@dataclass(frozen=True)
class MemoryEstimate:
    rows: tuple[tuple[str, int], ...]  # (component, bytes), in table order
    trainer_bytes: int
    vllm_budget_bytes: int
    kv_cache_bytes: int
    kv_tokens: int
    full_length_sequences: int  # KV cache / (prompt + completion) tokens: sequences that fit at the max length
    peak_bytes: int
    gpu_total_bytes: int
    n_params: int
    n_lora_params: int

    @property
    def headroom_bytes(self) -> int:
        return self.gpu_total_bytes - self.peak_bytes


def estimate(
    a: ArchSpec,
    *,
    lora_r: int,
    micro_batch: int,
    prompt_tokens: int,
    completion_tokens: int,
    vllm_util: float,
    gpu_total_bytes: int,
    assumptions: Assumptions = Assumptions(),
) -> MemoryEstimate:
    s = assumptions
    seq = prompt_tokens + completion_tokens
    n_params, n_lora = param_count(a), lora_param_count(a, lora_r)

    weights = n_params * s.weight_dtype_bytes
    lora = n_lora * s.lora_bytes_per_param
    ckpt = micro_batch * seq * a.hidden * 2 * a.layers if s.grad_checkpointing else 0
    live = micro_batch * seq * (10 * a.hidden + 3 * a.intermediate) * 2
    act = ckpt + live * (1 if s.grad_checkpointing else a.layers)
    logits = micro_batch * completion_tokens * a.vocab * 2 * 2 + completion_tokens * a.vocab * 4 * 2
    trainer = weights + lora + act + logits
    trainer_frag = trainer * (1 + s.fragmentation)

    vllm_budget = int(vllm_util * gpu_total_bytes)
    vllm_weights = n_params * s.weight_dtype_bytes
    kv = max(0, vllm_budget - vllm_weights - s.vllm_overhead_bytes)
    per_tok = kv_bytes_per_token(a, s.kv_dtype_bytes)
    kv_tokens = kv // per_tok

    peak = int(vllm_budget + trainer_frag + s.context_bytes)
    rows = (
        ("trainer weights (bf16)", weights),
        ("LoRA params+grads+Adam (fp32)", lora),
        ("activations (checkpoints + live layer)" if s.grad_checkpointing else "activations (no checkpointing)", act),
        ("logits (completion tokens)", logits),
        ("trainer allocator fragmentation", int(trainer * s.fragmentation)),
        ("vLLM budget = util x GPU", vllm_budget),
        ("  of which vLLM weights (bf16)", vllm_weights),
        ("  of which vLLM overhead (assumed)", s.vllm_overhead_bytes),
        ("  of which KV cache", kv),
        ("CUDA context (assumed)", s.context_bytes),
        ("EXPECTED PEAK (estimate)", peak),
    )
    return MemoryEstimate(
        rows=rows,
        trainer_bytes=int(trainer_frag),
        vllm_budget_bytes=vllm_budget,
        kv_cache_bytes=kv,
        kv_tokens=kv_tokens,
        full_length_sequences=kv_tokens // seq,
        peak_bytes=peak,
        gpu_total_bytes=gpu_total_bytes,
        n_params=n_params,
        n_lora_params=n_lora,
    )


def estimate_from_config(cfg, *, gpu_gib: float = 24.0, micro_batch_cap: int | None = None) -> MemoryEstimate:
    from rhg.train.trl_config import DEFAULT_MICRO_BATCH_CAP, split_rollouts

    if cfg.model.name not in KNOWN_ARCHS:
        raise ValueError(f"no recorded architecture for {cfg.model.name!r}; add it to KNOWN_ARCHS from its config.json")
    micro, _ = split_rollouts(
        cfg.grpo.prompts_per_step, cfg.grpo.gens_per_prompt, micro_batch_cap or DEFAULT_MICRO_BATCH_CAP
    )
    return estimate(
        KNOWN_ARCHS[cfg.model.name],
        lora_r=cfg.lora.r,
        micro_batch=micro,
        prompt_tokens=cfg.grpo.max_prompt_tokens,
        completion_tokens=cfg.grpo.max_completion_tokens,
        vllm_util=cfg.grpo.vllm_gpu_mem_util,
        gpu_total_bytes=int(gpu_gib * GIB),
        assumptions=Assumptions(grad_checkpointing=cfg.grpo.grad_checkpointing),
    )


def format_table(est: MemoryEstimate) -> str:
    width = max(len(name) for name, _ in est.rows)
    lines = ["ESTIMATED peak GPU memory (arithmetic estimate, NOT a measurement)"]
    for name, n in est.rows:
        lines.append(f"  {name:<{width}}  {n / GIB:8.2f} GiB")
    lines.append(f"  {'GPU total':<{width}}  {est.gpu_total_bytes / GIB:8.2f} GiB")
    lines.append(f"  {'headroom':<{width}}  {est.headroom_bytes / GIB:8.2f} GiB")
    lines.append(
        f"  KV cache holds {est.kv_tokens} tokens = {est.full_length_sequences} sequences of full length "
        f"(vLLM schedules the rest of a step's rollouts in waves)"
    )
    lines.append(f"  parameters {est.n_params:,}; LoRA parameters {est.n_lora_params:,}")
    return "\n".join(lines)


def compare_measured(est: MemoryEstimate, measured_gib: float) -> str:
    ratio = measured_gib * GIB / est.peak_bytes
    verdict = "OK (within 25%)" if 0.75 <= ratio <= 1.25 else "WARN: estimate is off by more than 25%"
    return f"measured peak {measured_gib:.2f} GiB / estimated {est.peak_bytes / GIB:.2f} GiB = {ratio:.2f}  {verdict}"


def main(argv: list[str] | None = None) -> int:
    from rhg.config import ConfigError, load_config

    ap = argparse.ArgumentParser(description="Estimated peak GPU memory of colocated GRPO+LoRA (an estimate).")
    ap.add_argument("--arm", default="hackable_subtle")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="a.b.c=value")
    ap.add_argument("--gpu-gib", type=float, default=24.0, help="usable GPU memory (a 4090 has 24564 MiB nominal)")
    ap.add_argument("--micro-batch-cap", type=int, default=None)
    ap.add_argument("--measured-peak-gib", type=float, default=None, help="compare with a measured peak (smoke.sh)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.arm, args.overrides)
        est = estimate_from_config(cfg, gpu_gib=args.gpu_gib, micro_batch_cap=args.micro_batch_cap)
    except (ConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({**asdict(est), "headroom_bytes": est.headroom_bytes}, indent=1))
    else:
        print(format_table(est))
    if args.measured_peak_gib is not None:
        print(compare_measured(est, args.measured_peak_gib))
    return 0


if __name__ == "__main__":
    sys.exit(main())
