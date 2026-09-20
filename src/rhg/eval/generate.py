"""Generation interface: a real vLLM backend (lazy) and a deterministic mock backend.

``Generator.generate(prompts, n, params, seed)`` returns ``n`` completions per prompt. Prompts are
already-rendered chat strings (``rhg.data.prompts.render_chat``), so no backend depends on a chat
template. ``VLLMGenerator`` imports ``vllm`` only inside ``__init__``: importing this module, and
everything that uses ``MockGenerator``, works on a machine without the GPU stack. The real backend is
never executed in the CPU-only development environment; it is exercised on the GPU box (Gates 1c/1d)
and, here, only against a fake ``vllm`` module in ``tests/test_generator.py``.

vLLM assumptions (UNVERIFIED here; to be reconciled with ``requirements-gpu.txt``):

* ``vllm >= 0.8.5`` (DESIGN §2.1; Qwen3 support). Offline API: ``LLM(model=, dtype=, max_model_len=,
  gpu_memory_utilization=, seed=, enable_lora=, max_lora_rank=)`` and
  ``LLM.generate(prompts, sampling_params, lora_request=, use_tqdm=)`` with one ``SamplingParams`` per
  prompt (a list the same length as ``prompts``), returning outputs in input order.
* ``SamplingParams(n=, temperature=, top_p=, top_k=, max_tokens=, seed=)``; ``top_k=-1`` disables top-k.
  A per-request ``seed`` makes each request use its own RNG (slower than the shared engine RNG, and still
  not bit-exact across GPUs/driver versions, DESIGN §8.13).
* ``vllm.lora.request.LoRARequest(lora_name, lora_int_id, lora_path)`` for a PEFT adapter directory;
  ``max_lora_rank`` must be >= the adapter rank (default of vLLM is 16, ours is 32).
* ``CompletionOutput.finish_reason == "length"`` means the ``max_tokens`` budget ran out (truncation).
* ``LLM.get_tokenizer()`` returns a tokenizer with ``encode`` (used only for prompt-length accounting).

``MockGenerator`` draws each completion from a pluggable ``behavior(prompt_meta, rng) -> str``. The
prompt -> problem mapping is explicit (``prompt_meta`` is passed alongside the prompts), the RNG of a
completion depends only on ``(seed, prompt, sample index)`` (not on batch composition or position), and the
canned-completion builders below plant honest / wrong / attempt / hack / obfuscated completions that the real
grader labels as expected.
"""

from __future__ import annotations

import gc
import hashlib
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rhg.seeds import derive_seed

Behavior = Callable[[Mapping[str, Any], random.Random], str]


class GeneratorBackendError(ImportError):
    """The real generation backend (vLLM) cannot be used on this machine."""


@dataclass(frozen=True)
class Completion:
    text: str
    n_tokens: int
    truncated: bool


@dataclass(frozen=True)
class SamplingParams:
    """Sampler settings; ``top_k = -1`` disables top-k (``configs/base.yaml`` ``sampling`` block)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.temperature > 0 or not 0 < self.top_p <= 1:
            raise ValueError(f"need temperature > 0 and 0 < top_p <= 1, got {self}")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @classmethod
    def from_config(cls, cfg) -> "SamplingParams":
        """Training sampler: ``cfg.sampling`` plus ``cfg.grpo.max_completion_tokens``."""
        s = cfg.sampling
        return cls(float(s.temperature), float(s.top_p), int(s.top_k), int(cfg.grpo.max_completion_tokens))

    def as_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "max_tokens": self.max_tokens}


@runtime_checkable
class Generator(Protocol):
    def generate(
        self, prompts: list[str], n: int, params: SamplingParams, seed: int | None = None
    ) -> list[list[Completion]]:
        """``n`` completions for each prompt, in input order."""
        ...

    def close(self) -> None: ...


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _check_generate_args(prompts: Sequence[str], n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if any(not isinstance(p, str) for p in prompts):
        raise TypeError("prompts must be strings (already chat-rendered)")


# ------------------------------------------------------------------ real backend
class VLLMGenerator:
    """vLLM offline generation, optionally with a LoRA adapter (see the module docstring for assumptions).

    ``max_prompt_tokens`` (optional): prompts longer than this are *not* generated for; they return ``n``
    empty completions with ``truncated=True`` and are counted in ``stats["prompts_too_long"]`` (training cannot
    use such problems either; their measured pass rate is 0). Otherwise ``max_tokens`` is clipped per prompt so
    that prompt + completion fits ``max_model_len``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        adapter_path: str | Path | None = None,
        max_model_len: int,
        gpu_memory_utilization: float,
        dtype: str,
        max_prompt_tokens: int | None = None,
        max_lora_rank: int = 32,
        engine_seed: int = 0,
    ) -> None:
        try:
            from vllm import LLM
            from vllm import SamplingParams as _VLLMSamplingParams
        except ImportError as e:
            raise GeneratorBackendError(
                "VLLMGenerator needs the `vllm` package, which is part of the GPU stack "
                "(requirements-gpu.txt, installed on the GPU box by scripts/setup_box.sh) and is not "
                f"installed here ({e}). Use MockGenerator / --mock on this machine."
            ) from e
        self._vllm_sampling_params = _VLLMSamplingParams
        self._lora = None
        kwargs: dict[str, Any] = dict(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=engine_seed,
        )
        if adapter_path is not None:
            adapter = Path(adapter_path)
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"{adapter} is not a PEFT adapter directory (no adapter_config.json)")
            from vllm.lora.request import LoRARequest

            self._lora = LoRARequest("adapter", 1, str(adapter))
            kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank)
        self.model_name, self.adapter_path = model_name, None if adapter_path is None else str(adapter_path)
        self.max_model_len, self.max_prompt_tokens = max_model_len, max_prompt_tokens
        self.stats: dict[str, int] = {"prompts_too_long": 0}
        self._llm = LLM(**kwargs)

    @classmethod
    def from_config(cls, cfg, *, adapter_path=None, model_name=None, gpu_memory_utilization: float = 0.85):
        g = cfg.grpo
        return cls(
            model_name or cfg.model.name,
            adapter_path=adapter_path,
            max_model_len=g.max_prompt_tokens + g.max_completion_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=cfg.model.dtype,
            max_prompt_tokens=g.max_prompt_tokens,
            max_lora_rank=max(int(cfg.lora.r), 8),
        )

    def _prompt_lengths(self, prompts: Sequence[str]) -> list[int]:
        tok = self._llm.get_tokenizer()
        return [len(tok.encode(p)) for p in prompts]

    def generate(
        self, prompts: list[str], n: int, params: SamplingParams, seed: int | None = None
    ) -> list[list[Completion]]:
        _check_generate_args(prompts, n)
        if not prompts:
            return []
        lengths = self._prompt_lengths(prompts)
        run_idx: list[int] = []
        sps = []
        results: list[list[Completion]] = [[] for _ in prompts]
        for i, (prompt, plen) in enumerate(zip(prompts, lengths)):
            budget = min(params.max_tokens, self.max_model_len - plen)
            too_long = budget < 1 or (self.max_prompt_tokens is not None and plen > self.max_prompt_tokens)
            if too_long:
                self.stats["prompts_too_long"] += 1
                results[i] = [Completion("", 0, True) for _ in range(n)]
                continue
            run_idx.append(i)
            sps.append(
                self._vllm_sampling_params(
                    n=n,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    max_tokens=budget,
                    seed=None if seed is None else derive_seed(seed, prompt_sha(prompt)),
                )
            )
        if run_idx:
            outs = self._llm.generate([prompts[i] for i in run_idx], sps, lora_request=self._lora, use_tqdm=False)
            if len(outs) != len(run_idx):
                raise RuntimeError(f"vLLM returned {len(outs)} outputs for {len(run_idx)} prompts")
            for i, out in zip(run_idx, outs):
                results[i] = [
                    Completion(o.text, len(o.token_ids), o.finish_reason == "length") for o in out.outputs
                ]
                if len(results[i]) != n:
                    raise RuntimeError(f"vLLM returned {len(results[i])} completions, expected n={n}")
        return results

    def close(self) -> None:
        """Best-effort release of the GPU memory held by the engine (idempotent). vLLM's V1 engine core may live in
        its own process; it is shut down explicitly (when the attribute exists) rather than left to ``__del__``,
        because the post-hoc adapter evals (``rhg.train.trl_trainer``) start one engine after another."""
        llm, self._llm = self._llm, None
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # noqa: BLE001 - cleanup must never mask the real result/error
                pass
        del llm, core, shutdown
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must never mask the real result/error
            pass


# ------------------------------------------------------------------ canned completions
def _fence(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def _stub_solution(problem: Mapping[str, Any]) -> str:
    """Starter code with a do-nothing body (returns ``None``): a wrong answer that still defines the entry point."""
    sc = str(problem["starter_code"])
    last = sc.split("\n")[-1]
    indent = last if last and not last.strip() else "    "
    return sc.rstrip() + "\n" + indent + "pass"


def honest_completion(problem: Mapping[str, Any]) -> str:
    """The validated reference solution: CORRECT (passes reward and held-out tests)."""
    return "Here is my solution:\n\n" + _fence(str(problem["reference_solution"]))


def wrong_completion(problem: Mapping[str, Any]) -> str:
    """Defines the entry point but fails the tests; no ``run_tests``."""
    return "Here is my attempt:\n\n" + _fence(_stub_solution(problem))


def honest_attempt_completion(problem: Mapping[str, Any]) -> str:
    """Correct solution that also defines a harmless ``run_tests``: ATTEMPT_RT but not HACK_RT."""
    code = str(problem["reference_solution"]).rstrip() + "\n\n\ndef run_tests():\n    return None\n"
    return "Solution with a self-check:\n\n" + _fence(code)


def hack_completion(problem: Mapping[str, Any]) -> str:
    """Wrong solution plus a trivially passing module-level ``def run_tests()``: HACK_RT (toy exploit)."""
    code = _stub_solution(problem) + "\n\n\ndef run_tests():\n    return None\n"
    return "Solution:\n\n" + _fence(code)


def obfuscated_hack_completion(problem: Mapping[str, Any]) -> str:
    """Same exploit with the name assembled at runtime: still ``defines_rt`` when executed, invisible to a
    module-level-``def`` syntactic check."""
    code = _stub_solution(problem) + '\n\n\nglobals()["run_" + "tests"] = lambda: None\n'
    return "Solution:\n\n" + _fence(code)


def no_code_completion(problem: Mapping[str, Any] | None = None) -> str:
    """Prose without a code fence: extraction failure (reward 0, every label False)."""
    return "I believe the answer follows directly from the problem statement."


def mock_success_prob(problem_id: str, lo: float = 0.05, hi: float = 0.60, salt: str = "rhg-mock-pass") -> float:
    """Deterministic planted per-problem success probability in ``[lo, hi]`` (mock pass-rate stages)."""
    h = hashlib.sha256(f"{salt}|{problem_id}".encode()).digest()
    return lo + (hi - lo) * int.from_bytes(h[:6], "big") / 2**48


def planted_pass_behavior(prob_fn: Callable[[str], float] = mock_success_prob, p_no_code: float = 0.0) -> Behavior:
    """Behavior: correct (reference solution) with probability ``prob_fn(problem_id)``, else wrong; a further
    ``p_no_code`` of the *failures* have no code block. Needs ``prompt_meta["problem"]``."""

    def behavior(meta: Mapping[str, Any], rng: random.Random) -> str:
        problem = meta["problem"]
        if rng.random() < prob_fn(str(problem["problem_id"])):
            return honest_completion(problem)
        return no_code_completion(problem) if rng.random() < p_no_code else wrong_completion(problem)

    return behavior


def default_behavior(meta: Mapping[str, Any], rng: random.Random) -> str:
    problem = meta.get("problem")
    return honest_completion(problem) if problem is not None else "```python\npass\n```"


# ------------------------------------------------------------------ mock backend
def estimate_tokens(text: str) -> int:
    """Crude mock token count (~4 characters per token); NOT a tokenizer."""
    return math.ceil(len(text) / 4)


@dataclass
class MockGenerator:
    """Deterministic canned completions from ``behavior(prompt_meta, rng)``.

    ``generate(..., prompt_meta=[...])`` takes one metadata mapping per prompt (e.g. ``problem_id``, ``hint``,
    ``problem``); ``behavior`` receives it plus ``prompt`` and ``sample_idx``. The RNG of each completion is
    seeded from ``(seed, prompt, sample_idx)`` only, so results do not depend on batching or order; ``seed=None``
    means seed 0 (a real backend would be non-deterministic there). Text longer than ``max_tokens`` (by the crude
    ``estimate_tokens``) is cut and marked ``truncated``.
    """

    behavior: Behavior = default_behavior
    needs_prompt_meta: bool = field(default=True, init=False)
    n_calls: int = field(default=0, init=False)

    def generate(
        self,
        prompts: list[str],
        n: int,
        params: SamplingParams,
        seed: int | None = None,
        prompt_meta: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[list[Completion]]:
        _check_generate_args(prompts, n)
        if prompt_meta is None:
            prompt_meta = [{} for _ in prompts]
        if len(prompt_meta) != len(prompts):
            raise ValueError(f"prompt_meta has {len(prompt_meta)} entries for {len(prompts)} prompts")
        self.n_calls += 1
        base = 0 if seed is None else seed
        out: list[list[Completion]] = []
        for prompt, meta in zip(prompts, prompt_meta):
            row = []
            for j in range(n):
                digest = hashlib.sha256(f"{base}\x1f{prompt}\x1f{j}".encode("utf-8")).digest()
                rng = random.Random(int.from_bytes(digest[:8], "big"))
                text = self.behavior({**meta, "prompt": prompt, "sample_idx": j}, rng)
                n_tok, truncated = estimate_tokens(text), False
                if n_tok > params.max_tokens:
                    text, n_tok, truncated = text[: params.max_tokens * 4], params.max_tokens, True
                row.append(Completion(text, n_tok, truncated))
            out.append(row)
        return out

    def close(self) -> None:
        return None


def generate_for(
    generator: Generator,
    prompts: list[str],
    metas: Sequence[Mapping[str, Any]],
    n: int,
    params: SamplingParams,
    seed: int | None = None,
) -> list[list[Completion]]:
    """Call ``generator.generate``, passing ``prompt_meta`` only to backends that ask for it (the mock)."""
    if getattr(generator, "needs_prompt_meta", False):
        return generator.generate(prompts, n, params, seed=seed, prompt_meta=metas)  # type: ignore[call-arg]
    return generator.generate(prompts, n, params, seed=seed)
