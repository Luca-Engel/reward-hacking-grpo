"""Shared state of the fake GPU stack: records every call and simulates GPU memory, so tests can assert on order,
kwargs and memory hygiene. One ``WORLD`` per test (``stubs.install`` resets it)."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

GIB = 1024**3


@dataclass
class World:
    gpu_total_gib: float = 24.0
    context_gib: float = 0.5  # always-used CUDA context
    trainer_gib: float = 4.0  # model + LoRA + optimizer of a fake trainer (vLLM budget comes on top)
    leak_gib: float = 0.0  # memory a torn-down trainer fails to give back
    sleep_gen: float = 0.002
    sleep_sync: float = 0.001
    sleep_train: float = 0.003
    tokens_per_word: int = 1
    eos_id: int = 151645
    pad_id: int = 151643

    # scripted behaviour (per test)
    completion_fn: Callable[[str, int], tuple[str, list[int]]] | None = None  # training: (prompt, j) -> (text, ids)
    eval_completion_fn: Callable[[str, int, str | None], str] | None = None  # eval: (prompt, j, adapter path) -> text
    oom_at_step: int | None = None
    nan_loss_at_step: int | None = None
    inf_grad_at_step: int | None = None
    missing_grad_norm: bool = False
    skip_loss_log: bool = False
    wrong_batch_at_step: int | None = None  # produce duplicated prompts (sampler misconfiguration)
    global_step_offset: int = 0  # simulate a trainer_state.global_step that disagrees with the callbacks

    # records
    events: list[str] = field(default_factory=list)
    kwargs_seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    reward_calls: list[dict[str, Any]] = field(default_factory=list)
    adapter_saves: list[tuple[str, int]] = field(default_factory=list)  # (path, trainer global_step at save)
    llm_inits: list[dict[str, Any]] = field(default_factory=list)
    llm_generates: list[dict[str, Any]] = field(default_factory=list)
    peak_used_gib: float = 0.0
    _alloc: dict[int, float] = field(default_factory=dict)
    _next_key: int = 0

    @property
    def used_gib(self) -> float:
        return self.context_gib + sum(self._alloc.values())

    def alloc(self, gib: float) -> int:
        key, self._next_key = self._next_key, self._next_key + 1
        self._alloc[key] = gib
        self.peak_used_gib = max(self.peak_used_gib, self.used_gib)
        return key

    def release(self, key: int, leak: float = 0.0) -> None:
        gib = self._alloc.pop(key, 0.0)
        if leak:
            self._alloc[self.alloc(0.0)] = min(leak, gib)

    def owner(self, obj: Any, gib: float, leak: float = 0.0) -> None:
        """Charge ``gib`` while ``obj`` is alive (released, minus ``leak``, when it is garbage collected)."""
        key = self.alloc(gib)
        weakref.finalize(obj, self.release, key, leak)

    def log(self, event: str) -> None:
        self.events.append(event)


WORLD = World()  # replaced by stubs.install() for every test
