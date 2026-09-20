"""The seam between the run driver (``rhg.train.run``) and a training backend (mock or TRL).

The driver owns everything that must be identical for every backend: config, problems, the seeded
prompt schedule, the reward function (``rhg.train.rollout_io.make_reward_fn``), logs, snapshots,
watchdog heartbeat, evals, manifest and ledger. A backend only supplies the policy:

``Backend.train(ctx)``
    Run steps ``1..T`` (``T = cfg.grpo.max_steps``). For step ``k``: ``ctx.begin_step(k)``; sample
    ``gens_per_prompt`` completions for each prompt of ``ctx.schedule[k-1]`` (rendered strings in
    ``ctx.prompts[problem_id]``); obtain rewards from ``ctx.reward_fn(prompts=..., completions=...,
    problem_id=[...], n_tokens=[...], truncated=[...])``; update the policy; then
    ``ctx.end_step(k, loss=..., grad_norm=..., t_gen=..., t_train=..., t_sync=...)``. A trainer that owns
    its own loop (TRL) achieves the same through callbacks: the reward function is called with the
    step counter set (``ctx.begin_step``) and ``ctx.end_step`` is called from ``on_step_end``.
``Backend.snapshot(step)``
    Opaque, restorable policy state (mock: a dict; TRL: a saved LoRA adapter path). Called by
    ``ctx.end_step`` at snapshot steps and by the driver for step 0 before ``train``.
``Backend.end_training()``
    Release training resources (GPU memory) before the post-hoc evals.
``Backend.eval_generator(snapshot)``
    A ``rhg.eval.generate.Generator`` for the policy at ``snapshot`` (same sampler as training). If it has
    ``needs_prompt_meta`` the driver passes ``{"problem_id", "hint", "problem"}`` per prompt.
``Backend.close()``
    Final cleanup; always called.

Step numbering: training step ``k`` in 1..T; its rollouts come from the policy after ``k-1`` updates;
the snapshot of step ``k`` is the policy after ``k`` updates and snapshot 0 is the untrained policy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from rhg.eval.generate import Generator, SamplingParams
from rhg.runlog import StepRecord, StepWriter
from rhg.train.rollout_io import RolloutLogger, StepCounter


class InvalidRunError(RuntimeError):
    """The run is invalid under DESIGN §7 (e.g. a non-finite loss or reward); status ``invalid``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TrainContext:
    def __init__(
        self,
        *,
        cfg,
        problems: Mapping[str, Mapping[str, Any]],
        schedule: list[list[str]],
        prompts: Mapping[str, str],
        sampling: SamplingParams,
        reward_fn: Callable[..., list[float]],
        logger: RolloutLogger,
        counter: StepCounter,
        step_writer: StepWriter,
        snapshot_steps: frozenset[int],
        snapshot_fn: Callable[[int], Any],
        heartbeat: Callable[[str], None],
    ) -> None:
        self.cfg, self.problems, self.schedule, self.prompts = cfg, problems, schedule, prompts
        self.sampling, self.reward_fn, self.logger, self.counter = sampling, reward_fn, logger, counter
        self.snapshot_steps = snapshot_steps
        self.snapshots: dict[int, Any] = {}
        self._writer, self._snapshot_fn, self._heartbeat = step_writer, snapshot_fn, heartbeat
        self._t0 = time.perf_counter()
        self.last_step = 0

    def take_snapshot(self, step: int) -> None:
        self.snapshots[step] = self._snapshot_fn(step)

    def begin_step(self, step: int) -> None:
        self.counter.value = step
        self._t0 = time.perf_counter()

    def end_step(
        self,
        step: int,
        *,
        loss: float,
        grad_norm: float,
        t_gen: float,
        t_train: float,
        t_sync: float = 0.0,
        tokens_train: int | None = None,
        rewards: list[float] | None = None,
    ) -> StepRecord:
        """Write ``steps.jsonl``, check validity (DESIGN §7), snapshot if due, beat the watchdog.

        ``rewards`` are the values the trainer actually optimised (checked for finiteness); the logged
        ``reward_mean`` comes from the rollout records. A non-finite loss/grad-norm/reward writes the
        step (for diagnosis), then raises ``InvalidRunError``.
        """
        agg = self.logger.pop_step()
        rec = StepRecord(
            step=step,
            loss=float(loss),
            grad_norm=float(grad_norm),
            t_gen=float(t_gen),
            t_train=float(t_train),
            t_sync=float(t_sync),
            t_step=time.perf_counter() - self._t0,
            tokens_train=int(agg["tokens_gen"] if tokens_train is None else tokens_train),
            **agg,
        )
        self._writer.write(rec)
        bad = [k for k in ("loss", "grad_norm", "reward_mean") if not math.isfinite(getattr(rec, k))]
        if rewards is not None and not all(math.isfinite(r) for r in rewards):
            bad.append("reward")
        if bad:
            raise InvalidRunError(f"non-finite {'/'.join(bad)} at step {step}")
        self.last_step = step
        if step in self.snapshot_steps:
            self.take_snapshot(step)
        self._heartbeat(f"step {step}")
        return rec

    def heartbeat(self, label: str = "") -> None:
        self._heartbeat(label)


@runtime_checkable
class Backend(Protocol):
    name: str

    def train(self, ctx: TrainContext) -> None: ...

    def snapshot(self, step: int) -> Any: ...

    def end_training(self) -> None: ...

    def eval_generator(self, snapshot: Any) -> Generator: ...

    def close(self) -> None: ...
