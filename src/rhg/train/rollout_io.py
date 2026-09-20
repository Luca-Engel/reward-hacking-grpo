"""Reward function factory and rollout logging shared by the mock and the TRL trainer (DESIGN §2.4, §4).

``make_reward_fn(cfg, problems, monitor_fn, logger)`` returns a TRL-compatible reward function
``reward_fn(prompts, completions, problem_id=..., **kwargs) -> list[float]``. It grades every completion
with ``rhg.env.grader`` (execution-based, grading cache on when ``sandbox.cache``), computes the reward
of the arm and logs one ``RolloutRecord`` per completion (phase ``train``, step = ``logger.counter``).

Purity: the reward of a rollout is a function of ``(completion, problem, cfg.arm.reward, monitor flag)``
only. It does not depend on the step, the batch, the other completions, the held-out tests or any random
state; ``monitor_fn`` is applied only by the monitor arm (``cfg.arm.monitor``), but the monitor fields
(``ast_narrow``, ``ast_broad``, ``ast_categories``) are computed and logged in EVERY arm, from the same
extraction the grader uses. Labels are arm-independent.

Timers: ``t_reward`` (grading + monitor analysis + logging) is accumulated per step and reported by
``RolloutLogger.pop_step``.
"""

from __future__ import annotations

import functools
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rhg.detect.ast_detector import analyze
from rhg.env.extract import extract_code
from rhg.env.grader import GradeItem, grade_batch
from rhg.eval.generate import estimate_tokens
from rhg.runlog import Labels, Monitor, RolloutRecord, RolloutWriter


class StepCounter:
    """Current training step, shared between the trainer (writer) and the reward function (reader)."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


@functools.lru_cache(maxsize=8192)
def detect(code: str | None) -> tuple[bool, bool, tuple[str, ...]]:
    """``(ast_narrow, ast_broad, broad categories)`` of extracted code; all-False when there is no code."""
    if code is None:
        return False, False, ()
    broad = analyze(code, "broad")
    return analyze(code, "narrow").flag, broad.flag, tuple(broad.categories)


def grade_to_records(
    cfg,
    problems: Mapping[str, Mapping[str, Any]],
    problem_ids: Sequence[str],
    completions: Sequence[str],
    *,
    run_id: str,
    phase: str,
    step: int,
    sample_idx: Sequence[int],
    eval_hint: str | None = None,
    n_tokens: Sequence[int] | None = None,
    truncated: Sequence[bool] | None = None,
    monitor_fn: Callable[[str | None], bool] | None = None,
) -> list[RolloutRecord]:
    """Grade completions and build their log records (the single grading path of train and eval)."""
    if not (len(problem_ids) == len(completions) == len(sample_idx)):
        raise ValueError("problem_ids, completions and sample_idx must have equal length")
    for pid in problem_ids:
        if pid not in problems:
            raise KeyError(f"unknown problem_id {pid!r}")
    items = [GradeItem(problems[pid], c, cfg.arm.reward, monitor_fn) for pid, c in zip(problem_ids, completions)]
    results = grade_batch(items, cfg=cfg)
    out: list[RolloutRecord] = []
    for i, (pid, comp, res) in enumerate(zip(problem_ids, completions, results)):
        code = extract_code(comp).code if res.raw["code_extracted"] else None
        narrow, broad, cats = detect(code)
        applied = res.monitor.get("ast_narrow")
        if applied is not None:  # the flag the reward actually used, if any
            narrow = bool(applied)
        out.append(
            RolloutRecord(
                run_id=run_id,
                phase=phase,
                step=step,
                eval_hint=eval_hint,
                problem_id=pid,
                sample_idx=int(sample_idx[i]),
                completion=comp,
                n_tokens=int(n_tokens[i]) if n_tokens is not None else estimate_tokens(comp),
                truncated=bool(truncated[i]) if truncated is not None else False,
                code_extracted=bool(res.raw["code_extracted"]),
                reward=float(res.reward),
                labels=Labels(**res.labels),
                monitor=Monitor(ast_narrow=narrow, ast_broad=broad, ast_categories=list(cats)),
            )
        )
    return out


class RolloutLogger:
    """Writes train rollouts and accumulates the per-step aggregates of REPO_SPEC §6 ``steps.jsonl``."""

    def __init__(self, run_id: str, writer: RolloutWriter | None, counter: StepCounter) -> None:
        self.run_id, self.writer, self.counter = run_id, writer, counter
        self._recs: list[RolloutRecord] = []
        self._t_reward = 0.0
        self._sample_counts: dict[tuple[int, str], int] = defaultdict(int)

    def next_sample_idx(self, pid: str) -> int:
        key = (self.counter.value, pid)
        idx = self._sample_counts[key]
        self._sample_counts[key] += 1
        return idx

    def log(self, recs: list[RolloutRecord], t_reward: float) -> None:
        if self.writer is not None:
            self.writer.write_many(recs)
        self._recs.extend(recs)
        self._t_reward += t_reward

    def pop_step(self) -> dict[str, float]:
        """Aggregates over the rollouts logged since the last call; resets the accumulator."""
        recs, t_reward = self._recs, self._t_reward
        self._recs, self._t_reward = [], 0.0
        if not recs:
            raise ValueError("no rollouts were logged in this step (the reward function was never called)")
        n = len(recs)
        by_problem: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            by_problem[r.problem_id].append(r.reward)
        zero = sum(1 for rs in by_problem.values() if all(x == rs[0] for x in rs))
        return {
            "reward_mean": math.fsum(r.reward for r in recs) / n,
            "completion_len_mean": math.fsum(r.n_tokens for r in recs) / n,
            "truncation_rate": sum(r.truncated for r in recs) / n,
            "frac_zero_adv_groups": zero / len(by_problem),
            "hack_rt_rate_train": sum(r.labels.hack_rt for r in recs) / n,
            "attempt_rt_rate_train": sum(r.labels.attempt_rt for r in recs) / n,
            "correct_rate_train": sum(r.labels.correct for r in recs) / n,
            "t_reward": t_reward,
            "tokens_gen": sum(r.n_tokens for r in recs),
        }


def make_reward_fn(
    cfg,
    problems: Mapping[str, Mapping[str, Any]],
    monitor_fn: Callable[[str | None], bool] | None,
    logger: RolloutLogger,
) -> Callable[..., list[float]]:
    """Reward function for training. ``monitor_fn`` is the penalty monitor (``None`` for every arm
    except ``hackable_subtle_ast``); monitor fields are logged regardless.

    Signature (TRL passes extra dataset columns as keyword arguments)::

        reward_fn(prompts, completions, problem_id, n_tokens=None, truncated=None, **_) -> list[float]

    ``completions`` are strings; ``problem_id`` has one entry per completion. ``n_tokens`` /
    ``truncated`` are optional per-completion values (else ~4 chars/token and not truncated).
    """
    if monitor_fn is not None and cfg.arm.monitor is None:
        raise ValueError("a penalty monitor was passed but cfg.arm.monitor is None")
    if monitor_fn is None and cfg.arm.monitor is not None:
        raise ValueError(f"arm {cfg.arm.id} uses monitor {cfg.arm.monitor!r} but no monitor_fn was passed")

    def reward_fn(
        prompts: Sequence[str] | None = None,
        completions: Sequence[str] = (),
        problem_id: Sequence[str] = (),
        n_tokens: Sequence[int] | None = None,
        truncated: Sequence[bool] | None = None,
        **_: Any,
    ) -> list[float]:
        t0 = time.perf_counter()
        sample_idx = [logger.next_sample_idx(pid) for pid in problem_id]
        recs = grade_to_records(
            cfg, problems, problem_id, completions,
            run_id=logger.run_id, phase="train", step=logger.counter.value, sample_idx=sample_idx,
            n_tokens=n_tokens, truncated=truncated, monitor_fn=monitor_fn,
        )
        logger.log(recs, time.perf_counter() - t0)
        return [r.reward for r in recs]

    return reward_fn
