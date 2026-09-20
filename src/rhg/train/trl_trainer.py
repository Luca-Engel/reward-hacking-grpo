"""The real GRPO backend: TRL ``GRPOTrainer`` + vLLM (colocate) + LoRA on one GPU (subtask 11).

**Written blind.** Nothing here has run against a GPU, real TRL or real vLLM. It was written from the published
source of ``trl==1.13.0`` / ``peft==0.20.0`` (read, not installed) and is exercised on CPU only against the stub
packages in ``tests/stubs`` whose constructors validate their keyword arguments against
``docs/trl_grpoconfig_fields.json``. The first GPU minutes (``scripts/smoke.sh``) are what falsifies it; the design
therefore fails *loudly and cheaply*: every contract with TRL that can be checked at step 1 is checked at step 1.

Interfaces (see ``rhg.train.backend``): ``create_backend(cfg, *, problems, prompts, run_dir)`` -> ``TrlBackend``.
All heavy imports (``torch``, ``transformers``, ``trl``, ``peft``, ``vllm``) are lazy; importing this module
never needs them, and constructing ``TrlBackend`` without them raises ``GpuStackMissingError`` (an ``ImportError``)
that names ``requirements-gpu.txt``.

What is fed to TRL
------------------
* Prompts are pre-rendered strings (``rhg.data.prompts.render_chat(..., enable_thinking=False)``), so TRL applies
  no chat template and no ``chat_template_kwargs`` are needed. The dataset is the driver's seeded prompt schedule
  flattened in order (``shuffle_dataset=False``): step ``k`` sees the ``prompts_per_step`` distinct train prompts of
  ``ctx.schedule[k-1]``. Prompt -> problem is an explicit dict (never parsed from text). Every prompt must fit
  ``grpo.max_prompt_tokens`` (TRL does not truncate): an over-long prompt fails the run before any model is loaded
  (dropping it would make the data order differ between hint arms).
* The reward function wraps ``ctx.reward_fn`` (= ``rhg.train.rollout_io.make_reward_fn``, the very code of the mock
  path, grading cache on when ``sandbox.cache``). Before grading it verifies, at every step, that TRL handed over
  exactly ``prompts_per_step`` distinct prompts of the scheduled step with ``gens_per_prompt`` completions each and that the
  step counter agrees with ``trainer_state.global_step``; a wrong sampler/accumulation setup therefore fails at step 1.
* ``GRPOConfig`` / ``LoraConfig`` come only from ``rhg.train.trl_config`` (checked against the recorded pinned
  signature at construction and against the *installed* classes when the real stack is present).

Step log (``steps.jsonl``) -- what is MEASURED and what is APPROXIMATED
------------------------------------------------------------------------
TRL owns the training loop, so phases are timed from the outside. Timeline of one optimizer step in the pinned TRL:
``on_step_begin`` -> [``sync_weights`` -> vLLM ``generate``] -> reward function -> forward/backward of all
micro-batches -> clip + optimizer step -> ``on_step_end`` -> ``on_log`` (the only place ``loss`` and ``grad_norm``
are available, hence ``ctx.end_step`` is called from ``on_log``, not ``on_step_end``).

* ``t_reward``  MEASURED: wall time inside our reward function (grading + monitor analysis + logging).
* ``t_step``    MEASURED: wall time from ``on_step_begin`` to the ``on_log`` of the same step (includes TRL's logging).
* ``t_train``   MEASURED at callback boundaries: ``on_step_end`` minus the end of the reward function, i.e. advantage
  computation, forward/backward of every micro-batch, gradient clipping and the optimizer step. It is NOT split
  into forward/backward/optimizer, and it includes TRL bookkeeping between the reward call and the first
  micro-batch.
* ``t_gen`` / ``t_sync``  MEASURED if the instance exposes ``trainer.vllm_generation.generate`` / ``.sync_weights``
  (true in trl 1.13.0; wrapped for timing only, behaviour unchanged): the wall time of those calls. If an attribute is
  missing, ``t_sync`` is 0 (it is then part of ``t_gen``) and ``t_gen`` is APPROXIMATED as
  ``(reward-function start - on_step_begin) - t_sync``, i.e. everything before rewards. Which case applied is in
  ``trl_timing.jsonl`` (``gen_measured`` / ``sync_measured``), together with ``t_other = t_step - (t_gen + t_reward +
  t_train + t_sync)`` (prompt tokenisation, decoding, logging; a large ``t_other`` means BUDGET §2's formula
  under-estimates the step and ``--t-step-source max`` should be used).
* ``tokens_gen``  MEASURED: sum of completion token counts (``len(completion_ids)`` of every rollout).
* ``tokens_train`` MEASURED, unpadded: sum over rollouts of prompt tokens (tokenised once per prompt) + completion
  tokens. Padding to the micro-batch maximum is not counted.
* ``truncated`` follows TRL's own rule (last completion token is neither EOS nor pad); ``reward_mean``,
  ``completion_len_mean``, ``truncation_rate``, ``frac_zero_adv_groups`` come from the rollout logger (a group has
  zero advantage iff all its rewards are equal), ``loss``/``grad_norm`` from TRL's ``on_log``.

Adapters and snapshots
----------------------
``snapshot(step)`` saves the LoRA adapter (``PeftModel.save_pretrained``) to ``<run_dir>/adapters/step_<N>/``. The
driver takes snapshot 0 *before* ``train``: the trainer does not exist yet, so the request is remembered and the
untrained adapter is written from ``on_train_begin`` (LoRA ``B`` is zero-initialised, so it equals the base model).
Other snapshots (steps in ``ctx.snapshot_steps``, default {0,20,40,60,80,100}) are saved from the ``on_log`` of that
step. Each save is verified (config + weights file exist).

Failure handling
----------------
CUDA OOM anywhere in training -> ``RunFailedError("oom")`` (status ``failed``, reason ``oom``). A non-finite loss,
grad norm or reward -> ``InvalidRunError`` (status ``invalid``; rewards are checked before TRL sees them). Hangs are
the stall watchdog's (``rhg.train.watchdog``): while the trainer/engine is being built or an eval engine is loading
(no step boundary), a keep-alive thread beats the heartbeat for at most ``STARTUP_GRACE_S``, so a slow model
download is not a false stall but a real hang still dies after that bound.

Post-hoc adapter evaluation, GPU memory and cost
------------------------------------------------
After the last step the driver calls ``end_training()``: the trainer (and with it the model, optimizer and the
colocated vLLM engine) is dropped, vLLM's parallel state is destroyed, then ``gc.collect()``,
``torch.cuda.synchronize()``, ``torch.cuda.empty_cache()``. **How we verify vLLM will see the memory:** we read
``torch.cuda.mem_get_info()`` -- the same driver-level free/total figure vLLM's own start-up check compares with
``gpu_memory_utilization * total`` -- print it, and refuse to continue (``RunFailedError("gpu_memory_not_released")``,
adapters kept on disk) unless free >= ``EVAL_GPU_MEM_UTIL + EVAL_FREE_MARGIN`` of total; the same check runs before every
eval engine. Every saved adapter is then evaluated with a **fresh** ``VLLMGenerator(adapter_path=...)`` (base weights
+ ``LoRARequest``; ``eval_generator``), closed before the next one. This uses no private TRL weight-sync API, at the
price of one extra engine start per snapshot: 6 per run at T=100 (base weights are read again from the disk cache,
engine profile + CUDA-graph capture, adapter load). The size of that cost is UNVERIFIED until measured: the bench
(``rhg.eval.bench``) times one eval pass *including* a fresh engine load (``t_eval_s``, and ``t_eval_load_s`` alone) so
the cost model contains it. Adapter directories (~0.14 GB each in fp32) are deleted by the driver after all evals
succeeded unless ``--keep-adapters``; after a failure they stay.

Ideas credited: the callback/reward/OOM structure is standard TRL usage (TRL docs); no code was copied from other
repositories.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import json
import shutil
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rhg.data.prompts import build_prompt, load_prompts_cfg, render_chat
from rhg.eval.generate import VLLMGenerator
from rhg.train import trl_config
from rhg.train.backend import InvalidRunError, RunFailedError, TrainContext

GIB = 1024**3
REQUIRED_MODULES = ("torch", "transformers", "trl", "peft", "vllm")
GPU_STACK_HINT = (
    "install requirements-gpu.txt (`uv pip install -r requirements-gpu.txt`, done on the GPU box by scripts/setup_box.sh); "
    "the GPU stack is deliberately not part of pyproject.toml / uv.lock"
)
EVAL_GPU_MEM_UTIL = 0.6  # eval engines run alone on the card; 0.6 leaves room for CUDA context/fragmentation leftovers
EVAL_FREE_MARGIN = 0.05
STARTUP_GRACE_S = 1200.0  # keep-alive bound for model download / engine start (no step boundary to beat the watchdog)
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
TIMING_FILE = "trl_timing.jsonl"


class GpuStackMissingError(ImportError):
    """The GPU stack (torch/transformers/trl/peft/vllm) is not installed."""


class PromptTooLongError(ValueError):
    """A train prompt exceeds ``grpo.max_prompt_tokens`` (TRL would not truncate it)."""


class RolloutBatchError(RuntimeError):
    """TRL called the reward function with a batch that does not match the seeded schedule."""


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def require_gpu_stack() -> None:
    missing = [m for m in REQUIRED_MODULES if not _module_available(m)]
    if missing:
        raise GpuStackMissingError(
            f"TrlBackend needs {', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} not installed: {GPU_STACK_HINT}. "
            "Use --mock / MockBackend on a machine without the GPU stack."
        )


# ------------------------------------------------------------------ prompts
def build_prompt_table(
    problems: Mapping[str, Mapping[str, Any]], pids: Sequence[str], hint: str
) -> tuple[dict[str, str], dict[str, str]]:
    """``(prompt by problem id, problem id by prompt)`` for ``pids`` rendered with thinking off. The reverse map
    must be injective: two problems with the same rendered prompt would be indistinguishable to the reward."""
    prompts_cfg = load_prompts_cfg()
    by_pid: dict[str, str] = {}
    by_prompt: dict[str, str] = {}
    for pid in sorted(set(pids)):
        text = render_chat(build_prompt(problems[pid], hint, prompts_cfg), enable_thinking=False)
        if by_prompt.setdefault(text, pid) != pid:
            raise ValueError(f"problems {by_prompt[text]!r} and {pid!r} render to the same prompt")
        by_pid[pid] = text
    return by_pid, by_prompt


def prompt_token_counts(prompts: Sequence[str], tokenizer: Any) -> list[int]:
    """Token count of each prompt exactly as TRL tokenises it (``processing_class(text=prompts)``)."""
    ids = tokenizer(text=list(prompts))["input_ids"]
    return [len(x) for x in ids]


def check_prompt_lengths(prompt_by_pid: Mapping[str, str], tokenizer: Any, max_tokens: int) -> dict[str, int]:
    pids = sorted(prompt_by_pid)
    lengths = dict(zip(pids, prompt_token_counts([prompt_by_pid[p] for p in pids], tokenizer)))
    too_long = {p: n for p, n in lengths.items() if n > max_tokens}
    if too_long:
        worst = ", ".join(f"{p} ({n})" for p, n in sorted(too_long.items(), key=lambda kv: -kv[1])[:5])
        raise PromptTooLongError(
            f"{len(too_long)} train prompt(s) exceed grpo.max_prompt_tokens={max_tokens} tokens and TRL does not truncate "
            f"prompts: {worst}. Fix the problem set (the same problems must be used by every arm), not this check."
        )
    return lengths


# ------------------------------------------------------------------ step instrumentation
class StepMonitor:
    """Turns TRL callback events, reward-function calls and vLLM timers into ``ctx.end_step`` calls (see the module
    docstring for what is measured). Pure Python: no torch/transformers, unit-testable."""

    def __init__(
        self,
        ctx: TrainContext,
        *,
        gpu_sampler: Callable[[], Mapping[str, float]] | None = None,
        timing_path: Path | None = None,
        on_train_begin: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.ctx, self.gpu_sampler, self.timing_path, self._on_begin, self.clock = ctx, gpu_sampler, timing_path, on_train_begin, clock
        self.measured = {"gen": False, "sync": False}
        self.done_steps = 0
        self.step = 0
        self.gpu_peak: dict[str, float] = {}
        self.first_step_begin: float | None = None
        self._pending: dict[str, Any] | None = None
        self._reset()

    def _reset(self) -> None:
        self._t_begin = self._t_reward0 = self._t_reward1 = self._t_end = None
        self._gen_s = self._sync_s = 0.0
        self._prompt_tokens = self._completion_tokens = 0
        self._rewards: list[float] | None = None

    # -- events
    def train_begin(self) -> None:
        if self._on_begin is not None:
            self._on_begin()
        self.ctx.heartbeat("train begin")

    def step_begin(self, global_step: int) -> None:
        if self._pending is not None:
            raise RuntimeError(
                f"step {self.step} ended but TRL never logged its loss (on_log without 'loss'): logging_steps must be 1"
            )
        if global_step != self.done_steps:
            raise RuntimeError(f"on_step_begin with global_step={global_step} but {self.done_steps} steps are complete")
        self.step = global_step + 1
        self._reset()
        self.ctx.begin_step(self.step)
        self._t_begin = self.clock()
        if self.first_step_begin is None:
            self.first_step_begin = self._t_begin

    def note_reward(self, t0: float, t1: float, rewards: Sequence[float], prompt_tokens: int, completion_tokens: int) -> None:
        if self._t_reward0 is not None:
            raise RolloutBatchError(f"the reward function was called twice in step {self.step}")
        self._t_reward0, self._t_reward1 = t0, t1
        self._rewards = list(rewards)
        self._prompt_tokens, self._completion_tokens = prompt_tokens, completion_tokens

    def add_gen(self, seconds: float) -> None:
        self._gen_s += seconds

    def add_sync(self, seconds: float) -> None:
        self._sync_s += seconds

    def step_end(self, global_step: int) -> None:
        if global_step != self.step:
            raise RuntimeError(f"on_step_end with global_step={global_step} during step {self.step}")
        if self._t_reward0 is None:
            raise RolloutBatchError(f"step {self.step} finished without a reward-function call")
        self._t_end = self.clock()
        if self.gpu_sampler is not None:
            for k, v in self.gpu_sampler().items():
                self.gpu_peak[k] = max(self.gpu_peak.get(k, 0.0), float(v))
        self._pending = {"step": self.step}

    def log(self, global_step: int, logs: Mapping[str, Any]) -> None:
        if self._pending is None or "loss" not in logs:
            return  # summary logs (train_runtime, ...) and logs of steps we do not track
        if global_step != self.step:
            raise RuntimeError(f"on_log with global_step={global_step} for step {self.step}")
        if logs.get("grad_norm") is None:
            raise RuntimeError(f"TRL logged no grad_norm at step {self.step}; the step log needs it")
        t_sync = self._sync_s
        pre_reward = self._t_reward0 - self._t_begin
        t_gen = self._gen_s if self.measured["gen"] else max(0.0, pre_reward - t_sync)
        t_train = max(0.0, self._t_end - self._t_reward1)
        rec = self.ctx.end_step(
            self.step,
            loss=float(logs["loss"]),
            grad_norm=float(logs["grad_norm"]),
            t_gen=t_gen,
            t_train=t_train,
            t_sync=t_sync,
            tokens_train=self._prompt_tokens + self._completion_tokens,
            rewards=self._rewards,
        )
        self._write_timing(rec, pre_reward, t_gen, t_sync, t_train)
        self._pending = None
        self.done_steps = self.step

    def train_end(self) -> None:
        if self._pending is not None:
            raise RuntimeError(f"training ended but step {self.step} was never finalised (no loss logged)")

    def _write_timing(self, rec, pre_reward: float, t_gen: float, t_sync: float, t_train: float) -> None:
        if self.timing_path is None:
            return
        row = {
            "step": rec.step, "t_step": rec.t_step, "pre_reward_s": pre_reward, "t_gen": t_gen, "t_sync": t_sync,
            "t_reward": rec.t_reward, "t_train": t_train,
            "t_other": rec.t_step - (t_gen + rec.t_reward + t_train + t_sync),
            "gen_measured": self.measured["gen"], "sync_measured": self.measured["sync"],
            "tokens_prompt": self._prompt_tokens, "tokens_completion": self._completion_tokens,
        }
        with open(self.timing_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")


def build_callback(monitor: StepMonitor) -> Any:
    """A ``transformers.TrainerCallback`` forwarding the events to ``monitor`` (class built lazily: needs transformers)."""
    from transformers import TrainerCallback

    class RhgStepCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            monitor.train_begin()

        def on_step_begin(self, args, state, control, **kwargs):
            monitor.step_begin(state.global_step)

        def on_step_end(self, args, state, control, **kwargs):
            monitor.step_end(state.global_step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            monitor.log(state.global_step, logs or {})

        def on_train_end(self, args, state, control, **kwargs):
            monitor.train_end()

    return RhgStepCallback()


def install_timers(trainer: Any, monitor: StepMonitor) -> None:
    """Wrap ``trainer.vllm_generation.generate`` / ``.sync_weights`` with wall-clock timers (behaviour unchanged).
    Silently skipped when the attribute does not exist; ``monitor.measured`` records what was wrapped."""
    target = getattr(trainer, "vllm_generation", None)
    for method, key, add in (("generate", "gen", monitor.add_gen), ("sync_weights", "sync", monitor.add_sync)):
        fn = getattr(target, method, None) if target is not None else None
        if not callable(fn):
            continue

        def make(fn=fn, add=add):
            def timed(*args, **kwargs):
                t0 = monitor.clock()
                try:
                    return fn(*args, **kwargs)
                finally:
                    add(monitor.clock() - t0)

            return timed

        try:
            setattr(target, method, make())
            monitor.measured[key] = True
        except Exception:  # noqa: BLE001 - timing is optional
            pass


def make_trl_reward(
    ctx: TrainContext,
    monitor: StepMonitor,
    prompt_to_pid: Mapping[str, str],
    prompt_tokens: Mapping[str, int],
    eos_ids: Callable[[], Sequence[int] | None],
) -> Callable[..., list[float]]:
    """The reward function handed to TRL: checks the batch against the schedule, then calls ``ctx.reward_fn``."""
    cfg, inner = ctx.cfg, ctx.reward_fn
    ppS, gens, max_new = cfg.grpo.prompts_per_step, cfg.grpo.gens_per_prompt, cfg.grpo.max_completion_tokens

    def rhg_reward(prompts=None, completions=(), completion_ids=None, problem_id=None, trainer_state=None, **_):
        t0 = monitor.clock()
        step = ctx.counter.value
        if not 1 <= step <= len(ctx.schedule):
            raise RolloutBatchError(f"reward function called outside a training step (step counter={step})")
        gs = getattr(trainer_state, "global_step", None)
        if gs is not None and gs + 1 != step:
            raise RolloutBatchError(f"step counter {step} disagrees with trainer_state.global_step={gs}: callback order changed")
        n = len(completions)
        if prompts is None or len(prompts) != n or completion_ids is None or len(completion_ids) != n:
            raise RolloutBatchError("reward function needs prompts, completions and completion_ids of equal length")
        if n != ppS * gens:
            raise RolloutBatchError(f"step {step}: {n} completions, expected prompts_per_step*gens_per_prompt={ppS * gens}")
        if not all(isinstance(c, str) for c in completions):
            raise RolloutBatchError("completions must be strings (prompts are pre-rendered strings, not chat messages)")
        try:
            pids = [prompt_to_pid[p] for p in prompts]
        except (KeyError, TypeError) as e:
            raise RolloutBatchError(f"step {step}: a prompt is not one of the scheduled train prompts ({e!r})") from e
        if problem_id is not None and list(problem_id) != pids:
            raise RolloutBatchError(f"step {step}: dataset problem_id column disagrees with the prompt -> problem map")
        counts = Counter(pids)
        if set(counts) != set(ctx.schedule[step - 1]) or set(counts.values()) != {gens}:
            raise RolloutBatchError(
                f"step {step}: got {len(counts)} distinct prompts with counts {sorted(set(counts.values()))}; expected exactly the "
                f"{ppS} scheduled problems x {gens} generations (TRL sampler/accumulation settings changed?)"
            )
        eos = set(eos_ids() or ())
        lens = [len(ids) for ids in completion_ids]
        if eos:
            truncated = [bool(len(ids) == 0 or ids[-1] not in eos) for ids in completion_ids]
        else:
            truncated = [ln >= max_new for ln in lens]
        rewards = inner(prompts=list(prompts), completions=list(completions), problem_id=pids, n_tokens=lens, truncated=truncated)
        if not all(isinstance(r, float) and r - r == 0 for r in rewards):  # NaN/inf/None never reach the loss
            raise InvalidRunError(f"non-finite reward at step {step}")
        monitor.note_reward(
            t0, monitor.clock(), rewards, sum(prompt_tokens[p] for p in pids), sum(lens)
        )
        return rewards

    return rhg_reward


# ------------------------------------------------------------------ helpers
@contextlib.contextmanager
def keepalive(beat: Callable[[str], None], label: str, *, grace_s: float = STARTUP_GRACE_S, interval_s: float = 30.0):
    """Beat the watchdog every ``interval_s`` for at most ``grace_s`` while a blocking call (model/engine start) runs."""
    stop = threading.Event()
    deadline = time.monotonic() + grace_s

    def loop() -> None:
        while not stop.wait(interval_s):
            if time.monotonic() > deadline:
                return
            beat(label)

    thread = threading.Thread(target=loop, name="rhg-keepalive", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def is_oom(exc: BaseException | None) -> bool:
    """CUDA OOM (``torch.cuda.OutOfMemoryError`` or a RuntimeError with the CUDA message) anywhere in the chain."""
    seen = 0
    while exc is not None and seen < 6:
        if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            return True
        exc, seen = exc.__cause__ or exc.__context__, seen + 1
    return False


@dataclass(frozen=True)
class AdapterSnapshot:
    step: int
    path: str


def _torch() -> Any:
    import torch

    return torch


# ------------------------------------------------------------------ the backend
class TrlBackend:
    """``rhg.train.backend.Backend`` on TRL + vLLM colocate + PEFT LoRA (see the module docstring)."""

    name = "trl"

    def __init__(
        self,
        cfg,
        *,
        problems: Mapping[str, Mapping[str, Any]],
        prompts: Any = None,
        run_dir: str | Path,
        micro_batch_cap: int = trl_config.DEFAULT_MICRO_BATCH_CAP,
        extra_grpo: dict[str, Any] | None = None,
    ) -> None:
        require_gpu_stack()
        if cfg.model.enable_thinking:
            raise ValueError("model.enable_thinking=true is not supported (DESIGN §2.1: thinking off, prompts pre-rendered)")
        self.cfg, self.problems = cfg, problems
        self.run_dir = Path(run_dir)
        self.adapters_dir = self.run_dir / "adapters"
        # every key is checked against the recorded pinned signature, then against the installed classes, then the
        # config objects are built: a wrong kwarg fails here, before any model is loaded
        self.grpo_kwargs = trl_config.build_grpo_config(
            cfg, output_dir=str(self.run_dir / "trl"), micro_batch_cap=micro_batch_cap, extra=extra_grpo
        )
        self.lora_kwargs = trl_config.build_lora_config(cfg)
        trl_config.check_against_installed(self.grpo_kwargs, self.lora_kwargs)
        import peft
        import trl

        self.grpo_args = trl.GRPOConfig(**self.grpo_kwargs)
        self.lora_config = peft.LoraConfig(**self.lora_kwargs)
        self._trainer: Any = None
        self.monitor: StepMonitor | None = None
        self._pending_snapshots: set[int] = set()
        self._heartbeat: Callable[[str], None] = lambda label="": None
        self.setup_s: float | None = None
        self.teardown_s: float | None = None
        self.gpu_free_after_teardown_gib: float | None = None
        self.gpu_total_gib: float | None = None
        self._trained = False

    # -- training
    def train(self, ctx: TrainContext) -> None:
        cfg = self.cfg
        self._heartbeat = ctx.heartbeat
        t_enter = time.perf_counter()
        pids = sorted({p for step in ctx.schedule for p in step})
        prompt_by_pid, prompt_to_pid = build_prompt_table(self.problems, pids, cfg.arm.hint)
        for pid in pids:
            if ctx.prompts.get(pid) != prompt_by_pid[pid]:
                raise RuntimeError(f"driver prompt for {pid!r} differs from the backend's thinking-off rendering")

        from datasets import Dataset
        from transformers import AutoTokenizer

        import trl

        with keepalive(ctx.heartbeat, "loading tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left", truncation_side="left")
        prompt_tokens = check_prompt_lengths(prompt_by_pid, tokenizer, cfg.grpo.max_prompt_tokens)
        dataset = Dataset.from_list([{"prompt": prompt_by_pid[p], "problem_id": p} for step in ctx.schedule for p in step])

        eos_holder: list[Sequence[int]] = []
        monitor = self.monitor = StepMonitor(
            ctx,
            gpu_sampler=self._gpu_sample,
            timing_path=self.run_dir / TIMING_FILE,
            on_train_begin=self._materialise_pending,
        )
        (self.run_dir / TIMING_FILE).unlink(missing_ok=True)
        reward = make_trl_reward(ctx, monitor, prompt_to_pid, prompt_tokens, lambda: eos_holder[0] if eos_holder else None)
        reward.__name__ = "rhg_reward"
        try:
            with keepalive(ctx.heartbeat, "building trainer (model load, vLLM engine)"):
                trainer = trl.GRPOTrainer(
                    model=cfg.model.name,
                    reward_funcs=[reward],
                    args=self.grpo_args,
                    train_dataset=dataset,
                    processing_class=tokenizer,
                    callbacks=[build_callback(monitor)],
                    peft_config=self.lora_config,
                )
            self._trainer = trainer
            proc = getattr(trainer, "processing_class", tokenizer)
            eos_holder.append([i for i in (getattr(proc, "eos_token_id", None), getattr(proc, "pad_token_id", None)) if i is not None])
            install_timers(trainer, monitor)
            self.setup_s = time.perf_counter() - t_enter
            print(
                f"[rhg.trl] trainer ready in {self.setup_s:.1f}s; t_gen measured={monitor.measured['gen']}, "
                f"t_sync measured={monitor.measured['sync']}",
                flush=True,
            )
            ctx.heartbeat("trainer ready")
            trainer.train()
        except (InvalidRunError, RunFailedError):
            raise
        except Exception as e:
            if is_oom(e):
                raise RunFailedError("oom", f"{type(e).__name__}: {str(e)[:300]}") from e
            raise
        if monitor.done_steps != cfg.grpo.max_steps:
            raise RuntimeError(f"TRL returned after {monitor.done_steps} of {cfg.grpo.max_steps} steps")
        self._trained = True

    # -- snapshots
    def snapshot(self, step: int) -> AdapterSnapshot:
        snap = AdapterSnapshot(step, str(self.adapters_dir / f"step_{step}"))
        if self._trainer is None:
            if step != 0 or self._trained:
                raise RuntimeError(f"cannot snapshot step {step}: no live trainer")
            self._pending_snapshots.add(step)  # untrained adapter: written from on_train_begin
            return snap
        self._save_adapter(snap)
        return snap

    def _materialise_pending(self) -> None:
        for step in sorted(self._pending_snapshots):
            self._save_adapter(AdapterSnapshot(step, str(self.adapters_dir / f"step_{step}")))
        self._pending_snapshots.clear()

    def _save_adapter(self, snap: AdapterSnapshot) -> None:
        trainer = self._trainer
        model = trainer.accelerator.unwrap_model(trainer.model)
        path = Path(snap.path)
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path))
        if not (path / ADAPTER_CONFIG).is_file() or not any((path / w).is_file() for w in ADAPTER_WEIGHTS):
            raise RuntimeError(f"saving the LoRA adapter for step {snap.step} produced no {ADAPTER_CONFIG}/weights in {path}")

    def delete_snapshots(self) -> None:
        d = self.adapters_dir
        if d.is_dir() and d.resolve().parent == self.run_dir.resolve():
            shutil.rmtree(d, ignore_errors=True)

    # -- GPU memory
    def _gpu_sample(self) -> dict[str, float]:
        torch = _torch()
        free, total = torch.cuda.mem_get_info()
        return {"used_gib": (total - free) / GIB, "reserved_gib": torch.cuda.max_memory_reserved() / GIB}

    def _require_free_memory(self, what: str) -> None:
        torch = _torch()
        free, total = torch.cuda.mem_get_info()
        self.gpu_free_after_teardown_gib, self.gpu_total_gib = free / GIB, total / GIB
        need = EVAL_GPU_MEM_UTIL + EVAL_FREE_MARGIN
        print(f"[rhg.trl] GPU memory before {what}: free {free / GIB:.2f} of {total / GIB:.2f} GiB (need >= {need:.0%})", flush=True)
        if free < need * total:
            raise RunFailedError(
                "gpu_memory_not_released",
                f"only {free / GIB:.2f} of {total / GIB:.2f} GiB free before {what} (need {need:.0%}); the adapters in "
                f"{self.adapters_dir} were kept, rerun the evals on a fresh process",
            )

    def _release(self) -> None:
        self._trainer = None
        gc.collect()
        with contextlib.suppress(Exception):  # vLLM's process-group state of the colocated engine
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()
        gc.collect()
        try:
            torch = _torch()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real result
            pass

    def end_training(self) -> None:
        t0 = time.perf_counter()
        self._release()
        self.teardown_s = time.perf_counter() - t0
        self._require_free_memory("post-hoc evals")

    # -- evals
    def eval_generator(self, snapshot: Any):
        snap = snapshot if isinstance(snapshot, AdapterSnapshot) else AdapterSnapshot(-1, str(snapshot))
        if not (Path(snap.path) / ADAPTER_CONFIG).is_file():
            raise FileNotFoundError(f"no saved adapter at {snap.path} (step {snap.step})")
        self._require_free_memory(f"eval engine for step {snap.step}")
        try:
            with keepalive(self._heartbeat, f"loading eval engine {snap.step}"):
                return VLLMGenerator.from_config(self.cfg, adapter_path=snap.path, gpu_memory_utilization=EVAL_GPU_MEM_UTIL)
        except Exception as e:
            if is_oom(e):
                raise RunFailedError("oom", f"eval engine for step {snap.step}: {str(e)[:300]}") from e
            raise

    def close(self) -> None:
        if self._trainer is not None:
            self._release()

    # -- reporting (used by rhg.eval.bench)
    def report(self) -> dict[str, Any]:
        m = self.monitor
        return {
            "setup_s": self.setup_s,
            "teardown_s": self.teardown_s,
            "timers_measured": dict(m.measured) if m else {},
            "gpu_peak_used_gib": (m.gpu_peak.get("used_gib") if m else None),
            "gpu_peak_reserved_gib": (m.gpu_peak.get("reserved_gib") if m else None),
            "gpu_total_gib": self.gpu_total_gib,
        }


def create_backend(cfg, *, problems: Mapping[str, Mapping[str, Any]], prompts: Any = None, run_dir: str | Path, **kwargs: Any) -> TrlBackend:
    return TrlBackend(cfg, problems=problems, prompts=prompts, run_dir=run_dir, **kwargs)
