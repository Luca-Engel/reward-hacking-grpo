"""Throughput benchmark: ``python -m rhg.eval.bench [--mock] [--steps 8] [--n-problems 64]`` (subtask 11; BUDGET §2, Gate 1b).

Runs N real GRPO steps of the configured backend on the ``hackable_subtle`` arm (production shape: 16 prompts x 8
generations, ``grpo.max_completion_tokens``) over >= 64 real problems of ``data/processed/candidates.jsonl`` (no band
selection: this measures speed, not learning; seed 9000 = pilot range), then times one evaluation pass and writes
``results/bench/throughput.json`` in exactly the schema documented in ``rhg.budget`` (every value MEASURED, never typed
in), which ``python -m rhg.budget cost_model`` prices.

How each field is obtained (steps 1-2 are warm-up, aggregates use steps >= 3, like BUDGET §2's "median over steps 3..N"):

``gen_tokens_per_step``, ``train_tokens_per_step``  mean of the logged ``tokens_gen`` / ``tokens_train`` (unpadded, see
    ``rhg.train.trl_trainer``).
``gen_tok_per_s``, ``train_tok_per_s``  sum of tokens over sum of ``t_gen`` / ``t_train`` (ratio of sums, not mean of ratios).
``sync_s_per_step``  mean ``t_sync`` (0 when ``sync_weights`` could not be timed separately; then it is inside ``t_gen``).
``exec_s_per_rollout``  **grading cache ON (production setting)**: sum of the steps' ``t_reward`` times ``cpu_workers``
    over the number of rollouts, so that BUDGET §2's ``t_reward = rollouts * exec_s / cpu_workers`` reproduces the
    measured reward time. It includes AST/log overhead in the trainer process and cache lookups. ``cache_hit_rate`` is the
    grading-cache hit rate over the same steps ((hits + in-batch duplicates) / lookups).
``exec_s_per_rollout_uncached``  from a *separate* sample of 64 rollouts (a seeded draw from the last step's real
    completions), graded with ``sandbox.cache=false`` in one ``grade_batch`` call; same normalisation.
    **Caveat printed with the results:** the cost model uses the cached figure because that is what production sees, but
    an N-step bench sees a cold cache (memory only, fresh) and under-estimates the steady-state hit rate of a 100-step run
    (the on-disk cache also persists across runs), so the cached figure is conservative; if the hit rate is high the
    figure says little about the *uncached* CPU cost, which is why both are reported.
``cpu_workers``  ``resolve_workers(sandbox.workers)`` (0 = cores - 2); ``cpu_cores`` = ``os.cpu_count()`` is reported too.
``t_eval_s``  one evaluation pass = fresh eval engine load (``eval_generator`` of the final snapshot: base + LoRA) +
    generation + grading of ``ceil(mean_eval_rollouts / val_samples_per_problem)`` problems x ``val_samples_per_problem``,
    where ``mean_eval_rollouts = (5 x val + test) / 6`` from the planned 40/60 val/test problems (BUDGET §2's
    ``n_eval = 5 val + 1 test``), so the single number stands for the average of those six passes. The load alone is
    ``t_eval_load_s``.
``t_startup_s``  once-per-run overhead: backend construction + trainer/engine setup (model load, vLLM start) + the first
    step's excess over the median of steps >= 3 (compile/warm-up) + trainer teardown before the evals.
``step_times_s``  ``t_step`` of every step; ``n_steps_measured``; ``rollouts_per_step``.
Extra informational keys (ignored by the cost model): ``peak_gpu_mem_gib`` (device-level used memory, max over steps),
``peak_torch_reserved_gib``, ``gpu_name``, ``mock``, ``arm``, ``seed``, ``n_problems``, ``timers_measured``, ``notes``.

``--mock`` uses the CPU mock policy: real sandbox grading (so CPU numbers are real for this machine) but instantaneous
"generation/training", so its output is **not** a measurement of the GPU stack. It is written to
``results/bench_mock/throughput.json`` (never the real path), carries ``"mock": true`` and ``rhg.budget cost_model``
refuses to write the default ``BUDGET_MEASURED.md``/``prereg/budget_decision.md`` from it.

Ledger: a real bench appends one ``bench`` entry (model load + steps + eval wall time) unless ``--no-ledger``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rhg import budget as budget_mod
from rhg import manifest as manifest_mod
from rhg import runlog
from rhg.config import Config, ConfigError, load_config
from rhg.env import cache as grade_cache
from rhg.env.grader import GradeItem, grade_batch, resolve_workers
from rhg.env.monitor import make_monitor
from rhg.eval.generate import SamplingParams
from rhg.seeds import derive_seed
from rhg.train import run as run_mod
from rhg.train.backend import InvalidRunError, RunFailedError, TrainContext
from rhg.train.rollout_io import RolloutLogger, StepCounter, make_reward_fn
from rhg.train.watchdog import Watchdog

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_GUARD = 0, 1, 2, 3
DEFAULT_ARM = "hackable_subtle"
BENCH_SEED = 9000  # pilot range (DESIGN §3)
MIN_PROBLEMS = 64
UNCACHED_SAMPLE = 64
WARMUP_STEPS = 2
PLANNED_VAL_PROBLEMS, PLANNED_TEST_PROBLEMS = 40, 60  # docs/SPEC_DEVIATIONS.md 04 (val/test sizes for >= 250 selected problems)
REAL_OUT = Path("results/bench/throughput.json")
MOCK_OUT = Path("results/bench_mock/throughput.json")


class BenchError(RuntimeError):
    """The bench cannot produce a trustworthy throughput.json."""


# ------------------------------------------------------------------ inputs
def load_candidates(processed_dir: Path) -> list[dict[str, Any]]:
    path = processed_dir / "candidates.jsonl"
    if not path.is_file():
        raise BenchError(f"{path} not found; build the dataset first (python -m rhg.data.build --stage fetch/tests/validate)")
    rows = run_mod.read_jsonl(path)
    return [r for r in rows if r.get("reward_tests") and r.get("heldout_tests")]


def filter_prompt_length(rows: list[dict[str, Any]], cfg: Config) -> tuple[list[dict[str, Any]], int]:
    """Drop problems whose (arm-hint) prompt exceeds ``grpo.max_prompt_tokens`` (needs the tokenizer: real path only)."""
    from transformers import AutoTokenizer

    from rhg.train.trl_trainer import prompt_token_counts

    book = run_mod.PromptBook({r["problem_id"]: r for r in rows}, cfg.model.enable_thinking)
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left", truncation_side="left")
    lengths = prompt_token_counts([book(r["problem_id"], cfg.arm.hint) for r in rows], tok)
    kept = [r for r, n in zip(rows, lengths) if n <= cfg.grpo.max_prompt_tokens]
    return kept, len(rows) - len(kept)


def select_problems(rows: list[dict[str, Any]], n: int, seed: int) -> dict[str, dict[str, Any]]:
    ids = sorted(str(r["problem_id"]) for r in rows)
    if len(set(ids)) != len(ids):
        raise BenchError("duplicate problem_id in the candidates")
    chosen = set(random.Random(derive_seed(seed, "bench")).sample(ids, n))
    return {str(r["problem_id"]): {**r, "split": "train"} for r in sorted(rows, key=lambda r: str(r["problem_id"])) if str(r["problem_id"]) in chosen}


def eval_pass_problems(cfg: Config) -> int:
    """Problems in the timed eval pass: the average of BUDGET §2's five val evals and one test eval, in problems."""
    val = PLANNED_VAL_PROBLEMS * cfg.eval.val_samples_per_problem
    test = PLANNED_TEST_PROBLEMS * cfg.eval.test_samples_per_problem
    return math.ceil((5 * val + test) / 6 / cfg.eval.val_samples_per_problem)


# ------------------------------------------------------------------ aggregation (pure)
def aggregate(
    rows: list[runlog.StepRecord],
    cache_deltas: Mapping[int, Mapping[str, int]],
    *,
    cpu_workers: int,
    rollouts_per_step: int,
) -> dict[str, Any]:
    """Cost-model fields from the step log (steps >= 3) and the per-step grading-cache deltas."""
    if len(rows) < WARMUP_STEPS + 1:
        raise BenchError(f"need at least {WARMUP_STEPS + 1} steps, got {len(rows)}")
    warm = rows[WARMUP_STEPS:]
    t_gen, t_train = sum(r.t_gen for r in warm), sum(r.t_train for r in warm)
    if not (t_gen > 0 and t_train > 0):
        raise BenchError("measured zero generation or training time; the step timers did not work")

    def count(steps: list[runlog.StepRecord], keys: tuple[str, ...]) -> int:
        return sum(cache_deltas.get(r.step, {}).get(k, 0) for r in steps for k in keys)

    def hit_rate(steps: list[runlog.StepRecord]) -> float:
        total = count(steps, ("hits", "dedup", "misses"))
        return count(steps, ("hits", "dedup")) / total if total else 0.0

    return {
        "gen_tokens_per_step": statistics.fmean(r.tokens_gen for r in warm),
        "gen_tok_per_s": sum(r.tokens_gen for r in warm) / t_gen,
        "train_tokens_per_step": statistics.fmean(r.tokens_train for r in warm),
        "train_tok_per_s": sum(r.tokens_train for r in warm) / t_train,
        "exec_s_per_rollout": sum(r.t_reward for r in warm) * cpu_workers / (len(warm) * rollouts_per_step),
        "sync_s_per_step": statistics.fmean(r.t_sync for r in warm),
        "cache_hit_rate": hit_rate(warm),
        "cache_hit_rate_all_steps": hit_rate(rows),
        "step_times_s": [r.t_step for r in rows],
        "t_step_median_s": statistics.median(r.t_step for r in warm),
        "t_gen_s": statistics.median(r.t_gen for r in warm),
        "t_reward_s": statistics.median(r.t_reward for r in warm),
        "t_train_s": statistics.median(r.t_train for r in warm),
    }


def time_uncached(cfg_nocache: Config, problems, pids: list[str], completions: list[str], monitor_fn, n: int, seed: int, workers: int):
    """Uncached seconds per rollout on a seeded ``n``-rollout draw of the given completions (cache off, one batch)."""
    idx = sorted(random.Random(derive_seed(seed, "bench-uncached")).sample(range(len(pids)), min(n, len(pids))))
    items = [GradeItem(problems[pids[i]], completions[i], cfg_nocache.arm.reward, monitor_fn) for i in idx]
    t0 = time.perf_counter()
    grade_batch(items, cfg=cfg_nocache)
    dt = time.perf_counter() - t0
    return dt * workers / len(items), len(items), dt


# ------------------------------------------------------------------ the bench
def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def run_bench(args: argparse.Namespace) -> dict[str, Any]:
    backend_name = args.backend or ("mock" if args.mock else "trl")
    mock = backend_name == "mock"
    overrides = list(args.overrides) + [f"grpo.max_steps={args.steps}", f"run.mode={'mock' if mock else 'train'}"]
    cfg = load_config(args.arm, overrides, seed=args.seed)
    cfg_nocache = load_config(args.arm, overrides + ["sandbox.cache=false"], seed=args.seed)
    ppS = cfg.grpo.prompts_per_step
    out_dir = args.out.parent
    run_dir = Path(args.run_dir) if args.run_dir else out_dir / "run"
    t_wall0 = time.perf_counter()

    processed = Path(args.processed_dir or cfg.data.processed_dir)
    if not processed.is_absolute() and not processed.is_dir() and (manifest_mod.REPO_ROOT / processed).is_dir():
        processed = manifest_mod.REPO_ROOT / processed
    if mock and not (processed / "candidates.jsonl").is_file():
        processed = run_mod.FIXTURE_PROCESSED  # the mock never needs (or touches) real data
    rows = load_candidates(processed)
    dropped_long = 0
    if not mock:
        rows, dropped_long = filter_prompt_length(rows, cfg)
    floor = ppS if (mock or args.allow_few_problems) else MIN_PROBLEMS
    n_problems = min(args.n_problems, len(rows)) if mock else args.n_problems
    if n_problems < floor or len(rows) < n_problems:
        raise BenchError(
            f"need >= {floor} usable problems, have {len(rows)} (requested {args.n_problems}); the bench must run on >= "
            f"{MIN_PROBLEMS} real problems (use --allow-few-problems only for tests)"
        )
    problems = select_problems(rows, n_problems, args.seed)
    schedule = run_mod.make_schedule(sorted(problems), args.steps, ppS, args.seed)

    grade_cache.configure_cache(None)  # fresh in-memory cache: cold start, as documented
    monitor_fn = make_monitor(cfg.arm.monitor)
    prompts = run_mod.PromptBook(problems, cfg.model.enable_thinking)
    prompt_by_pid = {pid: prompts(pid, cfg.arm.hint) for pid in problems}
    params = SamplingParams.from_config(cfg)
    workers = resolve_workers(cfg.sandbox.workers, cfg)

    for stale in (runlog.ROLLOUTS_FILE, runlog.STEPS_FILE, runlog.EVALS_FILE, "trl_timing.jsonl"):
        (run_dir / stale).unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    counter = StepCounter(0)
    logger = RolloutLogger(cfg.run_id, None, counter)
    base_reward = make_reward_fn(cfg, problems, monitor_fn, logger)
    cache_deltas: dict[int, dict[str, int]] = {}
    last: dict[str, list] = {"pids": [], "completions": []}

    def reward_fn(**kw):
        before = grade_cache.cache_stats()
        out = base_reward(**kw)
        after = grade_cache.cache_stats()
        cache_deltas[counter.value] = {k: int(after[k] - before[k]) for k in ("hits", "misses", "dedup")}
        last["pids"], last["completions"] = list(kw["problem_id"]), list(kw["completions"])
        return out

    step_writer = runlog.StepWriter(run_dir)
    rollout_writer = runlog.RolloutWriter(run_dir)
    wd = Watchdog(float(cfg.run.step_timeout_s), on_stall=lambda reason: _on_stall(args, cfg, t_wall0, reason))
    backend = None
    try:
        wd.start()
        print(f"[rhg.bench] backend={backend_name} arm={cfg.arm.id} steps={args.steps} problems={len(problems)} "
              f"(dropped {dropped_long} over-long prompts) rollouts/step={cfg.rollouts_per_step} workers={workers}", flush=True)
        t0 = time.perf_counter()
        backend = run_mod.load_backend_factory(backend_name)(cfg, problems=problems, prompts=prompts, run_dir=run_dir)
        t_backend_init = time.perf_counter() - t0
        ctx = TrainContext(
            cfg=cfg, problems=problems, schedule=schedule, prompts=prompt_by_pid, sampling=params, reward_fn=reward_fn,
            logger=logger, counter=counter, step_writer=step_writer, snapshot_steps=frozenset({args.steps}),
            snapshot_fn=backend.snapshot, heartbeat=lambda label="": wd.beat(label),
        )
        t0 = time.perf_counter()
        backend.train(ctx)
        t_train_total = time.perf_counter() - t0
        step_writer.close()
        if ctx.last_step != args.steps:
            raise BenchError(f"backend stopped after {ctx.last_step} of {args.steps} steps")
        t0 = time.perf_counter()
        backend.end_training()
        t_teardown = time.perf_counter() - t0

        # timed eval pass (fresh engine of the final adapter, generation, grading)
        n_eval_problems = min(eval_pass_problems(cfg), len(problems))
        acc = runlog.EvalAccumulator(cfg.run_id, cfg.arm.id, cfg.run.seed)
        wd.beat("eval load")
        t0 = time.perf_counter()
        gen = backend.eval_generator(ctx.snapshots[args.steps])
        t_eval_load = time.perf_counter() - t0
        try:
            run_mod.run_eval_point(
                cfg, gen, problems=problems, prompts=prompts, pids=sorted(problems)[:n_eval_problems], phase="eval_val",
                step=args.steps, prompt_hint=cfg.arm.hint, eval_hint=None, n=cfg.eval.val_samples_per_problem,
                eval_seed=derive_seed(cfg.run.seed, "eval"), params=params, monitor_fn=monitor_fn, acc=acc, writer=rollout_writer,
            )
        finally:
            gen.close()
        t_eval = time.perf_counter() - t0
        rollout_writer.close()
        wd.beat("eval done")

        step_rows = runlog.read_steps(run_dir)
        agg = aggregate(step_rows, cache_deltas, cpu_workers=workers, rollouts_per_step=cfg.rollouts_per_step)
        report = backend.report() if hasattr(backend, "report") else {}
        setup_s = report.get("setup_s") or 0.0
        warmup_excess = max(0.0, agg["step_times_s"][0] - agg["t_step_median_s"])
        exec_unc, n_unc, t_unc = time_uncached(
            cfg_nocache, problems, last["pids"], last["completions"], monitor_fn, UNCACHED_SAMPLE, args.seed, workers
        )
        hw = {} if mock else manifest_mod.hardware_info()
    finally:
        wd.stop()
        for closer in (step_writer.close, rollout_writer.close, (backend.close if backend is not None else lambda: None)):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass

    notes = [
        "steps 1-2 are warm-up; aggregates use steps >= 3 (BUDGET §2)",
        f"exec_s_per_rollout is the CACHED (production) figure, hit rate {agg['cache_hit_rate']:.1%} over steps >= 3 "
        f"({agg['cache_hit_rate_all_steps']:.1%} over all steps); a {args.steps}-step cold-cache bench under-estimates the "
        "steady-state hit rate, so the projection is conservative; the uncached figure is from a separate sample",
        f"t_eval_s is one pass of {n_eval_problems * cfg.eval.val_samples_per_problem} rollouts incl. a fresh eval-engine "
        f"load ({t_eval_load:.1f}s); t_startup_s includes backend build, trainer/engine setup, warm-up excess and teardown",
    ]
    if mock:
        notes.append("MOCK: CPU mock policy; only the sandbox timings are real; NOT a measurement of the GPU stack")
    tp: dict[str, Any] = {
        "gen_tokens_per_step": agg["gen_tokens_per_step"],
        "gen_tok_per_s": agg["gen_tok_per_s"],
        "train_tokens_per_step": agg["train_tokens_per_step"],
        "train_tok_per_s": agg["train_tok_per_s"],
        "exec_s_per_rollout": agg["exec_s_per_rollout"],
        "cpu_workers": workers,
        "rollouts_per_step": cfg.rollouts_per_step,
        "sync_s_per_step": agg["sync_s_per_step"],
        "t_eval_s": t_eval,
        "t_startup_s": t_backend_init + setup_s + warmup_excess + t_teardown,
        "n_steps_measured": len(step_rows),
        "step_times_s": agg["step_times_s"],
        "cache_hit_rate": agg["cache_hit_rate"],
        "exec_s_per_rollout_uncached": exec_unc,
        # informational (ignored by the cost model)
        "mock": mock, "arm": cfg.arm.id, "seed": cfg.run.seed, "backend": backend_name, "n_problems": len(problems),
        "cpu_cores": os.cpu_count(), "cache_hit_rate_all_steps": agg["cache_hit_rate_all_steps"],
        "n_uncached_rollouts": n_unc, "t_uncached_batch_s": t_unc, "t_eval_load_s": t_eval_load,
        "eval_rollouts": n_eval_problems * cfg.eval.val_samples_per_problem, "t_train_call_s": t_train_total,
        "t_step_median_s": agg["t_step_median_s"],
        "phase_median_s": {k: agg[f"t_{k}_s"] for k in ("gen", "reward", "train")},
        "peak_gpu_mem_gib": report.get("gpu_peak_used_gib"), "peak_torch_reserved_gib": report.get("gpu_peak_reserved_gib"),
        "gpu_name": hw.get("gpu_name") if hw else None, "timers_measured": report.get("timers_measured"),
        "notes": notes,
    }
    budget_mod.parse_throughput(tp)  # the file must satisfy the cost-model schema before it is written
    return tp


def _on_stall(args: argparse.Namespace, cfg: Config, t0: float, reason: str) -> None:
    print(f"[rhg.bench] STALL: no heartbeat within run.step_timeout_s; exiting 75 ({reason})", file=sys.stderr, flush=True)
    if args.record_ledger:
        try:
            budget_mod.record("bench", "bench", time.perf_counter() - t0, cfg.budget.usd_per_hour, note="stalled", ledger=cfg.budget.ledger)
        except Exception:  # noqa: BLE001
            pass


def print_summary(tp: dict[str, Any]) -> None:
    pm = tp["phase_median_s"]
    print(f"[rhg.bench] median step {tp['t_step_median_s']:.2f}s = gen {pm['gen']:.2f} + reward {pm['reward']:.2f} + train {pm['train']:.2f} "
          f"(+ sync {tp['sync_s_per_step']:.2f}); {tp['gen_tok_per_s']:.0f} gen tok/s, {tp['train_tok_per_s']:.0f} train tok/s")
    print(f"[rhg.bench] reward exec: {tp['exec_s_per_rollout']:.4f} s/rollout CACHED (production; hit rate {tp['cache_hit_rate']:.1%} over steps >= 3) "
          f"vs {tp['exec_s_per_rollout_uncached']:.4f} s/rollout UNCACHED (n={tp['n_uncached_rollouts']}), cpu_workers={tp['cpu_workers']} of {tp['cpu_cores']} cores")
    print("[rhg.bench] caveat: the cost model uses the cached figure (what production sees); a short cold-cache bench under-estimates the "
          "steady-state hit rate, so it is conservative, and a high hit rate hides the uncached CPU cost -- read both.")
    print(f"[rhg.bench] eval pass {tp['t_eval_s']:.1f}s (engine load {tp['t_eval_load_s']:.1f}s), startup {tp['t_startup_s']:.1f}s, "
          f"peak GPU {tp['peak_gpu_mem_gib']} GiB")
    if tp["mock"]:
        print("[rhg.bench] MOCK: not a measurement of the GPU stack; written to a mock path.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhg.eval.bench", description="GRPO throughput benchmark -> results/bench/throughput.json.")
    ap.add_argument("--mock", action="store_true", help="CPU mock policy (writes results/bench_mock/throughput.json)")
    ap.add_argument("--backend", choices=["mock", "trl"], default=None, help="default: mock with --mock, else trl")
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--seed", type=int, default=BENCH_SEED)
    ap.add_argument("--steps", type=int, default=8, help="GRPO steps to run (>= 3; steps 1-2 are warm-up)")
    ap.add_argument("--n-problems", type=int, default=MIN_PROBLEMS, help=f"problems sampled from candidates.jsonl (>= {MIN_PROBLEMS})")
    ap.add_argument("--allow-few-problems", action="store_true", help="test-only: waive the >= 64 problems floor")
    ap.add_argument("--processed-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help=f"default {REAL_OUT}, or {MOCK_OUT} with --mock")
    ap.add_argument("--run-dir", type=Path, default=None, help="scratch run directory (default: <out dir>/run)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="k=v", help="config override (repeatable)")
    ap.add_argument("--no-ledger", dest="record_ledger", action="store_false", help="do not append the bench to the spend ledger")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mock = args.mock or args.backend == "mock"
    args.out = args.out or (MOCK_OUT if mock else REAL_OUT)
    if args.steps < WARMUP_STEPS + 1:
        print(f"error: --steps must be >= {WARMUP_STEPS + 1} (the cost model medians steps >= 3)", file=sys.stderr)
        return EXIT_USAGE
    if args.out.exists() and not args.force:
        print(f"refused: {args.out} already exists; pass --force to overwrite it", file=sys.stderr)
        return EXIT_GUARD
    t0 = time.perf_counter()
    try:
        tp = run_bench(args)
    except (ConfigError, run_mod.RunSetupError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as e:  # noqa: BLE001 - a failed bench still cost GPU time: record it, then report
        if not isinstance(e, (BenchError, InvalidRunError, RunFailedError)):
            traceback.print_exc()
        print(f"error: bench run failed: {type(e).__name__}: {e}", file=sys.stderr)
        _record(args, time.perf_counter() - t0, mock, f"failed: {e}")
        return EXIT_ERROR
    _write_json_atomic(args.out, tp)
    _record(args, time.perf_counter() - t0, mock, "ok")
    print_summary(tp)
    print(f"[rhg.bench] wrote {args.out}; next: python -m rhg.budget cost_model --bench {args.out}")
    return EXIT_OK


def _record(args: argparse.Namespace, wall_s: float, mock: bool, note: str) -> None:
    if mock or not args.record_ledger:
        return
    cfg = load_config(args.arm, args.overrides, seed=args.seed)
    budget_mod.record("bench", "bench", wall_s, cfg.budget.usd_per_hour, note=note, ledger=cfg.budget.ledger)


if __name__ == "__main__":
    sys.exit(main())
