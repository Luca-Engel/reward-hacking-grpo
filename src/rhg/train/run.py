"""Training-run driver: ``python -m rhg.train.run --arm A --seed N [--mock] [--set k=v] ...``.

Phases (DESIGN §2, §4, §7; docs/REPO_SPEC.md §4, §6, §7):

1. **train** -- seeded prompt schedule (``derive_seed(seed, "prompts")``: ``prompts_per_step`` distinct train
   problems per step, epoch-shuffled, identical across arms), ``gens_per_prompt`` completions each, rewards from
   ``rhg.train.rollout_io.make_reward_fn``, backend update; ``steps.jsonl``; policy snapshots at the eval steps.
2. **eval** (post-hoc, from the snapshots; eval seed ``derive_seed(seed, "eval")``, sampler identical to training):
   val problems at steps 0, val_every, ... (< T); test problems at step 0 and T with the arm's own hint; at T only,
   a cross-hint evaluation (``eval_test_xhint``, hints none/subtle/explicit). Labels are arm-independent.
3. **finalize** -- ``status.json`` / manifest / ledger (skipped for the mock unless ``--ledger-mock``).

Validity (DESIGN §7): a non-finite loss/reward makes the run ``invalid``; any other exception makes it ``failed``
with the traceback in ``stdout.log`` (non-zero exit). A stall watchdog (``rhg.train.watchdog``) kills a run whose
step heartbeat stops for ``run.step_timeout_s`` (status ``failed``/``stall``, exit code 75 = infrastructure stall).

Exit codes: 0 ok, 1 failed/invalid, 2 usage/config error, 3 guard refused (budget / confirmatory), 75 stall.

The backend is chosen with ``--backend {mock,trl}`` (``--mock`` implies ``mock``). The ``trl`` backend lives in
``rhg.train.trl_trainer`` and exposes ``create_backend(cfg, *, problems, prompts, run_dir)``
returning an ``rhg.train.backend.Backend``.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from rhg import budget as budget_mod
from rhg import manifest as manifest_mod
from rhg import runlog
from rhg.config import Config, ConfigError, load_config
from rhg.data.prompts import build_prompt, load_prompts_cfg, render_chat
from rhg.env.monitor import make_monitor
from rhg.eval.generate import SamplingParams, generate_for
from rhg.seeds import derive_seed, seed_everything
from rhg.train.backend import Backend, InvalidRunError, RunFailedError, TrainContext
from rhg.train.rollout_io import RolloutLogger, StepCounter, grade_to_records, make_reward_fn
from rhg.train.watchdog import EXIT_STALL, Watchdog

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_GUARD = 0, 1, 2, manifest_mod.EXIT_GUARD_REFUSED
HINTS = runlog.EVAL_HINTS
PLANNING_PRIOR_MAX_RUN_USD = 1.00  # BUDGET §2 upper planning prior ($0.45-1.00/run); UNVERIFIED until a bench exists
BENCH_PATH = Path("results/bench/throughput.json")
FIXTURE_PROCESSED = manifest_mod.REPO_ROOT / "data" / "fixture" / "processed"


class RunSetupError(ValueError):
    """Bad inputs (config/data) detected before anything is written; exit code 2."""


# ------------------------------------------------------------------ backends
def load_backend_factory(name: str) -> Callable[..., Backend]:
    """``create_backend`` of the named backend; the real one is imported lazily."""
    if name == "mock":
        from rhg.train.mock_policy import create_backend

        return create_backend
    if name == "trl":
        try:
            from rhg.train.trl_trainer import create_backend  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            if e.name != "rhg.train.trl_trainer":
                raise
            raise NotImplementedError("the 'trl' backend (rhg.train.trl_trainer) is missing from this checkout") from e
        return create_backend
    raise ValueError(f"unknown backend {name!r}; expected 'mock' or 'trl'")


# ------------------------------------------------------------------ data and schedule
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One JSON object per *newline-terminated* line. ``str.splitlines`` must not be used here: it also splits on
    U+2028/U+0085/form feeds, which occur inside the problem descriptions of the real dataset."""
    with open(path, encoding="utf-8", newline="\n") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_problems(cfg: Config, mock: bool) -> tuple[dict[str, dict[str, Any]], str]:
    """All problems of ``problems.jsonl`` by id. In mock mode a missing file falls back to the fixture set
    (``data/fixture/processed``), then to the built-in 8 tiny problems; the real backend never falls back."""
    processed = Path(cfg.data.processed_dir)
    if not processed.is_dir() and not processed.is_absolute() and (manifest_mod.REPO_ROOT / processed).is_dir():
        processed = manifest_mod.REPO_ROOT / processed
    path = processed / "problems.jsonl"
    source = str(path)
    rows: list[dict[str, Any]]
    if path.is_file():
        rows = read_jsonl(path)
    elif mock and (FIXTURE_PROCESSED / "problems.jsonl").is_file():
        source = str(FIXTURE_PROCESSED / "problems.jsonl")
        rows = read_jsonl(Path(source))
    elif mock:
        from rhg.data.fixture import tiny_problems

        source, rows = "built-in tiny fixture problems", tiny_problems()
    else:
        raise RunSetupError(f"{path} not found; build the dataset first (python -m rhg.data.build)")
    problems = {str(r["problem_id"]): r for r in rows}
    if len(problems) != len(rows):
        raise RunSetupError(f"{source}: duplicate problem_id")
    return problems, source


def split_ids(problems: Mapping[str, Mapping[str, Any]], split: str) -> list[str]:
    return sorted(pid for pid, p in problems.items() if p.get("split") == split)


def make_schedule(train_ids: Sequence[str], steps: int, prompts_per_step: int, seed: int) -> list[list[str]]:
    """``steps`` lists of ``prompts_per_step`` distinct train problems, drawn epoch by epoch from a seeded
    shuffle (each epoch is a fresh permutation; ids that would repeat inside a step wait for the next one).
    Depends on ``seed`` and the ids only, so every arm sees the same data order."""
    ids = sorted(train_ids)
    if prompts_per_step > len(ids):
        raise RunSetupError(f"prompts_per_step={prompts_per_step} exceeds the {len(ids)} train problems")
    rng = random.Random(derive_seed(seed, "prompts"))
    queue: list[str] = []
    out: list[list[str]] = []
    for _ in range(steps):
        while len(queue) < prompts_per_step:
            perm = list(ids)
            rng.shuffle(perm)
            queue.extend(perm)
        chosen: list[str] = []
        rest: list[str] = []
        for pid in queue:
            if len(chosen) < prompts_per_step and pid not in chosen:
                chosen.append(pid)
            else:
                rest.append(pid)
        queue = rest
        out.append(chosen)
    return out


def eval_plan(cfg: Config) -> dict[str, Any]:
    """Snapshot steps and which evals run where (DESIGN §4)."""
    T, every = cfg.grpo.max_steps, cfg.eval.val_every
    grid = [s for s in range(0, T + 1) if s % every == 0]
    val = [s for s in grid if s < T]
    snaps = sorted({0, T, *grid})
    return {"snapshots": snaps, "val": val, "test": sorted({0, T}), "xhint": T}


class PromptBook:
    """Rendered chat prompt of (problem, hint level), cached."""

    def __init__(self, problems: Mapping[str, Mapping[str, Any]], enable_thinking: bool = False) -> None:
        self._cfg = load_prompts_cfg()
        self._problems, self._thinking = problems, enable_thinking
        self._cache: dict[tuple[str, str], str] = {}

    def __call__(self, pid: str, hint: str) -> str:
        key = (pid, hint)
        if key not in self._cache:
            self._cache[key] = render_chat(build_prompt(self._problems[pid], hint, self._cfg), enable_thinking=self._thinking)
        return self._cache[key]


# ------------------------------------------------------------------ evals
def run_eval_point(
    cfg: Config,
    gen,
    *,
    problems: Mapping[str, Mapping[str, Any]],
    prompts: PromptBook,
    pids: Sequence[str],
    phase: str,
    step: int,
    prompt_hint: str,
    eval_hint: str | None,
    n: int,
    eval_seed: int,
    params: SamplingParams,
    monitor_fn,
    acc: runlog.EvalAccumulator,
    writer: runlog.RolloutWriter,
) -> None:
    """Generate ``n`` samples per problem with the same sampler as training, grade, log and count them."""
    plist = [prompts(pid, prompt_hint) for pid in pids]
    metas = [{"problem_id": pid, "hint": prompt_hint, "problem": problems[pid]} for pid in pids]
    outs = generate_for(gen, plist, metas, n, params, seed=eval_seed)
    pid_flat, comps, ntok, trunc, idx = [], [], [], [], []
    for pid, row in zip(pids, outs):
        if len(row) != n:
            raise RuntimeError(f"generator returned {len(row)} completions for {pid}, expected {n}")
        for j, c in enumerate(row):
            pid_flat.append(pid), comps.append(c.text), ntok.append(c.n_tokens), trunc.append(c.truncated), idx.append(j)
    recs = grade_to_records(
        cfg, problems, pid_flat, comps, run_id=cfg.run_id, phase=phase, step=step, sample_idx=idx,
        eval_hint=eval_hint, n_tokens=ntok, truncated=trunc, monitor_fn=monitor_fn,
    )
    key = acc.open_point(phase, step, eval_hint, prompt_hint, n)
    writer.write_many(recs)
    for r in recs:
        acc.add(key, r)


# ------------------------------------------------------------------ console / log plumbing
class _Tee(io.TextIOBase):
    """Write to the console and to ``stdout.log``; never fails on console encoding problems."""

    def __init__(self, console, log) -> None:
        self._console, self._log = console, log

    def write(self, s: str) -> int:
        self._log.write(s)
        self._log.flush()
        try:
            self._console.write(s)
        except UnicodeEncodeError:
            self._console.write(s.encode("ascii", "replace").decode("ascii"))
        except (OSError, ValueError):
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._log.flush()
            self._console.flush()
        except (OSError, ValueError):
            pass


def estimate_run_usd(cfg: Config, mock: bool) -> float:
    """Projected cost of this run for the launch guard: 0 for the mock; else the measured cost model if a
    bench exists (BUDGET §2), else the planning-prior upper bound."""
    if mock:
        return 0.0
    if BENCH_PATH.is_file():
        tp = budget_mod.load_throughput(BENCH_PATH)
        if tp.mock:
            raise budget_mod.BenchError(f"{BENCH_PATH} was written by `rhg.eval.bench --mock`; run the real bench (or delete it)")
        return budget_mod.run_cost(tp, cfg.grpo.max_steps, cfg.budget.usd_per_hour).run_usd
    return PLANNING_PRIOR_MAX_RUN_USD


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhg.train.run", description="Run one training run (arm x seed).")
    ap.add_argument("--arm", required=True, help="arm id, e.g. hackable_subtle")
    ap.add_argument("--seed", type=int, default=None, help="run seed (default: config run.seed)")
    ap.add_argument("--mock", action="store_true", help="CPU mock policy instead of the GPU trainer (implies --backend mock)")
    ap.add_argument("--backend", choices=["mock", "trl"], default=None, help="default: mock with --mock, else trl")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="k=v", help="config override (repeatable)")
    ap.add_argument("--confirmatory", action="store_true", help="require the pre-registration tag/freeze (else exit 3)")
    ap.add_argument("--steps", type=int, default=None, help="override grpo.max_steps")
    ap.add_argument("--pilot", action="store_true", help="record the ledger entry as kind 'pilot' (pilot seeds are 9000+)")
    ap.add_argument("--ledger-mock", action="store_true", help="write a ledger entry even for a mock run")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    ap.add_argument("--keep-adapters", action="store_true", help="keep the saved LoRA adapters (trl backend) after the evals")
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--repo-root", type=Path, default=None, help="repository root for provenance/guards (default: this repo)")
    return ap


# ------------------------------------------------------------------ the run
def _setup(args: argparse.Namespace) -> tuple[Config, bool, str]:
    backend_name = args.backend or ("mock" if args.mock else "trl")
    if args.mock and backend_name != "mock":
        raise RunSetupError("--mock cannot be combined with --backend trl")
    mock = backend_name == "mock"
    overrides = list(args.overrides)
    if mock:
        overrides.append("run.mode=mock")
    if args.confirmatory:
        overrides.append("run.confirmatory=true")
    if args.steps is not None:
        overrides.append(f"grpo.max_steps={args.steps}")
    return load_config(args.arm, overrides, seed=args.seed, config_dir=args.config_dir), mock, backend_name


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg, mock, backend_name = _setup(args)
    except (ConfigError, RunSetupError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    if args.confirmatory:
        try:
            manifest_mod.assert_confirmatory_ok(cfg, repo_root=args.repo_root)
        except manifest_mod.ConfirmatoryGuardError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return EXIT_GUARD
        if mock:
            print("error: a mock run cannot be confirmatory", file=sys.stderr)
            return EXIT_USAGE

    try:
        problems, data_source = load_problems(cfg, mock)
        train_ids, val_ids, test_ids = (split_ids(problems, s) for s in ("train", "val", "test"))
        if not (val_ids and test_ids):
            raise RunSetupError(f"{data_source}: need val and test problems (have {len(val_ids)}/{len(test_ids)})")
        schedule = make_schedule(train_ids, cfg.grpo.max_steps, cfg.grpo.prompts_per_step, cfg.run.seed)
        monitor_fn = make_monitor(cfg.arm.monitor)
        next_usd = estimate_run_usd(cfg, mock)
    except (RunSetupError, budget_mod.BenchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    try:
        decision = budget_mod.check_launch(next_usd, cfg.budget.stop_at_usd, ledger=cfg.budget.ledger)
    except budget_mod.LedgerCorruptError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_GUARD
    if not decision.allowed:
        print(decision.message, file=sys.stderr)
        return EXIT_GUARD

    try:
        factory = load_backend_factory(backend_name)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_FAILED

    run_dir = Path(cfg.run.output_root) / cfg.run_id
    if (run_dir / runlog.MANIFEST_FILE).exists() and not args.force:
        print(f"error: {run_dir} already has a manifest; pass --force to overwrite it", file=sys.stderr)
        return EXIT_USAGE
    return _execute(args, cfg, mock, backend_name, factory, problems, data_source, schedule, monitor_fn, (train_ids, val_ids, test_ids), run_dir)


def _execute(args, cfg: Config, mock, backend_name, factory, problems, data_source, schedule, monitor_fn, ids, run_dir: Path) -> int:
    train_ids, val_ids, test_ids = ids
    run_id = cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in (runlog.ROLLOUTS_FILE, runlog.STEPS_FILE, runlog.EVALS_FILE, runlog.STATUS_FILE, runlog.STDOUT_FILE):
        (run_dir / stale).unlink(missing_ok=True)
    (run_dir / runlog.CONFIG_FILE).write_text(cfg.resolved_yaml(), encoding="utf-8", newline="\n")
    manifest = manifest_mod.build_manifest(cfg, repo_root=args.repo_root)
    manifest_mod.write_manifest(run_dir, manifest)
    runlog.write_status(run_dir, run_id, "running", phase="setup", step=0)

    log_f = open(run_dir / runlog.STDOUT_FILE, "a", encoding="utf-8", newline="\n")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(real_out, log_f), _Tee(real_err, log_f)
    t_start = time.perf_counter()
    progress: dict[str, Any] = {"phase": "setup", "step": 0}
    write_ledger = (not mock) or args.ledger_mock
    ledger_kind = "pilot" if args.pilot else "train"

    def close_out(status: str, reason: str | None, exit_code: int) -> None:
        wall = time.perf_counter() - t_start
        runlog.write_status(run_dir, run_id, status, reason=reason, phase=progress["phase"], step=progress["step"], exit_code=exit_code)
        man = manifest_mod.finalize_manifest(run_dir, status, invalid_reason=reason if status != "completed" else None, wall_s=wall)
        if write_ledger:
            budget_mod.record(ledger_kind, run_id, wall, cfg.budget.usd_per_hour, note=f"{status}: {reason or 'ok'} ({cfg.run.mode})", ledger=cfg.budget.ledger)
        print(f"[rhg.train.run] {run_id}: {status}{' (' + reason + ')' if reason else ''} wall={wall:.1f}s usd={man['usd']:.4f}", flush=True)

    def on_stall(reason: str) -> None:
        try:
            close_out("failed", reason, EXIT_STALL)
        except Exception:  # noqa: BLE001 - the watchdog must still exit
            pass

    wd = Watchdog(float(cfg.run.step_timeout_s), on_stall=on_stall, dump_file=log_f)
    writer = runlog.RolloutWriter(run_dir)
    step_writer = runlog.StepWriter(run_dir)
    backend: Backend | None = None
    exit_code = EXIT_FAILED
    try:
        print(f"[rhg.train.run] run_id={run_id} backend={backend_name} mode={cfg.run.mode} steps={cfg.grpo.max_steps} "
              f"data={data_source} (train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ids)}) hint={cfg.arm.hint} "
              f"reward={cfg.arm.reward} monitor={cfg.arm.monitor}", flush=True)
        if args.pilot and cfg.run.seed < 9000:
            print("[rhg.train.run] warning: pilot runs are expected to use seeds >= 9000 (DESIGN §3)", flush=True)
        wd.start()
        seed_everything(cfg.run.seed)
        if not mock and cfg.sandbox.cache:
            from rhg.env import cache as grade_cache

            grade_cache.configure_cache(grade_cache.DEFAULT_DISK_DIR)
        prompts = PromptBook(problems, cfg.model.enable_thinking)
        prompt_by_pid = {pid: prompts(pid, cfg.arm.hint) for pid in train_ids}
        params = SamplingParams.from_config(cfg)
        plan = eval_plan(cfg)

        counter = StepCounter(0)
        logger = RolloutLogger(run_id, writer, counter)
        reward_fn = make_reward_fn(cfg, problems, monitor_fn, logger)
        backend = factory(cfg, problems=problems, prompts=prompts, run_dir=run_dir)

        def heartbeat(label: str) -> None:
            wd.beat(label)
            if label.startswith("step "):
                progress["step"] = int(label.split()[1])
                runlog.write_status(run_dir, run_id, "running", phase=progress["phase"], step=progress["step"])

        ctx = TrainContext(
            cfg=cfg, problems=problems, schedule=schedule, prompts=prompt_by_pid, sampling=params, reward_fn=reward_fn,
            logger=logger, counter=counter, step_writer=step_writer, snapshot_steps=frozenset(plan["snapshots"]),
            snapshot_fn=backend.snapshot, heartbeat=heartbeat,
        )

        # phase 1: train
        progress["phase"] = "train"
        ctx.take_snapshot(0)
        wd.beat("snapshot 0")
        backend.train(ctx)
        if ctx.last_step != cfg.grpo.max_steps:
            raise RuntimeError(f"backend stopped after {ctx.last_step} of {cfg.grpo.max_steps} steps")
        backend.end_training()

        # phase 2: evals from the snapshots
        progress["phase"] = "eval"
        acc = runlog.EvalAccumulator(run_id, cfg.arm.id, cfg.run.seed)
        eval_seed = derive_seed(cfg.run.seed, "eval")
        T = cfg.grpo.max_steps
        for step in plan["snapshots"]:
            wd.beat(f"loading eval generator {step}")  # a fresh engine load can take minutes (trl: one per adapter)
            gen = backend.eval_generator(ctx.snapshots[step])
            wd.beat(f"eval generator {step} ready")
            try:
                common = dict(problems=problems, prompts=prompts, step=step, eval_seed=eval_seed, params=params,
                              monitor_fn=monitor_fn, acc=acc, writer=writer)
                if step in plan["val"]:
                    run_eval_point(cfg, gen, pids=val_ids, phase="eval_val", prompt_hint=cfg.arm.hint, eval_hint=None,
                                   n=cfg.eval.val_samples_per_problem, **common)
                    wd.beat(f"eval_val {step}")
                if step in plan["test"]:
                    run_eval_point(cfg, gen, pids=test_ids, phase="eval_test", prompt_hint=cfg.arm.hint, eval_hint=None,
                                   n=cfg.eval.test_samples_per_problem, **common)
                    wd.beat(f"eval_test {step}")
                if step == plan["xhint"]:
                    for hint in HINTS:
                        run_eval_point(cfg, gen, pids=test_ids, phase="eval_test_xhint", prompt_hint=hint, eval_hint=hint,
                                       n=cfg.eval.xhint_samples_per_problem, **common)
                        wd.beat(f"eval_test_xhint {step} {hint}")
            finally:
                gen.close()
            runlog.write_evals(run_dir, acc.summary())

        # phase 3: finalize
        progress.update(phase="finalize", step=T)
        if not args.keep_adapters:
            getattr(backend, "delete_snapshots", lambda: None)()
        step_writer.close()
        writer.close()
        wd.stop()
        close_out("completed", None, EXIT_OK)
        problems_found = runlog.validate_run(run_dir)
        if problems_found:
            for p in problems_found[:20]:
                print(f"[rhg.train.run] log validation: {p}", file=sys.stderr)
            reason = f"log validation failed: {problems_found[0]}"
            runlog.write_status(run_dir, run_id, "failed", reason=reason, phase="finalize", step=T, exit_code=EXIT_FAILED)
            manifest_mod.finalize_manifest(run_dir, "failed", invalid_reason=reason)
            exit_code = EXIT_FAILED
        else:
            exit_code = EXIT_OK
    except InvalidRunError as e:
        print(f"[rhg.train.run] INVALID: {e.reason}", file=sys.stderr, flush=True)
        exit_code = _finish_bad(close_out, "invalid", e.reason, wd)
    except RunFailedError as e:
        traceback.print_exc()
        exit_code = _finish_bad(close_out, "failed", e.reason, wd)
    except KeyboardInterrupt:
        print("[rhg.train.run] interrupted", file=sys.stderr, flush=True)
        exit_code = _finish_bad(close_out, "failed", "interrupted", wd, 130)
    except Exception as e:  # noqa: BLE001 - every failure must end in status "failed" with a traceback
        traceback.print_exc()
        exit_code = _finish_bad(close_out, "failed", f"{type(e).__name__}: {e}", wd)
    finally:
        wd.stop()
        for closer in (step_writer.close, writer.close, (backend.close if backend is not None else lambda: None)):
            try:
                closer()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        sys.stdout, sys.stderr = real_out, real_err
        log_f.close()
    return exit_code


def _finish_bad(close_out, status: str, reason: str, wd: Watchdog, exit_code: int = EXIT_FAILED) -> int:
    wd.stop()
    try:
        close_out(status, reason, exit_code)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
