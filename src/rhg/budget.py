"""Spend ledger, launch guard, measured-cost model and cut ladder (BUDGET.md).

CLI: ``python -m rhg.budget {check,record,status,cost_model} ...``; exit codes 0 ok,
2 usage / missing input (e.g. no bench file), 3 guard refused, 1 other.

Ledger ``results/ledger.jsonl``: one JSON object per line
``{ts, kind, run_id, wall_s, usd_per_hour, usd, note}``, appended with a single ``write`` on a
file opened in append mode, under an advisory cross-process lock. Reading raises on a corrupt
line (silently skipping it would under-count spend).

``results/bench/throughput.json`` (written by ``rhg.eval.bench``; every value MEASURED, never
hand-entered)::

    gen_tokens_per_step     float  completion tokens generated per training step
    gen_tok_per_s           float  vLLM generation throughput (tokens/s, colocated)
    train_tokens_per_step   float  tokens through LoRA fwd+bwd per step
    train_tok_per_s         float  training throughput (tokens/s)
    exec_s_per_rollout      float  sandbox seconds per rollout, grading cache ON (production)
    cpu_workers             int    parallel sandbox workers used for the measurement
    rollouts_per_step       int    prompts_per_step * gens_per_prompt
    sync_s_per_step         float  LoRA merge/sync to vLLM per step
    t_eval_s                float  one evaluation pass (see N_EVAL)
    t_startup_s             float  model load + vLLM start + first-step compile
    n_steps_measured        int    steps actually run in the bench
    step_times_s            [float] wall seconds of every measured step (>= 3 entries)
    cache_hit_rate          float  optional; grading-cache hit rate during the bench
    exec_s_per_rollout_uncached float  optional; uncached figure for comparison
    mock                    bool   optional; true = written by ``rhg.eval.bench --mock`` (not a measurement)
    (any other key -- peak_gpu_mem_gib, t_eval_load_s, notes, ... -- is informational and ignored here)

Formulas are BUDGET §2 exactly (``t_step`` from components; the median of ``step_times_s``
from step 3 onward is reported as a cross-check and can be selected with
``--t-step-source measured|max``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_LEDGER = Path("results/ledger.jsonl")
DEFAULT_STOP_AT_USD = 28.0
DEFAULT_MAIN_CAP_USD = 16.0
DEFAULT_MAX_WALL_H = 12.0
DEFAULT_BENCH = Path("results/bench/throughput.json")
DEFAULT_MEASURED_MD = Path("BUDGET_MEASURED.md")
DEFAULT_DECISION_MD = Path("prereg/budget_decision.md")
OVERHEAD = 0.08  # BUDGET §2: idle/overhead margin
N_EVAL = 6  # 5 val evals + 1 test eval
FLOOR_RUNS = 11
KINDS = ("train", "bench", "pilot", "passrate", "probe", "judge", "other")
_EPS = 1e-9

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_GUARD = 0, 1, 2, 3

# ---------------------------------------------------------------------------- ladder


@dataclass(frozen=True)
class LadderStep:
    step: int
    change: str
    seeds: dict[str, int]
    consequence: str

    @property
    def runs(self) -> int:
        return sum(self.seeds.values())


def _seeds(**kw: int) -> dict[str, int]:
    return dict(kw)


_S0 = _seeds(
    clean_none=2, clean_subtle=5, clean_explicit=2,
    hackable_none=2, hackable_subtle=5, hackable_explicit=3, hackable_subtle_ast=3,
)
_S1 = {**_S0, "hackable_none": 1}
_S2 = {**_S1, "clean_none": 0}
_S3 = {**_S2, "hackable_subtle_ast": 0}
_S4 = {**_S3, "clean_explicit": 0}
_S5 = {**_S4, "hackable_explicit": 2}
_S6 = {**_S5, "clean_subtle": 4, "hackable_subtle": 4}

# BUDGET §4, top to bottom; runs after each step: 22, 21, 19, 16, 14, 13, 11.
LADDER: tuple[LadderStep, ...] = (
    LadderStep(0, "full design", _S0, "-"),
    LadderStep(1, "hackable_none 2->1", _S1, "floor check only; H1 none-level has n=1"),
    LadderStep(2, "drop clean_none", _S2, "no unhinted clean baseline; clean_subtle is the learning-curve reference"),
    LadderStep(3, "drop hackable_subtle_ast", _S3, "H4 unevaluated (future work)"),
    LadderStep(
        4, "drop clean_explicit", _S4,
        "H1 loses its prompt-vs-reward control at the explicit level; H1 becomes 'hackable-arm trend, prompt confound unresolved'",
    ),
    LadderStep(5, "hackable_explicit 3->2", _S5, "H1 trend has n=2 at the top level"),
    LadderStep(
        6, "primary 5 v 5 -> 4 v 4", _S6,
        "primary significant only if 4/4 seeds emerge (p=0.0143; 3/4 gives 0.071)",
    ),
)
FLOOR_TEXT = "if 11 runs still exceed the cap: NO confirmatory study; report pilots/probes only"


def recommend_step(runs_done: int, affordable_runs: int) -> LadderStep | None:
    """Highest-priority ladder step whose run count fits ``runs_done + affordable_runs``.

    ``None`` means even the 11-run floor does not fit (no confirmatory study).
    """
    budget_runs = runs_done + max(0, affordable_runs)
    for step in LADDER:
        if step.runs <= budget_runs:
            return step
    return None


# ---------------------------------------------------------------------------- ledger


def resolve_ledger(ledger: str | Path | None = None) -> Path:
    """``--ledger`` if given, else ``budget.ledger`` of the base config, else the default."""
    if ledger is not None:
        return Path(ledger)
    try:
        from rhg.config import load_config

        return Path(load_config("clean_none").budget.ledger)
    except Exception:  # noqa: BLE001 - configs unavailable: fall back to the documented default
        return DEFAULT_LEDGER


@contextlib.contextmanager
def _file_lock(path: Path, timeout_s: float = 30.0) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        if os.name == "nt":
            import msvcrt

            def acquire() -> None:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

            def release() -> None:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(fd, fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                acquire()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not lock {lock_path} within {timeout_s}s") from None
                time.sleep(0.005)
        try:
            yield
        finally:
            release()
    finally:
        os.close(fd)


def _check_number(name: str, value: float | None) -> None:
    if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")


def record(
    kind: str,
    run_id: str,
    wall_s: float,
    usd_per_hour: float | None = None,
    usd: float | None = None,
    note: str = "",
    *,
    ledger: str | Path | None = None,
) -> dict[str, Any]:
    """Append one metered action. ``usd`` defaults to ``wall_s/3600 * usd_per_hour`` (no margin)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    for name, val in (("wall_s", wall_s), ("usd_per_hour", usd_per_hour), ("usd", usd)):
        _check_number(name, val)
    if usd is None:
        if usd_per_hour is None:
            raise ValueError("record needs usd, or usd_per_hour to price wall_s")
        usd = wall_s / 3600.0 * usd_per_hour
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "kind": kind,
        "run_id": str(run_id),
        "wall_s": float(wall_s),
        "usd_per_hour": None if usd_per_hour is None else float(usd_per_hour),
        "usd": float(usd),
        "note": str(note),
    }
    path = resolve_ledger(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(entry, sort_keys=False) + "\n").encode("utf-8")
    with _file_lock(path):
        with open(path, "ab", buffering=0) as f:
            f.write(line)  # one write call for the whole line
    return entry


class LedgerCorruptError(RuntimeError):
    """A ledger line could not be parsed; spend cannot be trusted."""


def read_entries(ledger: str | Path | None = None) -> list[dict[str, Any]]:
    path = resolve_ledger(ledger)
    if not path.exists():
        return []
    with _file_lock(path):
        raw = path.read_bytes().decode("utf-8")
    entries = []
    for n, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            float(e["usd"])
            str(e["kind"])
        except (ValueError, KeyError, TypeError) as exc:
            raise LedgerCorruptError(f"{path}:{n}: corrupt ledger line: {exc!r}") from exc
        entries.append(e)
    return entries


def spent_by_kind(ledger: str | Path | None = None) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for e in read_entries(ledger):
        by.setdefault(e["kind"], []).append(float(e["usd"]))
    return {k: math.fsum(v) for k, v in by.items()}


def spent_usd(ledger: str | Path | None = None) -> float:
    return math.fsum(float(e["usd"]) for e in read_entries(ledger))


def runs_done(ledger: str | Path | None = None) -> int:
    """Distinct run ids with a ``train`` entry (used only to phrase the ladder recommendation)."""
    return len({e["run_id"] for e in read_entries(ledger) if e["kind"] == "train"})


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    spent: float
    next_run_usd: float
    stop_at: float
    message: str
    recommended: LadderStep | None = None


def check_launch(
    next_run_usd: float,
    stop_at: float = DEFAULT_STOP_AT_USD,
    *,
    ledger: str | Path | None = None,
) -> GuardDecision:
    """Refuse iff ``spent + next_run_usd > stop_at`` (equality is allowed; 1e-9 float slack)."""
    _check_number("next_run_usd", next_run_usd)
    spent = spent_usd(ledger)
    total = spent + next_run_usd
    if total <= stop_at + _EPS:
        return GuardDecision(
            True, spent, next_run_usd, stop_at,
            f"OK: spent ${spent:.4f} + next run ${next_run_usd:.4f} = ${total:.4f} <= stop-at ${stop_at:.2f}",
        )
    remaining = max(0.0, stop_at - spent)
    affordable = int(math.floor(remaining / next_run_usd + _EPS)) if next_run_usd > 0 else 0
    done = runs_done(ledger)
    rec = recommend_step(done, affordable)
    msg = (
        f"REFUSED: spent ${spent:.4f} + next run ${next_run_usd:.4f} = ${total:.4f} > stop-at ${stop_at:.2f}. "
        f"${remaining:.4f} remains = {affordable} more run(s) at this cost; {done} train run(s) already in the ledger. "
    )
    if rec is None:
        msg += f"Recommended: below the {FLOOR_RUNS}-run floor - {FLOOR_TEXT}."
    else:
        msg += (
            f"Recommended ladder step {rec.step} ({rec.change}; {rec.runs} runs total): {rec.consequence}"
        )
    return GuardDecision(False, spent, next_run_usd, stop_at, msg, rec)


# ---------------------------------------------------------------------------- cost model


class BenchError(ValueError):
    """throughput.json is malformed."""


_POSITIVE_FLOAT = ("gen_tokens_per_step", "gen_tok_per_s", "train_tokens_per_step", "train_tok_per_s")
_NONNEG_FLOAT = ("exec_s_per_rollout", "sync_s_per_step", "t_eval_s", "t_startup_s")
_REQUIRED_INT = ("cpu_workers", "rollouts_per_step", "n_steps_measured")


@dataclass(frozen=True)
class Throughput:
    gen_tokens_per_step: float
    gen_tok_per_s: float
    train_tokens_per_step: float
    train_tok_per_s: float
    exec_s_per_rollout: float
    cpu_workers: int
    rollouts_per_step: int
    sync_s_per_step: float
    t_eval_s: float
    t_startup_s: float
    n_steps_measured: int
    step_times_s: tuple[float, ...]
    cache_hit_rate: float | None = None
    exec_s_per_rollout_uncached: float | None = None
    mock: bool = False

    @property
    def t_step_measured_s(self) -> float:
        """Robust median of the measured wall time of steps 3..N (steps 1-2 are warm-up)."""
        return statistics.median(self.step_times_s[2:])


def _num(d: dict[str, Any], key: str, positive: bool) -> float:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise BenchError(f"throughput.json: {key!r} missing or not a finite number (got {v!r})")
    if (v <= 0) if positive else (v < 0):
        raise BenchError(f"throughput.json: {key!r} must be {'> 0' if positive else '>= 0'}, got {v!r}")
    return float(v)


def parse_throughput(d: dict[str, Any]) -> Throughput:
    if not isinstance(d, dict):
        raise BenchError("throughput.json: top level must be an object")
    kw: dict[str, Any] = {}
    for k in _POSITIVE_FLOAT + _NONNEG_FLOAT:
        kw[k] = _num(d, k, positive=k in _POSITIVE_FLOAT)
    for k in _REQUIRED_INT:
        v = _num(d, k, positive=True)
        if v != int(v):
            raise BenchError(f"throughput.json: {k!r} must be an integer, got {v!r}")
        kw[k] = int(v)
    steps = d.get("step_times_s")
    if not isinstance(steps, list) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in steps):
        raise BenchError("throughput.json: 'step_times_s' must be a list of positive numbers")
    if len(steps) < 3:
        raise BenchError(f"throughput.json: need >= 3 measured steps (median is over steps >= 3), got {len(steps)}")
    kw["step_times_s"] = tuple(float(x) for x in steps)
    for opt in ("cache_hit_rate", "exec_s_per_rollout_uncached"):
        if d.get(opt) is not None:
            kw[opt] = _num(d, opt, positive=False)
    kw["mock"] = d.get("mock") is True
    return Throughput(**kw)


def load_throughput(path: str | Path) -> Throughput:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as e:
        raise BenchError(f"{path}: not valid JSON: {e}") from e
    return parse_throughput(raw)


@dataclass(frozen=True)
class RunCost:
    t_gen: float
    t_reward: float
    t_train: float
    t_sync: float
    t_step: float
    t_step_source: str
    run_s: float
    run_usd: float


def run_cost(
    tp: Throughput, T: int, usd_per_hour: float, *, t_step_source: str = "formula", overhead: float = OVERHEAD
) -> RunCost:
    """BUDGET §2: t_step = t_gen+t_reward+t_train+t_sync; run_s = T*t_step + N_EVAL*t_eval + t_startup;
    run_usd = run_s/3600 * usd_per_hour * (1 + overhead)."""
    t_gen = tp.gen_tokens_per_step / tp.gen_tok_per_s
    t_reward = tp.rollouts_per_step * tp.exec_s_per_rollout / tp.cpu_workers
    t_train = tp.train_tokens_per_step / tp.train_tok_per_s
    t_sync = tp.sync_s_per_step
    formula = t_gen + t_reward + t_train + t_sync
    if t_step_source == "formula":
        t_step = formula
    elif t_step_source == "measured":
        t_step = tp.t_step_measured_s
    elif t_step_source == "max":
        t_step = max(formula, tp.t_step_measured_s)
    else:
        raise ValueError(f"t_step_source must be formula|measured|max, got {t_step_source!r}")
    run_s = T * t_step + N_EVAL * tp.t_eval_s + tp.t_startup_s
    run_usd = run_s / 3600.0 * usd_per_hour * (1.0 + overhead)
    return RunCost(t_gen, t_reward, t_train, t_sync, t_step, t_step_source, run_s, run_usd)


@dataclass(frozen=True)
class LadderRow:
    step: LadderStep
    main_usd: float
    wall_h: float
    fits_cost: bool
    fits_wall: bool

    @property
    def fits(self) -> bool:
        return self.fits_cost and self.fits_wall


def evaluate_ladder(
    cost: RunCost, n_boxes: int, main_cap: float = DEFAULT_MAIN_CAP_USD, max_wall_h: float = DEFAULT_MAX_WALL_H
) -> list[LadderRow]:
    """main_usd = runs*run_usd (arms priced equal); wall_h = runs*run_s/3600/n_boxes."""
    rows = []
    for step in LADDER:
        main_usd = step.runs * cost.run_usd
        wall_h = step.runs * cost.run_s / 3600.0 / n_boxes
        rows.append(LadderRow(step, main_usd, wall_h, main_usd <= main_cap + _EPS, wall_h <= max_wall_h + _EPS))
    return rows


def select_step(rows: list[LadderRow]) -> LadderRow | None:
    return next((r for r in rows if r.fits), None)


def _md_ladder_table(rows: list[LadderRow], chosen: LadderRow | None) -> str:
    out = ["| step | change | runs | main_usd | wall_h | cost ok | wall ok | |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = "**selected**" if chosen is not None and r.step.step == chosen.step.step else ""
        out.append(
            f"| {r.step.step} | {r.step.change} | {r.step.runs} | {r.main_usd:.2f} | {r.wall_h:.2f} | "
            f"{'yes' if r.fits_cost else 'no'} | {'yes' if r.fits_wall else 'no'} | {mark} |"
        )
    return "\n".join(out)


def _seed_table(step: LadderStep) -> str:
    return "\n".join(["| arm | seeds |", "|---|---|", *(f"| {a} | {n} |" for a, n in step.seeds.items())])


def render_measured_md(
    tp: Throughput, cost: RunCost, rows: list[LadderRow], chosen: LadderRow | None, params: dict[str, Any]
) -> str:
    warn = ""
    ratio = tp.t_step_measured_s / (cost.t_gen + cost.t_reward + cost.t_train + cost.t_sync)
    if abs(ratio - 1.0) > 0.20:
        warn = (
            f"\n**WARNING:** median measured step time ({tp.t_step_measured_s:.2f} s) differs from the "
            f"component formula ({cost.t_gen + cost.t_reward + cost.t_train + cost.t_sync:.2f} s) by "
            f"{(ratio - 1) * 100:+.0f}%; inspect the bench (t_step source used: {cost.t_step_source}).\n"
        )
    extras = []
    if tp.cache_hit_rate is not None:
        extras.append(f"- grading-cache hit rate during bench: {tp.cache_hit_rate:.3f}")
    if tp.exec_s_per_rollout_uncached is not None:
        extras.append(f"- uncached exec_s_per_rollout: {tp.exec_s_per_rollout_uncached:.4f}")
    verdict = (
        f"**Selected: ladder step {chosen.step.step} - {chosen.step.change} ({chosen.step.runs} runs)**"
        if chosen is not None
        else f"**No ladder step fits. Below the floor of {FLOOR_RUNS} runs: no confirmatory study.**"
    )
    return f"""# BUDGET_MEASURED

Generated by `python -m rhg.budget cost_model` from `{params['bench']}`. All timings come from the
measured throughput file; nothing here is a planning prior.

## Inputs
- usd_per_hour = {params['usd_per_hour']}, T = {params['T']}, n_boxes = {params['n_boxes']}, main cap = ${params['main_cap']}, max wall = {params['max_wall_h']} h
- gen: {tp.gen_tokens_per_step:.0f} tok/step at {tp.gen_tok_per_s:.1f} tok/s; train: {tp.train_tokens_per_step:.0f} tok/step at {tp.train_tok_per_s:.1f} tok/s
- exec_s_per_rollout = {tp.exec_s_per_rollout}, cpu_workers = {tp.cpu_workers}, rollouts_per_step = {tp.rollouts_per_step}
- sync_s_per_step = {tp.sync_s_per_step}, t_eval_s = {tp.t_eval_s} (x{N_EVAL} evals), t_startup_s = {tp.t_startup_s}
- steps measured: {tp.n_steps_measured}; median step time over steps >= 3: {tp.t_step_measured_s:.2f} s
{chr(10).join(extras)}

## Per-run cost (BUDGET §2)
| quantity | value |
|---|---|
| t_gen | {cost.t_gen:.3f} s |
| t_reward | {cost.t_reward:.3f} s |
| t_train | {cost.t_train:.3f} s |
| t_sync | {cost.t_sync:.3f} s |
| t_step (source: {cost.t_step_source}) | {cost.t_step:.3f} s |
| run_s | {cost.run_s:.1f} s ({cost.run_s / 3600:.3f} h) |
| run_usd (incl. {OVERHEAD:.0%} margin) | ${cost.run_usd:.4f} |
{warn}
## Ladder (BUDGET §4)
{_md_ladder_table(rows, chosen)}

{verdict}
"""


def render_decision_md(
    cost: RunCost, rows: list[LadderRow], chosen: LadderRow | None, params: dict[str, Any]
) -> str:
    if chosen is None:
        body = (
            f"## Decision\n\nNO ladder step fits (cost cap ${params['main_cap']}, wall {params['max_wall_h']} h with "
            f"{params['n_boxes']} box(es)). Below the {FLOOR_RUNS}-run floor: **no confirmatory study**; "
            "report pilots/probes only (BUDGET §4 floor).\n"
        )
    else:
        s = chosen.step
        body = (
            f"## Decision\n\nRun ladder **step {s.step}: {s.change}** - {s.runs} runs, projected main-run cost "
            f"${chosen.main_usd:.2f} (cap ${params['main_cap']}), wall-clock {chosen.wall_h:.2f} h "
            f"(limit {params['max_wall_h']} h, {params['n_boxes']} box(es)).\n\n"
            f"Stated consequence: {s.consequence}\n\n### Seeds per arm\n\n{_seed_table(s)}\n"
        )
    return f"""# Budget decision (Gate 1b) - DRAFT

Status: **DRAFT** generated mechanically by `python -m rhg.budget cost_model`; a human must review it
before the freeze (BUDGET §3). Inputs: `{params['bench']}`.

- usd_per_hour = {params['usd_per_hour']}, T = {params['T']} (T is never a cost lever), n_boxes = {params['n_boxes']}
- run_s = {cost.run_s:.1f} s, run_usd = ${cost.run_usd:.4f} (incl. {OVERHEAD:.0%} margin)

{body}
## All ladder steps

{_md_ladder_table(rows, chosen)}
"""


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- CLI


def _cfg_defaults() -> dict[str, Any]:
    from rhg.config import load_config

    cfg = load_config("hackable_subtle")
    return {"usd_per_hour": cfg.budget.usd_per_hour, "T": cfg.grpo.max_steps}


def _cmd_check(args: argparse.Namespace) -> int:
    stop_at = args.stop_at if args.stop_at is not None else DEFAULT_STOP_AT_USD
    dec = check_launch(args.next_run_usd, stop_at, ledger=args.ledger)
    print(dec.message, file=sys.stdout if dec.allowed else sys.stderr)
    return EXIT_OK if dec.allowed else EXIT_GUARD


def _cmd_record(args: argparse.Namespace) -> int:
    e = record(args.kind, args.run_id, args.wall_s, args.usd_per_hour, args.usd, args.note, ledger=args.ledger)
    print(f"recorded {e['kind']} {e['run_id']}: ${e['usd']:.4f}")
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    by = spent_by_kind(args.ledger)
    for k in sorted(by):
        print(f"{k:10s} ${by[k]:.4f}")
    print(f"{'total':10s} ${spent_usd(args.ledger):.4f}")
    return EXIT_OK


def _cmd_cost_model(args: argparse.Namespace) -> int:
    bench = Path(args.bench)
    if not bench.is_file():
        print(
            f"error: throughput file {bench} not found.\n"
            "Run the throughput bench on the GPU box first (Gate 1b): `bash scripts/bench_throughput.sh` "
            "(i.e. `python -m rhg.eval.bench`), which writes results/bench/throughput.json, "
            "then re-run `python -m rhg.budget cost_model`.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    try:
        tp = load_throughput(bench)
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    defaults: dict[str, Any] = {}
    if args.usd_per_hour is None or args.T is None:
        try:
            defaults = _cfg_defaults()
        except Exception as e:  # noqa: BLE001 - config problems are usage errors here
            print(f"error: cannot read defaults from config ({e}); pass --usd-per-hour and --T", file=sys.stderr)
            return EXIT_USAGE
    usd_per_hour = args.usd_per_hour if args.usd_per_hour is not None else defaults["usd_per_hour"]
    T = args.T if args.T is not None else defaults["T"]
    if args.n_boxes < 1 or T < 1 or usd_per_hour < 0:
        print("error: --n-boxes and --T must be >= 1 and --usd-per-hour >= 0", file=sys.stderr)
        return EXIT_USAGE
    decision = Path(args.decision)
    if tp.mock:
        print("WARNING: the throughput file comes from `rhg.eval.bench --mock`; its numbers are NOT measurements.", file=sys.stderr)
        if Path(args.out_md) == DEFAULT_MEASURED_MD or decision == DEFAULT_DECISION_MD:
            print(
                f"refused: a mock throughput file must not produce {DEFAULT_MEASURED_MD} / {DEFAULT_DECISION_MD}; "
                "pass explicit --out-md and --decision paths (tests/dry runs only)",
                file=sys.stderr,
            )
            return EXIT_GUARD
    if decision.exists() and not args.force:
        print(f"refused: {decision} already exists; pass --force to overwrite the decision file", file=sys.stderr)
        return EXIT_GUARD
    cost = run_cost(tp, T, usd_per_hour, t_step_source=args.t_step_source)
    rows = evaluate_ladder(cost, args.n_boxes, args.main_cap, args.max_wall_h)
    chosen = select_step(rows)
    params = {
        "bench": str(bench).replace("\\", "/"), "usd_per_hour": usd_per_hour, "T": T,
        "n_boxes": args.n_boxes, "main_cap": args.main_cap, "max_wall_h": args.max_wall_h,
    }
    _write_atomic(Path(args.out_md), render_measured_md(tp, cost, rows, chosen, params))
    _write_atomic(decision, render_decision_md(cost, rows, chosen, params))
    print(f"t_step = {cost.t_step:.3f} s ({cost.t_step_source}); run_s = {cost.run_s:.1f} s; run_usd = ${cost.run_usd:.4f}")
    for r in rows:
        print(
            f"  step {r.step.step}: {r.step.runs:2d} runs  ${r.main_usd:7.2f}  {r.wall_h:6.2f} h  "
            f"cost {'ok' if r.fits_cost else 'NO'}  wall {'ok' if r.fits_wall else 'NO'}"
        )
    if chosen is None:
        print(f"NO ladder step fits: below the {FLOOR_RUNS}-run floor -> no confirmatory study")
    else:
        print(f"first fitting ladder step: {chosen.step.step} ({chosen.step.change}, {chosen.step.runs} runs)")
    print(f"wrote {args.out_md} and {decision} (draft; review before freeze)")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rhg.budget", description="Spend ledger, launch guard and cost model.")
    p.add_argument("--ledger", type=Path, default=None, help="ledger path (default: budget.ledger from config)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="exit 3 if spent + next run would exceed --stop-at")
    c.add_argument("--next-run-usd", type=float, required=True)
    c.add_argument("--stop-at", type=float, default=None, help=f"default {DEFAULT_STOP_AT_USD}")
    c.add_argument("--ledger", type=Path, default=argparse.SUPPRESS)
    c.set_defaults(fn=_cmd_check)

    r = sub.add_parser("record", help="append a metered action to the ledger")
    r.add_argument("--kind", required=True, choices=KINDS)
    r.add_argument("--run-id", required=True)
    r.add_argument("--wall-s", type=float, required=True)
    r.add_argument("--usd-per-hour", type=float, default=None)
    r.add_argument("--usd", type=float, default=None)
    r.add_argument("--note", default="")
    r.add_argument("--ledger", type=Path, default=argparse.SUPPRESS)
    r.set_defaults(fn=_cmd_record)

    s = sub.add_parser("status", help="print spend by kind")
    s.add_argument("--ledger", type=Path, default=argparse.SUPPRESS)
    s.set_defaults(fn=_cmd_status)

    m = sub.add_parser("cost_model", help="price runs from results/bench/throughput.json and pick a ladder step")
    m.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    m.add_argument("--usd-per-hour", type=float, default=None, help="default: budget.usd_per_hour from config")
    m.add_argument("--T", type=int, default=None, help="training steps (default: grpo.max_steps from config)")
    m.add_argument("--n-boxes", type=int, default=1)
    m.add_argument("--main-cap", type=float, default=DEFAULT_MAIN_CAP_USD)
    m.add_argument("--max-wall-h", type=float, default=DEFAULT_MAX_WALL_H)
    m.add_argument("--t-step-source", choices=("formula", "measured", "max"), default="formula")
    m.add_argument("--out-md", type=Path, default=DEFAULT_MEASURED_MD)
    m.add_argument("--decision", type=Path, default=DEFAULT_DECISION_MD)
    m.add_argument("--force", action="store_true", help="overwrite an existing decision file")
    m.set_defaults(fn=_cmd_cost_model)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, LedgerCorruptError, TimeoutError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE if isinstance(e, ValueError) else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
