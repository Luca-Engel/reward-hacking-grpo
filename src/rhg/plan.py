"""Run plan: which runs, in what order, how to shard, cut and replace them (DESIGN §3/§7, BUDGET §3-§5).

``configs/plan.yaml`` holds the planned runs as data; the cut ladder is ``rhg.budget.LADDER`` (imported, never
copied). CLI: ``python -m rhg.plan {list,shard,ladder,replacement,next,reconcile,health,next-run-usd,record-missing}``.

Priority order (BUDGET §4): tier 0 = the primary contrast (``hackable_subtle`` / ``clean_subtle``) as H/C pairs by
seed, in a pre-declared shuffled seed order (the hackable/clean order alternates from pair to pair so that
round-robin sharding puts both arms on every box), then ``hackable_explicit``, ``clean_explicit``,
``hackable_subtle_ast``, ``clean_none``, ``hackable_none``. Ladder step *N* keeps, per arm, the first
``LADDER[N].seeds[arm]`` runs of that arm in priority order, so a ladder step is a prefix per arm and running out of
money mid-way cuts the tail exactly like the ladder does.

Shards are 1-based (``--shard 1/2``): run *j* of the FULL priority list belongs to shard ``j % n + 1``. The
assignment is made before the ladder filter, so cutting the ladder never moves a run to another box.

Replacements: seed ``seed_base + k`` (k < ``max_total``) for a run whose ``status.json`` is ``invalid`` or ``failed``
for a reason that is not a deterministic setup error. Shard ``i`` of ``n`` may only use ``k % n == i - 1``, so boxes
that cannot talk to each other never collide and the total over all boxes cannot exceed the cap. Grants are recorded
in ``results/replacements.jsonl``. Decisions depend on validity only, never on outcomes (no metric is read).
Pilot seeds (>= 9000) never appear in the plan.

Health output (``results/RUNS_HEALTH.md``) deliberately holds status, steps, wall time, cost and the final *training*
reward only: no hack rate, by arm or otherwise (DESIGN §7.4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from rhg.budget import LADDER
from rhg.seeds import derive_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "configs" / "plan.yaml"
DEFAULT_RUNS_DIR = Path("results/runs")
DEFAULT_GRANTS = Path("results/replacements.jsonl")
DEFAULT_HEALTH = Path("results/RUNS_HEALTH.md")
DEFAULT_STALE_AFTER_S = 1800.0  # 2 x run.step_timeout_s (900): a live run rewrites status.json every step
STATES = ("pending", "running", "completed", "invalid", "failed")
# failed-run reasons that are deterministic setup/code errors: a fresh seed cannot fix them, so no replacement.
NON_REPLACEABLE_REASONS = ("PromptTooLongError", "NotImplementedError", "ConfigError", "RunSetupError",
                           "ConfirmatoryGuardError")
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_GUARD = 0, 1, 2, 3


class PlanError(ValueError):
    """The plan file is inconsistent with the design (or a CLI argument is invalid)."""


@dataclass(frozen=True)
class PlannedRun:
    arm: str
    seed: int
    priority: int  # index in the full 22-run priority list; -1 for replacement runs
    tier: int  # 0 = primary, 1.. = tail arms in order; -1 for replacement runs
    replaces: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.arm}__s{self.seed}"


@dataclass(frozen=True)
class Plan:
    arms: dict[str, int]
    primary_hackable: str
    primary_clean: str
    shuffle_seed: int
    tail: tuple[str, ...]
    replacement_seed_base: int
    replacement_cap: int
    pilot_seed_min: int


def load_plan(path: str | Path | None = None) -> Plan:
    """Read ``configs/plan.yaml`` and check it against ``rhg.budget.LADDER`` step 0 and DESIGN §3."""
    p = Path(path) if path is not None else PLAN_PATH
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        arms = {str(k): int(v) for k, v in raw["arms"].items()}
        prim = raw["primary"]
        plan = Plan(
            arms=arms,
            primary_hackable=str(prim["hackable"]),
            primary_clean=str(prim["clean"]),
            shuffle_seed=int(prim["shuffle_seed"]),
            tail=tuple(str(a) for a in raw["tail"]),
            replacement_seed_base=int(raw["replacement"]["seed_base"]),
            replacement_cap=int(raw["replacement"]["max_total"]),
            pilot_seed_min=int(raw["pilot_seed_min"]),
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise PlanError(f"{p}: cannot read plan: {e!r}") from e
    if plan.arms != LADDER[0].seeds:
        raise PlanError(f"{p}: arms {plan.arms} differ from rhg.budget.LADDER[0].seeds {LADDER[0].seeds}")
    if {plan.primary_hackable, plan.primary_clean, *plan.tail} != set(plan.arms) or len(plan.tail) != len(plan.arms) - 2:
        raise PlanError(f"{p}: primary + tail must list every arm exactly once")
    if plan.arms[plan.primary_hackable] != plan.arms[plan.primary_clean]:
        raise PlanError(f"{p}: the primary arms need equal seed counts (paired launch)")
    if not 0 < plan.replacement_cap <= 3:
        raise PlanError(f"{p}: replacement.max_total must be in 1..3 (DESIGN §7.2 hard cap is 3)")
    if plan.replacement_seed_base < max(plan.arms.values()) or plan.replacement_seed_base + plan.replacement_cap > plan.pilot_seed_min:
        raise PlanError(f"{p}: replacement seeds must lie between the planned seeds and the pilot seeds")
    return plan


def primary_seed_order(plan: Plan) -> list[int]:
    n = plan.arms[plan.primary_hackable]
    return sorted(range(n), key=lambda s: (derive_seed(plan.shuffle_seed, f"primary_order:{s}"), s))


def priority_list(plan: Plan | None = None) -> list[PlannedRun]:
    """The full design (22 runs) in launch-priority order."""
    plan = plan or load_plan()
    runs: list[tuple[str, int, int]] = []
    for k, seed in enumerate(primary_seed_order(plan)):
        pair = (plan.primary_hackable, plan.primary_clean)
        for arm in pair if k % 2 == 0 else pair[::-1]:
            runs.append((arm, seed, 0))
    for tier, arm in enumerate(plan.tail, start=1):
        runs.extend((arm, seed, tier) for seed in range(plan.arms[arm]))
    return [PlannedRun(arm, seed, i, tier) for i, (arm, seed, tier) in enumerate(runs)]


def ladder_runs(step: int, plan: Plan | None = None) -> list[PlannedRun]:
    """Runs kept at ladder step ``step`` (``rhg.budget.LADDER``), in priority order."""
    if not 0 <= step < len(LADDER):
        raise PlanError(f"ladder step must be in 0..{len(LADDER) - 1}, got {step}")
    keep = LADDER[step].seeds
    taken: dict[str, int] = {}
    out = []
    for r in priority_list(plan):
        if taken.get(r.arm, 0) < keep[r.arm]:
            taken[r.arm] = taken.get(r.arm, 0) + 1
            out.append(r)
    return out


def parse_shard(spec: str) -> tuple[int, int]:
    try:
        i_s, n_s = spec.split("/")
        i, n = int(i_s), int(n_s)
    except ValueError as e:
        raise PlanError(f"--shard must look like I/N (1-based), got {spec!r}") from e
    _check_shard(i, n)
    return i, n


def _check_shard(i: int, n: int) -> None:
    if n < 1 or not 1 <= i <= n:
        raise PlanError(f"shard {i}/{n} is invalid: need 1 <= i <= n")


def shard(i: int, n: int, ladder: int = 0, plan: Plan | None = None) -> list[PlannedRun]:
    """Shard ``i`` (1-based) of ``n``: round-robin over the full priority list, then the ladder filter."""
    _check_shard(i, n)
    kept = {r.run_id for r in ladder_runs(ladder, plan)}
    return [r for r in priority_list(plan) if r.priority % n == i - 1 and r.run_id in kept]


# ---------------------------------------------------------------------------- run state


@dataclass(frozen=True)
class RunState:
    state: str  # STATES
    reason: str | None = None
    exit_code: int | None = None
    updated_at: str | None = None


def read_state(run_id: str, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> RunState:
    """State of a run from its ``status.json``; a missing directory/file is ``pending``, a corrupt one ``failed``."""
    path = Path(runs_dir) / run_id / "status.json"
    if not path.is_file():
        return RunState("pending")
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        st = d["status"]
    except (OSError, ValueError, KeyError, TypeError):
        return RunState("failed", "status.json unreadable")
    if st not in STATES:
        return RunState("failed", f"unknown status {st!r}")
    code = d.get("exit_code")
    return RunState(st, d.get("reason"), code if isinstance(code, int) else None, d.get("updated_at"))


def is_replaceable(state: RunState) -> bool:
    """Invalid, or failed for anything except a deterministic setup error (DESIGN §7.2: crash/OOM/NaN/preemption)."""
    if state.state == "invalid":
        return True
    if state.state != "failed":
        return False
    reason = state.reason or ""
    return not any(reason.startswith(p) for p in NON_REPLACEABLE_REASONS)


def is_infrastructure(state: RunState) -> bool:
    """Stall watchdog (exit code 75), OOM and preemption-like reasons; used only to label log lines."""
    r = (state.reason or "").lower()
    return state.exit_code == 75 or any(w in r for w in ("stall", "oom", "out of memory", "memory_not_released", "interrupted", "preempt"))


# ---------------------------------------------------------------------------- replacements


def read_grants(path: str | Path = DEFAULT_GRANTS) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rec = json.loads(line)
                rec["run_id"], rec["replaces"], int(rec["k"])
            except (ValueError, KeyError, TypeError) as e:
                raise PlanError(f"{p}:{n}: corrupt replacement grant: {e!r}") from e
            out.append(rec)
    return out


def allowed_k(i: int, n: int, plan: Plan | None = None) -> list[int]:
    plan = plan or load_plan()
    _check_shard(i, n)
    return [k for k in range(plan.replacement_cap) if k % n == i - 1]


@dataclass(frozen=True)
class QueueEntry:
    run: PlannedRun
    state: RunState


def queue(i: int, n: int, ladder: int, runs_dir: str | Path = DEFAULT_RUNS_DIR, grants: str | Path = DEFAULT_GRANTS,
          plan: Plan | None = None) -> list[QueueEntry]:
    """The shard's runs in priority order, each failed replaceable run substituted by its granted replacement (chained)."""
    plan = plan or load_plan()
    by_replaced = {g["replaces"]: g for g in read_grants(grants)}
    out = []
    for r in shard(i, n, ladder, plan):
        cur, st = r, read_state(r.run_id, runs_dir)
        while st.state in ("invalid", "failed") and is_replaceable(st) and cur.run_id in by_replaced:
            g = by_replaced[cur.run_id]
            cur = PlannedRun(cur.arm, plan.replacement_seed_base + int(g["k"]), -1, -1, replaces=cur.run_id)
            st = read_state(cur.run_id, runs_dir)
        out.append(QueueEntry(cur, st))
    return out


def next_run(i: int, n: int, ladder: int, runs_dir: str | Path = DEFAULT_RUNS_DIR, grants: str | Path = DEFAULT_GRANTS,
             plan: Plan | None = None) -> PlannedRun | None:
    """First run of the shard that has not been started (``completed`` and dead runs are skipped, and so is a
    ``running`` one: call :func:`reconcile_stale` first if a crashed box may have left one behind)."""
    return next((e.run for e in queue(i, n, ladder, runs_dir, grants, plan) if e.state.state == "pending"), None)


@dataclass(frozen=True)
class Decision:
    failed_run_id: str
    arm: str
    granted: bool
    k: int | None
    seed: int | None
    reason: str

    @property
    def replacement_id(self) -> str | None:
        return None if self.seed is None else f"{self.arm}__s{self.seed}"


def decide_replacements(i: int, n: int, ladder: int, runs_dir: str | Path = DEFAULT_RUNS_DIR,
                        grants: str | Path = DEFAULT_GRANTS, plan: Plan | None = None) -> list[Decision]:
    """Decisions for every dead, replaceable, not yet replaced run of the shard (priority order). Pure: writes nothing."""
    plan = plan or load_plan()
    used = {int(g["k"]) for g in read_grants(grants)}
    free = [k for k in allowed_k(i, n, plan) if k not in used]
    out = []
    for e in queue(i, n, ladder, runs_dir, grants, plan):
        if e.state.state not in ("invalid", "failed"):
            continue
        if not is_replaceable(e.state):
            out.append(Decision(e.run.run_id, e.run.arm, False, None, None,
                                f"not replaceable: deterministic error ({e.state.reason})"))
        elif not free:
            out.append(Decision(e.run.run_id, e.run.arm, False, None, None,
                                f"replacement cap reached for shard {i}/{n} (k in {allowed_k(i, n, plan)}, max {plan.replacement_cap} in total)"))
        else:
            k = free.pop(0)
            out.append(Decision(e.run.run_id, e.run.arm, True, k, plan.replacement_seed_base + k,
                                f"{e.state.state}: {e.state.reason or 'no reason recorded'}"))
    return out


def apply_decisions(decisions: Iterable[Decision], grants: str | Path = DEFAULT_GRANTS) -> int:
    """Append the granted decisions to the grants file; returns how many were written."""
    granted = [d for d in decisions if d.granted]
    if granted:
        p = Path(grants)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            for d in granted:
                f.write(json.dumps({"ts": _now(), "replaces": d.failed_run_id, "run_id": d.replacement_id, "k": d.k,
                                    "reason": d.reason}) + "\n")
    return len(granted)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------- stale runs, health, cost


def reconcile_stale(run_ids: Iterable[str], runs_dir: str | Path = DEFAULT_RUNS_DIR,
                    stale_after_s: float = DEFAULT_STALE_AFTER_S, now: datetime | None = None) -> list[str]:
    """Mark ``running`` runs whose status has not been rewritten for ``stale_after_s`` as ``invalid`` (a killed or
    preempted box leaves ``running`` behind). ``stale_after_s=0`` marks every ``running`` run: only use it when no
    other process is training on this box. Returns the ids that were marked."""
    from rhg import manifest as manifest_mod
    from rhg import runlog

    now = now or datetime.now(timezone.utc)
    marked = []
    for rid in run_ids:
        st = read_state(rid, runs_dir)
        if st.state != "running":
            continue
        try:
            age = (now - datetime.fromisoformat(st.updated_at)).total_seconds() if st.updated_at else math.inf
        except ValueError:
            age = math.inf
        if age < stale_after_s:
            continue
        run_dir = Path(runs_dir) / rid
        reason = f"preempted/killed: status stuck at 'running' for {age:.0f}s"
        runlog.write_status(run_dir, rid, "invalid", reason=reason, exit_code=None)
        try:
            manifest_mod.finalize_manifest(run_dir, "invalid", invalid_reason=reason)
        except (OSError, KeyError, ValueError):
            pass  # no readable manifest: status.json is the record
        marked.append(rid)
    return marked


def _last_step(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "steps.jsonl"
    if not p.is_file():
        return None
    last = None
    with open(p, encoding="utf-8", newline="\n") as f:
        for line in f:
            if line.strip():
                try:
                    last = json.loads(line)
                except ValueError:
                    return None
    return last


def health_rows(entries: Sequence[PlannedRun], runs_dir: str | Path = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    rows = []
    for r in entries:
        run_dir = Path(runs_dir) / r.run_id
        st = read_state(r.run_id, runs_dir)
        try:
            man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            man = {}
        last = _last_step(run_dir)
        rows.append({
            "run_id": r.run_id, "arm": r.arm, "seed": r.seed, "replaces": r.replaces, "status": st.state,
            "reason": st.reason, "steps": None if last is None else last.get("step"),
            "wall_s": man.get("wall_s"), "usd": man.get("usd"),
            "final_train_reward": None if last is None else last.get("reward_mean"),
        })
    return rows


def render_health(rows: Sequence[dict[str, Any]], *, title: str = "RUNS_HEALTH") -> str:
    def f(v: Any, fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    lines = [
        f"# {title}", "",
        "Health only: validity, cost, step count and final *training* reward. No hack rate is shown, by arm or "
        "otherwise (DESIGN §7.4).", "",
        "| run | status | steps | wall (min) | usd | final train reward |", "|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = r["run_id"] + (f" (replaces {r['replaces']})" if r["replaces"] else "")
        wall = None if r["wall_s"] is None else r["wall_s"] / 60.0
        lines.append(f"| {name} | {r['status']} | {f(r['steps'], 'd')} | {f(wall, '.1f')} | {f(r['usd'], '.3f')} | "
                     f"{f(r['final_train_reward'], '.3f')} |")
    bad = [r for r in rows if r["status"] in ("invalid", "failed")]
    lines += ["", f"Completed: {sum(r['status'] == 'completed' for r in rows)} / {len(rows)}; "
                  f"invalid or failed: {len(bad)}."]
    if bad:
        lines += ["", "## Invalid / failed runs", ""]
        lines += [f"- {r['run_id']}: {r['status']} - {r['reason'] or 'no reason recorded'}" for r in bad]
    return "\n".join(lines) + "\n"


def estimate_next_run_usd(arm: str) -> tuple[float, str]:
    """Projected cost of one run for the launch guard: the measured cost model of ``results/bench/throughput.json``
    (BUDGET §2, same as the guard inside ``rhg.train.run``), else the planning-prior ceiling (UNVERIFIED)."""
    from rhg.config import load_config
    from rhg.train import run as train_run

    cfg = load_config(arm)
    usd = train_run.estimate_run_usd(cfg, mock=False)
    src = "measured cost model (results/bench/throughput.json)" if train_run.BENCH_PATH.is_file() else \
        "planning prior upper bound, UNVERIFIED (no bench yet)"
    return usd, src


def record_missing_ledger(run_id: str, arm: str, wall_s: float, ledger: str | Path | None = None) -> bool:
    """Append a ``train`` ledger entry for a run that left none (process killed before ``rhg.train.run`` could bill it)."""
    from rhg import budget
    from rhg.config import load_config

    if any(e["run_id"] == run_id and e["kind"] == "train" for e in budget.read_entries(ledger)):
        return False
    cfg = load_config(arm)
    budget.record("train", run_id, wall_s, cfg.budget.usd_per_hour, note="recorded by rhg.plan: run left no ledger entry",
                  ledger=ledger or cfg.budget.ledger)
    return True


# ---------------------------------------------------------------------------- CLI


def _runs_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--grants", type=Path, default=DEFAULT_GRANTS, help=f"default {DEFAULT_GRANTS}")


def _sel_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shard", default="1/1", help="I/N, 1-based (default 1/1)")
    p.add_argument("--ladder", type=int, default=0, help="ladder step 0..6 (rhg.budget.LADDER); default 0")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhg.plan", description="Run plan, shards, ladder and replacements.")
    ap.add_argument("--plan", type=Path, default=None, help=f"plan file (default {PLAN_PATH.relative_to(REPO_ROOT)})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="the runs of a shard/ladder step in priority order")
    _sel_args(p), _runs_arg(p)
    p.add_argument("--format", choices=("table", "tsv", "ids", "json"), default="table")
    p.add_argument("--state", choices=STATES, default=None, help="only runs in this state (with replacements applied)")

    p = sub.add_parser("shard", help="the runs of shard I/N (ladder step 0 unless --ladder)")
    p.add_argument("spec", metavar="I/N")
    p.add_argument("--ladder", type=int, default=0)
    p.add_argument("--format", choices=("table", "tsv", "ids", "json"), default="table")

    p = sub.add_parser("ladder", help="print the ladder; with N, the runs kept at step N")
    p.add_argument("step", type=int, nargs="?", default=None)
    p.add_argument("--format", choices=("table", "tsv", "ids", "json"), default="table")

    p = sub.add_parser("replacement", help="replacement decisions for dead runs of the shard (--apply records the grants)")
    _sel_args(p), _runs_arg(p)
    p.add_argument("--apply", action="store_true", help="append the granted replacements to the grants file")

    p = sub.add_parser("next", help="the next run of the shard that has not started (empty output = none left)")
    _sel_args(p), _runs_arg(p)
    p.add_argument("--format", choices=("tsv", "ids", "json"), default="tsv")

    p = sub.add_parser("reconcile", help="mark stale 'running' runs invalid (killed/preempted box)")
    _sel_args(p), _runs_arg(p)
    p.add_argument("--stale-after", type=float, default=DEFAULT_STALE_AFTER_S, help="seconds; 0 = every running run")

    p = sub.add_parser("health", help="write the health-only dashboard")
    _sel_args(p), _runs_arg(p)
    p.add_argument("--out", type=Path, default=DEFAULT_HEALTH)

    p = sub.add_parser("next-run-usd", help="projected cost of one run of ARM for `rhg.budget check`")
    p.add_argument("--arm", default="hackable_subtle")

    p = sub.add_parser("record-missing", help="ledger entry for a run that left none")
    p.add_argument("--run-id", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--wall-s", type=float, required=True)
    p.add_argument("--ledger", type=Path, default=None)
    return ap


def _emit(runs: Sequence[PlannedRun], fmt: str, states: dict[str, str] | None = None) -> None:
    states = states or {}
    if fmt == "ids":
        print("\n".join(r.run_id for r in runs))
    elif fmt == "tsv":
        for r in runs:
            print("\t".join([str(r.priority), str(r.tier), r.arm, str(r.seed), r.run_id, states.get(r.run_id, "")]))
    elif fmt == "json":
        print(json.dumps([{"priority": r.priority, "tier": r.tier, "arm": r.arm, "seed": r.seed, "run_id": r.run_id,
                           "replaces": r.replaces, "state": states.get(r.run_id)} for r in runs], indent=2))
    else:
        print(f"{'prio':>4} {'tier':>4}  {'run_id':<28} state")
        for r in runs:
            print(f"{r.priority:>4} {r.tier:>4}  {r.run_id:<28} {states.get(r.run_id, '')}")
        print(f"{len(runs)} run(s)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except PlanError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE


def _sel(args: argparse.Namespace) -> tuple[int, int]:
    return parse_shard(args.shard)


def _dispatch(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    cmd = args.cmd
    if cmd == "list":
        i, n = _sel(args)
        q = queue(i, n, args.ladder, args.runs_dir, args.grants, plan)
        entries = [e for e in q if args.state is None or e.state.state == args.state]
        _emit([e.run for e in entries], args.format, {e.run.run_id: e.state.state for e in q})
    elif cmd == "shard":
        i, n = parse_shard(args.spec)
        _emit(shard(i, n, args.ladder, plan), args.format)
    elif cmd == "ladder":
        if args.step is None:
            for s in LADDER:
                seeds = ", ".join(f"{a}={c}" for a, c in s.seeds.items())
                print(f"step {s.step}: {s.runs:2d} runs  {s.change}  [{seeds}]")
                print(f"        consequence: {s.consequence}")
        else:
            runs = ladder_runs(args.step, plan)
            if args.format == "table":
                print(f"ladder step {args.step}: {LADDER[args.step].change} ({len(runs)} runs)")
            _emit(runs, args.format)
    elif cmd == "replacement":
        i, n = _sel(args)
        decisions = decide_replacements(i, n, args.ladder, args.runs_dir, args.grants, plan)
        if not decisions:
            print("no dead runs need a replacement decision")
        for d in decisions:
            verdict = f"GRANTED {d.replacement_id} (k={d.k})" if d.granted else "DENIED"
            print(f"{d.failed_run_id}: {verdict} - {d.reason}")
        if args.apply:
            print(f"recorded {apply_decisions(decisions, args.grants)} grant(s) in {args.grants}")
    elif cmd == "next":
        i, n = _sel(args)
        r = next_run(i, n, args.ladder, args.runs_dir, args.grants, plan)
        if r is not None:
            _emit([r], args.format)
    elif cmd == "reconcile":
        i, n = _sel(args)
        ids = [e.run.run_id for e in queue(i, n, args.ladder, args.runs_dir, args.grants, plan)]
        marked = reconcile_stale(ids, args.runs_dir, args.stale_after)
        for rid in marked:
            print(f"marked invalid (stale running): {rid}")
    elif cmd == "health":
        i, n = _sel(args)
        runs = [e.run for e in queue(i, n, args.ladder, args.runs_dir, args.grants, plan)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_health(health_rows(runs, args.runs_dir)), encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    elif cmd == "next-run-usd":
        usd, src = estimate_next_run_usd(args.arm)
        print(f"{usd:.6f}")
        print(f"source: {src}", file=sys.stderr)
    elif cmd == "record-missing":
        wrote = record_missing_ledger(args.run_id, args.arm, args.wall_s, args.ledger)
        print("ledger entry recorded" if wrote else "ledger already has an entry for this run")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
