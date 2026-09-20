"""rhg.plan: the run list, priority order, ladder, shards, replacements, `next`, health output."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from rhg import plan as P
from rhg import runlog
from rhg.budget import LADDER

# BUDGET §4 / DESIGN §3 constants, written out independently of rhg.budget.LADDER.
SEEDS_PER_ARM = {"clean_none": 2, "clean_subtle": 5, "clean_explicit": 2, "hackable_none": 2, "hackable_subtle": 5,
                 "hackable_explicit": 3, "hackable_subtle_ast": 3}
LADDER_RUNS = [22, 21, 19, 16, 14, 13, 11]
TAIL_ORDER = ["hackable_explicit", "clean_explicit", "hackable_subtle_ast", "clean_none", "hackable_none"]


def _status(runs_dir: Path, run_id: str, status: str, reason: str | None = None, exit_code: int | None = None) -> None:
    d = runs_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    runlog.write_status(d, run_id, status, reason=reason, exit_code=exit_code)


@pytest.fixture
def plan() -> P.Plan:
    return P.load_plan()


def test_total_and_seeds_per_arm(plan):
    runs = P.priority_list(plan)
    assert len(runs) == 22
    assert len({r.run_id for r in runs}) == 22
    counts = Counter(r.arm for r in runs)
    assert dict(counts) == SEEDS_PER_ARM
    for arm, n in SEEDS_PER_ARM.items():
        assert sorted(r.seed for r in runs if r.arm == arm) == list(range(n))
    assert [r.priority for r in runs] == list(range(22))


def test_ladder_counts_and_consistency_with_budget(plan):
    assert [s.runs for s in LADDER] == LADDER_RUNS  # single source of truth agrees with BUDGET §4
    prev: set[str] | None = None
    for step, want in enumerate(LADDER_RUNS):
        runs = P.ladder_runs(step, plan)
        assert len(runs) == want
        assert dict(Counter(r.arm for r in runs)) == {a: c for a, c in LADDER[step].seeds.items() if c}
        ids = {r.run_id for r in runs}
        if prev is not None:
            assert ids < prev  # every step only removes runs
        prev = ids
    with pytest.raises(P.PlanError):
        P.ladder_runs(7, plan)
    with pytest.raises(P.PlanError):
        P.ladder_runs(-1, plan)


def test_ladder_cuts_the_expected_runs(plan):
    ids = lambda s: {r.run_id for r in P.ladder_runs(s, plan)}  # noqa: E731
    assert ids(0) - ids(1) == {"hackable_none__s1"}
    assert ids(1) - ids(2) == {"clean_none__s0", "clean_none__s1"}
    assert {r for r in ids(2) - ids(3)} == {f"hackable_subtle_ast__s{i}" for i in range(3)}
    assert ids(3) - ids(4) == {"clean_explicit__s0", "clean_explicit__s1"}
    assert ids(4) - ids(5) == {"hackable_explicit__s2"}
    cut = ids(5) - ids(6)
    assert len(cut) == 2 and {r.split("__")[0] for r in cut} == {"hackable_subtle", "clean_subtle"}
    assert len({r.split("__s")[1] for r in cut}) == 1  # one primary pair (same seed) is dropped


def test_priority_order_tier0_pairs_then_tail(plan):
    runs = P.priority_list(plan)
    head, tail = runs[:10], runs[10:]
    assert all(r.arm in ("hackable_subtle", "clean_subtle") and r.tier == 0 for r in head)
    for a, b in zip(head[0::2], head[1::2]):
        assert a.seed == b.seed and {a.arm, b.arm} == {"hackable_subtle", "clean_subtle"}
    order = [a.seed for a in head[0::2]]
    assert sorted(order) == list(range(5)) and order == P.primary_seed_order(plan)
    firsts = [a.arm for a in head[0::2]]
    assert firsts[0] != firsts[1]  # H/C order alternates from pair to pair
    assert all(x != y for x, y in zip(firsts, firsts[1:]))
    arms_in_tail = [r.arm for r in tail]
    seen = [a for i, a in enumerate(arms_in_tail) if i == 0 or a != arms_in_tail[i - 1]]
    assert seen == TAIL_ORDER
    assert [r.tier for r in tail] == [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    for arm in TAIL_ORDER:
        assert [r.seed for r in tail if r.arm == arm] == sorted(r.seed for r in tail if r.arm == arm)


def test_priority_list_is_deterministic_and_shuffle_seed_matters(plan, tmp_path):
    assert P.priority_list(plan) == P.priority_list(P.load_plan())
    orders = set()
    raw = yaml.safe_load(P.PLAN_PATH.read_text(encoding="utf-8"))
    for s in range(20260920, 20260920 + 12):
        raw["primary"]["shuffle_seed"] = s
        f = tmp_path / f"plan{s}.yaml"
        f.write_text(yaml.safe_dump(raw), encoding="utf-8")
        orders.add(tuple(P.primary_seed_order(P.load_plan(f))))
    assert len(orders) > 1  # the seed actually drives the shuffle


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("ladder", range(7))
def test_shards_are_an_exact_cover_in_priority_order(plan, n, ladder):
    shards = [P.shard(i, n, ladder, plan) for i in range(1, n + 1)]
    flat = [r.run_id for s in shards for r in s]
    assert len(flat) == len(set(flat))
    assert set(flat) == {r.run_id for r in P.ladder_runs(ladder, plan)}
    for s in shards:
        assert [r.priority for r in s] == sorted(r.priority for r in s)
    if ladder == 0:
        assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_shard_assignment_is_stable_under_the_ladder(plan):
    full = {r.run_id: i for i in (1, 2, 3) for r in P.shard(i, 3, 0, plan)}
    for ladder in range(1, 7):
        for i in (1, 2, 3):
            for r in P.shard(i, 3, ladder, plan):
                assert full[r.run_id] == i


def test_shards_split_the_primary_arms_across_boxes(plan):
    for i in (1, 2):
        arms = {r.arm for r in P.shard(i, 2, 0, plan) if r.tier == 0}
        assert arms == {"hackable_subtle", "clean_subtle"}


@pytest.mark.parametrize("bad", ["0/2", "3/2", "1/0", "x", "1-2", "1/2/3", "-1/2"])
def test_bad_shard_specs(bad):
    with pytest.raises(P.PlanError):
        P.parse_shard(bad)


def test_parse_shard():
    assert P.parse_shard("2/3") == (2, 3)


def test_pilot_seeds_never_in_plan_or_replacements(plan):
    assert all(r.seed < 9000 for r in P.priority_list(plan))
    assert plan.replacement_seed_base + plan.replacement_cap <= 9000
    assert plan.pilot_seed_min == 9000


def test_plan_file_validation(tmp_path):
    raw = yaml.safe_load(P.PLAN_PATH.read_text(encoding="utf-8"))
    bad = {**raw, "arms": {**raw["arms"], "clean_none": 3}}
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(P.PlanError, match="LADDER"):
        P.load_plan(f)
    for patch in ({"replacement": {"seed_base": 100, "max_total": 4}}, {"replacement": {"seed_base": 8999, "max_total": 3}}):
        f.write_text(yaml.safe_dump({**raw, **patch}), encoding="utf-8")
        with pytest.raises(P.PlanError):
            P.load_plan(f)
    with pytest.raises(P.PlanError):
        P.load_plan(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------- state, next, replacements


def test_read_state(tmp_path):
    assert P.read_state("x__s0", tmp_path).state == "pending"
    _status(tmp_path, "x__s0", "completed")
    assert P.read_state("x__s0", tmp_path).state == "completed"
    (tmp_path / "y__s0").mkdir()
    (tmp_path / "y__s0" / "status.json").write_text("{not json", encoding="utf-8")
    assert P.read_state("y__s0", tmp_path).state == "failed"


def test_next_skips_completed_and_running_and_follows_priority(plan, tmp_path):
    runs = tmp_path / "runs"
    grants = tmp_path / "grants.jsonl"
    order = P.shard(1, 2, 0, plan)
    assert P.next_run(1, 2, 0, runs, grants, plan) == order[0]
    _status(runs, order[0].run_id, "completed")
    _status(runs, order[1].run_id, "running")
    assert P.next_run(1, 2, 0, runs, grants, plan) == order[2]
    for r in order:
        _status(runs, r.run_id, "completed")
    assert P.next_run(1, 2, 0, runs, grants, plan) is None
    # the other shard is unaffected
    assert P.next_run(2, 2, 0, runs, grants, plan) == P.shard(2, 2, 0, plan)[0]


def test_next_respects_the_ladder(plan, tmp_path):
    runs = tmp_path / "runs"
    kept = {r.run_id for r in P.ladder_runs(6, plan)}
    for r in P.priority_list(plan):
        if r.run_id in kept:
            _status(runs, r.run_id, "completed")
    assert P.next_run(1, 1, 6, runs, tmp_path / "g.jsonl", plan) is None
    assert P.next_run(1, 1, 0, runs, tmp_path / "g.jsonl", plan) is not None


def test_replacement_only_for_invalid_or_failed(plan, tmp_path):
    runs, grants = tmp_path / "runs", tmp_path / "g.jsonl"
    order = P.shard(1, 1, 0, plan)
    _status(runs, order[0].run_id, "completed")
    _status(runs, order[1].run_id, "running")
    assert P.decide_replacements(1, 1, 0, runs, grants, plan) == []
    _status(runs, order[2].run_id, "invalid", reason="NaN loss at step 40")
    _status(runs, order[3].run_id, "failed", reason="oom", exit_code=1)
    _status(runs, order[4].run_id, "failed", reason="stall: no heartbeat for 900s", exit_code=75)
    ds = P.decide_replacements(1, 1, 0, runs, grants, plan)
    assert [d.failed_run_id for d in ds] == [order[2].run_id, order[3].run_id, order[4].run_id]
    assert all(d.granted for d in ds)
    assert [d.seed for d in ds] == [100, 101, 102]  # 100 + k
    assert [d.replacement_id for d in ds] == [f"{order[i].arm}__s10{i - 2}" for i in (2, 3, 4)]
    assert P.decide_replacements(1, 1, 0, runs, grants, plan) == ds  # pure: nothing was written


def test_replacement_hard_cap_of_three(plan, tmp_path):
    runs, grants = tmp_path / "runs", tmp_path / "g.jsonl"
    order = P.shard(1, 1, 0, plan)
    for r in order[:5]:
        _status(runs, r.run_id, "failed", reason="oom")
    ds = P.decide_replacements(1, 1, 0, runs, grants, plan)
    assert [d.granted for d in ds] == [True, True, True, False, False]
    assert "cap" in ds[3].reason
    assert P.apply_decisions(ds, grants) == 3
    assert [g["k"] for g in P.read_grants(grants)] == [0, 1, 2]
    # a later call (e.g. after a fourth failure) can never grant more
    again = P.decide_replacements(1, 1, 0, runs, grants, plan)
    assert not any(d.granted for d in again)
    # and a replacement that itself fails does not get a fourth seed either
    q = P.queue(1, 1, 0, runs, grants, plan)
    assert [e.run.seed for e in q[:3]] == [100, 101, 102]
    _status(runs, q[0].run.run_id, "invalid", reason="preempted")
    assert not any(d.granted for d in P.decide_replacements(1, 1, 0, runs, grants, plan))


def test_replacement_cap_holds_across_boxes(plan, tmp_path):
    """Boxes cannot see each other: shard i of n may only use k = i-1 (mod n), so 3 boxes total <= 3 grants."""
    for n in (2, 3):
        used = []
        for i in range(1, n + 1):
            runs, grants = tmp_path / f"r{n}{i}", tmp_path / f"g{n}{i}.jsonl"
            for r in P.shard(i, n, 0, plan)[:5]:
                _status(runs, r.run_id, "failed", reason="oom")
            ds = P.decide_replacements(i, n, 0, runs, grants, plan)
            granted = [d for d in ds if d.granted]
            assert all(d.k % n == i - 1 for d in granted)
            used += [d.k for d in granted]
        assert len(used) == len(set(used)) <= plan.replacement_cap
        assert sorted(used) == [0, 1, 2]


def test_no_replacement_for_deterministic_errors(plan, tmp_path):
    runs, grants = tmp_path / "runs", tmp_path / "g.jsonl"
    r0 = P.shard(1, 1, 0, plan)[0]
    _status(runs, r0.run_id, "failed", reason="PromptTooLongError: prompt of 900 tokens")
    (d,) = P.decide_replacements(1, 1, 0, runs, grants, plan)
    assert not d.granted and "deterministic" in d.reason
    assert P.apply_decisions([d], grants) == 0 and not grants.exists()


def test_queue_substitutes_replacement_and_next_launches_it(plan, tmp_path):
    runs, grants = tmp_path / "runs", tmp_path / "g.jsonl"
    order = P.shard(1, 1, 0, plan)
    _status(runs, order[0].run_id, "failed", reason="stall", exit_code=75)
    assert P.next_run(1, 1, 0, runs, grants, plan) == order[1]  # dead run is never re-launched under its own id
    P.apply_decisions(P.decide_replacements(1, 1, 0, runs, grants, plan), grants)
    nxt = P.next_run(1, 1, 0, runs, grants, plan)
    assert nxt.arm == order[0].arm and nxt.seed == 100 and nxt.replaces == order[0].run_id
    _status(runs, nxt.run_id, "completed")
    assert P.next_run(1, 1, 0, runs, grants, plan) == order[1]


def test_classification_helpers():
    assert P.is_infrastructure(P.RunState("failed", "stall watchdog", 75))
    assert P.is_infrastructure(P.RunState("failed", "oom"))
    assert not P.is_infrastructure(P.RunState("failed", "ValueError"))
    assert P.is_replaceable(P.RunState("invalid", "NaN"))
    assert not P.is_replaceable(P.RunState("completed"))
    assert not P.is_replaceable(P.RunState("pending"))


def test_reconcile_stale_running(tmp_path):
    runs = tmp_path / "runs"
    _status(runs, "a__s0", "running")
    _status(runs, "b__s0", "running")
    _status(runs, "c__s0", "completed")
    later = datetime.now(timezone.utc) + timedelta(seconds=4000)
    assert P.reconcile_stale(["a__s0", "b__s0", "c__s0", "d__s0"], runs, stale_after_s=1800, now=later) == ["a__s0", "b__s0"]
    st = P.read_state("a__s0", runs)
    assert st.state == "invalid" and "killed" in (st.reason or "")
    assert P.is_replaceable(st)
    _status(runs, "e__s0", "running")
    assert P.reconcile_stale(["e__s0"], runs, stale_after_s=1800) == []  # fresh heartbeat is left alone


def test_health_shows_no_hack_rate(tmp_path):
    runs = tmp_path / "runs"
    d = runs / "hackable_subtle__s0"
    _status(runs, "hackable_subtle__s0", "completed")
    (d / "manifest.json").write_text(json.dumps({"wall_s": 3600.0, "usd": 0.5}), encoding="utf-8")
    (d / "steps.jsonl").write_text(
        json.dumps({"step": 1, "reward_mean": 0.2, "hack_rt_rate_train": 0.9}) + "\n"
        + json.dumps({"step": 2, "reward_mean": 0.4321, "hack_rt_rate_train": 0.9}) + "\n", encoding="utf-8")
    _status(runs, "clean_subtle__s0", "invalid", reason="NaN loss")
    rows = P.health_rows([P.PlannedRun("hackable_subtle", 0, 0, 0), P.PlannedRun("clean_subtle", 0, 1, 0)], runs)
    text = P.render_health(rows)
    assert "0.432" in text and "60.0" in text and "0.500" in text
    assert "0.9" not in text  # the planted hack_rt_rate_train never reaches the dashboard
    assert "hack_rt" not in text and "attempt" not in text.lower()
    assert "| clean_subtle__s0 | invalid |" in text and "clean_subtle__s0: invalid - NaN loss" in text


def test_cli_list_shard_ladder_next(tmp_path, capsys):
    runs = tmp_path / "runs"
    assert P.main(["ladder"]) == 0
    out = capsys.readouterr().out
    for n in LADDER_RUNS:
        assert f"{n:2d} runs" in out
    for step, want in enumerate(LADDER_RUNS):
        assert P.main(["ladder", str(step), "--format", "ids"]) == 0
        assert len(capsys.readouterr().out.split()) == want
    assert P.main(["shard", "2/3", "--format", "ids"]) == 0
    assert capsys.readouterr().out.split() == [r.run_id for r in P.shard(2, 3)]
    assert P.main(["shard", "4/3"]) == P.EXIT_USAGE
    capsys.readouterr()
    assert P.main(["next", "--shard", "1/2", "--runs-dir", str(runs), "--grants", str(tmp_path / "g")]) == 0
    first = capsys.readouterr().out.split("\t")
    assert first[4] == P.shard(1, 2)[0].run_id
    assert P.main(["list", "--shard", "1/2", "--runs-dir", str(runs), "--format", "tsv", "--state", "completed"]) == 0
    assert capsys.readouterr().out == ""
    assert P.main(["replacement", "--runs-dir", str(runs), "--grants", str(tmp_path / "g")]) == 0
    assert "no dead runs" in capsys.readouterr().out
