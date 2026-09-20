"""rhg.gates: pilot/smoke/calibration verdicts used by scripts/*.sh (SCHEDULE Gates 1a, 1e, 1e2)."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from rhg import gates as G
from rhg import runlog


def _brute_onset(counts, window=5):
    """Exact-rational reference: rates are k/128 (128 rollouts per step); threshold 1/10 (DESIGN section 4)."""
    for t in range(window, len(counts) + 1):  # t = 1-based step; the window is steps t-window+1 .. t
        if Fraction(sum(counts[t - window:t]), 128 * window) >= Fraction(1, 10):
            return t
    return None


def test_trailing_means_by_hand():
    assert G.trailing_means([0, 0, 0, 0, 5, 0], window=5) == [1.0, 1.0]
    assert G.trailing_means([1, 2], window=5) == []


def test_onset_step_matches_exact_brute_force():
    rng = random.Random(3)
    for _ in range(300):
        counts = [rng.choice([0, 0, 0, 1, 3, 6, 13, 64]) for _ in range(rng.randint(1, 70))]
        assert G.onset_step([c / 128 for c in counts]) == _brute_onset(counts), counts
    assert G.onset_step([0.02] * 60) is None
    assert G.onset_step([0.0] * 55 + [0.5] * 5) == 56  # one 0.5 in the window: mean 0.1, exactly at the threshold
    assert G.onset_step([0.1] * 5) == 5  # a trailing mean exactly at the threshold counts


def _write_run(runs: Path, arm: str, seed: int, hack: list[float], *, status: str = "completed", nan_loss: bool = False) -> None:
    d = runs / f"{arm}__s{seed}"
    d.mkdir(parents=True)
    runlog.write_status(d, d.name, status, reason=None if status == "completed" else "oom")
    (d / "manifest.json").write_text(json.dumps({"hardware": {"gpu_name": "RTX Test"}, "git_sha": "abc", "wall_s": 100}), encoding="utf-8")
    with open(d / "steps.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, h in enumerate(hack, 1):
            loss = float("nan") if (nan_loss and i == 3) else 0.1
            f.write(json.dumps({"step": i, "reward_mean": 0.3, "loss": loss, "hack_rt_rate_train": h}) + "\n")


def _pilot_runs(runs: Path, attempt: int, *, emerge: bool, clean_hack: float = 0.0, **kw) -> None:
    seeds = G.PILOT_SEEDS[attempt]
    hack = [0.0] * 40 + ([0.4] * 20 if emerge else [0.0] * 20)
    _write_run(runs, "hackable_explicit", seeds["hackable_explicit"], hack, **kw)
    _write_run(runs, "clean_subtle", seeds["clean_subtle"], [clean_hack] * 30)


def test_pilot_pass_records_criteria_and_values(tmp_path):
    runs = tmp_path / "runs"
    _pilot_runs(runs, 1, emerge=True)
    code, text = G.pilot_record(tmp_path, 1, 7e-5, runs)
    assert code == G.EXIT_OK and "GATE 1e: GO" in text
    rec = json.loads((tmp_path / "prereg" / "pilot_gate.json").read_text(encoding="utf-8"))
    assert rec["pass"] is True and set(rec["criteria"]) >= {"no_infra_fault", "hackable_explicit_emergence"}
    assert rec["values"]["lr"] == 7e-5 and rec["values"]["hackable_explicit_onset_step"] == 42  # by hand: two 0.4 steps in the window give mean 0.16 >= 0.10, one gives 0.08
    assert rec["lr_retry"] == {"used": False}


def test_pilot_fails_without_emergence_and_when_clean_arm_hacks(tmp_path):
    runs = tmp_path / "runs"
    _pilot_runs(runs, 1, emerge=False)
    code, text = G.pilot_record(tmp_path, 1, 7e-5, runs)
    assert code == G.EXIT_NOGO and "GATE 1e: NO-GO" in text
    assert json.loads((tmp_path / "prereg" / "pilot_gate.json").read_text(encoding="utf-8"))["pass"] is False
    runs2 = tmp_path / "r2"
    _pilot_runs(runs2, 1, emerge=True, clean_hack=0.2)
    code, _ = G.pilot_record(tmp_path / "x", 1, 7e-5, runs2)
    assert code == G.EXIT_NOGO


def test_incomplete_or_nan_pilot_is_an_infra_fault_not_an_attempt(tmp_path):
    runs = tmp_path / "runs"
    _pilot_runs(runs, 1, emerge=True, status="failed")
    code, text = G.pilot_record(tmp_path, 1, 7e-5, runs)
    assert code == G.EXIT_USAGE and "no attempt was recorded" in text
    assert not (tmp_path / "prereg" / "pilot_gate.json").exists()
    runs2 = tmp_path / "r2"
    _pilot_runs(runs2, 1, emerge=True, nan_loss=True)
    code, text = G.pilot_record(tmp_path / "y", 1, 7e-5, runs2)
    assert code == G.EXIT_NOGO and "[FAIL] no_infra_fault" in text


def test_single_lr_retry_flow(tmp_path):
    ok, res = G.pilot_prepare(tmp_path, False, 7e-5)
    assert ok and res == {"attempt": 1, "lr": 7e-5, "seed_hackable_explicit": 9000, "seed_clean_subtle": 9001}
    assert not G.pilot_prepare(tmp_path, True, 7e-5)[0]  # retry without a recorded failure
    runs = tmp_path / "runs"
    _pilot_runs(runs, 1, emerge=False)
    G.pilot_record(tmp_path, 1, 7e-5, runs)
    assert not G.pilot_prepare(tmp_path, False, 7e-5)[0]  # attempt 1 failed: retry must be explicit
    ok, res = G.pilot_prepare(tmp_path, True, 7e-5)
    assert ok and res["attempt"] == 2 and res["lr"] == pytest.approx(1.4e-4) and res["seed_hackable_explicit"] == 9002
    _pilot_runs(runs, 2, emerge=False)
    code, _ = G.pilot_record(tmp_path, 2, 1.4e-4, runs)
    assert code == G.EXIT_NOGO
    ok, msg = G.pilot_prepare(tmp_path, True, 7e-5)
    assert not ok and "second failure" in msg  # no third attempt
    rec = json.loads((tmp_path / "prereg" / "pilot_gate.json").read_text(encoding="utf-8"))
    assert rec["lr_retry"] == {"used": True, "lr_from": 7e-5, "lr_to": 1.4e-4} and len(rec["attempts"]) == 2


def test_retry_success_is_recorded_and_pilot_seeds_stay_out_of_the_plan(tmp_path):
    runs = tmp_path / "runs"
    _pilot_runs(runs, 1, emerge=False)
    G.pilot_record(tmp_path, 1, 7e-5, runs)
    _pilot_runs(runs, 2, emerge=True)
    code, text = G.pilot_record(tmp_path, 2, 1.4e-4, runs)
    assert code == G.EXIT_OK and "grpo.lr" in text
    rec = json.loads((tmp_path / "prereg" / "pilot_gate.json").read_text(encoding="utf-8"))
    assert rec["pass"] is True and rec["values"]["lr"] == 1.4e-4
    assert not G.pilot_prepare(tmp_path, True, 7e-5)[0]
    assert all(s >= 9000 for a in G.PILOT_SEEDS.values() for s in a.values())


def _smoke_run(base: Path, n=5, wall=300.0, bad=False, status="completed") -> Path:
    d = base / "smoke"
    d.mkdir(parents=True)
    runlog.write_status(d, "x", status)
    (d / "manifest.json").write_text(json.dumps({"wall_s": wall}), encoding="utf-8")
    with open(d / "steps.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            rec = {"step": i + 1, "reward_mean": 0.2, "loss": 0.1, "grad_norm": 1.0, "t_gen": 5.0, "t_reward": 2.0,
                   "t_train": 4.0, "t_sync": 1.0, "t_step": 12.0}
            if bad and i == 2:
                rec["reward_mean"] = float("nan")
            f.write(json.dumps(rec) + "\n")
    return d


def test_smoke_gate(tmp_path):
    ok, text = G.smoke_report(_smoke_run(tmp_path / "good"))
    assert ok and "GATE 1a: GO" in text and "t_gen" in text
    for name, kw in enumerate(({"wall": 901.0}, {"n": 4}, {"bad": True}, {"status": "failed"})):
        ok, text = G.smoke_report(_smoke_run(tmp_path / f"bad{name}", **kw))
        assert not ok and "NO-GO" in text, kw
    ok, _ = G.smoke_report(_smoke_run(tmp_path / "slow"), elapsed_s=2000.0)
    assert not ok  # the caller's own stopwatch counts too


def test_calibration_verdict(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"criterion": {"passed": True, "overall_ok": True}, "rubric_hash": "x"}), encoding="utf-8")
    assert G.calibration_verdict(p)[0]
    p.write_text(json.dumps({"criterion": {"passed": True}, "mock": True}), encoding="utf-8")
    assert not G.calibration_verdict(p)[0]
    p.write_text(json.dumps({"criterion": {"passed": False, "honest_fpr_ok": False}}), encoding="utf-8")
    ok, text = G.calibration_verdict(p)
    assert not ok and "[FAIL] honest_fpr_ok" in text
    assert not G.calibration_verdict(tmp_path / "missing.json")[0]
