"""Run-log schemas, writers/readers, eval accumulation and the validator (subtask 09, REPO_SPEC §6)."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from rhg import runlog
from rhg.env.labels import derive_labels
from rhg.runlog import (
    EvalAccumulator,
    Labels,
    Monitor,
    RolloutRecord,
    RolloutWriter,
    StepRecord,
    StepWriter,
    iter_rollouts,
    validate_run,
    write_evals,
    write_status,
)

# REPO_SPEC §6, copied by hand (not derived from the models under test)
SPEC_ROLLOUT_KEYS = {"run_id", "phase", "step", "eval_hint", "problem_id", "sample_idx", "completion", "n_tokens",
                     "truncated", "code_extracted", "reward", "labels", "monitor"}
SPEC_LABEL_KEYS = {"defines_rt", "rt_ok", "visible_pass", "heldout_pass", "timeout", "crash", "hack_rt", "attempt_rt",
                   "correct", "gap_other"}
SPEC_STEP_KEYS = {"step", "reward_mean", "loss", "grad_norm", "completion_len_mean", "truncation_rate",
                  "frac_zero_adv_groups", "hack_rt_rate_train", "attempt_rt_rate_train", "correct_rate_train", "t_gen",
                  "t_reward", "t_train", "t_sync", "t_step", "tokens_gen", "tokens_train"}


def make_rec(run_id="r__s0", phase="train", step=1, problem_id="p1", sample_idx=0, eval_hint=None, reward=0.0, **raw) -> RolloutRecord:
    raw_full = dict(defines_rt=False, rt_ok=False, visible_pass=False, heldout_pass=False, timeout=False, crash=False,
                    code_extracted=True)
    raw_full.update(raw)
    labels = derive_labels(raw_full)
    return RolloutRecord(
        run_id=run_id, phase=phase, step=step, eval_hint=eval_hint, problem_id=problem_id, sample_idx=sample_idx,
        completion="```python\npass\n```", n_tokens=5, truncated=False, code_extracted=raw_full["code_extracted"],
        reward=reward, labels=Labels(**labels), monitor=Monitor(ast_narrow=False, ast_broad=False, ast_categories=[]),
    )


def make_step(step: int, **kw) -> StepRecord:
    base = dict(step=step, reward_mean=0.5, loss=0.1, grad_norm=1.0, completion_len_mean=10.0, truncation_rate=0.0,
                frac_zero_adv_groups=0.25, hack_rt_rate_train=0.0, attempt_rt_rate_train=0.0, correct_rate_train=0.5,
                t_gen=1.0, t_reward=0.5, t_train=1.0, t_sync=0.0, t_step=2.5, tokens_gen=100, tokens_train=100)
    base.update(kw)
    return StepRecord(**base)


def build_run(root: Path, run_id="r__s0", status="completed") -> Path:
    """A minimal, consistent run directory."""
    d = root / run_id
    d.mkdir(parents=True)
    (d / "config.resolved.yaml").write_text("run: {}\n")
    (d / "stdout.log").write_text("log\n")
    (d / "manifest.json").write_text(json.dumps({"run_id": run_id, "status": status}))
    write_status(d, run_id, status, phase="finalize", step=2)
    with StepWriter(d) as sw:
        sw.write(make_step(1))
        sw.write(make_step(2))
    acc = EvalAccumulator(run_id, "clean_none", 0)
    with RolloutWriter(d) as rw:
        for step in (1, 2):
            rw.write_many([make_rec(run_id, "train", step, "p1", j) for j in range(2)])
        key = acc.open_point("eval_test", 0, None, "none", 2)
        recs = [
            make_rec(run_id, "eval_test", 0, "p1", 0, defines_rt=True, rt_ok=True),
            make_rec(run_id, "eval_test", 0, "p1", 1, visible_pass=True, heldout_pass=True),
            make_rec(run_id, "eval_test", 0, "p2", 0, visible_pass=True),
            make_rec(run_id, "eval_test", 0, "p2", 1, code_extracted=False),
        ]
        rw.write_many(recs)
        for r in recs:
            acc.add(key, r)
        key = acc.open_point("eval_test_xhint", 2, "subtle", "subtle", 1)
        r = make_rec(run_id, "eval_test_xhint", 2, "p1", 0, eval_hint="subtle", defines_rt=True)
        rw.write(r)
        acc.add(key, r)
    write_evals(d, acc.summary())
    return d


# ------------------------------------------------------------------ schemas
def test_rollout_record_has_exactly_the_spec_fields():
    rec = make_rec()
    dump = json.loads(rec.model_dump_json())
    assert set(dump) == SPEC_ROLLOUT_KEYS
    assert set(dump["labels"]) == SPEC_LABEL_KEYS
    assert set(dump["monitor"]) == {"ast_narrow", "ast_broad", "ast_categories"}
    assert set(json.loads(make_step(1).model_dump_json())) == SPEC_STEP_KEYS


def test_schema_rejects_bad_records():
    good = json.loads(make_rec().model_dump_json())
    for mutate in (
        lambda d: d.update(phase="eval"),
        lambda d: d.update(step=-1),
        lambda d: d.update(extra=1),
        lambda d: d.pop("labels"),
        lambda d: d["labels"].pop("hack_rt"),
        lambda d: d.update(eval_hint="subtle"),  # only allowed for eval_test_xhint
        lambda d: d.update(phase="eval_test_xhint", eval_hint=None),
        lambda d: d.update(phase="eval_test_xhint", eval_hint="loud"),
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(ValidationError):
            RolloutRecord.model_validate(bad)
    with pytest.raises(ValidationError):
        StepRecord.model_validate({**json.loads(make_step(1).model_dump_json()), "step": 0})


def test_labels_derived_consistently():
    hack = make_rec(defines_rt=True, rt_ok=True, visible_pass=True)
    assert hack.labels.hack_rt and hack.labels.attempt_rt and not hack.labels.correct
    gap = make_rec(visible_pass=True)
    assert gap.labels.gap_other and not gap.labels.hack_rt
    none = make_rec(code_extracted=False, visible_pass=True)
    assert not any(none.labels.model_dump().values())


# ------------------------------------------------------------------ writers / readers
def test_rollout_writer_roundtrip_and_filters(tmp_path):
    recs = [make_rec(phase="train", step=s, problem_id=f"p{s}") for s in (1, 2, 3)]
    recs += [make_rec(phase="eval_val", step=0, problem_id="v"), make_rec(phase="eval_test", step=3, problem_id="t"),
             make_rec(phase="eval_test_xhint", step=3, problem_id="t", eval_hint="none")]
    with RolloutWriter(tmp_path) as w:
        w.write_many(recs)
        assert w.n_written == 6
    assert (tmp_path / "rollouts.jsonl.gz").is_file()
    assert list(iter_rollouts(tmp_path)) == recs
    assert [r.step for r in iter_rollouts(tmp_path, phase="train")] == [1, 2, 3]
    assert [r.problem_id for r in iter_rollouts(tmp_path, phase="train", step=2)] == ["p2"]
    assert [r.phase for r in iter_rollouts(tmp_path, phase=("eval_val", "eval_test"))] == ["eval_val", "eval_test"]
    assert list(iter_rollouts(tmp_path, step=99)) == []


def test_rollout_writer_flush_makes_records_readable_before_close(tmp_path):
    w = RolloutWriter(tmp_path)
    w.write_many([make_rec()])
    assert len(list(iter_rollouts(tmp_path, partial_ok=True))) == 1  # writer not closed: no end-of-stream marker yet
    with pytest.raises(EOFError):
        list(iter_rollouts(tmp_path))
    w.close()
    assert len(list(iter_rollouts(tmp_path))) == 1


def test_status_and_step_files(tmp_path):
    st = write_status(tmp_path, "r__s0", "failed", reason="stall", phase="train", step=7, exit_code=75)
    assert runlog.read_status(tmp_path) == st
    with pytest.raises(ValidationError):
        write_status(tmp_path, "r__s0", "exploded")
    with StepWriter(tmp_path) as sw:
        sw.write(make_step(1))
        sw.write(make_step(2, loss=float("nan")))
    steps = runlog.read_steps(tmp_path)
    assert [s.step for s in steps] == [1, 2] and math.isnan(steps[1].loss)  # NaN stays diagnosable


# ------------------------------------------------------------------ eval accumulation
def test_eval_accumulator_matches_brute_force_counts():
    recs = []
    for pid, spec in {"a": [dict(defines_rt=True, rt_ok=True), dict(visible_pass=True, heldout_pass=True), dict()],
                      "b": [dict(visible_pass=True), dict(code_extracted=False)]}.items():
        recs += [make_rec(phase="eval_val", step=20, problem_id=pid, sample_idx=i, **raw) for i, raw in enumerate(spec)]
    acc = EvalAccumulator("r__s0", "clean_none", 0)
    key = acc.open_point("eval_val", 20, None, "none", 3)
    for r in recs:
        acc.add(key, r)
    (pt,) = acc.summary().points
    assert pt.key == "eval_val|20|none" and pt.n_rollouts == 5 and pt.prompt_hint == "none" and pt.eval_hint is None
    for pid in ("a", "b"):
        mine = [r for r in recs if r.problem_id == pid]
        row = pt.per_problem[pid]
        assert row["n"] == len(mine)
        for name in runlog.LABEL_FIELDS:
            assert row[name] == sum(getattr(r.labels, name) for r in mine), (pid, name)
        assert row["code_extracted"] == sum(r.code_extracted for r in mine)
    assert pt.totals["hack_rt"] == 1 and pt.totals["correct"] == 1 and pt.totals["gap_other"] == 1
    assert pt.totals["n"] == 5
    with pytest.raises(ValueError):
        acc.open_point("eval_val", 20, None, "none", 3)  # duplicate point


# ------------------------------------------------------------------ validator
def test_validator_accepts_consistent_run_and_cli_reports_ok(tmp_path, capsys):
    d = build_run(tmp_path)
    assert validate_run(d) == []
    assert runlog.main(["validate", str(d)]) == 0
    assert "OK" in capsys.readouterr().out


def _rewrite_rollouts(d: Path, fn) -> None:
    rows = [json.loads(line) for line in gzip.open(d / "rollouts.jsonl.gz", "rt", encoding="utf-8")]
    rows = fn(rows)
    with gzip.open(d / "rollouts.jsonl.gz", "wt", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_validator_flags_corruptions(tmp_path):
    def errs_after(mutator, name):
        d = build_run(tmp_path, run_id=name)
        mutator(d)
        return validate_run(d)

    def flip_hack(d):
        def fn(rows):
            rows[0]["labels"]["hack_rt"] = True
            return rows

        _rewrite_rollouts(d, fn)

    def dup(d):
        _rewrite_rollouts(d, lambda rows: rows + [rows[0]])

    def wrong_run_id(d):
        def fn(rows):
            rows[1]["run_id"] = "other"
            return rows

        _rewrite_rollouts(d, fn)

    def drop_eval_rollout(d):
        def fn(rows):
            i = next(i for i, r in enumerate(rows) if r["phase"] == "eval_test")
            return rows[:i] + rows[i + 1:]

        _rewrite_rollouts(d, fn)

    def tamper_evals(d):
        ev = json.loads((d / "evals.json").read_text())
        ev["points"][0]["per_problem"]["p1"]["hack_rt"] += 1
        (d / "evals.json").write_text(json.dumps(ev))

    def nan_step(d):
        rows = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
        rows[1]["loss"] = float("nan")
        (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def gap_in_steps(d):
        lines = (d / "steps.jsonl").read_text().splitlines()
        (d / "steps.jsonl").write_text(lines[1] + "\n")

    def truncated_gz(d):
        raw = (d / "rollouts.jsonl.gz").read_bytes()
        (d / "rollouts.jsonl.gz").write_bytes(raw[: len(raw) // 2])

    cases = {
        "flip_hack": (flip_hack, "inconsistent"),
        "dup": (dup, "duplicate"),
        "wrong_run_id": (wrong_run_id, "run_id"),
        "drop_eval_rollout": (drop_eval_rollout, "differ"),
        "tamper_evals": (tamper_evals, "differ"),
        "nan_step": (nan_step, "non-finite"),
        "gap_in_steps": (gap_in_steps, "not 1.."),
        "truncated_gz": (truncated_gz, "unreadable"),
        "no_evals": (lambda d: (d / "evals.json").unlink(), "missing file evals.json"),
        "no_manifest": (lambda d: (d / "manifest.json").unlink(), "missing file manifest.json"),
        "status_mismatch": (lambda d: write_status(d, d.name, "invalid", reason="x"), "status"),
    }
    for name, (mutator, needle) in cases.items():
        errs = errs_after(mutator, name)
        assert errs, f"{name}: corruption not detected"
        assert any(needle in e for e in errs), (name, errs)


def test_validator_is_lenient_about_missing_files_of_failed_runs(tmp_path):
    d = tmp_path / "r__s0"
    d.mkdir()
    (d / "config.resolved.yaml").write_text("x: 1\n")
    (d / "manifest.json").write_text(json.dumps({"run_id": "r__s0", "status": "failed"}))
    write_status(d, "r__s0", "failed", reason="stall", exit_code=75)
    assert validate_run(d) == []
    (d / "steps.jsonl").write_text('{"step": 1}\n')
    assert validate_run(d)  # but whatever exists must still be well-formed


def test_validator_cli_exit_codes(tmp_path, capsys):
    d = build_run(tmp_path)
    (d / "evals.json").write_text("{not json")
    assert runlog.main(["validate", str(d)]) == 1
    assert "INVALID" in capsys.readouterr().err
    assert runlog.main(["validate", str(tmp_path / "nope")]) == 1
    with pytest.raises(SystemExit) as e:
        runlog.main([])
    assert e.value.code == 2
