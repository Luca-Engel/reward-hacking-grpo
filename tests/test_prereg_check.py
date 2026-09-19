import hashlib
import json
from pathlib import Path

import pytest

from fakerepo import commit_all, fake_repo, freeze_and_tag, frozen_repo, git  # noqa: F401
from rhg.analysis import prereg_check as pc
from rhg.analysis.prereg_check import CODE_GROUPS, PreregError

ARMS = pc.ARMS
CODE_FILE_FOR_GROUP = {
    "analysis": "src/rhg/analysis/stats.py",
    "env": "src/rhg/env/grader.py",
    "detect": "src/rhg/detect/ast_detector.py",
    "judge_rubric": "src/rhg/judge/rubric.py",
    "data_build": "src/rhg/data/build.py",
}


def _edit(repo: Path, rel: str, old: str, new: str) -> None:
    p = repo / rel
    text = p.read_text(encoding="utf-8")
    assert old in text
    p.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ code-group hashing


def test_hash_code_group_matches_hand_computation(tmp_path):
    d = tmp_path / "g"
    (d / "sub").mkdir(parents=True)
    (d / "b.py").write_bytes(b"B\n")
    (d / "a.py").write_bytes(b"A\n")
    (d / "sub" / "c.py").write_bytes(b"C\n")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"junk")
    (d / "sub" / "x.pyc").write_bytes(b"junk")

    def leaf(rel: str, content: bytes) -> bytes:
        return rel.encode() + b"\0" + hashlib.sha256(content).hexdigest().encode() + b"\n"

    expected = hashlib.sha256(leaf("a.py", b"A\n") + leaf("b.py", b"B\n") + leaf("sub/c.py", b"C\n")).hexdigest()
    assert pc.hash_code_group(d) == expected


def test_hash_code_group_sensitivity_and_normalisation(tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    (d / "a.py").write_bytes(b"x = 1\ny = 2\n")
    base = pc.hash_code_group(d)
    (d / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")  # CRLF checkout of the same text
    assert pc.hash_code_group(d) == base
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "z.py").write_bytes(b"ignored")
    assert pc.hash_code_group(d) == base
    (d / "a.py").write_bytes(b"x = 1\ny = 3\n")
    assert pc.hash_code_group(d) != base
    (d / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (d / "new.py").write_bytes(b"")
    assert pc.hash_code_group(d) != base
    assert pc.hash_code_group(tmp_path / "does_not_exist") is None


# ------------------------------------------------------------------ compute_freeze / write_freeze


def test_compute_freeze_contents(fake_repo):
    f = pc.compute_freeze(fake_repo / "configs", fake_repo)
    assert set(f["config_hashes"]) == set(ARMS) and len(ARMS) == 7
    assert all(len(h) == 64 for h in f["config_hashes"].values())
    assert f["prompts_sha256"] == hashlib.sha256((fake_repo / "configs/prompts.yaml").read_bytes()).hexdigest()
    assert f["split_hash"] == "splithash123"
    assert f["dataset_revision"] == "abc123rev"
    assert f["requirements_gpu_sha256"] == hashlib.sha256(b"torch==0.0.0\n").hexdigest()
    assert set(f["code_groups"]) == set(CODE_GROUPS) == {"analysis", "env", "detect", "judge_rubric", "data_build"}
    assert f["missing_code_groups"] == []
    hp = f["hyperparameters"]
    assert hp["T"] == 100 and hp["lr"] == 7.0e-5 and hp["max_completion_tokens"] == 1024
    assert hp["batch_shape"] == {"prompts_per_step": 16, "gens_per_prompt": 8}
    assert f["hint_selection"]["subtle_selected"] == "S1"
    assert f["hint_selection"]["wordings"]["none"] == ""
    assert "run_tests()" in f["hint_selection"]["wordings"]["subtle"]
    assert f["hint_selection"]["wordings"]["explicit"].startswith(f["hint_selection"]["wordings"]["subtle"])


def test_compute_freeze_reports_missing_group_and_write_refuses(fake_repo):
    git(fake_repo, "rm", "-q", "src/rhg/judge/rubric.py")
    f = pc.compute_freeze(fake_repo / "configs", fake_repo)
    assert f["code_groups"]["judge_rubric"] is None
    assert f["missing_code_groups"] == ["judge_rubric"]
    with pytest.raises(PreregError, match="judge_rubric"):
        pc.write_freeze(repo_root=fake_repo)
    assert not (fake_repo / "prereg" / "FREEZE.json").exists()


def test_compute_freeze_missing_optional_inputs_are_null(fake_repo):
    (fake_repo / "requirements-gpu.txt").unlink()
    (fake_repo / "data/processed/splits.json").unlink()
    (fake_repo / "data/processed/DATASET_REVISION").unlink()
    f = pc.compute_freeze(fake_repo / "configs", fake_repo)
    assert f["requirements_gpu_sha256"] is None and f["split_hash"] is None and f["dataset_revision"] is None


def test_write_freeze_guards(fake_repo):
    path = fake_repo / "prereg" / "FREEZE.json"
    written = pc.write_freeze(repo_root=fake_repo, gpu_type="RTX 4090")
    assert json.loads(path.read_text(encoding="utf-8"))["gpu_type"] == "RTX 4090"
    assert "missing_code_groups" not in written
    with pytest.raises(PreregError, match="exists"):
        pc.write_freeze(repo_root=fake_repo)
    pc.write_freeze(repo_root=fake_repo, force=True)
    commit_all(fake_repo, "freeze")
    git(fake_repo, "tag", "prereg-v1")
    with pytest.raises(PreregError, match="immutable"):
        pc.write_freeze(repo_root=fake_repo, force=True)


# ------------------------------------------------------------------ check: tag logic


def test_no_tag_fails(fake_repo):
    pc.write_freeze(repo_root=fake_repo)
    commit_all(fake_repo, "freeze")
    res = pc.check(fake_repo)
    assert not res.ok and not res
    assert "tag_exists" in res.failed_names and "tag_is_ancestor" in res.failed_names
    assert "FAIL" in res.format()


def test_tag_not_ancestor_fails(fake_repo):
    first = git(fake_repo, "rev-parse", "HEAD")
    git(fake_repo, "checkout", "-q", "-b", "side")
    pc.write_freeze(repo_root=fake_repo)
    commit_all(fake_repo, "freeze on side branch")
    git(fake_repo, "tag", "prereg-v1")
    git(fake_repo, "checkout", "-q", first)
    git(fake_repo, "checkout", "side", "--", "prereg/FREEZE.json")  # same freeze, so only ancestry can fail
    res = pc.check(fake_repo)
    by_name = {i.name: i for i in res.items}
    assert by_name["tag_exists"].status == "pass"
    assert by_name["tag_is_ancestor"].status == "fail"
    assert not res.ok


def test_missing_freeze_fails(fake_repo):
    git(fake_repo, "tag", "prereg-v1")
    res = pc.check(fake_repo)
    assert res.failed_names == ["freeze_exists"]


def test_tag_and_freeze_ok_passes(frozen_repo):
    res = pc.check(frozen_repo)
    assert res.ok, res.format()
    names = {i.name for i in res.items}
    assert {"tag_exists", "tag_is_ancestor", "freeze_exists", "freeze_matches_tag"} <= names
    assert {f"config:{a}" for a in ARMS} <= names
    assert {f"code:{g}" for g in CODE_GROUPS} <= names
    assert {"prompts", "hint_selection", "hyperparameters", "split", "dataset_revision", "requirements_gpu"} <= names
    assert res.warnings == [] and res.amendments == []
    assert pc.confirmatory_ok(frozen_repo)
    assert pc.main(["--repo-root", str(frozen_repo)]) == 0


def test_pycache_changes_do_not_break_the_check(frozen_repo):
    d = frozen_repo / "src/rhg/analysis/__pycache__"
    d.mkdir()
    (d / "stats.cpython-312.pyc").write_bytes(b"compiled")
    assert pc.check(frozen_repo).ok


def test_rewriting_freeze_after_tag_is_detected(frozen_repo):
    _edit(frozen_repo, "src/rhg/analysis/stats.py", "return 1", "return 99")
    freeze_path = frozen_repo / "prereg/FREEZE.json"
    data = json.loads(freeze_path.read_text(encoding="utf-8"))
    data["code_groups"]["analysis"] = pc.hash_code_group(frozen_repo / "src/rhg/analysis")
    freeze_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    res = pc.check(frozen_repo)
    assert res.failed_names == ["freeze_matches_tag"]  # hashes now agree, but the tagged file differs


# ------------------------------------------------------------------ check: drift is named


@pytest.mark.parametrize("group", sorted(CODE_FILE_FOR_GROUP))
def test_code_group_drift_names_the_group(frozen_repo, group):
    rel = CODE_FILE_FOR_GROUP[group]
    p = frozen_repo / rel
    p.write_text(p.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8", newline="\n")
    res = pc.check(frozen_repo)
    assert res.failed_names == [f"code:{group}"]
    assert group in res.format()
    assert pc.main(["--repo-root", str(frozen_repo)]) == 3


def test_new_file_in_group_is_drift(frozen_repo):
    (frozen_repo / "src/rhg/env/extra.py").write_text("X = 1\n", encoding="utf-8")
    assert pc.check(frozen_repo).failed_names == ["code:env"]


def test_deleting_a_group_after_freeze_fails(frozen_repo):
    (frozen_repo / "src/rhg/judge/rubric.py").unlink()
    res = pc.check(frozen_repo)
    assert res.failed_names == ["code:judge_rubric"]
    assert "missing" in res.failures[0].detail


def test_single_arm_config_drift(frozen_repo):
    p = frozen_repo / "configs/arms/hackable_subtle.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "sandbox: {timeout_s: 7.0}\n", encoding="utf-8", newline="\n")
    assert pc.check(frozen_repo).failed_names == ["config:hackable_subtle"]


def test_base_config_drift_names_every_arm(frozen_repo):
    _edit(frozen_repo, "configs/base.yaml", "timeout_s: 6.0", "timeout_s: 7.0")
    assert pc.check(frozen_repo).failed_names == [f"config:{a}" for a in ARMS]


def test_hyperparameter_drift_is_named_with_field(frozen_repo):
    _edit(frozen_repo, "configs/base.yaml", "lr: 7.0e-5", "lr: 5.0e-5")
    res = pc.check(frozen_repo)
    assert "hyperparameters" in res.failed_names
    assert all(f"config:{a}" in res.failed_names for a in ARMS)
    hp = next(i for i in res.failures if i.name == "hyperparameters")
    assert "lr" in hp.detail
    assert "prompts" not in res.failed_names and "code:analysis" not in res.failed_names


def test_prompts_drift(frozen_repo):
    _edit(frozen_repo, "configs/prompts.yaml", "called `run_tests()`.", "called `run_tests()` now.")
    res = pc.check(frozen_repo)
    assert "prompts" in res.failed_names and "hint_selection" in res.failed_names
    assert not any(n.startswith("code:") for n in res.failed_names)


def test_split_dataset_and_requirements_drift(frozen_repo):
    (frozen_repo / "data/processed/splits.json").write_text('{"split_hash": "other"}', encoding="utf-8")
    (frozen_repo / "data/processed/DATASET_REVISION").write_text("newrev\n", encoding="utf-8")
    (frozen_repo / "requirements-gpu.txt").write_text("torch==9.9.9\n", encoding="utf-8")
    assert set(pc.check(frozen_repo).failed_names) == {"split", "dataset_revision", "requirements_gpu"}


# ------------------------------------------------------------------ amendments


def test_amend_then_check_passes_with_warning(frozen_repo):
    freeze_before = (frozen_repo / "prereg/FREEZE.json").read_bytes()
    old = pc.hash_code_group(frozen_repo / "src/rhg/analysis")
    _edit(frozen_repo, "src/rhg/analysis/stats.py", "return 1", "return 2")
    assert not pc.check(frozen_repo).ok
    new = pc.hash_code_group(frozen_repo / "src/rhg/analysis")

    recs = pc.amend("fix off-by-one in stats", repo_root=frozen_repo, ts="2026-01-01T00:00:00+00:00")
    assert [r["group"] for r in recs] == ["analysis"]
    lines = (frozen_repo / "prereg/AMENDMENTS.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x) for x in lines] == [
        {"ts": "2026-01-01T00:00:00+00:00", "group": "analysis", "old_hash": old, "new_hash": new,
         "reason": "fix off-by-one in stats"}
    ]
    assert (frozen_repo / "prereg/FREEZE.json").read_bytes() == freeze_before  # in-memory update only

    res = pc.check(frozen_repo)
    assert res.ok, res.format()
    assert [w.name for w in res.warnings] == ["amendment:analysis"]
    assert "fix off-by-one in stats" in res.warnings[0].detail
    assert res.amendments == [json.loads(lines[0])]
    assert pc.main(["--repo-root", str(frozen_repo)]) == 0


def test_amendments_chain_and_further_drift_still_fails(frozen_repo):
    _edit(frozen_repo, "src/rhg/analysis/stats.py", "return 1", "return 2")
    pc.amend("first", repo_root=frozen_repo)
    _edit(frozen_repo, "src/rhg/analysis/stats.py", "return 2", "return 3")
    assert pc.check(frozen_repo).failed_names == ["code:analysis"]  # un-amended drift fails
    pc.amend("second", repo_root=frozen_repo)
    res = pc.check(frozen_repo)
    assert res.ok and len(res.warnings) == 2 and len(res.amendments) == 2


def test_amend_only_drifted_groups_and_cli(frozen_repo, capsys):
    _edit(frozen_repo, "src/rhg/env/grader.py", "return 2", "return 20")
    _edit(frozen_repo, "src/rhg/detect/ast_detector.py", "return 3", "return 30")
    assert pc.main(["--repo-root", str(frozen_repo), "--amend", "--reason", "grader+detector fix"]) == 0
    out = capsys.readouterr().out
    assert "amended detect" in out and "amended env" in out
    recs = pc.load_amendments(frozen_repo)
    assert sorted(r["group"] for r in recs) == ["detect", "env"]
    assert pc.check(frozen_repo).ok


def test_amend_specific_group_and_unknown_group(frozen_repo):
    _edit(frozen_repo, "src/rhg/env/grader.py", "return 2", "return 20")
    _edit(frozen_repo, "src/rhg/detect/ast_detector.py", "return 3", "return 30")
    with pytest.raises(PreregError, match="unknown"):
        pc.amend("x", groups=["nope"], repo_root=frozen_repo)
    with pytest.raises(PreregError, match="no drift"):
        pc.amend("x", groups=["analysis"], repo_root=frozen_repo)
    pc.amend("only env", groups=["env"], repo_root=frozen_repo)
    assert pc.check(frozen_repo).failed_names == ["code:detect"]


def test_amend_with_no_drift_is_a_noop(frozen_repo):
    assert pc.amend("nothing changed", repo_root=frozen_repo) == []
    assert not (frozen_repo / "prereg/AMENDMENTS.jsonl").exists()


def test_amend_before_tag_is_refused(fake_repo, capsys):
    pc.write_freeze(repo_root=fake_repo)
    commit_all(fake_repo, "freeze")  # no tag yet
    _edit(fake_repo, "src/rhg/analysis/stats.py", "return 1", "return 2")
    with pytest.raises(PreregError, match="does not exist yet"):
        pc.amend("too early", repo_root=fake_repo)
    assert pc.main(["--repo-root", str(fake_repo), "--amend", "--reason", "too early"]) == 3
    assert "refused" in capsys.readouterr().err
    assert not (fake_repo / "prereg/AMENDMENTS.jsonl").exists()


def test_amend_requires_reason(frozen_repo):
    _edit(frozen_repo, "src/rhg/env/grader.py", "return 2", "return 20")
    with pytest.raises(PreregError, match="reason"):
        pc.amend("   ", repo_root=frozen_repo)
    with pytest.raises(SystemExit) as exc:
        pc.main(["--repo-root", str(frozen_repo), "--amend"])
    assert exc.value.code == 2


def test_amend_refuses_to_amend_to_a_missing_value(frozen_repo):
    (frozen_repo / "src/rhg/judge/rubric.py").unlink()
    with pytest.raises(PreregError, match="missing"):
        pc.amend("deleted", repo_root=frozen_repo)


def test_forged_amendment_with_wrong_old_hash_breaks_the_chain(frozen_repo):
    _edit(frozen_repo, "src/rhg/analysis/stats.py", "return 1", "return 2")
    new = pc.hash_code_group(frozen_repo / "src/rhg/analysis")
    forged = {"ts": "t", "group": "analysis", "old_hash": "0" * 64, "new_hash": new, "reason": "forged"}
    path = frozen_repo / "prereg/AMENDMENTS.jsonl"
    path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
    res = pc.check(frozen_repo)
    assert res.failed_names == ["code:analysis"]
    assert "chain" in res.failures[0].detail


def test_malformed_amendments_file_raises(frozen_repo):
    (frozen_repo / "prereg/AMENDMENTS.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(PreregError, match="malformed"):
        pc.check(frozen_repo)


# ------------------------------------------------------------------ CLI misc


def test_cli_write_freeze(fake_repo, capsys):
    assert pc.main(["--repo-root", str(fake_repo), "--write-freeze", "--gpu-type", "A100"]) == 0
    assert (fake_repo / "prereg/FREEZE.json").is_file()
    assert pc.main(["--repo-root", str(fake_repo), "--write-freeze"]) == 3  # exists, no --force
    assert "refused" in capsys.readouterr().err


def test_cli_exit_code_on_failure_names_item(frozen_repo, capsys):
    _edit(frozen_repo, "src/rhg/env/grader.py", "return 2", "return 20")
    assert pc.main(["--repo-root", str(frozen_repo)]) == 3
    out = capsys.readouterr().out
    assert "[FAIL] code:env" in out and "PREREG CHECK: FAIL" in out


def test_module_runs_as_script(frozen_repo):
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "rhg.analysis.prereg_check", "--repo-root", str(frozen_repo)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PREREG CHECK: PASS" in proc.stdout
