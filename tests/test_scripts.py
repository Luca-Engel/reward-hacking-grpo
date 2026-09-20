"""scripts/*.sh: syntax, conventions, --dry-run behaviour, refusals. Nothing here trains, calls an API or needs a GPU:
python is replaced by RHG_PYTHON (the test interpreter or a recording stub) and the GPU check by a shell function."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from rhg import plan as P

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
EXPECTED = ["_common", "analyze", "bench_throughput", "calibrate_judge", "freeze_prereg", "judge_all", "measure_pass_rate",
            "package_results", "pilot", "probe_hints", "run_all", "run_arm", "setup_box", "smoke"]
GPU_SCRIPTS = ["setup_box", "smoke", "bench_throughput", "measure_pass_rate", "probe_hints", "pilot", "run_arm", "run_all"]
FAKE_KEY = "sk-ant-FAKE-KEY-FOR-TESTS"


def _find_bash() -> str | None:
    if os.name == "nt":  # `bash` on PATH may be the WSL launcher, which cannot read these paths
        for cand in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if Path(cand).is_file():
                return cand
    return shutil.which("bash")


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash found")


def _posix(p: Path) -> str:
    return p.as_posix()


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "BASH_ENV")}
    env["RHG_PYTHON"] = _posix(Path(sys.executable))
    env.update(extra)
    return env


def sh(script: Path, *args: str, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    assert BASH
    return subprocess.run([BASH, _posix(script), *args], capture_output=True, text=True, env=env or _env(),
                          cwd=cwd or script.parents[1], timeout=240, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------- static checks (no bash needed)


def test_all_expected_scripts_exist_and_nothing_else():
    assert sorted(p.stem for p in SCRIPTS.glob("*.sh")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_lf_endings_strict_mode_and_shebang(name):
    raw = (SCRIPTS / f"{name}.sh").read_bytes()
    assert b"\r" not in raw, "CRLF line endings would break the script on Linux"
    text = raw.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    if name != "_common":
        assert 'source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"' in text
        assert "--dry-run" in text


def test_gitattributes_forces_lf_for_shell_scripts():
    assert "*.sh text eol=lf" in (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", [n for n in EXPECTED if n != "_common"])
def test_scripts_never_export_the_api_key_and_local_scripts_check_it(name):
    text = (SCRIPTS / f"{name}.sh").read_text(encoding="utf-8")
    assert "export ANTHROPIC_API_KEY" not in text
    if name in ("judge_all", "calibrate_judge"):
        assert "require_local_machine" in text and "ANTHROPIC_API_KEY" in text
    if name in GPU_SCRIPTS:
        assert "drop_api_key" in text


def test_mock_and_real_train_paths_are_not_mixed_up():
    """The GPU scripts must use the trl backend (never --mock) and main runs must be confirmatory."""
    for name in GPU_SCRIPTS:
        assert "--mock" not in (SCRIPTS / f"{name}.sh").read_text(encoding="utf-8")
    assert "--confirmatory" in (SCRIPTS / "run_arm.sh").read_text(encoding="utf-8")
    assert "--pilot" in (SCRIPTS / "pilot.sh").read_text(encoding="utf-8")
    assert "run.tag=pilot" in (SCRIPTS / "pilot.sh").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------- bash checks


@needs_bash
@pytest.mark.parametrize("name", EXPECTED)
def test_bash_syntax(name):
    r = subprocess.run([BASH, "-n", _posix(SCRIPTS / f"{name}.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


DRY_ARGS = {
    "setup_box": [], "smoke": [], "bench_throughput": [], "measure_pass_rate": [], "probe_hints": [], "pilot": [],
    "run_arm": ["hackable_subtle", "0"], "run_all": [], "package_results": [],
}


@needs_bash
@pytest.mark.parametrize("name", GPU_SCRIPTS + ["package_results"])
def test_dry_run_prints_commands_and_exits_zero(name):
    r = sh(SCRIPTS / f"{name}.sh", *DRY_ARGS[name], "--dry-run")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "[dry-run]" in r.stdout or name == "package_results"
    if name == "package_results":
        assert "[dry-run] tar" in r.stdout or "[dry-run] mkdir" in r.stdout


@needs_bash
def test_dry_run_variants():
    r = sh(SCRIPTS / "measure_pass_rate.sh", "--generate-only", "--dry-run")
    assert r.returncode == 0 and "--stage A --generate-only" in r.stdout and "--grade-only" in r.stdout
    r = sh(SCRIPTS / "measure_pass_rate.sh", "--dry-run")
    assert r.returncode == 0
    assert "--stage split --select-only" in r.stdout and "--stage split --strict" in r.stdout
    r = sh(SCRIPTS / "pilot.sh", "--dry-run")
    assert r.returncode == 0
    assert "--seed 9000" in r.stdout and "--seed 9001" in r.stdout and "--steps 60" in r.stdout and "--steps 30" in r.stdout
    assert "run.tag=pilot" in r.stdout and "--pilot" in r.stdout
    r = sh(SCRIPTS / "smoke.sh", "--dry-run")
    assert "--steps 5" in r.stdout and "hackable_subtle" in r.stdout and "Gate 1a" in r.stdout
    r = sh(SCRIPTS / "bench_throughput.sh", "--usd-per-hour", "0.42", "--n-boxes", "2", "--dry-run")
    assert "budget.usd_per_hour=0.42" in r.stdout and "cost_model --usd-per-hour 0.42 --n-boxes 2" in r.stdout
    r = sh(SCRIPTS / "bench_throughput.sh")  # the rate is mandatory outside a dry run
    assert r.returncode == 2


@needs_bash
def test_pilot_retry_needs_a_recorded_failed_attempt(tmp_path):
    chk = _checkout(tmp_path / "chk")
    r = sh(chk / "scripts" / "pilot.sh", "--lr-retry", "--dry-run", cwd=chk)  # nothing recorded -> refused
    assert r.returncode == 3 and "attempt 1" in r.stderr


@needs_bash
def test_setup_box_dry_run_covers_the_setup_steps():
    r = sh(SCRIPTS / "setup_box.sh", "--dry-run")
    assert r.returncode == 0
    for needle in ("uv sync", "uv pip install -r requirements-gpu.txt", "--stage fetch", "--stage tests", "--stage validate"):
        assert needle in r.stdout
    assert "HF_HOME" in r.stdout


@needs_bash
def test_api_key_is_warned_about_and_never_printed():
    r = sh(SCRIPTS / "setup_box.sh", "--dry-run", env=_env(ANTHROPIC_API_KEY=FAKE_KEY))
    assert r.returncode == 0
    assert "ANTHROPIC_API_KEY is set" in r.stderr
    assert FAKE_KEY not in r.stdout + r.stderr


@needs_bash
@pytest.mark.parametrize("shard,ladder", [("1/2", 0), ("2/2", 0), ("1/1", 0), ("2/3", 3), ("3/3", 6)])
def test_run_all_dry_run_lists_the_shard_without_executing(shard, ladder, tmp_path):
    runs_dir = REPO_ROOT / "results" / "runs"
    before = sorted(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else []
    r = sh(SCRIPTS / "run_all.sh", "--shard", shard, "--ladder", str(ladder), "--dry-run")
    assert r.returncode == 0, r.stderr
    i, n = P.parse_shard(shard)
    want = [x.run_id for x in P.shard(i, n, ladder)]
    got = [ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith("[dry-run] RUN ")]
    assert got == want
    launches = [ln.split() for ln in r.stdout.splitlines() if "bash scripts/run_arm.sh" in ln]
    assert [f"{a[-2]}__s{a[-1]}" for a in launches] == want
    assert r.stdout.count("rhg.budget check --next-run-usd") == len(want)
    assert "rhg.plan health" in r.stdout
    after = sorted(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else []
    assert before == after


@needs_bash
def test_run_all_argument_errors():
    for bad in (["--shard", "3/2"], ["--shard", "x"], ["--ladder", "7"], ["--nope"]):
        r = sh(SCRIPTS / "run_all.sh", *bad, "--dry-run")
        assert r.returncode == 2, (bad, r.stderr)


@needs_bash
def test_run_all_refuses_without_the_prereg_tag_ancestor(tmp_path):
    """Real (non-dry) run in a checkout without the prereg-v1 tag: refuse (exit 3) before any GPU/training step."""
    chk = _checkout(tmp_path / "chk")
    r = sh(chk / "scripts" / "run_all.sh", cwd=chk)
    assert r.returncode == 3 and "prereg-v1" in r.stderr


@needs_bash
def test_run_arm_guards():
    r = sh(SCRIPTS / "run_arm.sh", "hackable_subtle", "0", "--dry-run")
    assert r.returncode == 0
    assert "--arm hackable_subtle --seed 0 --backend trl --confirmatory" in r.stdout
    assert sh(SCRIPTS / "run_arm.sh", "nonsense", "0", "--dry-run").returncode == 2
    assert sh(SCRIPTS / "run_arm.sh", "hackable_subtle", "-3", "--dry-run").returncode == 2
    assert sh(SCRIPTS / "run_arm.sh", "hackable_subtle", "9000", "--dry-run").returncode == 2  # pilot seed: never confirmatory
    assert sh(SCRIPTS / "run_arm.sh", "hackable_subtle").returncode == 2  # missing seed
    r = sh(SCRIPTS / "run_arm.sh", "hackable_subtle", "9000", "--exploratory", "--dry-run")
    assert r.returncode == 0 and "--confirmatory" not in r.stdout


# ---------------------------------------------------------------------------- local-machine scripts


def _gpu_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """An env in which `nvidia-smi -L` succeeds (a shell function loaded through BASH_ENV)."""
    f = tmp_path / "fake_gpu.sh"
    f.write_text("nvidia-smi() { echo 'GPU 0: Fake (UUID: x)'; return 0; }\nexport -f nvidia-smi\n", encoding="utf-8", newline="\n")
    return _env(BASH_ENV=_posix(f), **extra)


@needs_bash
@pytest.mark.parametrize("name", ["judge_all", "calibrate_judge"])
def test_local_scripts_refuse_without_key(name):
    r = sh(SCRIPTS / f"{name}.sh", "--yes", env=_env())
    assert r.returncode == 3, r.stdout + r.stderr
    assert "ANTHROPIC_API_KEY" in r.stderr


@needs_bash
@pytest.mark.parametrize("name", ["judge_all", "calibrate_judge"])
def test_local_scripts_refuse_on_a_gpu_box(name, tmp_path):
    r = sh(SCRIPTS / f"{name}.sh", "--yes", "--dry-run", env=_gpu_env(tmp_path, ANTHROPIC_API_KEY=FAKE_KEY))
    assert r.returncode == 3, r.stdout + r.stderr
    assert "NVIDIA" in r.stderr
    assert FAKE_KEY not in r.stdout + r.stderr


def _checkout(dst: Path, *, stub: Path | None = None) -> Path:
    """A minimal checkout (scripts + configs) so scripts run against a directory we control."""
    dst.mkdir(parents=True)
    shutil.copytree(SCRIPTS, dst / "scripts")
    shutil.copytree(REPO_ROOT / "configs", dst / "configs")
    return dst


def _stub_python(path: Path, log: Path) -> str:
    """RHG_PYTHON stub: records every invocation, delegates `rhg.plan` to the real interpreter, fails prereg_check."""
    real = _posix(Path(sys.executable))
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{_posix(log)}"\n'
        'case "$*" in\n'
        f'  "-m rhg.plan"*) exec "{real}" "$@" ;;\n'
        '  "-m rhg.analysis.prereg_check"*) exit 3 ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n", encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return _posix(path)


@needs_bash
def test_judge_all_flow_with_a_stub_python(tmp_path):
    chk = _checkout(tmp_path / "chk")
    runs = chk / "results" / "runs"
    from rhg import runlog

    for rid in ("hackable_subtle__s1", "clean_subtle__s1"):
        (runs / rid).mkdir(parents=True)
        runlog.write_status(runs / rid, rid, "completed")
    (runs / "clean_subtle__s2").mkdir(parents=True)
    runlog.write_status(runs / "clean_subtle__s2", "clean_subtle__s2", "invalid", reason="NaN")
    log = tmp_path / "calls.log"
    env = _env(RHG_PYTHON=_stub_python(tmp_path / "pystub", log), ANTHROPIC_API_KEY=FAKE_KEY)

    r = sh(chk / "scripts" / "judge_all.sh", env=env, cwd=chk)  # no --yes: estimate only, then refuse
    assert r.returncode == 3, r.stdout + r.stderr
    calls = log.read_text(encoding="utf-8")
    assert "--estimate-only" in calls and "--real" not in calls
    assert "clean_subtle__s1" in calls and "hackable_subtle__s1" in calls and "clean_subtle__s2" not in calls
    assert "--confirmatory" not in calls  # prereg_check fails in the stub -> EXPLORATORY labels
    assert "--yes" in r.stderr

    log.unlink()
    r = sh(chk / "scripts" / "judge_all.sh", "--yes", env=env, cwd=chk)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if "rhg.judge.run" in ln]
    assert len(lines) == 2 and "--estimate-only" in lines[0] and "--real" in lines[1]
    assert FAKE_KEY not in r.stdout + r.stderr + log.read_text(encoding="utf-8")


@needs_bash
def test_calibrate_judge_requires_yes(tmp_path):
    chk = _checkout(tmp_path / "chk")
    log = tmp_path / "calls.log"
    env = _env(RHG_PYTHON=_stub_python(tmp_path / "pystub", log), ANTHROPIC_API_KEY=FAKE_KEY)
    r = sh(chk / "scripts" / "calibrate_judge.sh", env=env, cwd=chk)
    assert r.returncode == 3 and "--yes" in r.stderr
    assert not log.exists() or "rhg.validate.calibrate" not in log.read_text(encoding="utf-8")
    r = sh(chk / "scripts" / "calibrate_judge.sh", "--yes", env=env, cwd=chk)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rhg.validate.calibrate --client anthropic --yes" in log.read_text(encoding="utf-8")


@needs_bash
def test_analyze_passes_confirmatory_only_if_prereg_check_passes(tmp_path):
    r = sh(SCRIPTS / "analyze.sh", "--dry-run")  # the real prereg_check fails here (no freeze in the working tree)
    assert r.returncode == 0
    assert "rhg.validate.harness" in r.stdout and "rhg.analysis.run" in r.stdout
    assert "--confirmatory" not in r.stdout
    chk = _checkout(tmp_path / "chk")
    stub = tmp_path / "ok_stub"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    r = sh(chk / "scripts" / "analyze.sh", "--dry-run", env=_env(RHG_PYTHON=_posix(stub)), cwd=chk)
    assert r.returncode == 0 and "rhg.analysis.run --runs results/runs --out results/analysis --confirmatory" in r.stdout


@needs_bash
def test_package_results_dry_run_excludes_adapters_and_prints_transfer_hints(tmp_path):
    chk = _checkout(tmp_path / "chk")
    run = chk / "results" / "runs" / "clean_subtle__s0"
    (run / "adapters" / "step_0").mkdir(parents=True)
    (run / "adapters" / "step_0" / "adapter_model.safetensors").write_bytes(b"x" * 10)
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    r = sh(chk / "scripts" / "package_results.sh", "--out", "out/pack.tar.gz", cwd=chk)
    assert r.returncode == 0, r.stderr
    assert "rsync" in r.stdout and "scp" in r.stdout
    names = subprocess.run(["tar", "-tzf", "out/pack.tar.gz"], cwd=chk, capture_output=True, text=True).stdout.split()
    assert any(n.endswith("manifest.json") for n in names)
    assert not any("adapter" in n for n in names)


# ---------------------------------------------------------------------------- freeze_prereg.sh

PUSH_1, PUSH_2 = "git push origin HEAD", "git push origin prereg-v1"
PREREQ_KEYS = ["hint_selection.json", "budget_decision.md", "pilot_gate.json", "splits.json", "judge_calibration.json"]


def _git_env(**extra: str) -> dict[str, str]:
    return _env(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com",
                GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="commit.gpgsign", GIT_CONFIG_VALUE_0="false", **extra)


def _freeze_repo(tmp_path: Path) -> Path:
    from fakerepo import build_repo, commit_all  # tests/fakerepo.py

    root = build_repo(tmp_path / "repo")
    shutil.copytree(SCRIPTS, root / "scripts")
    commit_all(root, "add scripts")
    (root / "data" / "processed" / "splits.json").unlink()  # gitignored artefact that fakerepo pre-creates; a gate output here
    return root


def _write_prereqs(root: Path, *, only: set[str] | None = None, pilot_pass: bool = True, rubric_hash: str | None = None) -> None:
    import yaml

    from rhg.config import load_config
    from rhg.judge.rubric import rubric_hash as current_hash

    cfg = load_config("hackable_subtle", config_dir=root / "configs")
    subtle = yaml.safe_load((root / "configs" / "prompts.yaml").read_text(encoding="utf-8"))["subtle_selected"]
    files = {
        "prereg/hint_selection.json": json.dumps({"selected_id": subtle, "mock": False}),
        "prereg/budget_decision.md": "# Budget decision\nfull design fits\n",
        "prereg/pilot_gate.json": json.dumps({"pass": pilot_pass, "criteria": {}, "values": {"lr": cfg.grpo.lr, "gpu_name": "RTX Test"}}),
        "data/processed/splits.json": json.dumps({"split_hash": "abc"}),
        "results/analysis/judge_calibration.json": json.dumps(
            {"passed": True, "criterion": {"passed": True}, "rubric_hash": rubric_hash or current_hash()}),
    }
    for rel, text in files.items():
        if only is not None and not any(rel.endswith(k) for k in only):
            continue
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")


@needs_bash
def test_freeze_dry_run_lists_every_missing_prerequisite(tmp_path):
    root = _freeze_repo(tmp_path)
    r = sh(root / "scripts" / "freeze_prereg.sh", "--dry-run", env=_git_env(), cwd=root)
    assert r.returncode != 0
    out = r.stdout
    for key in PREREQ_KEYS:
        assert f"[MISSING]" in out and key in out
    assert out.count("[MISSING]") == 5 and out.count("[OK]") >= 2  # tag-free and clean tree are fine
    assert "Publish the freeze" not in out  # nothing to publish yet
    assert not (root / "prereg" / "FREEZE.json").exists()


@needs_bash
@pytest.mark.parametrize("missing", PREREQ_KEYS)
def test_freeze_refuses_when_any_single_artifact_is_missing(tmp_path, missing):
    root = _freeze_repo(tmp_path)
    _write_prereqs(root, only=set(PREREQ_KEYS) - {missing})
    r = sh(root / "scripts" / "freeze_prereg.sh", "--dry-run", env=_git_env(), cwd=root)
    assert r.returncode != 0
    assert r.stdout.count("[MISSING]") == 1 and missing in [ln for ln in r.stdout.splitlines() if "[MISSING]" in ln][0]


@needs_bash
def test_freeze_refuses_failed_pilot_stale_rubric_dirty_tree(tmp_path):
    root = _freeze_repo(tmp_path)
    _write_prereqs(root, pilot_pass=False)
    r = sh(root / "scripts" / "freeze_prereg.sh", "--dry-run", env=_git_env(), cwd=root)
    assert r.returncode != 0 and "pilot_gate.json with pass: true" in r.stdout and r.stdout.count("[MISSING]") == 1

    _write_prereqs(root, rubric_hash="0" * 64)
    r = sh(root / "scripts" / "freeze_prereg.sh", "--dry-run", env=_git_env(), cwd=root)
    assert r.returncode != 0 and "judge_calibration.json" in r.stdout and "rubric" in r.stdout

    _write_prereqs(root)
    (root / "src" / "rhg" / "env" / "grader.py").write_text("def grade():\n    return 99\n", encoding="utf-8", newline="\n")
    r = sh(root / "scripts" / "freeze_prereg.sh", "--dry-run", env=_git_env(), cwd=root)
    assert r.returncode != 0 and "git tree clean" in r.stdout


@needs_bash
def test_freeze_dry_run_passes_when_all_present_and_prints_push_commands(tmp_path):
    root = _freeze_repo(tmp_path)
    _write_prereqs(root)
    r = sh(root / "scripts" / "freeze_prereg.sh", "--dry-run", env=_git_env(), cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[MISSING]" not in r.stdout and "all present" in r.stdout
    assert PUSH_1 in r.stdout and PUSH_2 in r.stdout
    assert "checkout that contains" in r.stdout and "precondition" in r.stdout
    assert "git tag -a prereg-v1" in r.stdout and "[dry-run]" in r.stdout
    assert not (root / "prereg" / "FREEZE.json").exists()
    assert subprocess.run(["git", "tag"], cwd=root, capture_output=True, text=True).stdout.strip() == ""


@needs_bash
def test_freeze_for_real_in_a_temp_checkout_commits_tags_and_never_pushes(tmp_path):
    root = _freeze_repo(tmp_path)
    _write_prereqs(root)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=root, check=True)
    head_before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

    r = sh(root / "scripts" / "freeze_prereg.sh", env=_git_env(), cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert PUSH_1 in r.stdout and PUSH_2 in r.stdout
    assert (root / "prereg" / "FREEZE.json").is_file()
    tags = subprocess.run(["git", "tag"], cwd=root, capture_output=True, text=True).stdout.split()
    assert tags == ["prereg-v1"]
    msg = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=root, capture_output=True, text=True).stdout.strip()
    assert msg == "Freeze pre-registration (prereg-v1)"
    head_after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    assert head_after != head_before
    files = subprocess.run(["git", "show", "--name-only", "--format=", "HEAD"], cwd=root, capture_output=True, text=True).stdout.split()
    assert files and all(f.startswith("prereg/") for f in files) and "prereg/FREEZE.json" in files
    assert subprocess.run(["git", "ls-remote", str(remote)], capture_output=True, text=True).stdout.strip() == ""  # not pushed

    r = sh(root / "scripts" / "freeze_prereg.sh", env=_git_env(), cwd=root)  # second freeze: refused, tag exists
    assert r.returncode != 0 and "does not exist yet" in r.stdout
    assert subprocess.run(["git", "tag"], cwd=root, capture_output=True, text=True).stdout.split() == ["prereg-v1"]
