import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path

import pytest

import rhg.manifest as mf
from fakerepo import commit_all, fake_repo, frozen_repo, git  # noqa: F401
from rhg.analysis import prereg_check as pc
from rhg.config import load_config
from rhg.manifest import ConfirmatoryGuardError

TOP_KEYS = {
    "run_id", "arm", "seed", "config_hash", "run_hash", "git_sha", "git_dirty", "git_diff_sha256",
    "prereg_tag_present", "prereg_tag_is_ancestor", "freeze_json_sha256", "split_hash", "prompts_hash",
    "dataset_revision", "libs", "hardware", "mode", "confirmatory", "started_at", "finished_at",
    "wall_s", "usd_per_hour", "usd", "status", "invalid_reason",
}
LIB_KEYS = {"python", "torch", "transformers", "trl", "peft", "vllm", "datasets", "numpy", "anthropic"}
HW_KEYS = {"gpu_name", "gpu_count", "driver", "cuda", "cpu_cores", "hostname_sha256"}


@pytest.fixture(scope="module")
def cfg(config_dir):
    return load_config("hackable_subtle", seed=3, config_dir=config_dir)


def _no_external(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such executable")

    monkeypatch.setattr(subprocess, "run", boom)


# ------------------------------------------------------------------ manifest content


def test_all_keys_present_and_identity_fields(cfg, fake_repo):
    m = mf.build_manifest(cfg, repo_root=fake_repo)
    assert set(m) == TOP_KEYS
    assert set(m["libs"]) == LIB_KEYS and set(m["hardware"]) == HW_KEYS
    assert m["run_id"] == "hackable_subtle__s3" and m["arm"] == "hackable_subtle" and m["seed"] == 3
    assert m["config_hash"] == cfg.config_hash and m["run_hash"] == cfg.run_hash
    assert m["mode"] == "train" and m["confirmatory"] is False
    assert m["status"] == "running" and m["finished_at"] is None and m["wall_s"] is None and m["usd"] is None
    assert m["usd_per_hour"] == cfg.budget.usd_per_hour and m["invalid_reason"] is None
    assert m["started_at"]
    json.dumps(m)  # serialisable


def test_git_fields_in_a_real_repo(cfg, fake_repo):
    head = git(fake_repo, "rev-parse", "HEAD")
    m = mf.build_manifest(cfg, repo_root=fake_repo)
    assert m["git_sha"] == head and m["git_dirty"] is False and m["git_diff_sha256"] is None
    assert m["prereg_tag_present"] is False and m["prereg_tag_is_ancestor"] is False
    assert m["freeze_json_sha256"] is None
    assert m["split_hash"] == "splithash123" and m["dataset_revision"] == "abc123rev"
    assert m["prompts_hash"] == hashlib.sha256((fake_repo / "configs/prompts.yaml").read_bytes()).hexdigest()


def test_dirty_tree_records_diff_hash(cfg, fake_repo):
    p = fake_repo / "src/rhg/env/grader.py"
    p.write_text(p.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8", newline="\n")
    raw_diff = subprocess.run(["git", "diff", "HEAD"], cwd=fake_repo, capture_output=True, check=True).stdout
    assert raw_diff  # sanity: there is a diff
    m = mf.build_manifest(cfg, repo_root=fake_repo)
    assert m["git_dirty"] is True
    assert m["git_diff_sha256"] == hashlib.sha256(raw_diff).hexdigest()


def test_untracked_file_counts_as_dirty(cfg, fake_repo):
    (fake_repo / "notes.txt").write_text("x", encoding="utf-8")
    assert mf.build_manifest(cfg, repo_root=fake_repo)["git_dirty"] is True


def test_ignored_files_do_not_dirty_the_tree(cfg, fake_repo):
    (fake_repo / "results").mkdir()
    (fake_repo / "results/ledger.jsonl").write_text("{}", encoding="utf-8")
    assert mf.build_manifest(cfg, repo_root=fake_repo)["git_dirty"] is False


def test_tag_and_freeze_fields(cfg, frozen_repo):
    m = mf.build_manifest(cfg, repo_root=frozen_repo)
    assert m["prereg_tag_present"] is True and m["prereg_tag_is_ancestor"] is True
    assert m["freeze_json_sha256"] == hashlib.sha256((frozen_repo / "prereg/FREEZE.json").read_bytes()).hexdigest()


def test_dataset_revision_absent_is_null(cfg, fake_repo):
    (fake_repo / "data/processed/DATASET_REVISION").unlink()
    (fake_repo / "data/processed/splits.json").unlink()
    m = mf.build_manifest(cfg, repo_root=fake_repo)
    assert m["dataset_revision"] is None and m["split_hash"] is None


# ------------------------------------------------------------------ robustness


def test_robust_when_git_and_nvidia_smi_missing(cfg, fake_repo, monkeypatch):
    _no_external(monkeypatch)
    m = mf.build_manifest(cfg, repo_root=fake_repo)
    assert set(m) == TOP_KEYS
    assert m["git_sha"] is None and m["git_dirty"] is None and m["git_diff_sha256"] is None
    assert m["prereg_tag_present"] is False and m["prereg_tag_is_ancestor"] is False
    hw = m["hardware"]
    assert hw["gpu_name"] is None and hw["gpu_count"] == 0 and hw["driver"] is None and hw["cuda"] is None
    assert hw["cpu_cores"] >= 1 and len(hw["hostname_sha256"]) == 64
    json.dumps(m)


def test_robust_when_not_a_git_repo(cfg, tmp_path):
    m = mf.build_manifest(cfg, repo_root=tmp_path)
    assert m["git_sha"] is None and m["prereg_tag_present"] is False
    assert m["split_hash"] is None and m["prompts_hash"] is None and m["freeze_json_sha256"] is None


def test_nvidia_smi_parsing(cfg, fake_repo, monkeypatch):
    def fake_run(cmd, cwd=None, text=True):
        if cmd[0] != "nvidia-smi":
            return None
        if len(cmd) > 1:
            out = "NVIDIA GeForce RTX 4090, 550.54.14\nNVIDIA GeForce RTX 4090, 550.54.14\n"
        else:
            out = "| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4 |\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(mf, "_run", fake_run)
    hw = mf.build_manifest(cfg, repo_root=fake_repo)["hardware"]
    assert hw["gpu_name"] == "NVIDIA GeForce RTX 4090" and hw["gpu_count"] == 2
    assert hw["driver"] == "550.54.14" and hw["cuda"] == "12.4"


def test_nvidia_smi_failure_gives_none(monkeypatch):
    monkeypatch.setattr(mf, "_run", lambda cmd, cwd=None, text=True: subprocess.CompletedProcess(cmd, 9, "", "err"))
    hw = mf.hardware_info()
    assert hw["gpu_name"] is None and hw["gpu_count"] == 0


def test_hostname_is_hashed_never_raw(cfg, fake_repo, monkeypatch):
    monkeypatch.setattr(mf.socket, "gethostname", lambda: "super-secret-host-01")
    m = mf.build_manifest(cfg, repo_root=fake_repo)
    assert m["hardware"]["hostname_sha256"] == hashlib.sha256(b"super-secret-host-01").hexdigest()
    assert "super-secret-host-01" not in json.dumps(m)


def test_library_versions_none_when_missing(monkeypatch):
    real = metadata.version

    def fake_version(name):
        if name == "numpy":
            return real("numpy")
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(mf.metadata, "version", fake_version)
    libs = mf.library_versions()
    assert libs["numpy"] == real("numpy") and libs["python"]
    assert all(libs[k] is None for k in LIB_KEYS - {"python", "numpy"})


# ------------------------------------------------------------------ write / finalize


def test_write_and_finalize_manifest(cfg, fake_repo, tmp_path):
    run_dir = tmp_path / "runs" / cfg.run_id
    m = mf.build_manifest(cfg, repo_root=fake_repo, started_at="2026-01-01T00:00:00+00:00")
    path = mf.write_manifest(run_dir, m)
    assert path == run_dir / "manifest.json" and mf.read_manifest(run_dir) == m
    out = mf.finalize_manifest(run_dir, "completed", finished_at="2026-01-01T01:00:00+00:00")
    assert out["status"] == "completed" and out["wall_s"] == 3600.0
    assert out["usd"] == pytest.approx(cfg.budget.usd_per_hour)  # 1 h at usd_per_hour
    assert out["finished_at"] == "2026-01-01T01:00:00+00:00" and out["invalid_reason"] is None
    assert mf.read_manifest(run_dir) == out
    assert not list(run_dir.glob("*.tmp*"))


def test_finalize_invalid_with_reason_and_explicit_usd(cfg, fake_repo, tmp_path):
    mf.write_manifest(tmp_path, mf.build_manifest(cfg, repo_root=fake_repo))
    out = mf.finalize_manifest(tmp_path, "invalid", invalid_reason="OOM at step 12", usd=0.25, wall_s=10.0)
    assert (out["status"], out["invalid_reason"], out["usd"], out["wall_s"]) == ("invalid", "OOM at step 12", 0.25, 10.0)


def test_finalize_rejects_non_final_status(cfg, fake_repo, tmp_path):
    mf.write_manifest(tmp_path, mf.build_manifest(cfg, repo_root=fake_repo))
    with pytest.raises(ValueError):
        mf.finalize_manifest(tmp_path, "running")


# ------------------------------------------------------------------ confirmatory guard


def test_guard_passes_on_clean_frozen_repo(cfg, frozen_repo):
    mf.assert_confirmatory_ok(cfg, repo_root=frozen_repo)


def test_guard_refuses_without_tag(cfg, fake_repo):
    pc.write_freeze(repo_root=fake_repo)
    commit_all(fake_repo, "freeze")
    with pytest.raises(ConfirmatoryGuardError) as exc:
        mf.assert_confirmatory_ok(cfg, repo_root=fake_repo)
    assert exc.value.exit_code == 3
    assert any("tag_exists" in f for f in exc.value.failures)


def test_guard_refuses_dirty_tree(cfg, frozen_repo):
    (frozen_repo / "scratch.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ConfirmatoryGuardError, match="not clean"):
        mf.assert_confirmatory_ok(cfg, repo_root=frozen_repo)


def test_guard_refuses_hash_drift_and_accepts_committed_amendment(cfg, frozen_repo):
    p = frozen_repo / "src/rhg/analysis/stats.py"
    p.write_text("def stat():\n    return 5\n", encoding="utf-8", newline="\n")
    commit_all(frozen_repo, "post-freeze edit")
    with pytest.raises(ConfirmatoryGuardError) as exc:
        mf.assert_confirmatory_ok(cfg, repo_root=frozen_repo)
    assert any("code:analysis" in f for f in exc.value.failures)
    pc.amend("logged", repo_root=frozen_repo)
    with pytest.raises(ConfirmatoryGuardError, match="not clean"):  # AMENDMENTS.jsonl still uncommitted
        mf.assert_confirmatory_ok(cfg, repo_root=frozen_repo)
    commit_all(frozen_repo, "log amendment")
    mf.assert_confirmatory_ok(cfg, repo_root=frozen_repo)


def test_guard_refuses_when_git_is_unavailable(cfg, frozen_repo, monkeypatch):
    _no_external(monkeypatch)
    with pytest.raises(ConfirmatoryGuardError, match="unknown"):
        mf.assert_confirmatory_ok(cfg, repo_root=frozen_repo)


def test_real_repo_manifest_builds(cfg):
    # smoke test against the actual checkout (whatever its git state is)
    m = mf.build_manifest(cfg)
    assert set(m) == TOP_KEYS
