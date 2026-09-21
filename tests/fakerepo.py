"""Throw-away git repositories with a fake repo layout for manifest / prereg-check tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CODE_FILES = {
    "src/rhg/analysis/stats.py": "def stat():\n    return 1\n",
    "src/rhg/env/grader.py": "def grade():\n    return 2\n",
    "src/rhg/detect/ast_detector.py": "def detect():\n    return 3\n",
    "src/rhg/judge/rubric.py": "RUBRIC = 'v1'\n",
    "src/rhg/data/build.py": "def build():\n    return 4\n",
    "src/rhg/prereg_constants.py": "ALPHA = 0.05\n",
    "src/rhg/train/rollout_io.py": "def log():\n    return 5\n",
    "src/rhg/eval/generate.py": "def sample():\n    return 6\n",
    "src/rhg/runlog.py": "SCHEMA = 1\n",
    "src/rhg/seeds.py": "def derive():\n    return 7\n",
    "src/rhg/config.py": "def load():\n    return 8\n",
    "src/rhg/plan.py": "def plan():\n    return 9\n",
    "src/rhg/budget.py": "LADDER = ()\n",
}


def git(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *args]
    out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def commit_all(repo: Path, msg: str = "c") -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def build_repo(root: Path) -> Path:
    """Fake layout + real configs, committed on a fresh git repo (no tag, no FREEZE.json)."""
    root.mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    for rel, text in CODE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    (root / "requirements-gpu.txt").write_text("torch==0.0.0\n", encoding="utf-8", newline="\n")
    (root / ".gitignore").write_text("data/processed/\nresults/\n__pycache__/\n", encoding="utf-8")
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "splits.json").write_text('{"split_hash": "splithash123"}', encoding="utf-8")
    (processed / "DATASET_REVISION").write_text("abc123rev\n", encoding="utf-8")
    git(root, "init", "-q")
    commit_all(root, "initial")
    return root


def freeze_and_tag(repo: Path) -> None:
    from rhg.analysis.prereg_check import write_freeze

    write_freeze(repo_root=repo)
    commit_all(repo, "freeze")
    git(repo, "tag", "prereg-v1")


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    return build_repo(tmp_path / "repo")


@pytest.fixture
def frozen_repo(fake_repo: Path) -> Path:
    freeze_and_tag(fake_repo)
    return fake_repo
