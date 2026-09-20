"""Shared pytest setup: gpu/network/api/slow tests are skipped unless selected with -m."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPT_IN_MARKERS = ("gpu", "network", "api", "slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected = config.getoption("markexpr", default="") or ""
    for marker in OPT_IN_MARKERS:
        if marker in selected:
            continue
        skip = pytest.mark.skip(reason=f"needs -m {marker} (skipped by default)")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture(scope="session")
def sim22(tmp_path_factory: pytest.TempPathFactory):
    """The full 22-run design simulated with planted truth, analysed once through the CLI (shared by the analysis tests)."""
    import json
    from types import SimpleNamespace

    import anafix

    from rhg.analysis import run as analysis_run

    root = tmp_path_factory.mktemp("sim22")
    sim = anafix.simulate_dir(root / "sim", anafix.scenario())
    repo = root / "repo"
    repo.mkdir()
    (repo / "DEVIATIONS.md").write_text("# DEVIATIONS\n\n| date | what | why | effect |\n|---|---|---|---|\n| 2099-01-01 | planted deviation row | test | none |\n",
                                        encoding="utf-8")
    out = root / "analysis"
    code = analysis_run.main(["--runs", str(sim.runs), "--out", str(out), "--problems", str(sim.problems), "--repo-root", str(repo)])
    assert code == 0
    doc = json.loads((out / "tests.json").read_text(encoding="utf-8"))
    return SimpleNamespace(root=root, sim=sim, runs=sim.runs, problems=sim.problems, truth=sim.truth, repo=repo, analysis=out, doc=doc,
                           tests={t["id"]: t for t in doc["tests"]})
