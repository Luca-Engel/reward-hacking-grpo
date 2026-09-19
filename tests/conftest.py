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
