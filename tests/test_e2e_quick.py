"""``rhg.e2e_mock --quick``: the whole pipeline on CPU with the mock policy recovers the planted truth."""

from __future__ import annotations

import shutil

import pytest

from rhg import e2e_mock

WORK = "results/e2e_test_quick"  # relative to the repo root: absolute home paths would trip the bundle's secret scan


@pytest.fixture(scope="module")
def quick_run(repo_root, request):
    work = repo_root / WORK
    request.addfinalizer(lambda: shutil.rmtree(work, ignore_errors=True))
    real_ledger = repo_root / "results" / "ledger.jsonl"
    before = real_ledger.read_bytes() if real_ledger.is_file() else None
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = e2e_mock.main(["--quick", "--work-dir", WORK])
    return {"code": code, "out": buf.getvalue(), "work": work, "ledger_before": before, "ledger": real_ledger}


def test_quick_e2e_passes_every_check(quick_run):
    assert quick_run["code"] == 0, quick_run["out"][-4000:]
    assert "E2E OK" in quick_run["out"] and "[FAIL]" not in quick_run["out"]


def test_planted_effect_and_both_h4b_paths_are_asserted(quick_run):
    out = quick_run["out"]
    for needle in ("primary is 'supported'", "H4b verdict matches the plant (displace=True -> displacement)",
                   "H4b verdict matches the plant (displace=False -> suppression_only)", "H1 direction (final rate)", "H1 direction (onset)",
                   "22 planned runs trained", "bundle passed its secret scan", "e2e ledger stayed at $0", "--confirmatory is refused"):
        assert f"[ok] {needle}" in out, needle


def test_outputs_exist_and_ledger_untouched(quick_run):
    work = quick_run["work"]
    for rel in ("analysis/REPORT.md", "analysis/examples.md", "analysis/tests.json", "analysis/validation.md", "results_public/README_RESULTS.md",
                "probe/hint_selection.mock.json", "processed/splits.json"):
        assert (work / rel).is_file(), rel
    assert len(list((work / "runs").glob("*__s*"))) == 22 and len(list((work / "runs_suppress").glob("*__s*"))) == 22
    ledger = quick_run["ledger"]
    assert (ledger.read_bytes() if ledger.is_file() else None) == quick_run["ledger_before"]
    assert not (work / "ledger.jsonl").exists() or (work / "ledger.jsonl").stat().st_size == 0


def test_report_is_exploratory_and_lists_min_p(quick_run):
    text = (quick_run["work"] / "analysis" / "REPORT.md").read_text(encoding="utf-8")
    assert text.splitlines()[0].endswith("[EXPLORATORY]") and "Minimum attainable p" in text and "Leave-one-seed-out" in text


def test_refuses_to_wipe_a_foreign_directory(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(e2e_mock.StageError):
        e2e_mock.prepare_tree(tmp_path)
    assert (tmp_path / "keep.txt").is_file()
    fresh = tmp_path / "new"
    e2e_mock.prepare_tree(fresh)
    assert (fresh / e2e_mock.MARKER).is_file()
    (fresh / "junk").write_text("y", encoding="utf-8")
    e2e_mock.prepare_tree(fresh)  # own tree: wiped and recreated
    assert not (fresh / "junk").exists()


def test_bad_arguments_exit_2():
    assert e2e_mock.main(["--jobs", "0"]) == 2
    assert e2e_mock.main(["--mock-lr", "-1"]) == 2


def test_checks_collects_every_failure():
    ck = e2e_mock.Checks()
    ck.ok(True, "a")
    ck.ok(False, "b", "why")
    ck.ok(False, "c")
    assert ck.failed == ["b [why]", "c"] and "[FAIL] b [why]" in ck.report()
