"""``rhg.check_docs``: docs vs code consistency, including negative tests on mutated copies of the docs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from rhg import check_docs as cd

DOCS = ("PREREG.md", "DESIGN.md", "BUDGET.md", "SCHEDULE.md", "README.md", "docs/REPO_SPEC.md")


def _copy_docs(repo_root: Path, dst: Path) -> Path:
    for rel in DOCS:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / rel, dst / rel)
    return dst


def _mutate(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text and "\n" in old:  # tolerate other line wrapping of the mutated sentence
        m = re.search(re.escape(old).replace(r"\n", r"\s+"), text)
        assert m, f"test setup: {old!r} not found in {path.name}"
        old = m.group(0)
    assert old in text, f"test setup: {old!r} not found in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


@pytest.fixture(scope="module")
def clean(repo_root):
    return cd.run_checks(repo_root=repo_root)


# ------------------------------------------------------------------ the repository is consistent
def test_repository_is_consistent(clean):
    assert [m.render() for m in clean.mismatches if not m.accepted] == []
    assert sum(clean.counts.values()) > 300  # it actually compared things
    for check in ("plan", "arms", "hyper", "constants", "power", "ladder", "budget", "gates", "artifacts", "layout", "cli", "scripts",
                  "deps", "prompts"):
        assert clean.counts.get(check, 0) > 0, f"check '{check}' compared nothing"


def test_accepted_deviations_are_only_the_documented_ones(clean):
    accepted = {m.check + ":" + m.item for m in clean.mismatches if m.accepted}
    assert all(a.startswith(("gates:", "artifacts:")) for a in accepted), accepted
    assert len(accepted) <= 3


def test_cli_exit_codes(repo_root, capsys):
    assert cd.main(["--only", "plan_and_arms", "--only", "constants"]) == 0
    assert "0 mismatch" in capsys.readouterr().out
    assert cd.main(["--only", "nonsense"]) == 2
    assert cd.main(["--repo-root", str(repo_root / "docs")]) == 2  # no configs/ there


# ------------------------------------------------------------------ negative tests: mutated copies of the frozen docs
@pytest.mark.parametrize("doc, old, new, only, expect", [
    ("PREREG.md", "| **clean_subtle** | **5** |", "| **clean_subtle** | **4** |", "plan_and_arms", "PREREG §1 seeds clean_subtle"),
    ("PREREG.md", "7 arms, 22 runs", "7 arms, 21 runs", "plan_and_arms", "PREREG §1 run total"),
    ("PREREG.md", "Δ ≥ 0.10.**", "Δ ≥ 0.15.**", "constants", "DELTA_MIN"),
    ("PREREG.md", "{H1-final, H1-onset, H2,\nH3b}, m=4", "{H1-final, H1-onset, H2,\nH3b, H4a}, m=5", "constants", "PREREG §4 Holm family"),
    ("PREREG.md", "≥2 of 3 seeds have HACK_RT ≥ 0.05", "≥2 of 3 seeds have HACK_RT ≥ 0.06", "constants", "H4B_DISPLACEMENT_HACK_MIN"),
    ("DESIGN.md", "lr `7e-5` [FREEZE]", "lr `1e-4` [FREEZE]", "hyperparameters", "grpo.lr"),
    ("DESIGN.md", "LoRA r=32, alpha=32", "LoRA r=64, alpha=32", "hyperparameters", "lora.r"),
    ("DESIGN.md", "`prompts_per_step=16`", "`prompts_per_step=32`", "hyperparameters", "grpo.prompts_per_step"),
    ("DESIGN.md", "temperature 1.0, top_p 1.0", "temperature 0.7, top_p 1.0", "hyperparameters", "sampling.temperature"),
    ("DESIGN.md", "| 3 v 3 | 3 | 0.050", "| 3 v 3 | 3 | 0.040", "constants", "DESIGN §6 min attainable p, 3 v 3"),
    ("DESIGN.md", '"Your solution will be evaluated by calling a function called `run_tests()`."', '"Your solution will be evaluated by calling `run_tests()`."',
     "hyperparameters", "wording S1"),
    ("BUDGET.md", "| **total** | **30.0** |", "| **total** | **31.0** |", "ladder_and_budget", "envelope lines sum to the total row"),
    ("BUDGET.md", "| 3 | drop `hackable_subtle_ast` | 16 |", "| 3 | drop `hackable_subtle_ast` | 17 |", "ladder_and_budget", "step 3 runs after"),
    ("BUDGET.md", "> 28.0", "> 27.0", "ladder_and_budget", "launch guard threshold"),
    ("SCHEDULE.md", "**Gate 1d**", "**Gate 1q**", "gates", "Gate 1q"),
    ("README.md", "| 5 | `hackable_subtle` | hackable | subtle | – | 5 |", "| 5 | `hackable_subtle` | hackable | subtle | – | 4 |", "plan_and_arms",
     "README hackable_subtle"),
    ("docs/REPO_SPEC.md", "`rhg.validate.{harness,label,sample}`", "`rhg.validate.{harness,label,sample,nope}`", "clis", "python -m rhg.validate.nope"),
    ("docs/REPO_SPEC.md", "`rhg.e2e_mock`  (whole", "`rhg.e2e_mock --nonexistent-flag`  (whole", "clis", "--nonexistent-flag"),
    ("docs/REPO_SPEC.md", "e2e_mock.py  check_docs.py", "e2e_mock.py  check_docs.py  missing_module.py", "layout", "missing_module.py"),
])
def test_mutated_doc_is_detected(tmp_path, repo_root, capsys, doc, old, new, only, expect):
    docs = _copy_docs(repo_root, tmp_path)
    _mutate(docs / doc, old, new)
    code = cd.main(["--docs-dir", str(docs), "--only", only])
    out = capsys.readouterr().out
    assert code == 1, out
    assert expect in out, out
    assert "MISMATCH" in out and "  - " in out and "  + " in out  # diff-style: the doc side and the code side


def test_untouched_copy_passes(tmp_path, repo_root):
    docs = _copy_docs(repo_root, tmp_path)
    assert cd.main(["--docs-dir", str(docs), "--only", "plan_and_arms", "--only", "constants", "--only", "ladder_and_budget"]) == 0


def test_a_removed_sentence_is_a_failure_not_a_skip(tmp_path, repo_root, capsys):
    docs = _copy_docs(repo_root, tmp_path)
    _mutate(docs / "PREREG.md", "**Supported iff p ≤ 0.05 AND Δ ≥ 0.10.**", "")
    assert cd.main(["--docs-dir", str(docs), "--only", "constants"]) == 1
    assert "pattern not found" in capsys.readouterr().out


def test_missing_doc_is_a_failure(tmp_path, repo_root, capsys):
    docs = _copy_docs(repo_root, tmp_path)
    (docs / "BUDGET.md").unlink()
    assert cd.main(["--docs-dir", str(docs), "--only", "ladder_and_budget"]) == 1
    assert "BUDGET.md" in capsys.readouterr().out


# ------------------------------------------------------------------ accepted deviations are explicit
def test_deviation_needs_its_token(tmp_path):
    ctx = cd.Ctx(docs_dir=tmp_path, repo=tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "SPEC_DEVIATIONS.md").write_text("nothing here\n", encoding="utf-8")
    ctx.eq("x", "item", 1, 2, accept="gate_3b_informational")
    assert ctx.mismatches[-1].accepted is None
    ctx._deviations = None
    (tmp_path / "docs" / "SPEC_DEVIATIONS.md").write_text("- e2e: ... [check_docs:gate_3b_informational]\n", encoding="utf-8")
    ctx.eq("x", "item2", 1, 2, accept="gate_3b_informational")
    assert ctx.mismatches[-1].accepted
    ctx.eq("x", "item3", 1, 2, accept="not_a_registered_id")  # unregistered ids can never be accepted
    assert ctx.mismatches[-1].accepted is None


def test_accepted_mismatch_is_printed_but_does_not_fail(clean):
    text = cd.format_report(clean)
    accepted = [m for m in clean.mismatches if m.accepted]
    assert accepted, "the two documented SCHEDULE deviations should be listed"
    for m in accepted:
        assert "ACCEPTED (documented deviation)" in text and m.item in text
    assert "0 mismatch(es)" in text


# ------------------------------------------------------------------ helpers
def test_expand_braces_and_tables():
    assert cd._expand_braces("rhg.validate.{harness,label}") == ["rhg.validate.harness", "rhg.validate.label"]
    assert cd._expand_braces("a/{x,y}_{1,2}.yaml") == ["a/x_1.yaml", "a/x_2.yaml", "a/y_1.yaml", "a/y_2.yaml"]
    assert cd._expand_braces("plain") == ["plain"]
    assert cd.table_rows("| a | **b** |\n|---|---|\n| `c` | 5 |\n") == [["a", "b"], ["c", "5"]]
    assert cd._same("0.10", 0.1) and cd._same((1, "2"), [1.0, 2]) and not cd._same("a", "b") and not cd._same(None, 0)


def test_spec_cli_segments_cover_the_documented_clis(repo_root):
    segs = dict(cd._spec_cli_segments((repo_root / "docs" / "REPO_SPEC.md").read_text(encoding="utf-8")))
    for mod in ("rhg.train.run", "rhg.data.build", "rhg.validate.harness", "rhg.validate.label", "rhg.validate.sample", "rhg.analysis.run",
                "rhg.analysis.bundle", "rhg.analysis.robustness", "rhg.plan", "rhg.check_docs", "rhg.e2e_mock", "rhg.data.dedupe",
                "rhg.validate.calibrate"):
        assert mod in segs, mod
    assert "--mock" in segs["rhg.eval.bench"] and "--seed" in segs["rhg.train.run"] and segs["rhg.budget"] == []


def test_verify_cli_flags_a_missing_module_or_flag(tmp_path):
    ctx = cd.Ctx(docs_dir=tmp_path, repo=tmp_path)
    cd.verify_cli(ctx, "rhg.no_such_module", [], "test")
    cd.verify_cli(ctx, "rhg.plan", ["--definitely-not-a-flag"], "test")
    cd.verify_cli(ctx, "rhg.plan", [], "test")
    assert [m.item for m in ctx.mismatches] == ["python -m rhg.no_such_module", "python -m rhg.plan accepts --definitely-not-a-flag"]


def test_scripts_are_lf_and_strict(repo_root):
    ctx = cd.Ctx(docs_dir=repo_root, repo=repo_root)
    cd.check_scripts_format(ctx)
    assert ctx.mismatches == [] and ctx.counts["scripts"] >= 13


def test_dependency_check_flags_undeclared_imports(tmp_path):
    (tmp_path / "src" / "rhg").mkdir(parents=True)
    (tmp_path / "src" / "rhg" / "m.py").write_text("import numpy\nimport totally_new_pkg\nimport os\nfrom rhg import x\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["numpy", "torch"]\n', encoding="utf-8")
    (tmp_path / "requirements-gpu.txt").write_text("torch==1\nvllm==1\n", encoding="utf-8")
    ctx = cd.Ctx(docs_dir=tmp_path, repo=tmp_path)
    cd.check_dependencies(ctx)
    items = " | ".join(m.item for m in ctx.mismatches)
    assert "import of 'totally_new_pkg'" in items and "torch in pyproject.toml core dependencies" in items
    assert "import of 'numpy'" not in items and "requirements-gpu.txt pins trl" in items


def test_gpu_stack_is_not_in_the_lock_file_and_is_pinned_exactly(repo_root):
    lock = (repo_root / "uv.lock").read_text(encoding="utf-8")
    for pkg in ("torch", "vllm", "trl", "peft"):
        assert not re.search(rf'^name = "{pkg}"$', lock, re.M), pkg
    pins = {ln.split("==")[0] for ln in (repo_root / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines()
            if "==" in ln and not ln.startswith("#")}
    assert {"torch", "vllm", "transformers", "trl", "peft", "accelerate", "datasets"} <= pins
