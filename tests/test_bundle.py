"""Public results bundle: contents, exclusions, secret scan, size cap, generated README table."""

from __future__ import annotations

import gzip
import json
import platform
import re
import shutil
from pathlib import Path

import pytest

from rhg import budget
from rhg.analysis import bundle


@pytest.fixture(scope="module")
def env(sim22, tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle_env")
    repo = root / "repo"
    (repo / "prereg").mkdir(parents=True)
    (repo / "prereg" / "FREEZE.json").write_text('{"schema_version": 1, "prereg_tag": "prereg-v1"}\n', encoding="utf-8")
    (repo / "prereg" / "AMENDMENTS.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00+00:00", "group": "analysis", "old_hash": "a", "new_hash": "b", "reason": "r"}) + "\n", encoding="utf-8")
    ledger = root / "ledger.jsonl"
    budget.record("train", "hackable_subtle__s0", 3600, 0.45, note="secret-looking note that must not be published", ledger=ledger)
    budget.record("train", "clean_subtle__s0", 1800, 0.45, ledger=ledger)
    budget.record("judge", "judge", 10, usd=0.25, ledger=ledger)
    analysis = root / "analysis"
    shutil.copytree(sim22.analysis, analysis)
    (analysis / "validation.md").write_text("# Validation\n\nplaceholder measurement-validity report\n", encoding="utf-8")
    out = root / "public"
    res = bundle.build_bundle(out, analysis_dir=analysis, runs_dir=sim22.runs, repo_root=repo, ledger=ledger)
    return {"root": root, "repo": repo, "ledger": ledger, "analysis": analysis, "out": out, "res": res, "runs": sim22.runs}


def _fresh_analysis(env, tmp_path):
    dst = tmp_path / "analysis"
    shutil.copytree(env["analysis"], dst)
    return dst


def _build(env, out, analysis, **kw):
    return bundle.build_bundle(out, analysis_dir=analysis, runs_dir=env["runs"], repo_root=env["repo"], ledger=env["ledger"], **kw)


# ------------------------------------------------------------------ contents
def test_expected_files_are_present_and_raw_material_is_excluded(env):
    out = env["out"]
    files = set(env["res"]["files"])
    for name in ("REPORT.md", "tests.json", "per_seed.csv", "examples.md", "validation.md", "FREEZE.json", "AMENDMENTS.jsonl",
                 "README_RESULTS.md", "ledger_summary.json", "rollouts_sample.jsonl.gz", "figures/figures.json", "tables/tables.json"):
        assert name in files, name
    assert len([f for f in files if f.startswith("figures/") and f.endswith(".png")]) >= 14
    run_ids = {d.name for d in env["runs"].iterdir()}
    assert len(run_ids) == 22
    assert {f[len("configs/"):-len(".yaml")] for f in files if f.startswith("configs/")} == run_ids
    assert {f[len("manifests/"):-len(".json")] for f in files if f.startswith("manifests/")} == run_ids
    assert not [f for f in files if f.endswith("rollouts.jsonl.gz") and f != "rollouts_sample.jsonl.gz"]  # no raw rollouts
    assert not [f for f in files if "stdout" in f or "adapter" in f or f.endswith((".safetensors", ".bin", ".pt"))]
    assert (out / "FREEZE.json").read_bytes() == (env["repo"] / "prereg" / "FREEZE.json").read_bytes()
    assert (out / "AMENDMENTS.jsonl").read_bytes() == (env["repo"] / "prereg" / "AMENDMENTS.jsonl").read_bytes()
    assert (out / "REPORT.md").read_bytes() == (env["analysis"] / "REPORT.md").read_bytes()
    assert env["res"]["bytes"] < bundle.MAX_BYTES == 25 * 1024 * 1024
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".bundle_stage_")]  # the staging directory is gone


def test_ledger_summary_contains_spend_by_kind_only(env):
    doc = json.loads((env["out"] / "ledger_summary.json").read_text(encoding="utf-8"))
    assert set(doc) == {"spend_usd_by_kind", "total_usd"}
    assert doc["spend_usd_by_kind"] == {"judge": 0.25, "train": pytest.approx(0.45 * 1.5)}
    assert doc["total_usd"] == pytest.approx(0.25 + 0.675, abs=1e-3)
    blob = "".join(p.read_text(encoding="utf-8", errors="replace") for p in env["out"].rglob("*") if p.suffix in {".md", ".json", ".csv"})
    assert "secret-looking note" not in blob


def _sample(out):
    with gzip.open(out / "rollouts_sample.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_rollout_sample_is_seeded_per_arm_final_test_eval_only(env, tmp_path):
    recs = _sample(env["out"])
    by_arm: dict[str, list[dict]] = {}
    for r in recs:
        by_arm.setdefault(r["arm"], []).append(r)
    assert len(by_arm) == 7 and all(len(v) == 200 for v in by_arm.values())  # 192 rollouts per seed x 2..5 seeds: capped at 200
    assert all(r["phase"] == "eval_test" and r["step"] == 60 and r["eval_hint"] is None for r in recs)
    assert len({(r["run_id"], r["problem_id"], r["sample_idx"]) for r in recs}) == len(recs)  # no duplicates
    assert {r["seed"] for r in by_arm["hackable_subtle"]} == {0, 1, 2, 3, 4}  # all seeds represented (random over the pooled arm)
    assert all("completion" in r and "labels" in r for r in recs)
    again = tmp_path / "again"
    _build(env, again, env["analysis"])
    assert (again / "rollouts_sample.jsonl.gz").read_bytes() == (env["out"] / "rollouts_sample.jsonl.gz").read_bytes()  # reproducible
    other = tmp_path / "other"
    _build(env, other, env["analysis"], seed=bundle.SAMPLE_SEED + 1)
    assert _sample(other) != recs


def _parse_table(text: str) -> list[dict[str, str]]:
    lines = text.split("## Pre-registered vs exploratory")[1].split("## Contents")[0].splitlines()
    rows = [ln for ln in lines if ln.startswith("|")]
    split = lambda ln: [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]  # noqa: E731
    head = split(rows[0])
    return [dict(zip(head, split(r))) for r in rows[2:]]


def test_preregistered_vs_exploratory_table_matches_tests_json(env):
    doc = json.loads((env["out"] / "tests.json").read_text(encoding="utf-8"))
    rows = _parse_table((env["out"] / "README_RESULTS.md").read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == [t["id"] for t in doc["tests"]] == ["primary", "H1_final", "H1_onset", "H2", "H3b", "H3a", "H4a", "H4b"]
    for row, t in zip(rows, doc["tests"]):
        assert row["pre-registered as"] == t["registered"] and row["stamped in this analysis"] == t["stamp"] == "EXPLORATORY"
        assert row["hypothesis"] == t["hypothesis"] and row["result"] == t["result"] and row["n"] == t["n_text"]
        assert row["p"] == ("n/a" if t["p"] is None else f"{t['p']:.4f}")
        assert row["min attainable p"] == ("n/a" if t["min_attainable_p"] is None else f"{t['min_attainable_p']:.4f}")
    pre = {r["id"] for r in rows if r["pre-registered as"] == "pre-registered confirmatory"}
    assert pre == {"primary", "H1_final", "H1_onset", "H2", "H3b"}
    text = (env["out"] / "README_RESULTS.md").read_text(encoding="utf-8")
    assert "Analysis mode: **EXPLORATORY**" in text and "Primary outcome (PREREG §2 wording): **supported**" in text
    assert "rollouts_sample.jsonl.gz" in text and "Not included by design" in text


def test_bundle_is_replaced_atomically_and_stale_files_disappear(env, tmp_path):
    out = tmp_path / "public"
    _build(env, out, env["analysis"])
    (out / "stale.txt").write_text("old", encoding="utf-8")
    _build(env, out, env["analysis"])
    assert not (out / "stale.txt").exists() and (out / "README_RESULTS.md").is_file()


# ------------------------------------------------------------------ secret scan
LOCAL_NAME = platform.node()
PLANTS = [
    ("api key prefix", "the key was sk-ant-api03-AAAABBBBCCCC"),
    ("env var name", "export ANTHROPIC_API_KEY=abc"),
    ("windows home path", r"see C:\Users\bob\work\reward-hacking"),
    ("windows home path (forward slashes)", "see C:/Users/bob/work/x"),
    ("posix home path", "logs in /home/alice/runs/x"),
    ("mac home path", "logs in /Users/carol/runs/x"),
    ("raw hostname", '"hostname": "gpu-box-17"'),
    ("current home directory", str(Path.home())),
]
PARAMS = [pytest.param(label, plant, id=label) for label, plant in PLANTS]
if len(LOCAL_NAME) >= 5:  # shorter names are not scanned for (too many false positives)
    PARAMS.append(pytest.param("current hostname", f"ran on {LOCAL_NAME}", id="current hostname"))


@pytest.mark.parametrize("label,plant", PARAMS)
def test_planted_secret_is_caught_and_nothing_is_published(env, tmp_path, label, plant):
    analysis = _fresh_analysis(env, tmp_path)
    with open(analysis / "examples.md", "a", encoding="utf-8") as f:
        f.write("\n" + plant + "\n")
    out = tmp_path / "public"
    with pytest.raises(bundle.BundleError) as e:
        _build(env, out, analysis)
    assert "secret scan failed" in str(e.value) and "examples.md" in str(e.value)
    assert plant not in str(e.value)  # the finding names the file and pattern, it does not echo the secret
    assert not out.exists() and not [p for p in tmp_path.iterdir() if p.name.startswith(".bundle_stage_")]
    code = bundle.main(["--out", str(out), "--analysis", str(analysis), "--runs", str(env["runs"]), "--repo-root", str(env["repo"]),
                        "--ledger", str(env["ledger"])])
    assert code == 3 and not out.exists()


def test_secret_in_a_non_markdown_file_a_gzip_and_a_config_is_found(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.json").write_text('{"k": "sk-ant-oops"}', encoding="utf-8")
    (tmp_path / "b.jsonl.gz").write_bytes(gzip.compress(b'{"completion": "os.environ[\\"ANTHROPIC_API_KEY\\"]"}\n'))
    (tmp_path / "c.yaml").write_text("path: /home/dave/data/x\n", encoding="utf-8")
    (tmp_path / "clean.md").write_text("# fine\n\nAnthropic is a company; hostname_sha256: abcdef; a forbidden-word\n", encoding="utf-8")  # neither is a hit
    findings = bundle.scan_tree(tmp_path)
    assert len(findings) == 3
    assert any(f.startswith("sub/a.json") for f in findings) and any(f.startswith("b.jsonl.gz") for f in findings)
    assert any(f.startswith("c.yaml") for f in findings) and not any("clean.md" in f for f in findings)
    assert bundle.scan_tree(tmp_path, extra_forbidden=["forbidden-word"]) != findings  # extra literals are honoured


def test_a_secret_in_a_run_manifest_or_the_rollout_sample_blocks_the_bundle(env, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(env["runs"], runs, ignore=shutil.ignore_patterns("rollouts.jsonl.gz", "stdout.log"))
    for d in runs.iterdir():  # rollouts are read only for the sample: give every run a minimal valid file
        with gzip.open(d / "rollouts.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(json.dumps({"phase": "eval_test", "step": 60, "eval_hint": None, "run_id": d.name, "problem_id": "p", "sample_idx": 0,
                                "completion": "x"}) + "\n")
    man = runs / "clean_subtle__s0" / "manifest.json"
    doc = json.loads(man.read_text(encoding="utf-8"))
    doc["hardware"]["hostname"] = "my-raw-host"  # a raw hostname where only hostname_sha256 belongs
    man.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="manifests/clean_subtle__s0.json"):
        bundle.build_bundle(tmp_path / "public", analysis_dir=env["analysis"], runs_dir=runs, repo_root=env["repo"], ledger=env["ledger"])
    doc["hardware"].pop("hostname")
    man.write_text(json.dumps(doc), encoding="utf-8")
    with gzip.open(runs / "clean_subtle__s0" / "rollouts.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "eval_test", "step": 60, "eval_hint": None, "run_id": "x", "problem_id": "p", "sample_idx": 0,
                            "completion": "key = 'sk-ant-leaked'"}) + "\n")
    with pytest.raises(bundle.BundleError, match="rollouts_sample.jsonl.gz"):
        bundle.build_bundle(tmp_path / "public", analysis_dir=env["analysis"], runs_dir=runs, repo_root=env["repo"], ledger=env["ledger"])
    assert not (tmp_path / "public").exists()


# ------------------------------------------------------------------ size cap and inputs
def test_size_cap_is_enforced_and_an_existing_bundle_is_left_intact(env, tmp_path):
    out = tmp_path / "public"
    _build(env, out, env["analysis"])
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(bundle.BundleError, match="over the .* MB cap"):
        _build(env, out, env["analysis"], max_bytes=10_000)
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before
    base = env["res"]["bytes"]
    _build(env, tmp_path / "fits", env["analysis"], max_bytes=base + 200_000)  # the cap is not simply tiny: the same bundle fits just above its size
    analysis = _fresh_analysis(env, tmp_path)
    (analysis / "figures" / "huge.png").write_bytes(b"\0" * 500_000)  # a planted big file pushes it over that cap
    with pytest.raises(bundle.BundleError, match="cap"):
        _build(env, tmp_path / "public2", analysis, max_bytes=base + 200_000)
    assert not (tmp_path / "public2").exists()
    assert bundle.main(["--out", str(tmp_path / "p3"), "--analysis", str(env["analysis"]), "--runs", str(env["runs"]), "--repo-root", str(env["repo"]),
                        "--ledger", str(env["ledger"]), "--max-mb", "0.01"]) == 3


def test_missing_analysis_outputs_and_foreign_output_directories_are_usage_errors(env, tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    with pytest.raises(bundle.BundleInputError, match="lacks REPORT.md, tests.json, per_seed.csv"):
        _build(env, tmp_path / "public", tmp_path / "empty")
    assert bundle.main(["--out", str(tmp_path / "public"), "--analysis", str(tmp_path / "empty")]) == 2
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "important.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(bundle.BundleInputError, match="not a results bundle"):
        _build(env, foreign, env["analysis"])
    assert (foreign / "important.txt").read_text(encoding="utf-8") == "keep me"


def test_missing_freeze_ledger_and_validation_are_tolerated_and_cli_succeeds(env, tmp_path, capsys):
    analysis = _fresh_analysis(env, tmp_path)
    (analysis / "validation.md").unlink()
    repo = tmp_path / "bare"
    repo.mkdir()
    out = tmp_path / "public"
    code = bundle.main(["--out", str(out), "--analysis", str(analysis), "--runs", str(env["runs"]), "--repo-root", str(repo),
                        "--ledger", str(tmp_path / "no_ledger.jsonl")])
    assert code == 0 and "wrote" in capsys.readouterr().out
    assert not (out / "validation.md").exists()
    assert "no pre-registration freeze" in (out / "FREEZE.json").read_text(encoding="utf-8")
    assert (out / "AMENDMENTS.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads((out / "ledger_summary.json").read_text(encoding="utf-8"))["total_usd"] == 0.0
