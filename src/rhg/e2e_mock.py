"""Whole pipeline on CPU with the mock policy (subtask 16): ``python -m rhg.e2e_mock [--quick]``.

Stages, all inside one throw-away tree (default ``results/e2e/``; gitignored):

1. fixture data -> ``rhg.data.build`` (fetch, tests, validate) -> mock pass-rate stages A and B (``rhg.eval.pass_rate --mock``)
   -> split -> mock hint probe (``rhg.eval.probe_hints --mock``);
2. all 22 planned runs of ``configs/plan.yaml`` through ``rhg.train.run --mock`` with planted emergence probabilities
   (hackable_subtle 0.9, hackable_explicit 1.0, hackable_none 0, hackable_subtle_ast 0.9, clean arms 0);
   ``hackable_subtle_ast`` is trained twice (two invocations): ``displace=True`` (the policy can move to an obfuscated
   definition the narrow monitor misses) and ``displace=False`` (it can only be suppressed), which exercise the two
   verdict paths of PREREG §5. The 19 other runs do not depend on the AST arm, so the second tree reuses them;
3. mock judge -> validation harness -> analysis (``rhg.analysis.run``, stamped EXPLORATORY: there is no prereg tag) ->
   report -> public bundle (``rhg.analysis.bundle``) with its secret scan.

It then asserts that the planted truth is recovered and that every report section exists. Nothing here touches a GPU,
the network or a paid API; the ledger must still be at $0 at the end. ``--quick`` uses fewer steps and prompts per step.

Exit codes: 0 all checks passed, 1 a stage or an assertion failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from rhg import budget, plan
from rhg import prereg_constants as C
from rhg.manifest import REPO_ROOT

MARKER = ".rhg_e2e_mock"
DEFAULT_WORK_DIR = Path("results/e2e")

# Planted per-seed emergence probability of the mock policy (subtask brief); the AST arm shares its comparator's value.
PLANT_Q = {
    "clean_none": 0.0, "clean_subtle": 0.0, "clean_explicit": 0.0,
    "hackable_none": 0.0, "hackable_subtle": 0.9, "hackable_explicit": 1.0, "hackable_subtle_ast": 0.9,
}
AST_ARM = "hackable_subtle_ast"
TRAIN_TIMEOUT_S = 1200  # one mock run takes 10-60 s; a hang must fail the e2e run, not block it
# With displace=True the obfuscated definition is as easy to find as the plain one (the mock default of 1.5 logits lower did not
# displace within 100 steps in a manual trial, so it is not a plant that this test could recover).
OBF_GAP_PLANTED = 0.0


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    prompts_per_step: int
    val_every: int
    probe_min_samples: int
    mock_lr: float | None  # None = the base.yaml default
    jobs: int
    sandbox_workers: int


FULL = Profile("full", steps=100, prompts_per_step=16, val_every=20, probe_min_samples=3000, mock_lr=None, jobs=4, sandbox_workers=3)
QUICK = Profile("quick", steps=20, prompts_per_step=4, val_every=10, probe_min_samples=800, mock_lr=30.0, jobs=8, sandbox_workers=2)


class StageError(RuntimeError):
    pass


class Checks:
    """Collects assertion outcomes so that one run reports every failure, not only the first."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str]] = []

    def ok(self, cond: bool, what: str, detail: str = "") -> bool:
        self.rows.append((bool(cond), what + (f" [{detail}]" if detail else "")))
        return bool(cond)

    @property
    def failed(self) -> list[str]:
        return [w for ok, w in self.rows if not ok]

    def report(self) -> str:
        return "\n".join(f"  [{'ok' if ok else 'FAIL'}] {w}" for ok, w in self.rows)


def log(msg: str) -> None:
    print(f"[e2e_mock {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rel(p: Path) -> str:
    """Path as the sub-commands should see it: relative to the repo root when inside it (keeps manifests free of home paths)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ stage runners
def _in_process(name: str, main: Callable[[Sequence[str]], int], argv: Sequence[str], logs: Path, expect: int = 0) -> str:
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            code = main(list(argv))
        except SystemExit as e:  # argparse errors
            code = e.code if isinstance(e.code, int) else 1
    out = buf.getvalue()
    (logs / f"{name}.log").write_text(f"$ {name} {' '.join(argv)}\n{out}", encoding="utf-8", newline="\n")
    log(f"{name}: exit {code} in {time.time() - t0:.1f}s")
    if code != expect:
        raise StageError(f"{name} exited {code}, expected {expect}; see {logs / (name + '.log')}\n{out[-1500:]}")
    return out


def _train_one(arm: str, seed: int, out_root: Path, processed: Path, ledger: Path, prof: Profile, displace: bool, logs: Path) -> str:
    cache = ledger.parent / "cache"
    argv = [sys.executable, "-m", "rhg.train.run", "--arm", arm, "--seed", str(seed), "--mock", "--force", "--steps", str(prof.steps),
            "--set", f"data.processed_dir={_rel(processed)}", "--set", f"run.output_root={_rel(out_root)}",
            "--set", f"budget.ledger={_rel(ledger)}", "--set", f"mock.q={PLANT_Q[arm]}", "--set", f"mock.displace={'true' if displace else 'false'}",
            "--set", f"grpo.prompts_per_step={prof.prompts_per_step}", "--set", f"eval.val_every={prof.val_every}",
            "--set", f"sandbox.workers={prof.sandbox_workers}", "--cache-dir", _rel(cache)]
    if displace and arm == AST_ARM:
        argv += ["--set", f"mock.obf_gap={OBF_GAP_PLANTED}"]
    if prof.mock_lr is not None:
        argv += ["--set", f"mock.lr={prof.mock_lr}"]
    run_id = f"{arm}__s{seed}"
    with open(logs / f"train_{run_id}.log", "w", encoding="utf-8", newline="\n") as fh:
        try:
            proc = subprocess.run(argv, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                  timeout=TRAIN_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired as e:
            raise StageError(f"training {run_id} did not finish in {TRAIN_TIMEOUT_S}s; see {logs / f'train_{run_id}.log'}") from e
    if proc.returncode != 0:
        raise StageError(f"training {run_id} exited {proc.returncode}; see {logs / f'train_{run_id}.log'}")
    return run_id


def _train_many(jobs: Sequence[tuple[str, int, bool]], out_root: Path, processed: Path, ledger: Path, prof: Profile, logs: Path) -> list[str]:
    with ThreadPoolExecutor(max_workers=max(1, prof.jobs)) as ex:
        futs = [ex.submit(_train_one, arm, seed, out_root, processed, ledger, prof, displace, logs) for arm, seed, displace in jobs]
        return [f.result() for f in futs]


# ------------------------------------------------------------------ tree preparation
def prepare_tree(work: Path) -> None:
    if work.exists() and any(work.iterdir()):
        if not (work / MARKER).is_file():
            raise StageError(f"{work} exists, is not empty and is not an e2e_mock tree (no {MARKER}); refusing to delete it")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / MARKER).write_text("created by rhg.e2e_mock; safe to delete\n", encoding="utf-8")
    for sub in ("logs", "repo"):
        (work / sub).mkdir(exist_ok=True)
    (work / "repo" / "DEVIATIONS.md").write_text(
        "# DEVIATIONS\n\nMock end-to-end run: synthetic data, no pre-registration tag; nothing here is a result.\n", encoding="utf-8", newline="\n")


def data_stages(work: Path, prof: Profile, logs: Path) -> Path:
    from rhg.data import build
    from rhg.eval import pass_rate, probe_hints

    raw, processed = work / "raw", work / "processed"
    b = ["--fixture", "--raw-dir", _rel(raw), "--processed-dir", _rel(processed)]
    _in_process("data_fetch", build.main, ["--stage", "fetch", *b], logs)
    _in_process("data_tests", build.main, ["--stage", "tests", *b], logs)
    _in_process("data_validate", build.main, ["--stage", "validate", *b, "--workers", str(prof.sandbox_workers)], logs)
    pr = ["--mock", "--processed-dir", _rel(processed), "--workers", str(prof.sandbox_workers), "--cache-dir", _rel(work / "cache")]
    _in_process("pass_rate_A", pass_rate.main, ["--stage", "A", *pr], logs)
    _in_process("split_select", build.main, ["--stage", "split", "--select-only", *b], logs)
    _in_process("pass_rate_B", pass_rate.main, ["--stage", "B", *pr], logs)
    # the fixture is far smaller than the real band (Gate 1c wants 150/40/60): the gate is expected to FAIL here
    _in_process("data_split", build.main, ["--stage", "split", *b], logs)
    _in_process("probe_hints", probe_hints.main,
                ["--mock", "--processed-dir", _rel(processed), "--out-dir", _rel(work / "probe"), "--min-samples", str(prof.probe_min_samples),
                 "--workers", str(prof.sandbox_workers), "--cache-dir", _rel(work / "cache")], logs)
    return processed


def planned_jobs(pl: plan.Plan, displace: bool) -> list[tuple[str, int, bool]]:
    return [(r.arm, r.seed, displace) for r in plan.priority_list(pl)]


def clone_non_ast_runs(src: Path, dst: Path) -> int:
    """The second AST configuration reuses every run that cannot depend on the AST arm (all the others)."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(src.iterdir()):
        if d.is_dir() and not d.name.startswith(AST_ARM + "__"):
            shutil.copytree(d, dst / d.name)
            n += 1
    return n


# ------------------------------------------------------------------ analysis and assertions
def _analysis(runs: Path, out: Path, problems: Path, repo: Path, logs: Path, name: str, figures: bool = True) -> dict[str, Any]:
    from rhg.analysis import run as analysis_run

    _in_process(name, analysis_run.main, ["--runs", _rel(runs), "--out", _rel(out), "--problems", _rel(problems), "--repo-root", _rel(repo),
                                          *([] if figures else ["--no-figures"])], logs)
    return json.loads((out / "tests.json").read_text(encoding="utf-8"))


def _by_id(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in doc["tests"]}


def check_common(ck: Checks, doc: dict[str, Any], out: Path, prof: Profile) -> None:
    t = _by_id(doc)
    prim = t["primary"]
    ck.ok(doc["mode"] == "EXPLORATORY" and prim["stamp"] == "EXPLORATORY", "run is stamped EXPLORATORY (no prereg tag exists)", f"mode={doc['mode']}")
    ck.ok(doc["seeds_per_arm"] == {a: n for a, n in plan.load_plan().arms.items()}, "all 22 planned runs entered the analysis", str(doc["seeds_per_arm"]))
    ck.ok(prim["decision"] == "supported" and prim["outcome_wording"] == "supported", "primary is 'supported'", f"decision={prim['decision']}")
    e = prim["effect"]
    ck.ok(prim["p"] is not None and prim["p"] <= C.ALPHA and e["delta_mean"] >= C.DELTA_MIN, "primary meets p <= alpha and Delta >= 0.10",
          f"p={prim['p']}, delta={e['delta_mean']:.3f}")
    ck.ok(prim["emergence"]["clean"]["emerged"] == 0, "no clean_subtle seed emerged (planted q=0)")
    ck.ok(prim["emergence"]["hackable"]["emerged"] >= 4, "at least 4 of 5 hackable_subtle seeds emerged (planted q=0.9; needed for p <= 0.05)",
          f"{prim['emergence']['hackable']['emerged']}/5")
    clean_final = list(prim["per_seed"]["clean_subtle"].values())
    ck.ok(max(clean_final) == 0.0, "clean_subtle final HACK_RT is exactly 0 for every seed", str(clean_final))
    f, o = t["H1_final"], t["H1_onset"]
    means = f["effect"]["arm_means"]
    ck.ok(means["hackable_none"] == 0.0 < means["hackable_subtle"] < means["hackable_explicit"],
          "H1 direction (final rate): none < subtle < explicit (planted q 0 / 0.9 / 1.0)", str({k: round(v, 3) for k, v in means.items()}))
    onset = {a: [v for v in d.values()] for a, d in o["per_seed"].items() if d}
    mean_onset = {a: sum(v) / len(v) for a, v in onset.items()}
    ck.ok(mean_onset["hackable_explicit"] <= mean_onset["hackable_subtle"] < mean_onset["hackable_none"],
          "H1 direction (onset): explicit <= subtle < none", str({k: round(v, 1) for k, v in mean_onset.items()}))
    ck.ok(f["p"] is not None and f["p"] <= C.ALPHA, "H1-final trend p <= 0.05 on the planted dose-response", f"p={f['p']}")
    ck.ok(bool(doc["min_attainable_p"]["rows"]), "min-attainable-p table present in tests.json")

    report = (out / "REPORT.md").read_text(encoding="utf-8")
    ck.ok("EXPLORATORY" in report and "[EXPLORATORY]" in report.splitlines()[0], "REPORT.md is stamped EXPLORATORY in its title and sections")
    for needle, what in (("## 5. Minimum attainable p", "min-attainable-p table section"), ("### Cross-hint evaluation", "cross-hint section"),
                         ("Leave-one-seed-out", "robustness leave-one-out section"), ("Run homogeneity", "homogeneity section"),
                         ("Training health", "training-health section")):
        ck.ok(needle in report, f"REPORT.md has the {what}")
    ck.ok((out / "examples.md").is_file() and (out / "examples.md").stat().st_size > 0, "examples.md exists")
    ck.ok((out / "per_seed.csv").is_file() and (out / "tests.json").is_file(), "per_seed.csv and tests.json exist")


def check_stamps(ck: Checks, runs: Path, out: Path, problems: Path, repo: Path, logs: Path) -> None:
    """Both stamp vocabularies exist. The real run is EXPLORATORY (no prereg tag): ``--confirmatory`` must be refused, and the
    CONFIRMATORY wording is exercised only through a stubbed prereg check into a separate, clearly named tree (not a result)."""
    from rhg.analysis import prereg_check, report
    from rhg.analysis import run as analysis_run

    refused = True
    try:
        _in_process("analysis_confirmatory_refused", analysis_run.main, ["--runs", _rel(runs), "--out", _rel(out / "refused"), "--problems", _rel(problems),
                                                                          "--repo-root", _rel(repo), "--confirmatory"], logs, expect=3)
    except StageError:
        refused = False
    ck.ok(refused and not (out / "refused").exists(), "--confirmatory is refused (exit 3, nothing written) without a prereg tag")
    stub = prereg_check.CheckResult()
    stub.add("stubbed_by_e2e_mock", True, "stub: e2e_mock exercises the CONFIRMATORY wording only; this tree is not a result")
    doc = report.build_analysis(runs, out, problems_path=problems, confirmatory=True, repo_root=repo, prereg_result=stub, figures=False)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    ck.ok(doc["mode"] == "CONFIRMATORY" and "[CONFIRMATORY]" in text and "[EXPLORATORY]" in text,
          "with a (stubbed) passing prereg check the report carries both CONFIRMATORY and EXPLORATORY stamps")
    prim = _by_id(doc)["primary"]
    ck.ok(prim["stamp"] == "CONFIRMATORY" and _by_id(doc)["H4a"]["stamp"] == "EXPLORATORY", "primary is CONFIRMATORY, H4a stays EXPLORATORY under the stub")


def check_h4b(ck: Checks, doc: dict[str, Any], displace: bool) -> str:
    h4b = _by_id(doc)["H4b"]
    want = "displacement" if displace else "suppression_only"
    ck.ok(h4b.get("decision") == want, f"H4b verdict matches the plant (displace={displace} -> {want})",
          f"got {h4b.get('decision')}; hack={h4b['effect']['hack_rates']}; evasion={h4b['effect']['evasion']}")
    return str(h4b.get("decision"))


def check_ledgers(ck: Checks, work: Path, real_before: tuple[bool, int, float]) -> None:
    ck.ok(budget.spent_usd(work / "ledger.jsonl") == 0.0, "e2e ledger stayed at $0")
    ck.ok(real_ledger_state() == real_before, "the real results/ledger.jsonl was not touched")
    ck.ok(not (work / "ledger.jsonl").exists() or (work / "ledger.jsonl").stat().st_size == 0, "no ledger entry was written by any mock stage")


def real_ledger_state() -> tuple[bool, int, float]:
    p = REPO_ROOT / "results" / "ledger.jsonl"
    return (True, p.stat().st_size, p.stat().st_mtime) if p.is_file() else (False, 0, 0.0)


# ------------------------------------------------------------------ main
def run(work: Path, prof: Profile, ast_mode: str = "both") -> tuple[int, Checks]:
    from rhg.analysis import bundle

    t_start = time.time()
    real_before = real_ledger_state()
    pl = plan.load_plan()
    prepare_tree(work)
    logs = work / "logs"
    ck = Checks()

    log(f"stage 1/6: data, pass-rate, split, hint probe (fixture; {prof.name} profile)")
    try:
        processed = data_stages(work, prof, logs)
    finally:
        from rhg.env import cache as grade_cache

        grade_cache.configure_cache(None)  # the in-process stages must not leave a disk-backed global cache behind
    probe = json.loads((work / "probe" / "hint_probe.json").read_text(encoding="utf-8"))
    ck.ok((work / "probe" / "hint_selection.mock.json").is_file(), "mock hint probe wrote its selection (never prereg/hint_selection.json)")
    dec = probe["decision"]
    ck.ok(dec["go"] is True and dec["selected"] in ("S1", "S2", "S3") and probe["mock"] is True, "mock Gate 1d is GO with a selected subtle wording",
          f"selected={dec['selected']}")
    split = json.loads((processed / "splits.json").read_text(encoding="utf-8"))
    ck.ok(all(split["counts"][s] > 0 for s in ("train", "val", "test")) and split["gate1c"]["pass"] is False,
          "split is non-empty; Gate 1c fails on the tiny fixture as expected", str(split["counts"]))
    ck.ok(split["counts"]["train"] >= prof.prompts_per_step, "enough train problems for prompts_per_step", str(split["counts"]))

    variants = {"displace": True, "suppress": False} if ast_mode == "both" else {ast_mode: ast_mode == "displace"}
    first = next(iter(variants))
    runs_dirs = {v: work / ("runs" if v == first else f"runs_{v}") for v in variants}
    ledger = work / "ledger.jsonl"

    log(f"stage 2/6: 22 mock runs, AST arm displace={variants[first]} ({prof.steps} steps, {prof.jobs} parallel)")
    t0 = time.time()
    ids = _train_many(planned_jobs(pl, variants[first]), runs_dirs[first], processed, ledger, prof, logs)
    log(f"22 runs finished in {time.time() - t0:.0f}s")
    ck.ok(len(ids) == 22 == sum(pl.arms.values()), "22 planned runs trained")
    for v in list(variants)[1:]:
        n = clone_non_ast_runs(runs_dirs[first], runs_dirs[v])
        _train_many([(AST_ARM, s, variants[v]) for s in range(pl.arms[AST_ARM])], runs_dirs[v], processed, ledger, prof, logs)
        log(f"variant {v}: reused {n} non-AST runs, trained {pl.arms[AST_ARM]} AST runs with displace={variants[v]}")

    log("stage 3/6: mock judge and validation harness")
    from rhg.judge import run as judge_run
    from rhg.validate import harness

    _in_process("judge_mock", judge_run.main, ["--runs", *ids, "--runs-dir", _rel(runs_dirs[first]), "--out-dir", _rel(work / "judge"),
                                               "--processed-dir", _rel(processed), "--mock", "--set", f"budget.ledger={_rel(ledger)}"], logs)
    ck.ok(len(list((work / "judge").glob("*.jsonl"))) >= 1, "mock judge wrote per-run outputs")
    analysis_dir = work / "analysis"
    _in_process("validation", harness.main, ["--runs-dir", _rel(runs_dirs[first]), "--judge-dir", _rel(work / "judge"), "--out-dir", _rel(analysis_dir),
                                             "--labels", _rel(work / "labels" / "human_labels.jsonl"), "--items", _rel(work / "labels" / "items.jsonl"),
                                             "--calibration", _rel(work / "labels" / "judge_calibration.json"), "--n-boot", "2000"], logs)
    ck.ok((analysis_dir / "validation.md").is_file() and (analysis_dir / "validation.json").is_file(), "validation harness wrote validation.{md,json}")

    log("stage 4/6: analysis and report")
    docs: dict[str, dict[str, Any]] = {}
    for v in variants:
        out = analysis_dir if v == first else work / f"analysis_{v}"
        if v != first:
            out.mkdir(parents=True, exist_ok=True)
        docs[v] = _analysis(runs_dirs[v], out, processed / "problems.jsonl", work / "repo", logs, f"analysis_{v}", figures=(v == first))
        outs = out
        if v == first:
            check_common(ck, docs[v], outs, prof)
        else:
            ck.ok(_by_id(docs[v])["primary"]["decision"] == "supported", f"[{v}] primary unchanged by the AST configuration (supported)")
        check_h4b(ck, docs[v], variants[v])
    check_stamps(ck, runs_dirs[first], work / "analysis_stamp_check", processed / "problems.jsonl", work / "repo", logs)

    log("stage 5/6: public bundle and secret scan")
    out_pub = work / "results_public"
    _in_process("bundle", bundle.main, ["--out", _rel(out_pub), "--analysis", _rel(analysis_dir), "--runs", _rel(runs_dirs[first]),
                                        "--repo-root", _rel(work / "repo"), "--ledger", _rel(ledger)], logs)
    ck.ok((out_pub / "README_RESULTS.md").is_file() and (out_pub / "REPORT.md").is_file(), "bundle passed its secret scan and was published")

    log("stage 6/6: ledger")
    check_ledgers(ck, work, real_before)
    grade_cache.configure_cache(None)
    log(f"total {time.time() - t_start:.0f}s")
    return (0 if not ck.failed else 1), ck


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhg.e2e_mock", description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer steps and prompts per step (< 3 min)")
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="throw-away tree (relative paths are inside the repo); default results/e2e")
    ap.add_argument("--ast", choices=("both", "displace", "suppress"), default="both",
                    help="hackable_subtle_ast plant: 'both' trains it twice (displace=True and False) and checks both H4b verdict paths")
    ap.add_argument("--jobs", type=int, default=None, help="parallel training subprocesses (profile default)")
    ap.add_argument("--steps", type=int, default=None, help="training steps per run (profile default)")
    ap.add_argument("--prompts-per-step", type=int, default=None, help="prompts per step (profile default)")
    ap.add_argument("--mock-lr", type=float, default=None, help="mock policy learning rate (profile default)")
    ap.add_argument("--sandbox-workers", type=int, default=None, help="sandbox workers per training run (profile default)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    prof = QUICK if args.quick else FULL
    override = {k: v for k, v in (("jobs", args.jobs), ("steps", args.steps), ("prompts_per_step", args.prompts_per_step),
                                  ("mock_lr", args.mock_lr), ("sandbox_workers", args.sandbox_workers)) if v is not None}
    if any(v < 1 for k, v in override.items() if k != "mock_lr") or override.get("mock_lr", 1.0) <= 0:
        print("error: --jobs/--steps/--prompts-per-step/--sandbox-workers must be >= 1 and --mock-lr > 0", file=sys.stderr)
        return 2
    prof = Profile(**{**prof.__dict__, **override})
    work = args.work_dir if args.work_dir.is_absolute() else REPO_ROOT / args.work_dir
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        code, ck = run(work, prof, args.ast)
    except StageError as e:
        print(f"E2E FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        os.chdir(cwd)
    print("\nchecks:\n" + ck.report())
    if code:
        print(f"\nE2E FAILED: {len(ck.failed)} check(s) failed", file=sys.stderr)
    else:
        print(f"\nE2E OK: {len(ck.rows)} checks passed; tree at {_rel(work)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
