"""Dataset build CLI: ``python -m rhg.data.build --stage {fetch,tests,validate,split} [--fixture] [--widen]``.

Pipeline (files under ``data/raw/`` and ``data/processed/``, both gitignored):

``fetch``     raw dataset -> ``data/raw/LeetCodeDataset.jsonl`` + ``data/processed/DATASET_REVISION``
``tests``     parse ``check(candidate)`` into per-assert tests, split reward/held-out, drop unsupported
              problems, compute near-duplicate ``cluster_id`` -> ``converted.jsonl`` + ``tests_report.json``
``validate``  execute every reference solution against ALL its tests in the sandbox (and once more through
              the real grader on the reward+held-out tests) and drop failures -> ``candidates.jsonl`` +
              ``validation_report.json``
``split``     read ``passrate_A.jsonl`` / ``passrate_B.jsonl`` (produced later on the GPU box), select the
              ``p_A`` band, stratify by ``p_A`` tercile, assign whole near-duplicate clusters to
              train/val/test -> ``problems.jsonl`` + ``splits.json``; prints the Gate-1c checklist.

``--fixture`` runs everything on the synthetic problems under ``data/fixture/`` (never touches real data;
``split`` fabricates deterministic pass-rate files there when none exist). Exit codes: 0 ok, 1 other
failure (or a failed Gate 1c with ``--strict``), 2 usage / missing input.

Pass-rate files: one JSON object per line ``{"problem_id", "n", "k_visible", "k_full"}`` where ``n`` samples
were drawn, ``k_visible`` passed the reward (visible) tests and ``k_full`` passed reward + held-out tests
(so ``k_full <= k_visible <= n``). Stage A covers all candidates, stage B only the band-selected ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rhg.data import load as _load
from rhg.data.dedupe import cluster_histogram

WIDEN_BAND = (0.05, 0.50)  # the single pre-declared fallback (DESIGN §2.2)
SPLIT_SEED = "rhg-split-v1"  # fixed constant: the split must not depend on the training seed
TARGET_TEST, TARGET_VAL = 60, 40  # sizes once >= 250 problems are selected
TEST_FRAC, VAL_FRAC = 0.24, 0.16  # 60/250 and 40/250: capped ratios when fewer problems are available
GATE = {"min_train": 150, "min_val": 40, "min_test": 60, "min_ref_validity": 0.95}
BALANCE_MIN_P = 0.10  # Kruskal-Wallis p on p_A across splits must be >= this
REF_TIMEOUT_S = 60.0
FIXTURE_ROOT = Path("data/fixture")
SPLITS = ("train", "val", "test")


class BuildError(RuntimeError):
    """Missing/inconsistent input; ``main`` maps it to exit code 2."""


# ------------------------------------------------------------------ paths / io
def default_dirs(fixture: bool, raw_dir: Path | None = None, processed_dir: Path | None = None,
                 cfg_processed: str = "data/processed") -> tuple[Path, Path]:
    if fixture:
        return Path(raw_dir or FIXTURE_ROOT / "raw"), Path(processed_dir or FIXTURE_ROOT / "processed")
    return Path(raw_dir or "data/raw"), Path(processed_dir or cfg_processed)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        raise BuildError(f"{path} not found")
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def ensure_raw(raw_dir: Path, processed_dir: Path, fixture: bool) -> None:
    if fixture:
        from rhg.data.fixture import fixture_revision, write_fixture_raw

        write_fixture_raw(raw_dir)
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        (Path(processed_dir) / _load.REVISION_FILE).write_text(f"{fixture_revision()} datasets=n/a downloaded=fixture\n", encoding="utf-8")
    elif not (Path(raw_dir) / _load.RAW_FILE).is_file():
        _load.fetch_raw(raw_dir, processed_dir)


def _current_commit(processed_dir: Path) -> str | None:
    return _load.parse_revision_commit(_load.read_revision_line(processed_dir))


def _warn_revision(stage: str, report_path: Path, processed_dir: Path) -> None:
    try:
        upstream = json.loads(Path(report_path).read_text(encoding="utf-8")).get("dataset_commit")
    except (OSError, ValueError):
        return
    msg = _load.revision_warning(stage, upstream, processed_dir)
    if msg:
        print(msg, file=sys.stderr)


def _load_cfg(overrides: Sequence[str]):
    from rhg.config import load_config

    return load_config("clean_none", overrides=list(overrides))


# ------------------------------------------------------------------ stage: tests
def stage_tests(raw_dir: Path, processed_dir: Path, cfg, fixture: bool, limit: int | None = None) -> dict:
    records = _load.read_raw_records(raw_dir, fixture)
    if limit:
        records = records[:limit]
    k, max_h = cfg.data.k_reward_tests, cfg.data.max_heldout_tests
    source = "rhg/fixture" if fixture else _load.SOURCE
    problems, drops, report = _load.convert_records(records, k, max_h, source=source)
    report.update(dataset_commit=_current_commit(processed_dir), k_reward_tests=k, max_heldout_tests=max_h,
                  drops=drops, fixture=fixture)
    write_jsonl(processed_dir / "converted.jsonl", sorted(problems, key=lambda p: p["problem_id"]))
    write_json(processed_dir / "tests_report.json", report)
    return report


# ------------------------------------------------------------------ stage: validate
_REF_HARNESS = r'''
import json, sys, time

def main():
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    ns = {"__name__": "__rhg_reference__"}
    res = {"errors": {}, "passed": [], "durations": [], "stage_error": None}
    for name, src in (("prefix", payload["prefix"]), ("reference", payload["code"])):
        try:
            exec(compile(src, "<" + name + ">", "exec"), ns)
        except BaseException as e:
            res["stage_error"] = name + ":" + type(e).__name__
            break
    cand = None
    if res["stage_error"] is None:
        try:
            ep = payload["entry_point"]
            cand = ns[ep] if ep in ns else eval(ep, ns)
        except BaseException as e:
            res["stage_error"] = "entry_point:" + type(e).__name__
    for src in payload["tests"]:
        ok, t0 = False, time.perf_counter()
        if cand is not None:
            tns = dict(ns)
            tns["candidate"] = cand
            try:
                exec(compile(src, "<test>", "exec"), tns)
                ok = True
            except BaseException as e:
                res["errors"][type(e).__name__] = res["errors"].get(type(e).__name__, 0) + 1
        res["passed"].append(ok)
        res["durations"].append(round(time.perf_counter() - t0, 4))
    sys.stdout.write("\nRHG_REF_RESULT:" + json.dumps(res) + "\n")
    sys.stdout.flush()

main()
'''


def check_reference(problem: Mapping, timeout_s: float = REF_TIMEOUT_S, mem_mb: int = 2048) -> dict:
    """Run the reference against every test in ``problem["_all_tests"]`` (all of the dataset's asserts,
    not only the reward/held-out selection) in one sandbox process; returns a verdict dict."""
    from rhg.env.sandbox import run_python

    tests = [t["src"] for t in problem["_all_tests"]]
    payload = {"prefix": problem.get("import_prefix") or "", "code": problem["reference_solution"],
               "entry_point": problem["entry_point"], "tests": tests}
    sb = run_python(_REF_HARNESS, timeout_s=timeout_s, mem_mb=mem_mb, stdin_data=json.dumps(payload))
    out: dict[str, Any] = {"problem_id": problem["problem_id"], "n_tests": len(tests), "wall_s": round(sb.wall_s, 3)}
    if sb.status == "timeout":
        out.update(verdict="reference_timeout", detail=f">{timeout_s:g}s")
        return out
    result = None
    for line in reversed(sb.stdout.splitlines()):
        if line.startswith("RHG_REF_RESULT:"):
            result = json.loads(line[len("RHG_REF_RESULT:"):])
            break
    if result is None:
        out.update(verdict="reference_crash", detail=f"{sb.status}: {sb.stderr.strip()[-160:]}")
        return out
    passed = result["passed"]
    out["max_test_s"] = max(result["durations"], default=0.0)
    if result["stage_error"]:
        out.update(verdict="reference_crash", detail=result["stage_error"])
    elif not all(passed) or len(passed) != len(tests):
        out.update(verdict="reference_fails_tests", detail=f"{passed.count(False)}/{len(tests)} failed {result['errors']}")
    else:
        out["verdict"] = "ok"
    return out


def _all_tests_for(raw: Mapping, problem: Mapping) -> list[dict]:
    """Every parsed assert of the raw record (not only the K + <=20 selected)."""
    from rhg.data import tests_split as ts

    try:
        parsed = ts.parse_check(str(raw.get("test") or ""))
        if parsed.tests:
            return [{"id": i, "src": s, "kind": "assert"} for i, s in enumerate(parsed.tests)]
    except ts.CheckParseError:
        pass
    parsed = ts.tests_from_input_output(raw.get("input_output") or [])
    return [{"id": i, "src": s, "kind": "assert"} for i, s in enumerate(parsed.tests)]


def stage_validate(raw_dir: Path, processed_dir: Path, cfg, fixture: bool, workers: int = 0,
                   timeout_s: float = REF_TIMEOUT_S, limit: int | None = None, progress: bool = True) -> dict:
    conv_path = processed_dir / "converted.jsonl"
    _warn_revision("validate", processed_dir / "tests_report.json", processed_dir)
    problems = read_jsonl(conv_path)
    if limit:
        problems = problems[:limit]
    raw_by_id = {str(r["task_id"]): r for r in _load.read_raw_records(raw_dir, fixture)}
    n_workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 2)

    jobs = []
    for p in problems:
        q = dict(p)
        q["_all_tests"] = _all_tests_for(raw_by_id[p["problem_id"]], p)
        jobs.append(q)
    t0 = time.monotonic()
    done = 0

    def run(q: dict) -> dict:
        nonlocal done
        r = check_reference(q, timeout_s=timeout_s)
        done += 1
        if progress and done % 200 == 0:
            print(f"  validated {done}/{len(jobs)} ({time.monotonic() - t0:.0f}s)", file=sys.stderr, flush=True)
        return r

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        stage1 = list(pool.map(run, jobs))
    verdicts = {r["problem_id"]: r for r in stage1}

    # stage 2: the reference must also pass the selected tests through the real grader (real timeout)
    survivors = [p for p in problems if verdicts[p["problem_id"]]["verdict"] == "ok"]
    grader_failures = _grader_path_check(survivors, cfg, n_workers)
    for pid, detail in grader_failures.items():
        verdicts[pid] = {**verdicts[pid], "verdict": "reference_fails_in_grader", "detail": detail}

    drops = [{"problem_id": pid, "reason": v["verdict"], "detail": v.get("detail", "")}
             for pid, v in sorted(verdicts.items()) if v["verdict"] != "ok"]
    valid = [p for p in problems if verdicts[p["problem_id"]]["verdict"] == "ok"]
    max_test = sorted(v.get("max_test_s", 0.0) for v in stage1)
    report = {
        "dataset_commit": _current_commit(processed_dir),
        "fixture": fixture,
        "n_input": len(problems),
        "n_valid": len(valid),
        "reference_validity": len(valid) / len(problems) if problems else 0.0,
        "drop_reasons": dict(Counter(d["reason"] for d in drops)),
        "drops": drops,
        "ref_timeout_s": timeout_s,
        "grader_timeout_s": float(cfg.sandbox.timeout_s),
        "wall_s": round(time.monotonic() - t0, 1),
        "workers": n_workers,
        "slowest_single_test_s": {"max": max_test[-1] if max_test else 0.0,
                                  "p99": max_test[int(0.99 * (len(max_test) - 1))] if max_test else 0.0},
        "n_problems_with_test_over_1s": sum(1 for v in stage1 if v.get("max_test_s", 0) > 1.0),
    }
    write_jsonl(processed_dir / "candidates.jsonl", valid)
    write_json(processed_dir / "validation_report.json", report)
    return report


def _grader_path_check(problems: Sequence[dict], cfg, workers: int) -> dict[str, str]:
    """Problems whose reference does not pass reward+held-out tests via ``grade_batch`` (retried once
    sequentially so that a loaded machine does not cause spurious timeouts)."""
    from rhg.config import load_config
    from rhg.env.grader import GradeItem, grade_batch

    gcfg = load_config("clean_none", overrides=[f"sandbox.timeout_s={float(cfg.sandbox.timeout_s)}",
                                                f"sandbox.mem_mb={int(cfg.sandbox.mem_mb)}", "sandbox.cache=false"])

    def items(ps: Sequence[dict]):
        return [GradeItem(p, f"```python\n{p['reference_solution']}\n```", "clean", cfg=gcfg) for p in ps]

    def failed(ps: Sequence[dict], results) -> dict[str, str]:
        out = {}
        for p, r in zip(ps, results):
            ok = r.raw["visible_pass"] and r.raw["heldout_pass"] and not r.raw["timeout"] and not r.raw["crash"]
            if not ok:
                out[p["problem_id"]] = (f"visible={r.raw['visible_pass']} heldout={r.raw['heldout_pass']} "
                                        f"timeout={r.raw['timeout']} crash={r.raw['crash']}")
        return out

    if not problems:
        return {}
    first = failed(problems, grade_batch(items(problems), workers=workers, cfg=gcfg))
    if not first:
        return {}
    retry = [p for p in problems if p["problem_id"] in first]
    second = failed(retry, grade_batch(items(retry), workers=1, cfg=gcfg))
    return second


# ------------------------------------------------------------------ stage: split
def read_passrates(path: Path, known_ids: set[str] | None = None) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for r in read_jsonl(path):
        pid = r.get("problem_id")
        try:
            n, kv, kf = int(r["n"]), int(r["k_visible"]), int(r["k_full"])
        except (KeyError, TypeError, ValueError) as e:
            raise BuildError(f"{path}: bad row {r!r}: {e}") from e
        if not (n > 0 and 0 <= kf <= kv <= n):
            raise BuildError(f"{path}: {pid}: need 0 <= k_full <= k_visible <= n and n > 0, got n={n} k_visible={kv} k_full={kf}")
        if pid in rows:
            raise BuildError(f"{path}: duplicate problem_id {pid!r}")
        if known_ids is not None and pid not in known_ids:
            raise BuildError(f"{path}: unknown problem_id {pid!r} (not in candidates.jsonl; stale file?)")
        rows[pid] = {"n": n, "k_visible": kv, "k_full": kf}
    return rows


def in_band(k: int, n: int, low: float, high: float) -> bool:
    """``low <= k/n <= high`` with exact rational arithmetic (0.10 and 0.40 are inclusive)."""
    return Fraction(str(low)) <= Fraction(k, n) <= Fraction(str(high))


def select_band_ids(candidates: Sequence[Mapping], pass_a: Mapping[str, Mapping], low: float, high: float) -> list[str]:
    """Problem ids with ``low <= p_A <= high`` (``p_A = k_visible / n`` on stage A), sorted."""
    return sorted(p["problem_id"] for p in candidates
                  if p["problem_id"] in pass_a and in_band(pass_a[p["problem_id"]]["k_visible"], pass_a[p["problem_id"]]["n"], low, high))


def _h(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()


def split_hash(assignment: Mapping[str, str]) -> str:
    """sha256 of the sorted ``id:split`` lines joined by newlines."""
    return hashlib.sha256("\n".join(sorted(f"{pid}:{s}" for pid, s in assignment.items())).encode("utf-8")).hexdigest()


def _largest_remainder(total: int, weights: Sequence[int]) -> list[int]:
    s = sum(weights)
    if s == 0:
        return [0] * len(weights)
    raw = [total * w / s for w in weights]
    out = [int(math.floor(x)) for x in raw]
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in order[: total - sum(out)]:
        out[i] += 1
    return out


def size_targets(n: int) -> tuple[int, int]:
    """``(n_test, n_val)``: 60/40 when >= 250 problems, else the same 24%/16% ratios."""
    return min(TARGET_TEST, round(TEST_FRAC * n)), min(TARGET_VAL, round(VAL_FRAC * n))


def assign_splits(selected: Sequence[Mapping], seed: str = SPLIT_SEED) -> tuple[dict[str, str], dict[str, int]]:
    """Cluster-level stratified split. ``selected`` items need ``problem_id``, ``cluster_id``, ``p_A`` (Fraction-
    or float-valued). Returns ``({problem_id: split}, {problem_id: stratum})``.

    Strata are terciles of the cluster-mean ``p_A`` (cut on cumulative problem counts). Inside a stratum
    clusters are visited in a seeded pseudo-random order and placed into val/test/train by largest relative
    deficit when they fit; a fix-up pass moves the smallest train clusters into val/test until both reach
    their size target (or nothing is left), so val/test are never smaller than targeted when avoidable.
    """
    n = len(selected)
    clusters: dict[str, list[Mapping]] = defaultdict(list)
    for p in selected:
        clusters[p["cluster_id"]].append(p)
    mean_pa = {c: sum(Fraction(str(m["p_A"])) for m in ms) / len(ms) for c, ms in clusters.items()}
    ordered = sorted(clusters, key=lambda c: (mean_pa[c], _h(seed, "tie", c)))
    stratum_of: dict[str, int] = {}
    cum = 0
    for c in ordered:
        size = len(clusters[c])
        stratum_of[c] = min(2, int(3 * (cum + size / 2) / n)) if n else 0
        cum += size
    strata = [[c for c in ordered if stratum_of[c] == s] for s in range(3)]
    n_stratum = [sum(len(clusters[c]) for c in cs) for cs in strata]
    t_test, t_val = size_targets(n)
    test_t = _largest_remainder(t_test, n_stratum)
    val_t = _largest_remainder(t_val, n_stratum)

    assign: dict[str, str] = {}
    counts = {s: {"train": 0, "val": 0, "test": 0} for s in range(3)}
    for s in range(3):
        target = {"test": test_t[s], "val": val_t[s], "train": max(0, n_stratum[s] - test_t[s] - val_t[s])}
        for c in sorted(strata[s], key=lambda c: _h(seed, "order", c)):
            size = len(clusters[c])
            fits = [k for k in ("test", "val", "train") if target[k] - counts[s][k] >= size]
            if fits:
                pick = max(fits, key=lambda k: (Fraction(target[k] - counts[s][k], max(target[k], 1)), k == "test", k == "val"))
            else:
                pick = "train"
            counts[s][pick] += size
            for m in clusters[c]:
                assign[m["problem_id"]] = pick
    # fix-up: fill any remaining val/test deficit from the smallest train clusters (same stratum first)
    for split, total_target in (("test", t_test), ("val", t_val)):
        while sum(1 for v in assign.values() if v == split) < total_target:
            train_clusters = [c for c in clusters if all(assign[m["problem_id"]] == "train" for m in clusters[c])]
            if not train_clusters:
                break
            have = {s: counts[s][split] for s in range(3)}
            tgt = test_t if split == "test" else val_t
            c = min(train_clusters, key=lambda c: (have[stratum_of[c]] - tgt[stratum_of[c]], len(clusters[c]), _h(seed, "fix", c)))
            for m in clusters[c]:
                assign[m["problem_id"]] = split
            counts[stratum_of[c]][split] += len(clusters[c])
            counts[stratum_of[c]]["train"] -= len(clusters[c])
    strat = {m["problem_id"]: stratum_of[c] for c, ms in clusters.items() for m in ms}
    return assign, strat


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def balance_report(problems: Sequence[Mapping], strat: Mapping[str, int]) -> dict:
    """Per-split summary plus balance tests (Kruskal-Wallis on ``p_A``, chi-square on difficulty)."""
    from scipy import stats

    per = {}
    for s in SPLITS:
        ps = [p for p in problems if p["split"] == s]
        per[s] = {
            "n": len(ps),
            "mean_p_A": _mean([p["p_A"] for p in ps]),
            "mean_p_B_full": _mean([p["p_B_full"] for p in ps]),
            "mean_p_B_visible": _mean([p["p_B_visible"] for p in ps]),
            "difficulty": dict(Counter(p["difficulty"] for p in ps)),
            "tercile": dict(Counter(strat[p["problem_id"]] for p in ps)),
        }
    out: dict[str, Any] = {"per_split": per}
    groups = [[p["p_A"] for p in problems if p["split"] == s] for s in SPLITS]
    groups = [g for g in groups if g]
    try:
        out["p_A_kruskal_p"] = float(stats.kruskal(*groups).pvalue) if len(groups) > 1 else None
    except ValueError:  # all values identical
        out["p_A_kruskal_p"] = 1.0
    levels = sorted({p["difficulty"] for p in problems if p.get("difficulty")})
    table = [[sum(1 for p in problems if p["split"] == s and p["difficulty"] == d) for d in levels] for s in SPLITS]
    table = [row for row in table if sum(row)]
    cols = [j for j in range(len(levels)) if any(row[j] for row in table)]
    if len(table) > 1 and len(cols) > 1:
        out["difficulty_chi2_p"] = float(stats.chi2_contingency([[row[j] for j in cols] for row in table])[1])
    else:
        out["difficulty_chi2_p"] = None
    return out


def gate1c(counts: Mapping[str, int], ref_validity: float | None, balance: Mapping, clusters_intact: bool,
           band: Mapping) -> dict:
    items = []

    def add(name: str, status: str, value: Any, note: str = "") -> None:
        items.append({"item": name, "status": status, "value": value, "note": note})

    add("train >= %d" % GATE["min_train"], "PASS" if counts["train"] >= GATE["min_train"] else "FAIL", counts["train"])
    add("val >= %d" % GATE["min_val"], "PASS" if counts["val"] >= GATE["min_val"] else "FAIL", counts["val"])
    add("test >= %d" % GATE["min_test"], "PASS" if counts["test"] >= GATE["min_test"] else "FAIL", counts["test"])
    if ref_validity is None:
        add("reference validity >= %.0f%%" % (100 * GATE["min_ref_validity"]), "FAIL", None, "validation_report.json missing")
    else:
        add("reference validity >= %.0f%%" % (100 * GATE["min_ref_validity"]),
            "PASS" if ref_validity >= GATE["min_ref_validity"] else "FAIL", round(ref_validity, 4),
            "share of validated problems whose reference passed all its tests")
    p = balance.get("p_A_kruskal_p")
    add("p_A balanced across splits (Kruskal-Wallis p >= %.2f)" % BALANCE_MIN_P,
        "PASS" if p is None or p >= BALANCE_MIN_P else "FAIL", None if p is None else round(p, 4))
    d = balance.get("difficulty_chi2_p")
    add("difficulty mix across splits (chi-square p >= 0.05; informational)",
        "PASS" if d is None or d >= 0.05 else "WARN", None if d is None else round(d, 4))
    add("no near-duplicate cluster spans splits", "PASS" if clusters_intact else "FAIL", clusters_intact)
    add("band", "INFO", f"[{band['low']}, {band['high']}]" + (" (widened fallback)" if band["widened"] else ""))
    passed = all(i["status"] in ("PASS", "INFO", "WARN") for i in items)
    return {"pass": passed, "items": items}


def stage_split(processed_dir: Path, cfg, widen: bool = False, fixture: bool = False, select_only: bool = False) -> dict:
    cands_path = processed_dir / "candidates.jsonl"
    _warn_revision("split", processed_dir / "validation_report.json", processed_dir)
    candidates = read_jsonl(cands_path)
    ids = {p["problem_id"] for p in candidates}
    low, high = (WIDEN_BAND if widen else (cfg.data.band_low, cfg.data.band_high))
    band = {"low": low, "high": high, "widened": bool(widen),
            "default_low": cfg.data.band_low, "default_high": cfg.data.band_high}

    pa_path, pb_path = processed_dir / "passrate_A.jsonl", processed_dir / "passrate_B.jsonl"
    if fixture:
        from rhg.data.fixture import synthetic_passrates

        for path, stage in ((pa_path, "A"), (pb_path, "B")):
            if not path.is_file():
                write_jsonl(path, synthetic_passrates(candidates, stage))
    if not pa_path.is_file():
        raise BuildError(f"{pa_path} not found: run `python -m rhg.eval.pass_rate --stage A` on the GPU box first")
    pass_a = read_passrates(pa_path, ids)
    selected_ids = select_band_ids(candidates, pass_a, low, high)
    if select_only:
        sel = {"band": band, "n_candidates": len(candidates), "n_stage_A": len(pass_a), "problem_ids": selected_ids}
        write_json(processed_dir / "selected_A.json", sel)
        return {"select_only": True, "n_selected": len(selected_ids), "band": band}
    if not pb_path.is_file():
        raise BuildError(f"{pb_path} not found: run stage B on the {len(selected_ids)} band-selected problems "
                         f"(`--stage split --select-only` writes selected_A.json), then re-run split")
    pass_b = read_passrates(pb_path, ids)
    missing_b = [pid for pid in selected_ids if pid not in pass_b]
    if missing_b:
        raise BuildError(f"{pb_path} lacks stage-B rows for {len(missing_b)} selected problems, e.g. {missing_b[:3]}"
                         f" (band {band['low']}-{band['high']}{'; did you forget --widen?' if not widen else ''})")

    by_id = {p["problem_id"]: p for p in candidates}
    selected = []
    for pid in selected_ids:
        a, b = pass_a[pid], pass_b[pid]
        p = dict(by_id[pid])
        p.update(p_A=a["k_visible"] / a["n"], p_B_full=b["k_full"] / b["n"], p_B_visible=b["k_visible"] / b["n"])
        selected.append(p)
    assignment, strat = assign_splits(selected)
    for p in selected:
        p["split"] = assignment[p["problem_id"]]
    counts = {s: sum(1 for p in selected if p["split"] == s) for s in SPLITS}

    spans = defaultdict(set)
    for p in selected:
        spans[p["cluster_id"]].add(p["split"])
    intact = all(len(v) == 1 for v in spans.values())
    balance = balance_report(selected, strat)
    ref_validity = None
    try:
        ref_validity = json.loads((processed_dir / "validation_report.json").read_text(encoding="utf-8"))["reference_validity"]
    except (OSError, ValueError, KeyError):
        pass
    gate = gate1c(counts, ref_validity, balance, intact, band)
    sel_clusters = {p["problem_id"]: p["cluster_id"] for p in selected}
    splits = {
        "split_hash": split_hash(assignment),
        **{s: sorted(p["problem_id"] for p in selected if p["split"] == s) for s in SPLITS},
        "counts": counts,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "band": band,
        "split_seed": SPLIT_SEED,
        "size_rule": {"target_test": TARGET_TEST, "target_val": TARGET_VAL, "fallback_fractions": [TEST_FRAC, VAL_FRAC]},
        "dataset_revision": _load.read_revision_line(processed_dir),
        "cluster_stats_selected": cluster_histogram(sel_clusters),
        "balance": balance,
        "gate1c": gate,
    }
    write_jsonl(processed_dir / "problems.jsonl", selected)
    write_json(processed_dir / "splits.json", splits)
    return splits


def format_gate(splits: Mapping) -> str:
    g = splits["gate1c"]
    lines = [f"Gate 1c checklist (band [{splits['band']['low']}, {splits['band']['high']}]"
             f"{', WIDENED fallback' if splits['band']['widened'] else ''}; {splits['n_selected']} of "
             f"{splits['n_candidates']} candidates selected):"]
    for it in g["items"]:
        note = f"  ({it['note']})" if it["note"] else ""
        lines.append(f"  [{it['status']:<4}] {it['item']}: {it['value']}{note}")
    lines.append(f"GATE 1c: {'PASS' if g['pass'] else 'FAIL'}   split_hash={splits['split_hash']}")
    if not g["pass"] and not splits["band"]["widened"]:
        lines.append("  Pre-declared fallback: re-run with --widen (band [0.05, 0.50], once); then add MBPP-sanitized.")
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rhg.data.build", description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["fetch", "tests", "validate", "split"])
    ap.add_argument("--fixture", action="store_true", help="synthetic problems under data/fixture/ (no network)")
    ap.add_argument("--widen", action="store_true", help="split: use the single fallback band [0.05, 0.50]")
    ap.add_argument("--select-only", action="store_true", help="split: only write selected_A.json (band selection from stage A)")
    ap.add_argument("--strict", action="store_true", help="split: exit 1 when Gate 1c fails")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--processed-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=0, help="validate: parallel sandbox workers (0 = cores-2)")
    ap.add_argument("--ref-timeout", type=float, default=REF_TIMEOUT_S, help="validate: per-problem timeout for all tests (s)")
    ap.add_argument("--limit", type=int, default=None, help="tests/validate: only the first N records (smoke runs)")
    ap.add_argument("--revision", default=None, help="fetch: dataset commit to pin (default: current main)")
    ap.add_argument("--force", action="store_true", help="fetch: re-download")
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        cfg = _load_cfg(args.overrides)
    except Exception as e:  # ConfigError and friends
        print(f"error: {e}", file=sys.stderr)
        return 2
    raw_dir, processed_dir = default_dirs(args.fixture, args.raw_dir, args.processed_dir, cfg.data.processed_dir)
    try:
        if args.stage == "fetch":
            old = _current_commit(processed_dir)
            if args.fixture:
                ensure_raw(raw_dir, processed_dir, True)
                print(f"fixture raw written to {raw_dir} ({_load.read_revision_line(processed_dir)})")
            else:
                info = _load.fetch_raw(raw_dir, processed_dir, revision=args.revision, force=args.force)
                print(json.dumps(info))
                if old and old != info["commit"]:
                    print(f"WARNING: dataset revision changed ({old} -> {info['commit']}); re-run tests/validate/split.", file=sys.stderr)
        elif args.stage == "tests":
            rep = stage_tests(raw_dir, processed_dir, cfg, args.fixture, args.limit)
            print(json.dumps({k: rep[k] for k in ("n_input", "n_kept", "drop_reasons", "tests_source", "n_tests_per_problem")}, indent=2))
        elif args.stage == "validate":
            rep = stage_validate(raw_dir, processed_dir, cfg, args.fixture, args.workers, args.ref_timeout, args.limit)
            print(json.dumps({k: rep[k] for k in ("n_input", "n_valid", "reference_validity", "drop_reasons", "wall_s")}, indent=2))
        else:
            res = stage_split(processed_dir, cfg, args.widen, args.fixture, args.select_only)
            if res.get("select_only"):
                print(f"selected {res['n_selected']} problems in band [{res['band']['low']}, {res['band']['high']}] -> selected_A.json")
                return 0
            print(format_gate(res))
            if args.strict and not res["gate1c"]["pass"]:
                return 1
    except (BuildError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
