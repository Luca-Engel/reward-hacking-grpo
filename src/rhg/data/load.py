"""Fetch ``newfacade/LeetCodeDataset`` and convert its records to the internal problem schema.

Real schema (inspected 2026-09, see ``docs/dataset_notes.md``): ``task_id`` (slug), ``question_id``,
``difficulty``, ``tags``, ``problem_description``, ``starter_code``, ``estimated_date``, ``prompt`` (the
*import prefix*: imports plus ``ListNode``/``TreeNode`` helpers, only 4 distinct values), ``completion``
(reference solution), ``entry_point`` (``Solution().method``), ``test`` (``def check(candidate)`` made of
``assert`` statements), ``input_output``, ``query``/``response`` (unused: LLM-written explanations).

Mapping: ``problem_id=task_id``, ``description=problem_description`` (NBSP -> space, trailing blanks
trimmed), ``import_prefix=prompt``, ``reference_solution=completion``, ``date=estimated_date``.

CLI: ``python -m rhg.data.load [--fixture] [--inspect]`` prints the schema/field summary of the raw file
(fetching it first unless ``--fixture``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from rhg.data import tests_split as ts
from rhg.data.dedupe import cluster_histogram, cluster_problems

DATASET_ID = "newfacade/LeetCodeDataset"
SOURCE = DATASET_ID
RAW_FILE = "LeetCodeDataset.jsonl"
REVISION_FILE = "DATASET_REVISION"

DROP_MISSING_FIELD = "missing_field"
DROP_PARSE_ERROR = "test_parse_error"
DROP_TOO_FEW = "too_few_tests"
DROP_HELPER = "unsupported_helper"
DROP_PREFIX = "prefix_import_unavailable"

_MULTI_BLANK = re.compile(r"\n{3,}")


# ------------------------------------------------------------------ provenance
def revision_line(commit: str, datasets_version: str, downloaded: str) -> str:
    return f"{DATASET_ID}@{commit} datasets={datasets_version} downloaded={downloaded}"


def parse_revision_commit(line: str | None) -> str | None:
    """Commit token of a ``DATASET_REVISION`` line (``name@commit key=val ...``)."""
    if not line or not line.strip():
        return None
    head = line.split()[0]
    return head.split("@", 1)[1] if "@" in head else head


def read_revision_line(processed_dir: Path) -> str | None:
    try:
        return (Path(processed_dir) / REVISION_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def revision_warning(stage: str, upstream_commit: str | None, processed_dir: Path) -> str | None:
    """Message when the revision recorded by an upstream stage differs from the current one."""
    current = parse_revision_commit(read_revision_line(processed_dir))
    if upstream_commit and current and upstream_commit != current:
        return (
            f"WARNING: dataset revision changed between stages: upstream output was built from "
            f"{upstream_commit}, {REVISION_FILE} now says {current} (stage: {stage}). Re-run the earlier stages."
        )
    return None


def _jsonable(v: Any) -> Any:
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return v


def fetch_raw(raw_dir: Path, processed_dir: Path, revision: str | None = None, force: bool = False) -> dict:
    """Download the dataset (pinned revision) into ``raw_dir`` and write ``DATASET_REVISION``."""
    import datasets
    from datasets import load_dataset
    from huggingface_hub import HfApi

    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    commit = revision or HfApi().dataset_info(DATASET_ID).sha
    out = raw_dir / RAW_FILE
    if out.is_file() and not force and parse_revision_commit(read_revision_line(processed_dir)) == commit:
        return {"commit": commit, "path": str(out), "downloaded": False}
    ds = load_dataset(DATASET_ID, revision=commit, cache_dir=str(raw_dir / "hf"))
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for split_name, part in ds.items():
            for row in part:
                rec = {k: _jsonable(v) for k, v in row.items()}
                rec["hf_split"] = split_name
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    (processed_dir / REVISION_FILE).write_text(revision_line(commit, datasets.__version__, today) + "\n", encoding="utf-8")
    return {"commit": commit, "path": str(out), "downloaded": True, "n_records": n}


def read_raw_records(raw_dir: Path, fixture: bool = False) -> list[dict]:
    path = Path(raw_dir) / ("fixture_raw.jsonl" if fixture else RAW_FILE)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m rhg.data.build --stage fetch"
                                f"{' --fixture' if fixture else ''}` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


# ------------------------------------------------------------------ conversion
def clean_description(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_BLANK.sub("\n\n", text).strip()


@dataclass
class Conversion:
    problem: dict | None
    drop_reason: str | None = None
    detail: str = ""
    tests_source: str = ""
    n_tests: int = 0
    n_duplicates: int = 0
    n_compound: int = 0
    has_preamble: bool = False
    output_share: float = 0.0
    notes: list[str] = field(default_factory=list)


def _expected_share(tests: list[str]) -> float:
    """Share of the most common right-hand side among ``assert candidate(...) == <rhs>`` tests.

    Text-based (the text after the last `` == ``) and therefore approximate; it is a diagnostic for
    degenerate problems (a constant answer passing most tests), never a drop rule.
    """
    counts: Counter[str] = Counter()
    for t in tests:
        head, sep, rhs = t.strip().rpartition(" == ")
        if sep and head.startswith("assert candidate("):
            counts[rhs.strip()] += 1
    return max(counts.values()) / len(tests) if counts and tests else 0.0


def convert_record(
    raw: Mapping[str, Any],
    k_reward: int,
    max_heldout: int,
    source: str = SOURCE,
    prefix_names: Callable[[str], Iterable[str]] | None = None,
) -> Conversion:
    """One raw record -> problem dict (no ``cluster_id`` yet) or a drop with reason."""
    pid = raw.get("task_id")
    required = ("task_id", "problem_description", "starter_code", "entry_point", "completion")
    missing = [f for f in required if not str(raw.get(f) or "").strip()]
    if missing:
        return Conversion(None, DROP_MISSING_FIELD, ",".join(missing))

    parsed: ts.ParsedTests | None = None
    err = ""
    try:
        parsed = ts.parse_check(str(raw.get("test") or ""))
        if not parsed.tests:
            err, parsed = "check() has no assert", None
    except ts.CheckParseError as e:
        err = str(e)
    if parsed is None:
        io = raw.get("input_output") or []
        uses_helpers = re.search(r"\b(ListNode|TreeNode|Node)\b", str(raw.get("starter_code", "")))
        if io and not uses_helpers:
            parsed = ts.tests_from_input_output(io)
            if not parsed.tests:
                parsed = None
        if parsed is None:
            return Conversion(None, DROP_PARSE_ERROR, err or "no usable tests")

    conv = Conversion(
        None,
        tests_source=parsed.source,
        n_tests=len(parsed.tests),
        n_duplicates=parsed.n_duplicates,
        n_compound=parsed.n_compound,
        has_preamble=bool(parsed.preamble),
        output_share=_expected_share(parsed.tests),
    )
    if len(parsed.tests) < k_reward + ts.MIN_EXTRA_TESTS:
        conv.drop_reason, conv.detail = DROP_TOO_FEW, f"{len(parsed.tests)} < K+{ts.MIN_EXTRA_TESTS}"
        return conv

    prefix = str(raw.get("prompt") or "")
    names_fn = prefix_names or ts.prefix_names
    try:
        known = set(names_fn(prefix))
    except ts.PrefixError as e:
        conv.drop_reason, conv.detail = DROP_PREFIX, str(e)
        return conv
    missing_names = ts.free_names("\n".join(parsed.tests)) - known
    if missing_names:
        conv.drop_reason, conv.detail = DROP_HELPER, ",".join(sorted(missing_names))
        return conv

    reward, heldout = ts.split_tests(str(pid), parsed.tests, k_reward, max_heldout)
    conv.problem = {
        "problem_id": str(pid),
        "source": source,
        "difficulty": raw.get("difficulty"),
        "tags": list(raw.get("tags") or []),
        "date": str(raw.get("estimated_date") or "")[:10] or None,
        "description": clean_description(str(raw["problem_description"])),
        "starter_code": str(raw["starter_code"]),
        "import_prefix": prefix,
        "entry_point": str(raw["entry_point"]).strip(),
        "reference_solution": str(raw["completion"]),
        "reward_tests": reward,
        "heldout_tests": heldout,
    }
    return conv


_WORKER_PREFIXES: dict[str, Any] = {}


def _init_worker(table: dict[str, Any]) -> None:
    _WORKER_PREFIXES.update(table)


def _prefix_from_table(prefix: str) -> frozenset[str]:
    v = _WORKER_PREFIXES[prefix]
    if isinstance(v, str):
        raise ts.PrefixError(v)
    return v


def _convert_worker(args: tuple) -> Conversion:
    raw, k_reward, max_heldout, source = args
    return convert_record(raw, k_reward, max_heldout, source, _prefix_from_table)


def _convert_all(records, k_reward, max_heldout, source, prefix_names, workers):
    """``convert_record`` over all records; in a process pool for big inputs (AST work dominates).

    Import prefixes are executed once each, in the parent's sandbox, and handed to the workers.
    """
    n_workers = workers if workers is not None else max(1, min(12, (os.cpu_count() or 2) - 2))
    if prefix_names is not None or n_workers <= 1 or len(records) < 64:
        return [convert_record(r, k_reward, max_heldout, source, prefix_names) for r in records]
    table: dict[str, Any] = {}
    for prefix in {str(r.get("prompt") or "") for r in records}:
        try:
            table[prefix] = ts.prefix_names(prefix)
        except ts.PrefixError as e:
            table[prefix] = str(e)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(table,)) as pool:
        return list(pool.map(_convert_worker, [(r, k_reward, max_heldout, source) for r in records], chunksize=8))


def convert_records(
    records: Iterable[Mapping[str, Any]],
    k_reward: int,
    max_heldout: int,
    source: str = SOURCE,
    prefix_names: Callable[[str], Iterable[str]] | None = None,
    workers: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Convert all records; returns ``(problems, drops, report)``; problems carry ``cluster_id``.

    Clusters are computed over *every* record with a description (also dropped ones) so ``cluster_id``
    does not depend on which problems later stages drop.
    """
    records = list(records)
    problems: list[dict] = []
    drops: list[dict] = []
    src_counts: Counter[str] = Counter()
    n_tests: list[int] = []
    n_dup = n_compound = n_preamble = n_degenerate = 0
    helper_names: Counter[str] = Counter()
    for raw, c in zip(records, _convert_all(records, k_reward, max_heldout, source, prefix_names, workers)):
        if c.tests_source:
            src_counts[c.tests_source] += 1
        n_dup += c.n_duplicates
        n_compound += c.n_compound
        n_preamble += c.has_preamble
        if c.problem is None:
            drops.append({"problem_id": raw.get("task_id"), "reason": c.drop_reason, "detail": c.detail, "n_tests": c.n_tests})
            if c.drop_reason == DROP_HELPER:
                helper_names.update(c.detail.split(","))
            continue
        n_tests.append(c.n_tests)
        n_degenerate += c.output_share >= 0.9
        problems.append(c.problem)

    cluster_input = [
        {"problem_id": str(r.get("task_id")), "description": clean_description(str(r.get("problem_description") or ""))}
        for r in records
        if str(r.get("task_id") or "")
    ]
    assign, cstats = cluster_problems(cluster_input, return_stats=True)
    kept_assign = {}
    for p in problems:
        p["cluster_id"] = assign[p["problem_id"]]
        kept_assign[p["problem_id"]] = p["cluster_id"]
    report = {
        "n_input": len(records),
        "n_kept": len(problems),
        "drop_reasons": dict(Counter(d["reason"] for d in drops)),
        "unsupported_helper_names": dict(helper_names),
        "tests_source": dict(src_counts),
        "n_tests_per_problem": (
            {"min": min(n_tests), "median": statistics.median(n_tests), "max": max(n_tests)} if n_tests else {}
        ),
        "n_duplicate_asserts_removed": n_dup,
        "n_compound_tests": n_compound,
        "n_problems_with_preamble": n_preamble,
        "n_degenerate_output_share_ge_0.9": n_degenerate,
        "clusters_all_records": cluster_histogram(assign),
        "clusters_kept": cluster_histogram(kept_assign),
        "cluster_edges": cstats,
    }
    drops.sort(key=lambda d: str(d["problem_id"]))
    return problems, drops, report


# ------------------------------------------------------------------ CLI
def inspect_records(records: list[dict]) -> dict:
    keys: dict[str, Counter] = {}
    for r in records:
        for k, v in r.items():
            keys.setdefault(k, Counter())[type(v).__name__] += 1
    return {
        "n_records": len(records),
        "fields": {k: dict(c) for k, c in keys.items()},
        "difficulty": dict(Counter(r.get("difficulty") for r in records)),
        "distinct_import_prefixes": len({r.get("prompt") for r in records}),
        "distinct_entry_point_shapes": dict(Counter("Solution()." if str(r.get("entry_point", "")).startswith("Solution().") else "other" for r in records)),
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rhg.data.load", description="Fetch/inspect the raw dataset.")
    ap.add_argument("--fixture", action="store_true", help="use the synthetic fixture (no network)")
    ap.add_argument("--inspect", action="store_true", help="print a schema summary of the raw records")
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--processed-dir", type=Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    from rhg.data.build import default_dirs, ensure_raw

    raw_dir, processed_dir = default_dirs(args.fixture, args.raw_dir, args.processed_dir)
    ensure_raw(raw_dir, processed_dir, args.fixture)
    records = read_raw_records(raw_dir, args.fixture)
    print(json.dumps(inspect_records(records) if args.inspect else {"n_records": len(records)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
