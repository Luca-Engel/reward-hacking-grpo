"""CLI of the pre-registered analysis: ``python -m rhg.analysis.run --runs results/runs [--out results/analysis] [--confirmatory]``.

Without ``--confirmatory`` every output is stamped EXPLORATORY. With it, the pre-registration check
(``rhg.analysis.prereg_check``) must pass, else nothing is written and the exit code is 3; so does a study below the
11-run floor of PREREG §6. Exit codes: 0 ok, 2 usage / no runs, 3 guard refused, 1 other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rhg.analysis import endpoints as E
from rhg.analysis import prereg_check
from rhg.analysis.examples import EXAMPLES_SEED
from rhg.budget import FLOOR_RUNS
from rhg.manifest import EXIT_GUARD_REFUSED, REPO_ROOT

DEFAULT_PROBLEMS = Path("data/processed/problems.jsonl")


def _resolve_problems(arg: Path | None, runs: Path) -> Path | None:
    if arg is not None:
        return arg
    if DEFAULT_PROBLEMS.is_file():
        return DEFAULT_PROBLEMS
    sibling = runs.parent / "problems.jsonl"  # rhg.analysis.simulate writes it next to runs/
    return sibling if sibling.is_file() else None


def _count_completed(runs: Path) -> int:
    pilot_min = E._pilot_seed_min()
    n = 0
    for d in runs.iterdir() if runs.is_dir() else []:
        ident = E.split_run_id(d.name)
        st = E._read_json(d / "status.json").get("status")
        if ident and ident[1] < pilot_min and st == "completed":
            n += 1
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhg.analysis.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("results/runs"), help="directory of run directories")
    ap.add_argument("--out", type=Path, default=Path("results/analysis"))
    ap.add_argument("--confirmatory", action="store_true", help="stamp the primary and the Holm family CONFIRMATORY (needs a passing prereg check)")
    ap.add_argument("--problems", type=Path, default=None, help="problems.jsonl with p_B_full (default data/processed/problems.jsonl, else next to --runs)")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="checkout holding DEVIATIONS.md and prereg/")
    ap.add_argument("--examples-seed", type=int, default=EXAMPLES_SEED)
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.runs.is_dir():
        print(f"error: {args.runs} is not a directory", file=sys.stderr)
        return 2
    n_done = _count_completed(args.runs)
    if n_done == 0:
        print(f"error: no completed run under {args.runs}", file=sys.stderr)
        return 2
    check = prereg_check.check(args.repo_root)
    if args.confirmatory:
        if not check.ok:
            print(check.format(), file=sys.stderr)
            print("refused: --confirmatory needs a passing pre-registration check (output would be EXPLORATORY)", file=sys.stderr)
            return EXIT_GUARD_REFUSED
        if n_done < FLOOR_RUNS:
            print(f"refused: {n_done} completed runs is below the {FLOOR_RUNS}-run floor of PREREG §6; not run confirmatorily", file=sys.stderr)
            return EXIT_GUARD_REFUSED
    from rhg.analysis.report import build_analysis

    try:
        doc = build_analysis(args.runs, args.out, problems_path=_resolve_problems(args.problems, args.runs), confirmatory=args.confirmatory,
                             repo_root=args.repo_root, prereg_result=check, examples_seed=args.examples_seed, figures=not args.no_figures)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    prim = next(t for t in doc["tests"] if t["id"] == "primary")
    print(f"[{doc['mode']}] wrote {args.out}/REPORT.md, tests.json, per_seed.csv; primary outcome: {prim.get('result') or 'not testable'}")
    print(json.dumps({"seeds_per_arm": doc["seeds_per_arm"], "robustness_flags": doc["robustness_flags"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
