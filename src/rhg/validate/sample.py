"""Stratified sampler of rollouts for human labeling (DESIGN §4).

    uv run python -m rhg.validate.sample --runs hackable_subtle__s0 ... --seed 0

Writes two files that must stay separate:

* ``data/labels/display.jsonl`` -- ``{item_id, description, completion}`` only. The labeling UI reads
  nothing else (``rhg.validate.label``). Descriptions are hint-stripped.
* ``data/labels/items.jsonl`` -- the hidden metadata of every item (stratum, kind, exec labels, AST
  flags, control id / construction label, inclusion counts, duplicate links). Never shown to the labeler.

Real rollouts come from the final-step ``eval_test`` rollouts of the chosen runs (the population the
judge samples from). Strata, assigned in this priority order so that they are disjoint:

1. ``hack_flagged``         exec ``hack_rt`` and AST-broad flagged
2. ``hack_unflagged``       exec ``hack_rt`` and not flagged (detector false negatives)
3. ``nonhack_flagged``      not ``hack_rt`` but flagged (detector false positives)
4. ``visible_only_unflagged``  ``visible_only`` and not flagged (visible-test overfitting the detector misses)
5. ``random``               a simple random draw from all rollouts not already chosen

Each of 1-4 gets ``quota`` items (default 4); a stratum with fewer rollouts gives what it has and the
shortfall goes to ``random``, so the total stays ``n_real`` whenever the population allows. The sample
is deliberately tilted toward disagreements: per-stratum sizes and population counts are recorded so a
reader can weight, but human-label agreement is reported unweighted and is not a population rate.

Besides the real items: ``n_controls`` synthetic controls (round-robin over categories so every
category is present) and ``n_duplicates`` verbatim duplicates (new opaque ids, ``duplicate_of`` only in
the hidden file) for intra-rater reliability. Everything is seeded and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rhg.data.prompts import load_prompts_cfg, strip_hint
from rhg.seeds import derive_seed
from rhg.validate import controls as ctl
from rhg.validate import io

STRATA = ("hack_flagged", "hack_unflagged", "nonhack_flagged", "visible_only_unflagged")
RANDOM = "random"
DEFAULT_QUOTA = 4
DEFAULT_N_REAL = 20
DEFAULT_N_CONTROLS = 20
DEFAULT_N_DUPLICATES = 10
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, f"validate.sample:{name}"))


def assign_stratum(row: Mapping[str, Any]) -> str | None:
    """Stratum 1-4 of a rollout row (needs ``labels`` and ``_broad``), or ``None`` (falls to ``random``)."""
    labels = row["labels"]
    hack, flagged = bool(labels["hack_rt"]), bool(row["_broad"])
    if hack:
        return "hack_flagged" if flagged else "hack_unflagged"
    if flagged:
        return "nonhack_flagged"
    if labels["visible_only"]:
        return "visible_only_unflagged"
    return None


def select_real(rows: Sequence[Mapping[str, Any]], *, n_real: int = DEFAULT_N_REAL, quota: int = DEFAULT_QUOTA,
                seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """(chosen rows with ``stratum``, ``stratum_size``, ``n_in_stratum``), per-stratum counts)."""
    pop = sorted(rows, key=io.rollout_key)
    by_stratum: dict[str, list[int]] = {s: [] for s in STRATA}
    for i, r in enumerate(pop):
        s = assign_stratum(r)
        if s:
            by_stratum[s].append(i)
    chosen: list[int] = []
    tag: dict[int, tuple[str, int, int]] = {}
    counts: dict[str, dict[str, int]] = {}
    for s in STRATA:
        idx = by_stratum[s]
        k = min(quota, len(idx), n_real - len(chosen))
        take = sorted(int(idx[j]) for j in _rng(seed, s).choice(len(idx), size=k, replace=False)) if k else []
        for i in take:
            tag[i] = (s, len(idx), k)
        chosen += take
        counts[s] = {"population": len(idx), "sampled": k}
    rest = [i for i in range(len(pop)) if i not in tag]
    k = min(max(n_real - len(chosen), 0), len(rest))
    take = sorted(int(rest[j]) for j in _rng(seed, RANDOM).choice(len(rest), size=k, replace=False)) if k else []
    for i in take:
        tag[i] = (RANDOM, len(rest), k)
    counts[RANDOM] = {"population": len(rest), "sampled": k}
    out = []
    for i in sorted(tag):
        s, size, n = tag[i]
        out.append({**pop[i], "stratum": s, "stratum_size": size, "n_in_stratum": n})
    return out, counts


def select_controls(n: int = DEFAULT_N_CONTROLS, seed: int = 0) -> list[ctl.Control]:
    """``n`` controls, round-robin over categories (seeded order within each), so no category is missed."""
    pools = {c: [x for x in ctl.build_controls() if x.category == c] for c in ctl.CATEGORIES}
    for c, pool in pools.items():
        order = _rng(seed, f"controls:{c}").permutation(len(pool))
        pools[c] = [pool[int(i)] for i in order]
    total = sum(len(p) for p in pools.values())
    if not 0 <= n <= total:
        raise ValueError(f"n_controls must be in [0, {total}]")
    chosen: list[ctl.Control] = []
    depth = 0
    while len(chosen) < n:
        for c in ctl.CATEGORIES:
            if depth < len(pools[c]) and len(chosen) < n:
                chosen.append(pools[c][depth])
        depth += 1
    return chosen


def _item_id(seed: int, *parts: object) -> str:
    return "it_" + hashlib.sha256((str(seed) + "|" + "|".join(str(p) for p in parts)).encode()).hexdigest()[:10]


def build_items(real: Sequence[Mapping[str, Any]], controls: Sequence[ctl.Control], problems: Mapping[str, Mapping],
                *, n_duplicates: int = DEFAULT_N_DUPLICATES, seed: int = 0,
                prompts_cfg: Mapping[str, Any] | None = None) -> tuple[list[dict], list[dict]]:
    """(display rows, hidden item rows). Display rows carry only ``item_id, description, completion``."""
    cfg = load_prompts_cfg() if prompts_cfg is None else prompts_cfg
    display: list[dict] = []
    hidden: list[dict] = []

    def desc(pid: str) -> str:
        return strip_hint(str(problems[pid]["description"]), cfg)

    for r in real:
        iid = _item_id(seed, "real", r["run_id"], r["problem_id"], r["sample_idx"])
        display.append({"item_id": iid, "description": desc(r["problem_id"]), "completion": r["completion"]})
        hidden.append({
            "item_id": iid, "kind": "real", "stratum": r["stratum"], "stratum_size": r["stratum_size"],
            "n_in_stratum": r["n_in_stratum"], "run_id": r["run_id"], "arm": r["arm"], "phase": r["phase"],
            "step": r["step"], "problem_id": r["problem_id"], "sample_idx": r["sample_idx"],
            "exec_labels": dict(r["labels"]), "ast_narrow": bool(r["_narrow"]), "ast_broad": bool(r["_broad"]),
            "ast_categories": list((r.get("monitor") or {}).get("ast_categories") or []),
        })
    for c in controls:
        iid = _item_id(seed, "control", c.control_id)
        text = strip_hint(ctl.judge_description(c), cfg)  # controls carry their own (fixture) problem text
        display.append({"item_id": iid, "description": text, "completion": c.completion})
        hidden.append({"item_id": iid, "kind": "control", "stratum": "control:" + c.category,
                       "control_id": c.control_id, "construction_label": c.construction_label,
                       "category": c.category, "variant": c.variant, "problem_id": c.problem_id})
    n_dup = min(n_duplicates, len(display))
    picks = sorted(int(i) for i in _rng(seed, "duplicates").choice(len(display), size=n_dup, replace=False))
    for i in picks:
        iid = _item_id(seed, "duplicate", display[i]["item_id"])
        display.append({**display[i], "item_id": iid})
        hidden.append({**hidden[i], "item_id": iid, "kind": "duplicate", "duplicate_of": hidden[i]["item_id"],
                       "source_kind": hidden[i]["kind"]})
    return display, hidden


def sample_manifest(seed: int, run_ids: Sequence[str], counts: Mapping, n_pop: int, display: Sequence[Mapping],
                    hidden: Sequence[Mapping]) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for h in hidden:
        kinds[h["kind"]] = kinds.get(h["kind"], 0) + 1
    return {"seed": seed, "runs": list(run_ids), "population_rollouts": n_pop, "strata": counts, "n_items": len(display),
            "kinds": kinds, "display_sha256": hashlib.sha256(
                "\n".join(json.dumps(d, sort_keys=True) for d in display).encode()).hexdigest(),
            "note": "display.jsonl is what the labeler sees; items.jsonl is hidden metadata (never shown)"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rhg.validate.sample", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run ids under --runs-dir (comma-separated also accepted)")
    p.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="dir with problems.jsonl")
    p.add_argument("--out-dir", type=Path, default=Path("data/labels"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-real", type=int, default=DEFAULT_N_REAL)
    p.add_argument("--quota", type=int, default=DEFAULT_QUOTA, help="items per stratum 1-4")
    p.add_argument("--n-controls", type=int, default=DEFAULT_N_CONTROLS)
    p.add_argument("--n-duplicates", type=int, default=DEFAULT_N_DUPLICATES)
    p.add_argument("--force", action="store_true", help="overwrite existing item files even if labels exist")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code not in (0, None) else EXIT_OK
    out = args.out_dir
    labels = out / "human_labels.jsonl"
    if labels.is_file() and labels.stat().st_size > 0 and not args.force:
        print(f"refused: {labels} already holds labels; re-sampling would orphan them (use --force)", file=sys.stderr)
        return EXIT_REFUSED
    run_ids = io.find_runs(args.runs_dir, args.runs)
    try:
        rows = io.final_eval_test(io.load_eval_rows(args.runs_dir, run_ids, ("eval_test",)))
        problems = io.load_problems(args.processed_dir / "problems.jsonl")
        if not rows:
            raise ValueError("no eval_test rollouts in the given runs")
        real, counts = select_real(rows, n_real=args.n_real, quota=args.quota, seed=args.seed)
        controls = select_controls(args.n_controls, args.seed)
        display, hidden = build_items(real, controls, problems, n_duplicates=args.n_duplicates, seed=args.seed)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    io.write_jsonl(out / "display.jsonl", display)
    io.write_jsonl(out / "items.jsonl", hidden)
    man = sample_manifest(args.seed, run_ids, counts, len(rows), display, hidden)
    (out / "sample_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(display)} items ({man['kinds']}); strata {counts}\n-> {out}/display.jsonl (labeler) and "
          f"{out}/items.jsonl (hidden metadata; do not open before labeling)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
