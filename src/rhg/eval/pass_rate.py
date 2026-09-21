"""Base-model pass-rate measurement, stages A and B (DESIGN §2.2).

``python -m rhg.eval.pass_rate --stage A|B [--mock] [--n 16] [--limit N] [--generate-only | --grade-only]``

Renders the **no-hint** prompt of every candidate (stage A, ``candidates.jsonl``) or band-selected problem
(stage B, ``selected_A.json`` written by ``rhg.data.build --stage split --select-only``), samples ``n`` completions at
the *training* sampler settings, grades them honestly (``clean`` mode, no monitor) and writes

* ``passrate_{S}.jsonl``: ``{problem_id, n, k_visible, k_full}`` (schema of ``docs/dataset_notes.md`` §5, consumed by
  ``rhg.data.build --stage split``; ``k_visible`` = reward tests pass, ``k_full`` = reward and held-out tests pass);
* ``passrate_{S}_stats.jsonl``: per problem completion-length / truncation / extraction-failure / timeout / crash
  counts (side file for the data notebook);
* ``completions_{S}.jsonl.gz``: the raw completions with prompt hashes and sampling seeds (see ``rhg.eval.pipeline``).

Stage B uses a different sampling seed from stage A (selecting on one sample and reporting on another avoids
winner's-curse bias, DESIGN §2.2); this is enforced in code. ``--generate-only`` skips grading (GPU box),
``--grade-only`` grades an existing completions file on CPU; the default grades in a background thread overlapped
with generation. Ledger entry (kind ``passrate``) covers model load + generation wall time only and is skipped in
``--mock``. ``--mock`` never touches real data: without an explicit directory it works in ``data/fixture/processed``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rhg.data import build
from rhg.data.load import read_revision_line
from rhg.data.prompts import load_prompts_cfg
from rhg.env.sandbox import grading_environment
from rhg.eval import pipeline as pl
from rhg.eval.generate import GeneratorBackendError, SamplingParams, planted_pass_behavior

# Fixed, distinct sampling seeds (independent of the training seed). Guarded by check_seed_independence().
STAGE_SEEDS = {"A": 61001, "B": 62002}
DEFAULT_N = 16
FIXTURE_PROCESSED = Path("data/fixture/processed")
REAL_PROCESSED = "data/processed"
EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2


def stage_seed(stage: str, override: int | None = None) -> int:
    if stage not in STAGE_SEEDS:
        raise ValueError(f"stage must be 'A' or 'B', got {stage!r}")
    return STAGE_SEEDS[stage] if override is None else int(override)


def recorded_seed(stage: str, processed_dir: Path) -> int | None:
    """Sampling seed an earlier run of ``stage`` used (completions header, else stats side file), if any."""
    comp = Path(processed_dir) / f"completions_{stage}.jsonl.gz"
    try:
        if comp.is_file():
            with gzip.open(comp, "rt", encoding="utf-8") as f:
                return int(json.loads(f.readline())["groups"][0]["seed"])
        stats = Path(processed_dir) / f"passrate_{stage}_stats.jsonl"
        if stats.is_file():
            with stats.open(encoding="utf-8") as f:
                return int(json.loads(f.readline())["seed"])
    except (OSError, EOFError, ValueError, KeyError, IndexError, TypeError):
        return None
    return None


def check_seed_independence(stage: str, seed: int, processed_dir: Path) -> None:
    """Stage A and B must sample with different seeds; refuse otherwise (never an ``assert``: survives ``-O``)."""
    other = "B" if stage == "A" else "A"
    if STAGE_SEEDS["A"] == STAGE_SEEDS["B"]:
        raise ValueError("STAGE_SEEDS['A'] and STAGE_SEEDS['B'] must differ")
    seen = recorded_seed(other, processed_dir)
    if seen is not None and seen == seed:
        raise ValueError(f"stage {stage} would use sampling seed {seed}, the same as the recorded stage {other} run; "
                         "stage B must be an independent sample of stage A")
    if seed == STAGE_SEEDS[other]:
        raise ValueError(f"seed {seed} is reserved for stage {other}")


def load_stage_problems(stage: str, processed_dir: Path, limit: int | None = None) -> tuple[list[dict], dict[str, dict]]:
    """(problems to sample for this stage, all candidates by id)."""
    cands = build.read_jsonl(processed_dir / "candidates.jsonl")
    by_id = {p["problem_id"]: p for p in cands}
    if stage == "A":
        problems = list(cands)
    else:
        sel_path = processed_dir / "selected_A.json"
        if not sel_path.is_file():
            raise pl.PipelineError(f"{sel_path} not found: run `python -m rhg.data.build --stage split --select-only` "
                                   "after stage A")
        ids = json.loads(sel_path.read_text(encoding="utf-8"))["problem_ids"]
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise pl.PipelineError(f"selected_A.json names problems missing from candidates.jsonl, e.g. {unknown[:3]}")
        problems = [by_id[i] for i in ids]
    if limit is not None:
        if limit < 1:
            raise pl.PipelineError("--limit must be >= 1")
        problems = problems[:limit]
    if not problems:
        raise pl.PipelineError(f"no problems to sample for stage {stage}")
    return problems, by_id


def aggregate(stage: str, groups: Sequence[pl.Group], graded: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Per-problem pass-rate rows and stats rows, in the groups' problem order."""
    (group,) = groups
    cells = pl.collect(graded, groups)
    rate_rows, stat_rows = [], []
    for p in group.problems:
        cell = cells[(group.hint, p["problem_id"])]
        n = len(cell)
        lab = [r["labels"] for r in cell]
        rate_rows.append({
            "problem_id": p["problem_id"], "n": n,
            "k_visible": sum(1 for x in lab if x["visible_pass"]),
            "k_full": sum(1 for x in lab if x["correct"]),
        })
        toks = [int(r["n_tokens"]) for r in cell]
        stat_rows.append({
            "problem_id": p["problem_id"], "stage": stage, "seed": group.seed, "n": n,
            "n_tokens_mean": round(sum(toks) / n, 4), "n_tokens_max": max(toks),
            "n_truncated": sum(1 for r in cell if r["truncated"]),
            "n_extract_fail": sum(1 for r in cell if not r["code_extracted"]),
            "n_timeout": sum(1 for x in lab if x["timeout"]),
            "n_crash": sum(1 for x in lab if x["crash"]),
            "n_defines_rt": sum(1 for x in lab if x["defines_rt"]),
        })
    return rate_rows, stat_rows


def write_outputs(stage: str, processed_dir: Path, rate_rows: list[dict], stat_rows: list[dict]) -> dict[str, Path]:
    paths = {"passrate": processed_dir / f"passrate_{stage}.jsonl", "stats": processed_dir / f"passrate_{stage}_stats.jsonl"}
    build.write_jsonl(paths["passrate"], rate_rows)
    build.write_jsonl(paths["stats"], stat_rows)
    return paths


def summarize(stage: str, cfg, rate_rows: Sequence[Mapping], stat_rows: Sequence[Mapping]) -> dict[str, Any]:
    n_prob = len(rate_rows)
    total = sum(r["n"] for r in rate_rows)
    lo, hi = cfg.data.band_low, cfg.data.band_high
    s = {
        "stage": stage, "n_problems": n_prob, "n_samples": total,
        "mean_pass_visible": sum(r["k_visible"] for r in rate_rows) / total,
        "mean_pass_full": sum(r["k_full"] for r in rate_rows) / total,
        "n_in_band": sum(1 for r in rate_rows if build.in_band(r["k_visible"], r["n"], lo, hi)),
        "band": [lo, hi],
        "truncation_rate": sum(r["n_truncated"] for r in stat_rows) / total,
        "extract_fail_rate": sum(r["n_extract_fail"] for r in stat_rows) / total,
        "timeout_rate": sum(r["n_timeout"] for r in stat_rows) / total,
        "crash_rate": sum(r["n_crash"] for r in stat_rows) / total,
    }
    return s


def run_stage(
    stage: str,
    *,
    cfg,
    processed_dir: Path,
    mock: bool,
    n: int | None = None,
    limit: int | None = None,
    seed: int | None = None,
    generate_only: bool = False,
    grade_only: bool = False,
    workers: int | None = None,
    chunk_prompts: int = pl.DEFAULT_CHUNK_PROMPTS,
    generator=None,
    behavior=None,
    model_name: str | None = None,
    gpu_mem_util: float = 0.85,
    record_ledger: bool | None = None,
    ledger: str | Path | None = None,
    prompts_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one stage end to end; returns a summary dict (``paths``, ``gen_wall_s``, ``summary`` when graded)."""
    if generate_only and grade_only:
        raise pl.PipelineError("--generate-only and --grade-only are mutually exclusive")
    processed_dir = Path(processed_dir)
    prompts_cfg = prompts_cfg if prompts_cfg is not None else load_prompts_cfg()
    record_ledger = (not mock) if record_ledger is None else record_ledger
    comp_path = processed_dir / f"completions_{stage}.jsonl.gz"
    result: dict[str, Any] = {"stage": stage, "completions": comp_path, "gen_wall_s": 0.0, "load_wall_s": 0.0}

    if grade_only:
        if n is not None or limit is not None or seed is not None:
            raise pl.PipelineError("--n/--limit/--seed are fixed by the completions file in --grade-only mode")
        header, rows = pl.read_completions(comp_path)
        if header.get("stage") != stage or header.get("purpose") != f"passrate_{stage}":
            raise pl.PipelineError(f"{comp_path} was not generated for stage {stage}")
        _, by_id = load_stage_problems("A", processed_dir)
        groups = pl.groups_from_header(header, by_id)
        if len(groups) != 1 or groups[0].hint != "none":
            raise pl.PipelineError(f"{comp_path}: a pass-rate file holds exactly one no-hint group")
        check_seed_independence(stage, groups[0].seed, processed_dir)
        graded = pl.grade_completions(header, rows, groups, cfg=cfg, prompts_cfg=prompts_cfg, workers=workers,
                                      chunk_prompts=chunk_prompts)
        result["mock"] = bool(header.get("generator", {}).get("kind") == "mock")
    else:
        n = DEFAULT_N if n is None else n
        if n < 1:
            raise pl.PipelineError("--n must be >= 1")
        seed = stage_seed(stage, seed)
        check_seed_independence(stage, seed, processed_dir)
        problems, _ = load_stage_problems(stage, processed_dir, limit)
        groups = [pl.Group("none", seed, n, problems)]
        params = SamplingParams.from_config(cfg)
        t0 = time.perf_counter()
        if generator is None:
            beh = behavior if behavior is not None else (planted_pass_behavior() if mock else None)
            generator, info = pl.load_generator(cfg, mock=mock, behavior=beh, model_name=model_name, gpu_mem_util=gpu_mem_util)
        else:
            info = {"kind": "custom", "model": None, "adapter": None}
        result["load_wall_s"] = time.perf_counter() - t0
        try:
            revision = read_revision_line(processed_dir)
        except Exception:  # noqa: BLE001 - provenance only
            revision = None
        header = pl.make_header(f"passrate_{stage}", groups, params, prompts_cfg, bool(cfg.model.enable_thinking), info,
                                {"stage": stage, "dataset_revision": revision})
        result["mock"] = info["kind"] == "mock"
        try:
            with pl.CompletionWriter(comp_path, header) as writer:
                out = pl.run_groups(groups, generator=generator, params=params, cfg=cfg, prompts_cfg=prompts_cfg,
                                    writer=writer, grade=not generate_only, workers=workers, chunk_prompts=chunk_prompts,
                                    purpose=f"passrate_{stage}")
        finally:
            generator.close()
        result["gen_wall_s"] = out.gen_wall_s
        graded = None if generate_only else out.graded
        if record_ledger:
            from rhg import budget

            budget.record("passrate", f"passrate_{stage}", result["load_wall_s"] + out.gen_wall_s,
                          usd_per_hour=cfg.budget.usd_per_hour, ledger=ledger or cfg.budget.ledger,
                          note=f"stage {stage}: load {result['load_wall_s']:.0f}s + generate {out.gen_wall_s:.0f}s, "
                               f"{len(problems)} problems x n={n}")

    if graded is not None:
        rate_rows, stat_rows = aggregate(stage, groups, graded)
        env = {**grading_environment(), "mock": bool(result["mock"])}
        for r in stat_rows:
            r["grader_env"] = env
        result["paths"] = write_outputs(stage, processed_dir, rate_rows, stat_rows)
        result["summary"] = summarize(stage, cfg, rate_rows, stat_rows)
        result["rates"] = rate_rows
    return result


def resolve_processed_dir(args: argparse.Namespace, cfg) -> Path:
    if args.processed_dir:
        return Path(args.processed_dir)
    if args.fixture or (args.mock and cfg.data.processed_dir == REAL_PROCESSED):
        return FIXTURE_PROCESSED  # mock output is fake data: never write it over the real pass-rate files
    return Path(cfg.data.processed_dir)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rhg.eval.pass_rate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--mock", action="store_true", help="MockGenerator with planted per-problem pass rates; no ledger entry")
    ap.add_argument("--n", type=int, default=None, help=f"samples per problem (default {DEFAULT_N})")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (smoke runs)")
    ap.add_argument("--seed", type=int, default=None, help="override the fixed stage sampling seed")
    ap.add_argument("--generate-only", action="store_true", help="write completions, do not grade (GPU box)")
    ap.add_argument("--grade-only", action="store_true", help="grade an existing completions file on CPU")
    ap.add_argument("--fixture", action="store_true", help="use data/fixture/processed")
    ap.add_argument("--processed-dir", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    ap.add_argument("--workers", type=int, default=None, help="sandbox workers (default: sandbox.workers, 0 = cores-2)")
    ap.add_argument("--chunk-prompts", type=int, default=pl.DEFAULT_CHUNK_PROMPTS, help="prompts per generate call / grading chunk")
    ap.add_argument("--model", default=None, help="model name override (default: model.name)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85, help="vLLM gpu_memory_utilization (no trainer is colocated)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="persist the grading cache (sqlite) here")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    from rhg.config import load_config

    try:
        cfg = load_config("clean_none", overrides=list(args.overrides))
    except Exception as e:  # noqa: BLE001 - ConfigError and friends
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    processed_dir = resolve_processed_dir(args, cfg)
    if args.cache_dir is not None:
        from rhg.env.cache import configure_cache

        configure_cache(args.cache_dir)
    try:
        res = run_stage(
            args.stage, cfg=cfg, processed_dir=processed_dir, mock=args.mock, n=args.n, limit=args.limit, seed=args.seed,
            generate_only=args.generate_only, grade_only=args.grade_only, workers=args.workers,
            chunk_prompts=args.chunk_prompts, model_name=args.model, gpu_mem_util=args.gpu_mem_util,
        )
    except (pl.PipelineError, FileNotFoundError, build.BuildError, GeneratorBackendError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    tag = "MOCK (fake data) " if res.get("mock") else ""
    if "summary" not in res:
        print(f"{tag}stage {args.stage}: completions -> {res['completions']} (generation {res['gen_wall_s']:.1f}s, not graded)")
        return EXIT_OK
    s = res["summary"]
    print(f"{tag}stage {args.stage}: {s['n_problems']} problems x n={s['n_samples'] // s['n_problems']}; "
          f"mean pass visible {s['mean_pass_visible']:.3f}, full {s['mean_pass_full']:.3f}; "
          f"{s['n_in_band']} in band {s['band']}; truncated {s['truncation_rate']:.3f}, "
          f"no-code {s['extract_fail_rate']:.3f}, timeout {s['timeout_rate']:.3f}, crash {s['crash_rate']:.3f}")
    print(f"wrote {res['paths']['passrate']} and {res['paths']['stats']}")
    if s["extract_fail_rate"] > 0.2 or s["truncation_rate"] > 0.1:
        print("WARNING: high extraction-failure/truncation rate; inspect passrate_*_stats.jsonl before trusting the band.",
              file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
