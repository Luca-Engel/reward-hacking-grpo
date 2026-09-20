"""Shared generation -> grading machinery for ``pass_rate`` and ``probe_hints``.

The paid GPU box should only *generate*; grading is CPU work that can run elsewhere. Hence three paths over one
code base:

* default: generate chunk by chunk, write each chunk to the completions file, and grade it in a background
  thread *while the next chunk is generating* (the GPU never waits for the sandbox);
* ``generate_only``: no grading;
* ``grade_only``: read a completions file and grade it on CPU.

All three end in the same aggregation over the same ``graded`` rows, ordered by the header's groups, so the
outputs of ``grade_only`` are byte-identical to those of the default path for the same completions (no
timestamps or wall times are ever written into output files; wall time goes to the ledger only).

Completions file (``*.jsonl.gz``, gzip mtime 0): line 1 is a header
``{"kind": "header", version, purpose, generator, sampling, enable_thinking, prompts_hash, groups: [{hint, seed, n,
problem_ids}], ...extra}``; then one ``{"kind": "completion", problem_id, hint, sample_idx, seed, prompt_sha,
completion, n_tokens, truncated}`` per sample. ``hint`` is a wording id (``none``, ``S1`` ...); ``seed`` is the
group's sampling seed. The file is written to ``<path>.part`` and renamed once complete.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rhg.data.prompts import build_prompt, prompts_hash, render_chat
from rhg.env.grader import GradeItem, grade_batch
from rhg.eval.generate import Completion, Generator, SamplingParams, generate_for, prompt_sha

COMPLETIONS_VERSION = 1
DEFAULT_CHUNK_PROMPTS = 128


class PipelineError(ValueError):
    """Bad or inconsistent input files (CLI exit code 2)."""


@dataclass
class Group:
    """One sampling job: ``n`` completions of the ``hint`` prompt of every problem, with one sampling seed."""

    hint: str
    seed: int
    n: int
    problems: list[Mapping[str, Any]]

    def header(self) -> dict[str, Any]:
        return {"hint": self.hint, "seed": self.seed, "n": self.n, "problem_ids": [p["problem_id"] for p in self.problems]}


def render_prompt(problem: Mapping[str, Any], hint: str, prompts_cfg: Mapping[str, Any], enable_thinking: bool) -> str:
    return render_chat(build_prompt(problem, hint, prompts_cfg), enable_thinking=enable_thinking)


# ------------------------------------------------------------------ completions file
class CompletionWriter:
    """Streaming writer for the completions file; renames ``.part`` to the final name on a clean close."""

    def __init__(self, path: str | Path, header: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = open(self.part, "wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = io.TextIOWrapper(self._gz, encoding="utf-8", newline="\n")
        self._dump({"kind": "header", **header})

    def _dump(self, obj: Mapping[str, Any]) -> None:
        self._text.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")

    def write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self._dump({"kind": "completion", **r})
        self._text.flush()

    def close(self, ok: bool = True) -> None:
        self._text.close()  # closes the gzip stream
        self._raw.close()
        if ok:
            os.replace(self.part, self.path)

    def __enter__(self) -> "CompletionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(ok=exc_type is None)


def read_completions(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"{path} not found: run with --generate-only (or the default path) first")
    rows: list[dict[str, Any]] = []
    header: dict[str, Any] | None = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if header is None:
                    if obj.get("kind") != "header":
                        raise PipelineError(f"{path}: first line is not a header")
                    header = obj
                elif obj.get("kind") == "completion":
                    rows.append(obj)
                else:
                    raise PipelineError(f"{path}: unexpected record kind {obj.get('kind')!r}")
    except (EOFError, OSError, ValueError) as e:
        if isinstance(e, PipelineError):
            raise
        raise PipelineError(f"{path}: unreadable or truncated completions file ({e})") from e
    if header is None:
        raise PipelineError(f"{path}: empty completions file")
    if header.get("version") != COMPLETIONS_VERSION:
        raise PipelineError(f"{path}: unsupported completions version {header.get('version')!r}")
    return header, rows


def make_header(purpose: str, groups: Sequence[Group], params: SamplingParams, prompts_cfg: Mapping[str, Any],
                enable_thinking: bool, generator_info: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": COMPLETIONS_VERSION,
        "purpose": purpose,
        "generator": dict(generator_info),
        "sampling": params.as_dict(),
        "enable_thinking": enable_thinking,
        "prompts_hash": prompts_hash(prompts_cfg),
        "groups": [g.header() for g in groups],
        **(dict(extra) if extra else {}),
    }


def groups_from_header(header: Mapping[str, Any], problems_by_id: Mapping[str, Mapping[str, Any]]) -> list[Group]:
    out = []
    for g in header["groups"]:
        missing = [pid for pid in g["problem_ids"] if pid not in problems_by_id]
        if missing:
            raise PipelineError(f"{len(missing)} problems named in the completions file are unknown here, e.g. {missing[:3]} "
                                "(different candidates/problems file than at generation time?)")
        out.append(Group(g["hint"], int(g["seed"]), int(g["n"]), [problems_by_id[pid] for pid in g["problem_ids"]]))
    return out


# ------------------------------------------------------------------ grading
def _slim(row: Mapping[str, Any], res) -> dict[str, Any]:
    return {
        "problem_id": row["problem_id"],
        "hint": row["hint"],
        "sample_idx": row["sample_idx"],
        "n_tokens": row["n_tokens"],
        "truncated": bool(row["truncated"]),
        "code_extracted": bool(res.raw["code_extracted"]),
        "labels": dict(res.labels),
    }


def grade_rows(rows: Sequence[Mapping[str, Any]], problems_by_id: Mapping[str, Mapping[str, Any]], cfg, workers: int | None) -> list[dict[str, Any]]:
    """Honest grading (``clean`` mode, no monitor) of completion rows; labels do not depend on the arm."""
    items = [GradeItem(problems_by_id[r["problem_id"]], r["completion"], "clean", None, cfg) for r in rows]
    results = grade_batch(items, workers=workers, cfg=cfg)
    return [_slim(r, res) for r, res in zip(rows, results)]


def make_rows(group: Group, chunk: Sequence[Mapping[str, Any]], prompts: Sequence[str], comps: Sequence[Sequence[Completion]]) -> list[dict[str, Any]]:
    rows = []
    for problem, prompt, cs in zip(chunk, prompts, comps):
        sha = prompt_sha(prompt)[:16]
        for j, c in enumerate(cs):
            rows.append({"problem_id": problem["problem_id"], "hint": group.hint, "sample_idx": j, "seed": group.seed,
                         "prompt_sha": sha, "completion": c.text, "n_tokens": int(c.n_tokens), "truncated": bool(c.truncated)})
    return rows


@dataclass
class RunOutput:
    graded: list[dict[str, Any]] = field(default_factory=list)
    gen_wall_s: float = 0.0  # time inside generator.generate only


def run_groups(
    groups: Sequence[Group],
    *,
    generator: Generator,
    params: SamplingParams,
    cfg,
    prompts_cfg: Mapping[str, Any],
    writer: CompletionWriter | None,
    grade: bool,
    workers: int | None = None,
    chunk_prompts: int = DEFAULT_CHUNK_PROMPTS,
    purpose: str = "",
) -> RunOutput:
    """Generate every group chunk by chunk; grade chunks in one background thread overlapped with generation."""
    if chunk_prompts < 1:
        raise ValueError("chunk_prompts must be >= 1")
    enable_thinking = bool(cfg.model.enable_thinking)
    by_id = {p["problem_id"]: p for g in groups for p in g.problems}
    out = RunOutput()
    futures: list[Future] = []
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rhg-grade") if grade else None
    ok = False
    try:
        for group in groups:
            for lo in range(0, len(group.problems), chunk_prompts):
                chunk = group.problems[lo:lo + chunk_prompts]
                prompts = [render_prompt(p, group.hint, prompts_cfg, enable_thinking) for p in chunk]
                metas = [{"problem_id": p["problem_id"], "hint": group.hint, "problem": p, "purpose": purpose} for p in chunk]
                t0 = time.perf_counter()
                comps = generate_for(generator, prompts, metas, group.n, params, group.seed)
                out.gen_wall_s += time.perf_counter() - t0
                if len(comps) != len(prompts) or any(len(cs) != group.n for cs in comps):
                    raise RuntimeError("generator returned a wrong number of prompts/completions")
                rows = make_rows(group, chunk, prompts, comps)
                if writer is not None:
                    writer.write(rows)
                if pool is not None:
                    futures.append(pool.submit(grade_rows, rows, by_id, cfg, workers))
                    for f in futures:  # fail fast: do not keep renting the GPU after a grading error
                        if f.done() and f.exception() is not None:
                            raise f.exception()  # type: ignore[misc]
        ok = True
    finally:
        if pool is not None:
            pool.shutdown(wait=ok, cancel_futures=not ok)
    for f in futures:
        out.graded.extend(f.result())
    return out


def grade_completions(
    header: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Group],
    *,
    cfg,
    prompts_cfg: Mapping[str, Any],
    workers: int | None = None,
    chunk_prompts: int = DEFAULT_CHUNK_PROMPTS,
) -> list[dict[str, Any]]:
    """CPU-only grading of a completions file's rows (``--grade-only``); validates them against ``groups``."""
    enable_thinking = bool(header.get("enable_thinking", cfg.model.enable_thinking))
    by_id = {p["problem_id"]: p for g in groups for p in g.problems}
    expected: dict[tuple[str, str], str] = {}
    for g in groups:
        for p in g.problems:
            expected[(g.hint, p["problem_id"])] = prompt_sha(render_prompt(p, g.hint, prompts_cfg, enable_thinking))[:16]
    seen: set[tuple[str, str, int]] = set()
    n_of = {g.hint: g.n for g in groups}
    seeds = {g.hint: g.seed for g in groups}
    for r in rows:
        key = (r["hint"], r["problem_id"])
        if key not in expected:
            raise PipelineError(f"completion for unknown (hint, problem) {key}")
        if r["prompt_sha"] != expected[key]:
            raise PipelineError(f"prompt for {key} differs from the one used at generation time (prompts.yaml or problems changed)")
        idx = (r["hint"], r["problem_id"], int(r["sample_idx"]))
        if idx in seen or not 0 <= idx[2] < n_of[r["hint"]]:
            raise PipelineError(f"duplicate or out-of-range sample {idx}")
        if r.get("seed") != seeds[r["hint"]]:
            raise PipelineError(f"sample {idx} has seed {r.get('seed')}, header says {seeds[r['hint']]}")
        seen.add(idx)
    want = sum(g.n * len(g.problems) for g in groups)
    if len(seen) != want:
        raise PipelineError(f"completions file is incomplete: {len(seen)} of {want} samples")
    step = max(1, chunk_prompts * max(g.n for g in groups))
    graded: list[dict[str, Any]] = []
    for lo in range(0, len(rows), step):
        graded.extend(grade_rows(rows[lo:lo + step], by_id, cfg, workers))
    return graded


def collect(graded: Iterable[Mapping[str, Any]], groups: Sequence[Group]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """``(hint, problem_id) -> graded rows`` sorted by sample index; validates that every cell has ``n`` samples."""
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {(g.hint, p["problem_id"]): [] for g in groups for p in g.problems}
    for r in graded:
        cells[(r["hint"], r["problem_id"])].append(r)
    for g in groups:
        for p in g.problems:
            cell = cells[(g.hint, p["problem_id"])]
            cell.sort(key=lambda r: r["sample_idx"])
            if [r["sample_idx"] for r in cell] != list(range(g.n)):
                raise PipelineError(f"cell {(g.hint, p['problem_id'])} does not hold samples 0..{g.n - 1}")
    return cells


# ------------------------------------------------------------------ generator construction
def load_generator(cfg, *, mock: bool, behavior=None, model_name: str | None = None, gpu_mem_util: float = 0.85):
    """(generator, info dict). ``mock`` -> ``MockGenerator``; else the real vLLM backend (needs the GPU stack)."""
    from rhg.eval.generate import MockGenerator, VLLMGenerator

    if mock:
        return (MockGenerator(behavior) if behavior is not None else MockGenerator()), {"kind": "mock", "model": None, "adapter": None}
    gen = VLLMGenerator.from_config(cfg, model_name=model_name, gpu_memory_utilization=gpu_mem_util)
    return gen, {"kind": "vllm", "model": model_name or cfg.model.name, "adapter": None}
