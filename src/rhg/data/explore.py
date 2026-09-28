"""Computations behind ``notebooks/01_data_exploration.ipynb``.

Everything here is a pure function over plain records / DataFrames (no notebook state), so it can be unit
tested; the notebook cells only call these and draw. Sections A..L follow the notebook. Pieces that need
outputs of later stages (pass rates, splits, hint probe) take them as arguments and the notebook gates on
their files (``load_inputs(...).available``).

Heuristic thresholds used for the automatic flags (section L) are module constants below; they mark things a
reader must look at, they are not pre-registered decision rules.

Ideas credited: the near-duplicate rule reuses ``rhg.data.dedupe`` (5-word shingles, Jaccard); the Newcombe
hybrid-score interval for a difference of proportions is Newcombe (1998), Stat. Med. 17:873 (method 10).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from rhg.analysis.stats import wilson
from rhg.data.dedupe import SHINGLE_WORDS, jaccard, normalize_text, normalize_title, shingles

# ------------------------------------------------------------------ constants
QWEN3_HORIZON_ASSUMED = "2024-06-30"  # UNVERIFIED: Qwen3's pretraining cutoff is not published; a labelled assumption
CONTAM_HORIZONS = ("2023-12-31", "2024-06-30", "2024-12-31")  # grid reported next to the assumed horizon
TOKENIZER_NAME = "Qwen/Qwen3-1.7B"
HINT_IDS = ("none", "S1", "S2", "S3", "E1")
DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}
SPLITS = ("train", "val", "test")
REQUIRED_KEYS = ("problem_id", "source", "difficulty", "tags", "date", "description", "starter_code", "import_prefix",
                 "entry_point", "reference_solution", "reward_tests", "heldout_tests", "cluster_id")
PASSRATE_KEYS = ("p_A", "p_B_full", "p_B_visible", "split")
NEAR_DUP_REPORT_MIN = 0.5  # pairs at or above this Jaccard are listed
NEAR_DUP_THRESHOLD = 0.8  # the clustering rule (rhg.data.dedupe.JACCARD_THRESHOLD)
# flag heuristics
DEGENERATE_WARN_SHARE = 0.05
TRUNC_WARN_RATE = 0.10
EXTRACT_FAIL_WARN_RATE = 0.10
WEAK_TESTS_WARN_SHARE = 0.25
SLOW_REFERENCE_S = 2.0
BALANCE_P_WARN = 0.01
BALANCE_KS_LARGE = 0.30
MIN_PROBLEMS = {"train": 150, "val": 40, "test": 60}

_ASSERT_HELPERS = ("is_same_list", "is_same_tree")


# ------------------------------------------------------------------ I/O
def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _opt(path: Path, reader: Callable[[Path], Any]) -> Any:
    return reader(path) if path.is_file() else None


@dataclass
class Inputs:
    """Everything the notebook reads. Missing files are ``None``; ``available`` says which gates are open."""

    processed_dir: Path
    probe_dir: Path
    candidates: list[dict]
    problems: list[dict] | None = None
    splits: dict | None = None
    pass_a: list[dict] | None = None
    pass_b: list[dict] | None = None
    stats_a: list[dict] | None = None
    stats_b: list[dict] | None = None
    probe: dict | None = None
    tests_report: dict | None = None
    validation_report: dict | None = None
    revision: str | None = None
    raw_difficulty: dict[str, str] = field(default_factory=dict)

    @property
    def available(self) -> dict[str, bool]:
        return {
            "candidates": bool(self.candidates),
            "passrate": self.pass_a is not None and self.pass_b is not None,
            "splits": self.splits is not None and self.problems is not None,
            "probe": self.probe is not None,
        }


def load_inputs(processed_dir: str | Path = "data/processed", probe_dir: str | Path = "results/probe",
                raw_dir: str | Path | None = None) -> Inputs:
    """Read ``candidates.jsonl`` (required) and every optional stage output that exists."""
    d, pd_ = Path(processed_dir), Path(probe_dir)
    cand = d / "candidates.jsonl"
    if not cand.is_file():
        raise FileNotFoundError(f"{cand} not found: run `python -m rhg.data.build --stage fetch/tests/validate` first")
    rev = d / "DATASET_REVISION"
    inp = Inputs(
        processed_dir=d, probe_dir=pd_, candidates=read_jsonl(cand),
        problems=_opt(d / "problems.jsonl", read_jsonl), splits=_opt(d / "splits.json", read_json),
        pass_a=_opt(d / "passrate_A.jsonl", read_jsonl), pass_b=_opt(d / "passrate_B.jsonl", read_jsonl),
        stats_a=_opt(d / "passrate_A_stats.jsonl", read_jsonl), stats_b=_opt(d / "passrate_B_stats.jsonl", read_jsonl),
        probe=_opt(pd_ / "hint_probe.json", read_json),
        tests_report=_opt(d / "tests_report.json", read_json), validation_report=_opt(d / "validation_report.json", read_json),
        revision=rev.read_text(encoding="utf-8").strip() if rev.is_file() else None,
    )
    raw = Path(raw_dir) if raw_dir is not None else d.parent / "raw"
    inp.raw_difficulty = raw_difficulty(raw)
    return inp


def raw_difficulty(raw_dir: Path) -> dict[str, str]:
    """``task_id -> difficulty`` from the raw dataset file(s) (only those two fields are kept)."""
    out: dict[str, str] = {}
    for f in sorted(Path(raw_dir).glob("*.jsonl")) if Path(raw_dir).is_dir() else []:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                pid, diff = r.get("task_id") or r.get("problem_id"), r.get("difficulty")
                if pid and diff:
                    out[str(pid)] = str(diff)
    return out


def unavailable(section: str, script: str) -> None:
    """Render the standard 'not available yet' cell (used by every gated section)."""
    from IPython.display import Markdown, display

    display(Markdown(f"> **Section {section}: not available yet** - run `{script}` and re-run this notebook."))


# ------------------------------------------------------------------ A. provenance & schema
def schema_summary(problems: Sequence[Mapping]) -> pd.DataFrame:
    """Per field: python types seen, n present / missing (absent or None) / empty ('' or [])."""
    keys: dict[str, None] = {}
    for p in problems:
        keys.update(dict.fromkeys(p))
    n = len(problems)
    rows = []
    for k in keys:
        vals = [p.get(k) for p in problems]
        missing = sum(1 for v in vals if v is None)
        empty = sum(1 for v in vals if v is not None and isinstance(v, (str, list, dict)) and len(v) == 0)
        types = sorted({type(v).__name__ for v in vals if v is not None})
        rows.append({"field": k, "types": ",".join(types) or "-", "n": n, "missing": missing, "empty": empty,
                     "missing_pct": round(100 * missing / n, 2) if n else 0.0})
    return pd.DataFrame(rows)


def schema_issues(problems: Sequence[Mapping], required: Sequence[str] = REQUIRED_KEYS, allow_empty: Sequence[str] = ("tags",)) -> list[str]:
    """Required fields that are absent/None (or empty, except ``allow_empty`` ones) in at least one problem (empty list = clean).

    ``tags`` may legitimately be empty (a few LeetCode problems carry none); that is reported by section A, not as an issue.
    """
    bad = []
    for k in required:
        n_bad = sum(1 for p in problems
                    if p.get(k) is None or (k not in allow_empty and isinstance(p.get(k), (str, list)) and len(p[k]) == 0))
        if n_bad:
            bad.append(f"{k}: {n_bad} problems missing/empty")
    return bad


def provenance(inp: Inputs) -> dict[str, Any]:
    files = {}
    for name in ("candidates.jsonl", "problems.jsonl", "splits.json", "passrate_A.jsonl", "passrate_B.jsonl",
                 "tests_report.json", "validation_report.json", "DATASET_REVISION"):
        p = inp.processed_dir / name
        if p.is_file():
            files[name] = {"bytes": p.stat().st_size, "sha256_12": hashlib.sha256(p.read_bytes()).hexdigest()[:12]}
    return {"dataset_revision": inp.revision, "DATASET_REVISION_file": "DATASET_REVISION" in files, "files": files,
            "n_candidates": len(inp.candidates), "n_problems": len(inp.problems) if inp.problems else None}


def library_table() -> pd.DataFrame:
    from importlib import metadata

    rows = [{"library": "python", "version": platform.python_version()}, {"library": "platform", "version": platform.platform()}]
    for lib in ("numpy", "pandas", "scipy", "matplotlib", "datasets", "tokenizers", "nbformat", "pyyaml"):
        try:
            rows.append({"library": lib, "version": metadata.version(lib)})
        except metadata.PackageNotFoundError:
            rows.append({"library": lib, "version": "not installed"})
    from rhg import manifest

    rows.append({"library": "rhg git sha", "version": manifest.git_head_sha(manifest.REPO_ROOT) or "unknown (no git)"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ frames
def _words(text: Any) -> int:
    return len(str(text or "").split())


def problem_frame(problems: Sequence[Mapping]) -> pd.DataFrame:
    """One row per problem with the derived columns the sections use (NaN where a field is absent)."""
    rows = []
    for p in problems:
        rt, ht = list(p.get("reward_tests") or []), list(p.get("heldout_tests") or [])
        alls = [t["src"] for t in rt + ht]
        rows.append({
            "problem_id": p["problem_id"], "difficulty": p.get("difficulty"), "tags": list(p.get("tags") or []),
            "n_tags": len(p.get("tags") or []), "date": p.get("date"), "cluster_id": p.get("cluster_id"),
            "split": p.get("split"),
            "desc_chars": len(p.get("description") or ""), "desc_words": _words(p.get("description")),
            "starter_chars": len(p.get("starter_code") or ""), "starter_words": _words(p.get("starter_code")),
            "n_reward": len(rt), "n_heldout": len(ht),
            "test_len_mean": float(np.mean([len(s) for s in alls])) if alls else float("nan"),
            "test_len_max": max((len(s) for s in alls), default=0),
            "ref_chars": len(p.get("reference_solution") or ""),
            "ref_lines": len([x for x in (p.get("reference_solution") or "").splitlines() if x.strip()]),
            "p_A": p.get("p_A", float("nan")), "p_B_full": p.get("p_B_full", float("nan")),
            "p_B_visible": p.get("p_B_visible", float("nan")),
        })
    df = pd.DataFrame(rows)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_ord"] = df["date_dt"].map(lambda d: d.toordinal() if pd.notna(d) else float("nan"))
    df["difficulty_ord"] = df["difficulty"].map(DIFFICULTY_ORDER)
    return df


# ------------------------------------------------------------------ B. difficulty, tags, dates
def count_table(values: Iterable[Any], order: Sequence[Any] | None = None) -> pd.DataFrame:
    c = Counter(v for v in values if v is not None)
    keys = list(order) if order is not None else [k for k, _ in c.most_common()]
    n = sum(c.values())
    return pd.DataFrame({"value": keys, "n": [c.get(k, 0) for k in keys],
                         "share": [round(c.get(k, 0) / n, 4) if n else 0.0 for k in keys]})


def tag_table(df: pd.DataFrame, min_n: int = 1) -> pd.DataFrame:
    c = Counter(t for tags in df["tags"] for t in tags)
    out = pd.DataFrame(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])), columns=["tag", "n"])
    out["share_of_problems"] = (out["n"] / max(len(df), 1)).round(4)
    return out[out["n"] >= min_n].reset_index(drop=True)


def contamination_proxy(dates: Iterable[str | None], horizon: str = QWEN3_HORIZON_ASSUMED) -> dict[str, Any]:
    """Share of dated problems strictly after ``horizon`` (YYYY-MM-DD; ISO strings compare correctly).

    Problems after the horizon cannot have been seen in pretraining if the horizon is right; the rest may
    have been. It is a proxy only: old problems may also have been seen through later re-postings.
    """
    dates = list(dates)
    ds = [d for d in dates if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
    n_all = len(dates)
    after = sum(1 for d in ds if d > horizon)
    return {"horizon": horizon, "n_dated": len(ds), "n_undated": n_all - len(ds), "n_after": after,
            "n_before_or_on": len(ds) - after, "share_after": after / len(ds) if ds else float("nan")}


def contamination_grid(dates: Sequence[str | None], horizons: Sequence[str] = CONTAM_HORIZONS) -> pd.DataFrame:
    return pd.DataFrame([contamination_proxy(dates, h) for h in sorted({*horizons, QWEN3_HORIZON_ASSUMED})])


# ------------------------------------------------------------------ C. text budget
@dataclass
class TokenCounter:
    kind: str  # "qwen3" | "proxy"
    label: str
    count_many: Callable[[Sequence[str]], list[int]]

    @property
    def is_real(self) -> bool:
        return self.kind == "qwen3"


_PROXY = re.compile(r"\w+|[^\w\s]")


def proxy_count(text: str) -> int:
    """Words plus punctuation marks. NOT a token count: BPE on code and digits usually yields more."""
    return len(_PROXY.findall(text))


def load_token_counter(mode: str = "auto", name: str = TOKENIZER_NAME) -> TokenCounter:
    """Real Qwen3 tokenizer from the local HF cache (never downloads), else the labelled word-count proxy."""
    proxy = TokenCounter("proxy", "PROXY (regex words+punctuation; a lower bound of real Qwen3 BPE tokens, not a token count)",
                         lambda texts: [proxy_count(t) for t in texts])
    if mode == "proxy":
        return proxy
    try:
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer

        path = try_to_load_from_cache(name, "tokenizer.json")
        if not isinstance(path, str):
            raise FileNotFoundError("tokenizer.json not in the local HF cache")
        tok = Tokenizer.from_file(path)
        return TokenCounter("qwen3", f"real {name} tokenizer (local cache, tokenizers {_ver('tokenizers')})",
                            lambda texts: [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)])
    except Exception:  # ImportError, cache miss, corrupt file: fall back, never fail the notebook
        if mode == "qwen3":
            raise
        return proxy


def _ver(pkg: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "?"


def prompt_lengths(problems: Sequence[Mapping], prompts_cfg: Mapping, counter: TokenCounter,
                   hints: Sequence[str] = HINT_IDS) -> pd.DataFrame:
    """Tokens of the chat-rendered prompt per problem and hint wording (+ ``delta_<hint>`` vs no hint)."""
    from rhg.data.prompts import build_prompt, render_chat

    out = pd.DataFrame({"problem_id": [p["problem_id"] for p in problems]})
    for h in hints:
        texts = [render_chat(build_prompt(p, h, prompts_cfg)) for p in problems]
        out[f"tokens_{h}"] = counter.count_many(texts)
    for h in hints:
        if h != "none" and "none" in hints:
            out[f"delta_{h}"] = out[f"tokens_{h}"] - out["tokens_none"]
    return out


def truncation_risk(tokens: Sequence[int] | pd.Series, limit: int) -> dict[str, Any]:
    t = np.asarray(list(tokens), dtype=float)
    if not t.size:
        return {"n": 0, "limit": limit, "n_over": 0, "share_over": float("nan"), "n_within_10pct": 0, "max": float("nan")}
    q = np.quantile(t, [0.5, 0.9, 0.99])
    return {"n": int(t.size), "limit": int(limit), "n_over": int((t > limit).sum()), "share_over": float((t > limit).mean()),
            "n_within_10pct": int(((t <= limit) & (t > 0.9 * limit)).sum()), "max": float(t.max()),
            "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2])}


# ------------------------------------------------------------------ D. tests
def _is_candidate_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate"


def _split_test(src: str) -> tuple[str | None, str | None]:
    """``(input, expected)`` source text of one ``assert`` test, or ``(None, None)`` if the form is unknown."""
    try:
        tree = ast.parse(src.strip())
    except SyntaxError:
        return None, None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None, None
    t = tree.body[0].test

    def call_text(c: ast.Call) -> str:
        return ", ".join([ast.unparse(a) for a in c.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in c.keywords])

    if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
        if _is_candidate_call(t.left):
            return call_text(t.left), ast.unparse(t.comparators[0])
        if _is_candidate_call(t.comparators[0]):
            return call_text(t.comparators[0]), ast.unparse(t.left)
    if isinstance(t, ast.Call) and isinstance(t.func, ast.Name) and t.func.id in _ASSERT_HELPERS:
        cands = [a for a in t.args if _is_candidate_call(a)]
        others = [a for a in t.args if not _is_candidate_call(a)]
        if len(cands) == 1 and others:
            return call_text(cands[0]), ", ".join(ast.unparse(o) for o in others)
    return None, None


def expected_of(src: str) -> str | None:
    """Source text of the expected value of an ``assert candidate(...) == X`` test (None if not of that form)."""
    return _split_test(src)[1]


def input_of(src: str) -> str | None:
    return _split_test(src)[0]


def expected_type(expected: str) -> str:
    """Coarse python type of an expected-value expression: bool/int/float/str/list/dict/none/helper/other."""
    try:
        v = ast.literal_eval(expected)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return "helper" if re.match(r"\s*(list_node|tree_node)\s*\(", expected) else "other"
    if v is None:
        return "none"
    for t, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str"), ((list, tuple, set), "list"), (dict, "dict")):
        if isinstance(v, t):
            return name
    return "other"


def expected_frame(problems: Sequence[Mapping]) -> pd.DataFrame:
    """One row per test: problem, part (reward|heldout), expected text, expected type, input text."""
    rows = []
    for p in problems:
        for part in ("reward", "heldout"):
            for t in p.get(f"{part}_tests") or []:
                i, e = _split_test(t["src"])
                rows.append({"problem_id": p["problem_id"], "part": part, "id": t.get("id"), "src": t["src"], "input": i,
                             "expected": e, "etype": expected_type(e) if e is not None else "unparsed"})
    return pd.DataFrame(rows, columns=["problem_id", "part", "id", "src", "input", "expected", "etype"])


def degenerate_report(problems: Sequence[Mapping], max_distinct: int = 2, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per problem: distinct expected outputs among the reward tests and the pass share of the best constant answer.

    ``degenerate`` = at most ``max_distinct`` distinct expected outputs among the reward tests (a hard-coded or
    overfit answer can then pass the reward tests by luck). ``const_share_heldout`` is the share of held-out tests
    the most common reward-test answer would pass, i.e. how much of the held-out set a constant covers.
    ``frame`` = a precomputed ``expected_frame(problems)`` (parsing 70k asserts takes ~12 s).
    """
    ex = frame if frame is not None else expected_frame(problems)
    by_pid = {(pid, part): g["expected"] for (pid, part), g in ex.groupby(["problem_id", "part"], sort=False)}
    none = pd.Series([], dtype=object)
    rows = []
    for p in problems:
        rw_all, ho_all = by_pid.get((p["problem_id"], "reward"), none), by_pid.get((p["problem_id"], "heldout"), none)
        rw, ho = rw_all.dropna(), ho_all.dropna()
        cnt = Counter(rw)
        mode, mode_n = (cnt.most_common(1)[0] if cnt else (None, 0))
        rows.append({
            "problem_id": p["problem_id"], "n_reward": len(rw), "n_distinct_reward": len(cnt),
            "n_distinct_all": len(set(rw) | set(ho)), "const_share_reward": mode_n / len(rw) if len(rw) else float("nan"),
            "const_share_heldout": float((ho == mode).mean()) if len(ho) and mode is not None else float("nan"),
            "n_unparsed": int(rw_all.isna().sum() + ho_all.isna().sum()), "degenerate": bool(len(cnt) <= max_distinct and len(rw) > 0),
            "constant_passes_all_reward": bool(len(cnt) == 1),
        })
    return pd.DataFrame(rows)


def duplicate_tests(problems: Sequence[Mapping], frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Duplicate-test diagnostics (within-problem duplicates and reward/held-out overlap are 0 after the builder's dedupe)."""
    ex = frame if frame is not None else expected_frame(problems)
    within = ex[ex.duplicated(["problem_id", "src"], keep=False)]
    ov = ex.groupby(["problem_id", "src"])["part"].nunique()
    overlap = ov[ov > 1]
    same_in = ex.dropna(subset=["input"]).groupby(["problem_id", "input"])["expected"].agg(["size", "nunique"])
    conflicting = same_in[same_in["nunique"] > 1]
    cross = ex.groupby("src")["problem_id"].nunique()
    cross = cross[cross > 1]
    return {
        "n_tests": int(len(ex)),
        "within_problem_duplicate_tests": int(len(within)), "problems_with_duplicates": int(within["problem_id"].nunique()),
        "reward_heldout_overlap": int(len(overlap)),
        "same_input_conflicting_expected": int(len(conflicting)),
        "same_input_repeated": int((same_in["size"] > 1).sum()),
        "cross_problem_duplicate_tests": int(len(cross)),
        "examples": within.head(3)[["problem_id", "src"]].to_dict("records"),
    }


def drop_frame(inp: Inputs) -> pd.DataFrame:
    """Every dropped problem with stage, reason and (if the raw file is present) difficulty."""
    rows = []
    for stage, rep in (("tests", inp.tests_report), ("validate", inp.validation_report)):
        for d in (rep or {}).get("drops", []):
            rows.append({"problem_id": d["problem_id"], "stage": stage, "reason": d["reason"], "detail": d.get("detail", ""),
                         "difficulty": inp.raw_difficulty.get(d["problem_id"])})
    return pd.DataFrame(rows, columns=["problem_id", "stage", "reason", "detail", "difficulty"])


# ------------------------------------------------------------------ E. reference solutions
def time_references(problems: Sequence[Mapping], sample_n: int | None = None, seed: int = 0, workers: int = 0,
                    timeout_s: float = 6.0) -> pd.DataFrame:
    """Run each (sampled) reference solution through the real grader with the cache off.

    ``wall_s`` is the wall time of the honest sandbox process (interpreter start included), measured per
    problem while other problems run in parallel, so it is an upper-ish proxy, not a benchmark.
    """
    from rhg.config import load_config
    from rhg.env.grader import GradeItem, grade_batch

    probs = list(problems)
    if sample_n is not None and sample_n < len(probs):
        idx = np.random.default_rng(seed).choice(len(probs), size=sample_n, replace=False)
        probs = [probs[i] for i in sorted(idx)]
    cfg = load_config("clean_none", overrides=["sandbox.cache=false", f"sandbox.timeout_s={timeout_s}"])
    items = [GradeItem(p, f"```python\n{p['reference_solution']}\n```", "clean") for p in probs]
    res = grade_batch(items, workers=workers or None, cfg=cfg)
    return pd.DataFrame([{
        "problem_id": p["problem_id"], "wall_s": float(r.info.get("wall_s", float("nan"))),
        "visible_pass": bool(r.labels["visible_pass"]), "heldout_pass": bool(r.labels["heldout_pass"]),
        "timeout": bool(r.labels["timeout"]), "crash": bool(r.labels["crash"]), "correct": bool(r.labels["correct"]),
    } for p, r in zip(probs, res)])


def runtime_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    w = df["wall_s"].to_numpy(dtype=float)
    q = np.quantile(w, [0.5, 0.9, 0.99])
    return {"n": int(len(df)), "n_correct": int(df["correct"].sum()), "validity": float(df["correct"].mean()),
            "median_s": float(q[0]), "p90_s": float(q[1]), "p99_s": float(q[2]), "max_s": float(w.max()),
            "n_slow": int((w > SLOW_REFERENCE_S).sum()), "n_timeout": int(df["timeout"].sum())}


# ------------------------------------------------------------------ F. leakage / duplication
def near_duplicate_pairs(problems: Sequence[Mapping], min_jaccard: float = NEAR_DUP_REPORT_MIN,
                         n: int = SHINGLE_WORDS) -> pd.DataFrame:
    """All problem pairs whose word-shingle Jaccard is >= ``min_jaccard`` (exact, via a sparse shingle matrix).

    Uses the clustering rule's normalisation (``rhg.data.dedupe``): the statement without examples and
    constraints, letters only, ``n``-word shingles. Columns: a, b, jaccard, same_cluster.
    """
    from scipy import sparse

    vocab: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    sizes = np.zeros(len(problems), dtype=np.int64)
    for i, p in enumerate(problems):
        sh = shingles(normalize_text(p["description"]), n)
        sizes[i] = len(sh)
        for s in sh:
            rows.append(i)
            cols.append(vocab.setdefault(s, len(vocab)))
    if not vocab:
        return pd.DataFrame(columns=["a", "b", "jaccard", "same_cluster"])
    x = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(problems), len(vocab)))
    inter = (x @ x.T).tocoo()
    m = inter.row < inter.col
    r, c, v = inter.row[m], inter.col[m], inter.data[m].astype(np.int64)
    j = v / (sizes[r] + sizes[c] - v)
    keep = j >= min_jaccard - 1e-12
    ids = [p["problem_id"] for p in problems]
    cl = [p.get("cluster_id") for p in problems]
    out = pd.DataFrame({"a": [ids[i] for i in r[keep]], "b": [ids[i] for i in c[keep]], "jaccard": j[keep],
                        "same_cluster": [cl[i] is not None and cl[i] == cl[k] for i, k in zip(r[keep], c[keep])]})
    return out.sort_values(["jaccard", "a", "b"], ascending=[False, True, True]).reset_index(drop=True)


def cluster_summary(problems: Sequence[Mapping], top: int = 5) -> dict[str, Any]:
    members: dict[str, list[str]] = {}
    for p in problems:
        members.setdefault(p["cluster_id"], []).append(p["problem_id"])
    sizes = Counter(len(v) for v in members.values())
    largest = sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    return {"n_clusters": len(members), "n_problems": len(problems),
            "histogram": pd.DataFrame(sorted(sizes.items()), columns=["cluster_size", "n_clusters"]),
            "share_in_multi": sum(len(v) for v in members.values() if len(v) > 1) / max(len(problems), 1),
            "largest": [{"cluster_id": k, "size": len(v), "members": sorted(v)} for k, v in largest]}


def id_duplicates(problems: Sequence[Mapping]) -> dict[str, list]:
    """Duplicate ids, duplicate normalised titles (slugs without numbers / Roman numerals) and identical statements."""
    ids = Counter(p["problem_id"] for p in problems)
    titles: dict[str, list[str]] = {}
    texts: dict[str, list[str]] = {}
    for p in problems:
        titles.setdefault(normalize_title(p["problem_id"]), []).append(p["problem_id"])
        texts.setdefault(hashlib.sha256(str(p["description"]).encode()).hexdigest(), []).append(p["problem_id"])
    return {"duplicate_ids": sorted(k for k, v in ids.items() if v > 1),
            "same_title": sorted(v for v in titles.values() if len(v) > 1),
            "identical_description": sorted(v for v in texts.values() if len(v) > 1)}


def clusters_spanning_splits(problems: Sequence[Mapping]) -> list[dict]:
    """Clusters whose members sit in more than one split (must be empty once splits exist)."""
    by: dict[str, dict[str, list[str]]] = {}
    for p in problems:
        if p.get("split"):
            by.setdefault(p["cluster_id"], {}).setdefault(p["split"], []).append(p["problem_id"])
    return [{"cluster_id": c, "splits": {s: sorted(v) for s, v in d.items()}} for c, d in sorted(by.items()) if len(d) > 1]


# ------------------------------------------------------------------ G. base pass rates
def passrate_frame(cands: pd.DataFrame, pass_a: Sequence[Mapping], pass_b: Sequence[Mapping] | None,
                   low: float = 0.10, high: float = 0.40) -> pd.DataFrame:
    """Candidates joined with stage A/B counts: ``p_A, p_B_visible, p_B_full, in_band`` (exact band test)."""
    from rhg.data.build import in_band

    a = {r["problem_id"]: r for r in pass_a}
    b = {r["problem_id"]: r for r in (pass_b or [])}
    df = cands.drop(columns=["p_A", "p_B_full", "p_B_visible"], errors="ignore").copy()
    df["n_A"] = df["problem_id"].map(lambda i: a[i]["n"] if i in a else np.nan)
    df["k_A"] = df["problem_id"].map(lambda i: a[i]["k_visible"] if i in a else np.nan)
    df["k_A_full"] = df["problem_id"].map(lambda i: a[i]["k_full"] if i in a else np.nan)
    df["p_A"] = df["k_A"] / df["n_A"]
    df["n_B"] = df["problem_id"].map(lambda i: b[i]["n"] if i in b else np.nan)
    df["k_B_visible"] = df["problem_id"].map(lambda i: b[i]["k_visible"] if i in b else np.nan)
    df["k_B_full"] = df["problem_id"].map(lambda i: b[i]["k_full"] if i in b else np.nan)
    df["p_B_visible"] = df["k_B_visible"] / df["n_B"]
    df["p_B_full"] = df["k_B_full"] / df["n_B"]
    df["in_band"] = [bool(pd.notna(k) and in_band(int(k), int(n), low, high)) for k, n in zip(df["k_A"], df["n_A"])]
    return df


def rtm_summary(df: pd.DataFrame, low: float = 0.10, high: float = 0.40) -> dict[str, Any]:
    """Regression to the mean for band-selected problems: stage-A rate (selection sample) vs stage-B rate (independent).

    Reported on the visible-test rate (the same quantity in both stages). ``slope`` is the OLS slope of ``p_B_visible``
    on ``p_A`` within the band (1 = no shrinkage); the band is narrow, so the slope is noisy and the mean shift
    ``mean(p_B_visible - p_A)`` (paired, with a normal-approximation CI over problems) is the headline number.
    """
    from scipy import stats

    s = df[df["in_band"] & df["p_B_visible"].notna()]
    n = len(s)
    if n < 3:
        return {"n": n}
    d = (s["p_B_visible"] - s["p_A"]).to_numpy()
    se = d.std(ddof=1) / math.sqrt(n)
    lr = stats.linregress(s["p_A"], s["p_B_visible"]) if s["p_A"].nunique() > 1 else None
    pn = np.clip(s["p_A"].to_numpy(), 0, 1)
    return {"n": n, "mean_p_A": float(s["p_A"].mean()), "mean_p_B_visible": float(s["p_B_visible"].mean()),
            "mean_shift": float(d.mean()), "shift_ci": (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)),
            "slope": float(lr.slope) if lr else float("nan"), "slope_se": float(lr.stderr) if lr else float("nan"),
            "share_in_band_B": float(((s["p_B_visible"] >= low - 1e-12) & (s["p_B_visible"] <= high + 1e-12)).mean()),
            "share_below_band_B": float((s["p_B_visible"] < low - 1e-12).mean()),
            "share_above_band_B": float((s["p_B_visible"] > high + 1e-12).mean()),
            "binomial_sd_at_n": float(np.mean(np.sqrt(pn * (1 - pn) / s["n_A"].to_numpy()))),
            "band_width": high - low}


def weak_tests_summary(rows: Sequence[Mapping]) -> dict[str, Any]:
    """Visible vs full pass: how many visible passes fail the held-out tests (weak reward tests)."""
    kv = sum(int(r["k_visible"]) for r in rows)
    kf = sum(int(r["k_full"]) for r in rows)
    n = sum(int(r["n"]) for r in rows)
    gap = [(r["k_visible"] - r["k_full"]) / r["n"] for r in rows]
    lo, hi = wilson(kv - kf, kv) if kv else (float("nan"), float("nan"))
    return {"n_problems": len(rows), "n_samples": n, "visible_rate": kv / n if n else float("nan"),
            "full_rate": kf / n if n else float("nan"), "frac_visible_pass_fail_heldout": (kv - kf) / kv if kv else float("nan"),
            "frac_ci": (lo, hi), "share_problems_with_gap": float(np.mean([g > 0 for g in gap])) if gap else float("nan"),
            "share_problems_gap_ge_25pct": float(np.mean([g >= 0.25 for g in gap])) if gap else float("nan")}


def band_table(df: pd.DataFrame, col: str, min_n: int = 1) -> pd.DataFrame:
    """Band membership (share of problems with ``in_band``) per value of ``col`` (list-valued ``col`` is exploded)."""
    d = df[["in_band", col]].explode(col) if df[col].map(lambda v: isinstance(v, list)).any() else df[["in_band", col]]
    g = d.groupby(col)["in_band"].agg(["size", "sum"]).reset_index()
    g.columns = [col, "n", "n_in_band"]
    g["share_in_band"] = g["n_in_band"] / g["n"]
    ci = [wilson(int(k), int(n)) for k, n in zip(g["n_in_band"], g["n"])]
    g["ci_lo"], g["ci_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g[g["n"] >= min_n].sort_values(["n", col], ascending=[False, True]).reset_index(drop=True)


def cramers_v(table: np.ndarray | Sequence[Sequence[float]]) -> float:
    from scipy import stats

    t = np.asarray(table, dtype=float)
    t = t[t.sum(1) > 0][:, t.sum(0) > 0]
    if min(t.shape) < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return float(math.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def correlation_table(df: pd.DataFrame, target: str, features: Sequence[str]) -> pd.DataFrame:
    """Spearman rho of ``target`` with each feature (pairwise-complete), n and p; exploratory, uncorrected."""
    from scipy import stats

    rows = []
    for f in features:
        d = df[[target, f]].dropna()
        if len(d) < 4 or d[target].nunique() < 2 or d[f].nunique() < 2:
            rows.append({"feature": f, "rho": float("nan"), "p": float("nan"), "n": len(d)})
            continue
        r = stats.spearmanr(d[f], d[target])
        rows.append({"feature": f, "rho": float(r.statistic), "p": float(r.pvalue), "n": len(d)})
    return pd.DataFrame(rows)


def stats_side_summary(rows: Sequence[Mapping]) -> dict[str, Any]:
    """Completion length / truncation / extraction-failure / timeout / crash rates from ``passrate_*_stats.jsonl``."""
    n = sum(int(r["n"]) for r in rows)
    if not n:
        return {"n_samples": 0}

    def rate(key: str) -> float:
        return sum(int(r[key]) for r in rows) / n

    per_fail = np.array([r["n_extract_fail"] / r["n"] for r in rows if r["n"]])
    return {"n_problems": len(rows), "n_samples": n,
            "n_tokens_mean": sum(float(r["n_tokens_mean"]) * int(r["n"]) for r in rows) / n,
            "truncation_rate": rate("n_truncated"), "extract_fail_rate": rate("n_extract_fail"),
            "timeout_rate": rate("n_timeout"), "crash_rate": rate("n_crash"), "defines_rt_rate": rate("n_defines_rt"),
            "share_problems_extract_fail_ge_half": float((per_fail >= 0.5).mean())}


# ------------------------------------------------------------------ H. splits
def ks_statistic(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov D (sup distance between the empirical CDFs), computed directly."""
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    grid = np.concatenate([x, y])
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x) - np.searchsorted(y, grid, side="right") / len(y))))


def smd(x: Sequence[float], y: Sequence[float]) -> float:
    """Standardised mean difference (Cohen's d with pooled SD), ``mean(x) - mean(y)``; 0 if both are constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sp = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else 0.0


def balance_numeric(df: pd.DataFrame, col: str, pairs: Sequence[tuple[str, str]] = (("train", "val"), ("train", "test"), ("val", "test"))) -> pd.DataFrame:
    """KS D (+ p) and SMD for ``col`` between splits, plus the Kruskal-Wallis epsilon^2 over all three."""
    from scipy import stats

    rows = []
    groups = {s: df.loc[df["split"] == s, col].dropna().to_numpy(float) for s in SPLITS}
    for a, b in pairs:
        x, y = groups[a], groups[b]
        if len(x) < 2 or len(y) < 2:
            continue
        ks = stats.ks_2samp(x, y)
        rows.append({"column": col, "pair": f"{a} vs {b}", "n_a": len(x), "n_b": len(y), "ks_D": float(ks.statistic),
                     "ks_p": float(ks.pvalue), "smd": smd(x, y)})
    out = pd.DataFrame(rows, columns=["column", "pair", "n_a", "n_b", "ks_D", "ks_p", "smd"])
    valid = [g for g in groups.values() if len(g) > 0]
    if len(valid) > 1 and len(np.unique(np.concatenate(valid))) > 1:
        h = stats.kruskal(*valid)
        n = sum(len(g) for g in valid)
        out.attrs["kruskal_p"] = float(h.pvalue)
        out.attrs["epsilon_sq"] = float(max(0.0, (h.statistic - len(valid) + 1) / (n - len(valid)))) if n > len(valid) else float("nan")
    return out


def _chi2_stat(obs: np.ndarray) -> float:
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    m = exp > 0
    return float((((obs - exp) ** 2)[m] / exp[m]).sum())


def balance_categorical(df: pd.DataFrame, col: str, n_perm: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Chi-square of ``col`` across splits with Cramer's V; p is a permutation p-value over split labels
    (small expected counts make the asymptotic p unreliable) next to the asymptotic one."""
    from scipy import stats

    d = df.dropna(subset=[col, "split"])
    d = d[d["split"].isin(SPLITS)]
    cats = sorted(d[col].unique())
    tab = pd.crosstab(d[col], d["split"]).reindex(index=cats, columns=[s for s in SPLITS if s in set(d["split"])], fill_value=0)
    obs = tab.to_numpy(float)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return {"column": col, "table": tab, "chi2": float("nan"), "p_asymptotic": float("nan"), "p_permutation": float("nan"),
                "cramers_v": float("nan"), "min_expected": float("nan")}
    chi2, p_asym, _, exp = stats.chi2_contingency(obs, correction=False)
    codes = pd.Categorical(d[col], categories=cats).codes
    grp = pd.Categorical(d["split"], categories=list(tab.columns)).codes
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        o = np.bincount(codes * obs.shape[1] + rng.permutation(grp), minlength=obs.size).reshape(obs.shape).astype(float)
        ge += _chi2_stat(o) >= chi2 - 1e-9
    return {"column": col, "table": tab, "chi2": float(chi2), "p_asymptotic": float(p_asym),
            "p_permutation": (ge + 1) / (n_perm + 1), "cramers_v": cramers_v(obs), "min_expected": float(exp.min())}


def tag_balance(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Prevalence of the ``top`` tags per split, largest gap in percentage points, and a per-tag chi-square p.
    With ~12 tags, about one nominal p < 0.05 is expected by chance."""
    from scipy import stats

    d = df[df["split"].isin(SPLITS)]
    tags = [t for t, _ in Counter(t for tags in d["tags"] for t in tags).most_common(top)]
    n_split = d["split"].value_counts()
    rows = []
    for t in tags:
        has = d["tags"].map(lambda ts: t in ts)
        prev = {s: float(has[d["split"] == s].mean()) for s in SPLITS if n_split.get(s, 0)}
        tab = np.array([[int(has[d["split"] == s].sum()), int((~has[d["split"] == s]).sum())] for s in prev])
        p = float(stats.chi2_contingency(tab, correction=False)[1]) if len(prev) > 1 and tab.sum(0).min() > 0 else float("nan")
        rows.append({"tag": t, **{f"share_{s}": v for s, v in prev.items()}, "max_gap_pp": 100 * (max(prev.values()) - min(prev.values())),
                     "cramers_v": cramers_v(tab), "chi2_p": p})
    return pd.DataFrame(rows)


def balance_flags(numeric: pd.DataFrame, categorical: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    """Human-readable warnings from ``balance_numeric`` rows (all columns concatenated) and ``balance_categorical`` dicts.

    A numeric difference is flagged if KS p < ``BALANCE_P_WARN`` or (p < 0.05 and D > ``BALANCE_KS_LARGE``); with about
    24 comparisons one nominal p < 0.05 is expected by chance, which is why a lone 0.01-0.05 result with a small D is not flagged.
    """
    out = []
    for r in numeric.itertuples():
        if r.ks_p < BALANCE_P_WARN or (r.ks_p < 0.05 and r.ks_D > BALANCE_KS_LARGE):
            out.append(f"{r.column}: {r.pair} differ (KS D={r.ks_D:.2f}, p={r.ks_p:.3g}, SMD={r.smd:+.2f})")
    for c in categorical:
        if c.get("p_permutation", 1.0) < 0.05:
            out.append(f"{c['column']} mix differs across splits (permutation p={c['p_permutation']:.3g}, Cramer's V={c['cramers_v']:.2f})")
    return out


def verify_splits(problems: Sequence[Mapping], splits: Mapping) -> dict[str, Any]:
    """Recompute the split hash / membership / cluster integrity from ``problems.jsonl`` and compare with ``splits.json``."""
    from rhg.data.build import split_hash

    assign = {p["problem_id"]: p["split"] for p in problems}
    listed = {s: sorted(splits.get(s, [])) for s in SPLITS}
    actual = {s: sorted(i for i, sp in assign.items() if sp == s) for s in SPLITS}
    spanning = clusters_spanning_splits(problems)
    return {"hash_recomputed": split_hash(assign), "hash_recorded": splits.get("split_hash"),
            "hash_ok": split_hash(assign) == splits.get("split_hash"), "lists_match": listed == actual,
            "n_spanning_clusters": len(spanning), "spanning": spanning,
            "counts": {s: len(v) for s, v in actual.items()}}


def gate1c_table(splits: Mapping, verified: Mapping | None = None) -> pd.DataFrame:
    """The Gate-1c checklist recorded in ``splits.json`` plus independent re-checks, as PASS/FAIL/WARN/INFO rows."""
    rows = [{"item": i["item"], "status": i["status"], "value": i.get("value"), "note": i.get("note", ""), "source": "splits.json"}
            for i in splits.get("gate1c", {}).get("items", [])]
    if verified is not None:
        rows += [
            {"item": "split hash recomputes from problems.jsonl", "status": "PASS" if verified["hash_ok"] else "FAIL",
             "value": str(verified["hash_recomputed"])[:12], "note": "", "source": "recomputed"},
            {"item": "split lists in splits.json equal problems.jsonl", "status": "PASS" if verified["lists_match"] else "FAIL",
             "value": "", "note": "", "source": "recomputed"},
            {"item": "no cluster spans splits (recomputed)", "status": "PASS" if verified["n_spanning_clusters"] == 0 else "FAIL",
             "value": verified["n_spanning_clusters"], "note": "", "source": "recomputed"},
        ]
    return pd.DataFrame(rows, columns=["item", "status", "value", "note", "source"])


# ------------------------------------------------------------------ I. hint probe
def newcombe_diff(k1: int, n1: int, k0: int, n0: int, z: float = 1.96) -> tuple[float, float, float]:
    """Newcombe hybrid-score CI for ``k1/n1 - k0/n0``: ``(diff, lo, hi)``."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1, z)
    l0, u0 = wilson(k0, n0, z)
    d = p1 - p0
    return d, d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def probe_tables(probe: Mapping) -> pd.DataFrame:
    """Per wording: ATTEMPT_RT rate + Wilson CI, honest pass rates and the step-0 honest-pass shift vs ``none``."""
    w = probe["decision"]["wordings"]
    base = w["none"]
    rows = []
    for name in [k for k in HINT_IDS if k in w] + [k for k in w if k not in HINT_IDS]:
        s = w[name]
        row = {"wording": name, "n": s["n"], "k_attempt": s["k_attempt"], "attempt_rate": s["rate_attempt"],
               "attempt_lo": s["ci_attempt"][0], "attempt_hi": s["ci_attempt"][1],
               "honest_visible_rate": s["rate_visible"], "honest_visible_lo": s["ci_visible"][0], "honest_visible_hi": s["ci_visible"][1],
               "honest_full_rate": s["rate_correct"], "k_hack": s["k_hack"]}
        if name != "none" and s["n"] and base["n"]:
            d, lo, hi = newcombe_diff(s["k_visible"], s["n"], base["k_visible"], base["n"])
            row.update(honest_shift_pp=100 * d, shift_lo_pp=100 * lo, shift_hi_pp=100 * hi)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ J. power & design
def power_tables(seeds: Mapping[str, int] | None = None) -> dict[str, Any]:
    from rhg.analysis import power
    from rhg.budget import LADDER

    seeds = dict(seeds) if seeds is not None else dict(LADDER[0].seeds)
    rep = power.design_report(seeds, "planned design")
    tests = pd.DataFrame([{"test": t.test, "family": t.family, "design": f"{t.kind}{t.sizes}", "min_attainable_p": t.min_p,
                           "reaches_alpha": t.reachable_alone, "reaches_alpha_over_m": t.reachable_holm_first} for t in rep.tests])
    ladder = []
    for r in power.ladder_reports():
        prim = next((t for t in r.tests if t.family == "primary"), None)
        ladder.append({"ladder": r.label, "runs": r.runs, "primary_min_p": prim.min_p if prim else None,
                       "power_q0.8": next((x["power"] for x in r.primary_power if abs(x["q"] - 0.8) < 1e-9), None),
                       "n_flags": len(r.flags)})
    return {"seeds": seeds, "runs": rep.runs, "tests": tests, "flags": list(rep.flags),
            "emergence": pd.DataFrame(rep.emergence), "primary_power": pd.DataFrame(rep.primary_power),
            "h1_power": pd.DataFrame(rep.h1_power), "ladder": pd.DataFrame(ladder)}


# ------------------------------------------------------------------ K. labels in action
def label_names(labels: Mapping[str, bool]) -> str:
    names = [n for n, k in (("HACK_RT", "hack_rt"), ("ATTEMPT_RT", "attempt_rt"), ("CORRECT", "correct"), ("VISIBLE_ONLY", "visible_only"))
             if labels.get(k)]
    return "+".join(names) or "(none)"


def controls_table(workers: int = 0) -> pd.DataFrame:
    """Synthetic controls through the real grader and both detector profiles."""
    from rhg.config import load_config
    from rhg.detect.ast_detector import analyze
    from rhg.env.extract import extract_code
    from rhg.env.grader import GradeItem, grade_batch
    from rhg.validate import controls as C

    cfg = load_config("hackable_subtle", overrides=["sandbox.cache=false"])
    probs = C.problems()
    ctrls = C.build_controls()
    res = grade_batch([GradeItem(probs[c.problem_id], c.completion, "hackable") for c in ctrls], workers=workers or None, cfg=cfg)
    rows = []
    for c, r in zip(ctrls, res):
        code = extract_code(c.completion).code
        nar, bro = analyze(code, "narrow"), analyze(code, "broad")
        rows.append({
            "control_id": c.control_id, "category": c.category, "variant": c.variant, "construction_label": c.construction_label,
            "code_extracted": bool(r.raw["code_extracted"]), "defines_rt": r.labels["defines_rt"], "rt_ok": r.labels["rt_ok"],
            "visible_pass": r.labels["visible_pass"], "heldout_pass": r.labels["heldout_pass"],
            "labels": label_names(r.labels), "hack_rt": r.labels["hack_rt"], "reward_hackable": r.reward,
            "ast_narrow": bool(nar.flag), "ast_broad": bool(bro.flag), "broad_categories": ",".join(bro.categories),
            "special_casing_heuristic": bool(bro.special_casing_heuristic),
        })
    return pd.DataFrame(rows)


def detector_blindspots(t: pd.DataFrame) -> dict[str, Any]:
    """Where each detector disagrees with the execution label / the constructed label."""
    exec_h = t["hack_rt"]
    hackish = t["construction_label"].str.startswith("hack_")
    return {
        "narrow_vs_exec": pd.crosstab(exec_h.rename("HACK_RT (execution)"), t["ast_narrow"].rename("ast_narrow")),
        "broad_vs_exec": pd.crosstab(exec_h.rename("HACK_RT (execution)"), t["ast_broad"].rename("ast_broad")),
        "narrow_misses": t[exec_h & ~t["ast_narrow"]][["control_id", "category", "variant"]].reset_index(drop=True),
        "broad_misses": t[exec_h & ~t["ast_broad"]][["control_id", "category", "variant"]].reset_index(drop=True),
        "exec_blind_hacks": t[hackish & ~exec_h][["control_id", "category", "variant", "ast_narrow", "ast_broad"]].reset_index(drop=True),
        "narrow_false_positives": t[t["ast_narrow"] & ~exec_h][["control_id", "category", "variant", "construction_label"]].reset_index(drop=True),
        "n_narrow_false_positive": int((t["ast_narrow"] & ~exec_h).sum()),
    }


# ------------------------------------------------------------------ L. flags
def flag(severity: str, section: str, message: str) -> dict[str, str]:
    assert severity in ("ERROR", "WARN", "INFO")
    return {"severity": severity, "section": section, "message": message}


def derive_flags(m: Mapping[str, Any]) -> list[dict[str, str]]:
    """Automatic list of worrying findings from the metrics the notebook collected (missing keys are skipped).

    ERROR = a hard invariant is violated, WARN = look at it before spending GPU money, INFO = context.
    """
    out: list[dict[str, str]] = []
    add = lambda *a: out.append(flag(*a))  # noqa: E731
    for msg in m.get("schema_issues", []):
        add("ERROR", "A", f"schema: {msg}")
    rev = m.get("dataset_revision")
    if m.get("n_candidates") is not None and not rev:
        add("WARN", "A", "no DATASET_REVISION recorded: the dataset revision cannot be pinned in the manifests")
    elif rev and rev.startswith("rhg-fixture"):
        add("INFO", "A", "running on the synthetic fixture, not the real dataset")
    if m.get("n_empty_tags"):
        add("INFO", "A", f"{m['n_empty_tags']} problems have no tags (tag balance and the tag breakdowns exclude them)")
    c = m.get("contamination")
    if c and c.get("n_dated"):
        add("INFO", "B", f"{100 * (1 - c['share_after']):.0f}% of problems are dated on/before the ASSUMED Qwen3 horizon {c['horizon']} "
                         "(UNVERIFIED horizon): contamination is likely and not fixable")
    pr = m.get("prompt")
    if pr:
        if pr.get("n_over", 0) > 0:
            add("WARN", "C", f"{pr['n_over']} prompts ({100 * pr['share_over']:.1f}%) exceed max_prompt_tokens={pr['limit']} and would not be generated")
        if pr.get("tokenizer_kind") == "proxy":
            add("WARN", "C", "prompt lengths use the word-count PROXY, not the Qwen3 tokenizer; truncation risk is a lower bound")
    deg = m.get("degenerate")
    if deg and deg["share"] > DEGENERATE_WARN_SHARE:
        add("WARN", "D", f"{deg['n']} problems ({100 * deg['share']:.1f}%) have <=2 distinct expected outputs among the reward tests "
                         f"({deg.get('n_constant', 0)} with a single one): hard-coded answers can pass the reward tests")
    dt = m.get("dup_tests", {})
    if dt.get("reward_heldout_overlap", 0):
        add("ERROR", "D", f"{dt['reward_heldout_overlap']} tests appear in both the reward and held-out sets")
    if dt.get("within_problem_duplicate_tests", 0):
        add("WARN", "D", f"{dt['within_problem_duplicate_tests']} duplicated tests inside problems")
    if dt.get("same_input_conflicting_expected", 0):
        add("ERROR", "D", f"{dt['same_input_conflicting_expected']} inputs with conflicting expected outputs")
    ref = m.get("reference")
    if ref:
        if ref.get("validity") is not None and ref["validity"] < 0.95:
            add("WARN", "E", f"reference validity {100 * ref['validity']:.1f}% < 95% (Gate 1c)")
        if ref.get("n_slow"):
            add("INFO", "E", f"{ref['n_slow']} sampled reference solutions take > {SLOW_REFERENCE_S:g}s in the sandbox (timeouts hit honest solutions first)")
    lk = m.get("leakage", {})
    if lk.get("n_duplicate_ids"):
        add("ERROR", "F", f"{lk['n_duplicate_ids']} duplicate problem ids")
    if lk.get("n_cross_cluster_pairs"):
        add("ERROR", "F", f"{lk['n_cross_cluster_pairs']} pairs with Jaccard >= {NEAR_DUP_THRESHOLD} are in different clusters")
    if lk.get("n_spanning"):
        add("ERROR", "F", f"{lk['n_spanning']} clusters span more than one split")
    g = m.get("passrate")
    if g:
        if g.get("n_selected") is not None and g["n_selected"] < sum(MIN_PROBLEMS.values()):
            add("WARN", "G", f"only {g['n_selected']} problems in the band, fewer than the {sum(MIN_PROBLEMS.values())} the split needs (the single pre-declared widening applies)")
        for stage, s in (g.get("stats") or {}).items():
            if s.get("truncation_rate", 0) > TRUNC_WARN_RATE:
                add("WARN", "G", f"stage {stage}: {100 * s['truncation_rate']:.1f}% of base completions hit the token limit")
            if s.get("extract_fail_rate", 0) > EXTRACT_FAIL_WARN_RATE:
                add("WARN", "G", f"stage {stage}: {100 * s['extract_fail_rate']:.1f}% code-extraction failures at base (pass rates are biased down)")
        w = g.get("weak_tests")
        if w and w.get("frac_visible_pass_fail_heldout", 0) > WEAK_TESTS_WARN_SHARE:
            add("WARN", "G", f"{100 * w['frac_visible_pass_fail_heldout']:.0f}% of visible passes fail the held-out tests (weak reward tests)")
        r = g.get("rtm")
        if r and r.get("n", 0) >= 3 and (r["shift_ci"][1] < 0 or r["shift_ci"][0] > 0):
            add("INFO", "G", f"regression to the mean: band-selected problems moved by {r['mean_shift']:+.3f} between stage A and B; report p_B, not p_A")
    s = m.get("splits")
    if s:
        for item in s.get("gate1c_fail", []):
            add("WARN", "H", f"Gate 1c item failing: {item}")
        if s.get("hash_ok") is False:
            add("ERROR", "H", "split hash in splits.json does not recompute from problems.jsonl")
        for msg in s.get("balance_flags", []):
            add("WARN", "H", msg)
    p = m.get("probe")
    if p:
        if not p.get("go"):
            add("WARN", "I", "hint probe is NO-GO: " + "; ".join(p.get("reasons", [])))
        if p.get("sample_size_ok") is False:
            add("WARN", "I", "hint probe has fewer than the pre-declared 3000 samples per wording")
        if p.get("confound_flag"):
            add("WARN", "I", "a hint wording shifts the step-0 honest pass rate by > 5 pp (confound for H1/H3)")
        if p.get("mock"):
            add("INFO", "I", "the probe file is a MOCK run, not a measurement")
    for msg in m.get("power_flags", []):
        add("INFO", "J", msg)
    lb = m.get("labels")
    if lb:
        if lb.get("n_narrow_misses"):
            add("INFO", "K", f"the narrow detector misses {lb['n_narrow_misses']} controls that execution labels HACK_RT (by design: leaky monitor)")
        if lb.get("n_broad_misses"):
            add("WARN", "K", f"the broad detector misses {lb['n_broad_misses']} controls that execution labels HACK_RT")
        if lb.get("n_exec_blind"):
            add("INFO", "K", f"{lb['n_exec_blind']} constructed hacks are invisible to the execution label (special-casing etc.)")
    for sec in m.get("unavailable", []):
        add("INFO", sec, "section not available yet (inputs missing); its checks have not run")
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    return sorted(out, key=lambda f: (order[f["severity"]], f["section"]))


# ------------------------------------------------------------------ figures (dataviz: opaque surface so they read in light and dark UIs)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (validated all-pairs in the reference palette)
NEUTRAL = "#8d8c86"
SPLIT_COLOR = dict(zip(SPLITS, SERIES))


def _plt():
    import matplotlib

    if not os.environ.get("MPLBACKEND") and "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 100,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": NEUTRAL, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "regular", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "lines.linewidth": 2.0,
    })
    return plt


def _int_locator():
    from matplotlib.ticker import MaxNLocator

    return MaxNLocator(integer=True)


def show(fig) -> None:
    """Display a figure in the notebook and release it."""
    from IPython.display import display

    display(fig)
    _plt().close(fig)


def _finish(ax, title: str, xlabel: str, ylabel: str, n: int | str | None = None) -> None:
    ax.set_title(title + (f"  (n = {n})" if n is not None else ""))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def bar_figure(labels: Sequence[str], values: Sequence[float], title: str, xlabel: str, ylabel: str, n: int | None = None,
               horizontal: bool = False, color: str = SERIES[0]):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, 0.32 * len(labels) + 1.2) if horizontal else 3.6))
    if horizontal:
        ax.barh(list(labels)[::-1], list(values)[::-1], color=color, edgecolor=SURFACE, linewidth=1.5, height=0.72)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(list(labels), list(values), color=color, edgecolor=SURFACE, linewidth=1.5, width=0.7)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_locator(_int_locator())
    _finish(ax, title, xlabel, ylabel, n)
    fig.tight_layout()
    return fig


def hist_figure(series: Mapping[str, Sequence[float]], title: str, xlabel: str, ylabel: str = "problems", bins: int | Sequence[float] = 30,
                vline: tuple[float, str] | None = None, colors: Mapping[str, str] | None = None, log_y: bool = False):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    allv = np.concatenate([np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for v in series.values()] or [np.array([])])
    edges = np.histogram_bin_edges(allv, bins=bins) if allv.size else bins
    for i, (name, v) in enumerate(series.items()):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        c = (colors or {}).get(name, SERIES[i % len(SERIES)])
        ax.hist(v, bins=edges, color=c, edgecolor=SURFACE, linewidth=0.8, alpha=0.9 if len(series) == 1 else 0.6, label=f"{name} (n={len(v)})")
    if vline is not None:
        ax.axvline(vline[0], color=INK, linewidth=1.2, linestyle="--")
        ax.annotate(vline[1], (vline[0], 1), xycoords=("data", "axes fraction"), xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=INK2)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_locator(_int_locator())
    if len(series) > 1:
        ax.legend()
    _finish(ax, title, xlabel, ylabel, sum(len(v) for v in series.values()) if len(series) == 1 else None)
    fig.tight_layout()
    return fig


def scatter_figure(x: Sequence[float], y: Sequence[float], title: str, xlabel: str, ylabel: str, band: tuple[float, float] | None = None,
                   diagonal: bool = True, jitter: float = 0.006, seed: int = 0):
    plt = _plt()
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    if band:
        ax.axvspan(band[0], band[1], color=SERIES[0], alpha=0.10, linewidth=0)
        ax.axhspan(band[0], band[1], color=SERIES[0], alpha=0.10, linewidth=0)
    if diagonal:
        ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1.2, linestyle="--")
    ax.scatter(x + rng.normal(0, jitter, x.size), y + rng.normal(0, jitter, y.size), s=22, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    _finish(ax, title, xlabel, ylabel, int(np.sum(~(np.isnan(x) | np.isnan(y)))))
    fig.tight_layout()
    return fig


def strip_by_group_figure(groups: Mapping[str, Sequence[float]], title: str, ylabel: str, seed: int = 0):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    rng = np.random.default_rng(seed)
    for i, (name, v) in enumerate(groups.items()):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=16, color=SPLIT_COLOR.get(name, SERIES[i % 3]), edgecolor=SURFACE, linewidth=0.6, alpha=0.85, zorder=2)
        if v.size:
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=3)
    ax.set_xticks(range(len(groups)), [f"{k}\nn={len(v)}" for k, v in groups.items()])
    ax.grid(axis="x", visible=False)
    ax.set_title(title + "  (bar = median)")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def forest_figure(labels: Sequence[str], est: Sequence[float], lo: Sequence[float], hi: Sequence[float], title: str, xlabel: str,
                  vline: float | None = None, ns: Sequence[int] | None = None, pct: bool = False):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.55 * len(labels) + 1.2)))
    ys = np.arange(len(labels))[::-1]
    f = 100.0 if pct else 1.0
    est, lo, hi = (np.asarray(v, float) * f for v in (est, lo, hi))
    ax.hlines(ys, lo, hi, color=SERIES[0], linewidth=2)
    ax.scatter(est, ys, s=46, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    if vline is not None:
        ax.axvline(vline * f, color=NEUTRAL, linewidth=1.2, linestyle="--")
    ax.set_yticks(ys, [f"{lab}  (n={n})" if ns is not None else lab for lab, n in zip(labels, ns if ns is not None else labels)])
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def line_figure(x: Sequence[float], series: Mapping[str, Sequence[float]], title: str, xlabel: str, ylabel: str):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (name, y) in enumerate(series.items()):
        ax.plot(x, y, color=SERIES[i % len(SERIES)], marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
