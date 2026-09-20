"""Public results bundle (subtask 14): ``python -m rhg.analysis.bundle --out results_public/``.

Assembles what is safe and useful to share: ``REPORT.md``, ``tests.json``, ``per_seed.csv``, ``tables/``, ``figures/``,
``examples.md``, ``validation.md`` (if present), ``FREEZE.json``, ``AMENDMENTS.jsonl``, resolved configs, run manifests,
``ledger_summary.json`` (spend by kind only), ``README_RESULTS.md`` (a pre-registered vs exploratory table generated from
``tests.json``) and ``rollouts_sample.jsonl.gz`` (a seeded random sample of final-test-eval rollouts per arm).
Raw rollouts, adapters, caches and stdout logs are never copied.

The bundle is staged in a temporary directory next to ``--out`` and only moved into place when every text file passes the
secret scan (``sk-ant``, ``ANTHROPIC``, absolute home paths, raw hostnames) and the total size is within the cap
(25 MB); otherwise nothing is published and the exit code is 3.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import re
import shutil
import socket
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rhg import runlog
from rhg.analysis import endpoints as E
from rhg.manifest import EXIT_GUARD_REFUSED, REPO_ROOT
from rhg.seeds import derive_seed

SAMPLE_PER_ARM = 200
SAMPLE_SEED = 20260920
MAX_BYTES = 25 * 1024 * 1024
REQUIRED = ("REPORT.md", "tests.json", "per_seed.csv")
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".cff", ".log"}
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sk-ant", re.compile(r"sk-ant")),
    ("ANTHROPIC", re.compile(r"ANTHROPIC")),
    ("windows home path", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", re.I)),
    ("posix home path", re.compile(r"(?<![\w.])/(?:home|Users)/[A-Za-z0-9_.-]+/")),
    ("root home path", re.compile(r"(?<![\w.])/root/")),
    ("raw hostname field", re.compile(r"hostname(?!_sha256)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9][A-Za-z0-9._-]{2,}", re.I)),
)


class BundleError(RuntimeError):
    """The bundle was refused (secret found, size cap); nothing was published."""

    exit_code = EXIT_GUARD_REFUSED


class BundleInputError(ValueError):
    """Missing analysis outputs (usage error)."""


# ------------------------------------------------------------------ generated README
def results_table(doc: dict[str, Any]) -> list[dict[str, str]]:
    """One row per test of ``tests.json``: pre-registered class vs the stamp of this analysis, p, min attainable p, n, result."""
    def f(x: Any) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    return [{"id": t["id"], "hypothesis": t["hypothesis"], "pre-registered as": t["registered"], "stamped in this analysis": t["stamp"],
             "p": f(t.get("p")), "min attainable p": f(t.get("min_attainable_p")), "n": str(t.get("n_text") or ""),
             "result": str(t.get("result") or "")} for t in doc["tests"]]


def render_results_table(rows: Sequence[dict[str, str]]) -> str:
    cols = list(rows[0]) if rows else []
    esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def render_readme(doc: dict[str, Any], files: Sequence[str], ledger: dict[str, Any], sizes: dict[str, int]) -> str:
    prim = next(t for t in doc["tests"] if t["id"] == "primary")
    return "\n".join([
        "# Results bundle",
        "",
        f"Analysis mode: **{doc['mode']}**. " + ("Only the primary and the Holm family are confirmatory; everything else is exploratory."
                                                  if doc["mode"] == "CONFIRMATORY" else "The analysis was not run with `--confirmatory`, so everything is stamped EXPLORATORY."),
        f"Primary outcome (PREREG §2 wording): **{prim.get('result') or 'not testable'}**.",
        "",
        "## Pre-registered vs exploratory",
        "",
        "Generated from `tests.json`. 'pre-registered as' is the class fixed in PREREG.md; 'stamped in this analysis' is what this run may claim.",
        "A p-value is only meaningful next to its minimum attainable value and n.",
        "",
        render_results_table(results_table(doc)),
        "## Contents",
        "",
        *[f"- `{f}`" for f in files],
        "",
        "Not included by design: raw rollouts (`rollouts.jsonl.gz` of each run), adapters, caches, stdout logs. "
        f"`rollouts_sample.jsonl.gz` is a seeded random sample (up to {SAMPLE_PER_ARM} rollouts per arm, final test eval, seed {SAMPLE_SEED}).",
        "",
        "## Spend",
        "",
        "```json",
        json.dumps(ledger, indent=2),
        "```",
        "",
    ])


# ------------------------------------------------------------------ scanning
def _forbidden_literals(extra: Sequence[str]) -> list[str]:
    lits = {str(Path.home()), str(Path.home()).replace("\\", "/"), socket.gethostname(), platform.node(), *extra}
    return sorted((x for x in lits if x and len(x) >= 5 and x.lower() not in {"localhost", "root", "runner"}), key=len, reverse=True)


def scan_text(name: str, text: str, literals: Sequence[str]) -> list[str]:
    findings = []
    for label, pat in SECRET_PATTERNS:
        m = pat.search(text)
        if m:
            findings.append(f"{name}: matches '{label}' (line {text.count(chr(10), 0, m.start()) + 1})")
    low = text.lower()
    for lit in literals:
        if lit.lower() in low:
            findings.append(f"{name}: contains a local identifier ({lit[:2]}... , {len(lit)} chars: home directory or hostname)")
    return findings


def scan_tree(root: Path, extra_forbidden: Sequence[str] = ()) -> list[str]:
    literals = _forbidden_literals(extra_forbidden)
    findings: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes()
        if p.suffix == ".gz":
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError):
                findings.append(f"{rel}: unreadable gzip")
                continue
        text = data.decode("utf-8", errors="replace") if (p.suffix in TEXT_SUFFIXES or p.suffix == ".gz") else data.decode("latin-1")
        findings += scan_text(rel, text, literals)
    return findings


# ------------------------------------------------------------------ assembling
def _final_rollouts_by_arm(runs_dir: Path) -> dict[str, list[dict[str, Any]]]:
    pilot_min = E._pilot_seed_min()
    out: dict[str, list[dict[str, Any]]] = {}
    for d in sorted(x for x in runs_dir.iterdir() if x.is_dir()) if runs_dir.is_dir() else []:
        ident = E.split_run_id(d.name)
        if not ident or ident[1] >= pilot_min or E._read_json(d / "status.json").get("status") != "completed":
            continue
        T = E._configured_T(d)
        if T is None:
            continue
        with gzip.open(d / runlog.ROLLOUTS_FILE, "rt", encoding="utf-8") as f:  # already validated by the analysis run
            for line in f:
                if '"eval_test"' not in line:
                    continue
                rec = json.loads(line)
                if rec["phase"] == "eval_test" and rec["step"] == T:
                    out.setdefault(ident[0], []).append({"arm": ident[0], "seed": ident[1], **rec})
    return out


def write_rollout_sample(runs_dir: Path, path: Path, seed: int = SAMPLE_SEED, per_arm: int = SAMPLE_PER_ARM) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for arm, recs in sorted(_final_rollouts_by_arm(runs_dir).items()):
            recs.sort(key=lambda r: (r["seed"], r["problem_id"], r["sample_idx"]))
            rng = np.random.default_rng(derive_seed(seed, f"bundle_sample:{arm}"))
            idx = sorted(rng.choice(len(recs), size=min(per_arm, len(recs)), replace=False).tolist())
            for i in idx:
                gz.write((json.dumps(recs[i], sort_keys=True) + "\n").encode("utf-8"))
            sizes[arm] = len(idx)
    return sizes


def _copy(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _ledger_summary(ledger: Path | None) -> dict[str, Any]:
    from rhg import budget

    if ledger is None or not Path(ledger).is_file():
        return {"spend_usd_by_kind": {}, "total_usd": 0.0, "note": "no ledger found"}
    by = budget.spent_by_kind(ledger)
    return {"spend_usd_by_kind": {k: round(v, 4) for k, v in sorted(by.items())}, "total_usd": round(sum(by.values()), 4)}


def build_bundle(out: str | Path, *, analysis_dir: str | Path = "results/analysis", runs_dir: str | Path = "results/runs",
                 repo_root: str | Path | None = None, ledger: str | Path | None = "results/ledger.jsonl",
                 validation: str | Path | None = None, seed: int = SAMPLE_SEED, max_bytes: int = MAX_BYTES,
                 extra_forbidden: Sequence[str] = ()) -> dict[str, Any]:
    """Stage, scan, size-check and publish the bundle. Raises ``BundleInputError`` / ``BundleError`` (nothing published)."""
    out, ana, runs = Path(out), Path(analysis_dir), Path(runs_dir)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    missing = [n for n in REQUIRED if not (ana / n).is_file()]
    if missing:
        raise BundleInputError(f"{ana} lacks {', '.join(missing)}: run `python -m rhg.analysis.run` first")
    if out.exists() and any(out.iterdir()) and not (out / "README_RESULTS.md").is_file():
        raise BundleInputError(f"{out} exists, is not empty and is not a results bundle; refusing to replace it")
    doc = json.loads((ana / "tests.json").read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".bundle_stage_", dir=out.parent))
    try:
        for n in REQUIRED + ("examples.md",):
            _copy(ana / n, stage / n)
        for sub in ("figures", "tables"):
            if (ana / sub).is_dir():
                for f in sorted((ana / sub).iterdir()):
                    if f.is_file() and f.suffix in {".png", ".json", ".csv"}:
                        _copy(f, stage / sub / f.name)
        val = Path(validation) if validation else ana / "validation.md"
        _copy(val, stage / "validation.md")
        if not _copy(root / "prereg" / "FREEZE.json", stage / "FREEZE.json"):
            (stage / "FREEZE.json").write_text('{"note": "prereg/FREEZE.json not present: no pre-registration freeze has been made"}\n', encoding="utf-8")
        if not _copy(root / "prereg" / "AMENDMENTS.jsonl", stage / "AMENDMENTS.jsonl"):
            (stage / "AMENDMENTS.jsonl").write_text("", encoding="utf-8")
        for d in sorted(x for x in runs.iterdir() if x.is_dir()) if runs.is_dir() else []:
            _copy(d / runlog.CONFIG_FILE, stage / "configs" / f"{d.name}.yaml")
            _copy(d / runlog.MANIFEST_FILE, stage / "manifests" / f"{d.name}.json")
        ledger_doc = _ledger_summary(Path(ledger) if ledger else None)
        (stage / "ledger_summary.json").write_text(json.dumps(ledger_doc, indent=2) + "\n", encoding="utf-8")
        sample_sizes = write_rollout_sample(runs, stage / "rollouts_sample.jsonl.gz", seed)
        files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
        (stage / "README_RESULTS.md").write_text(render_readme(doc, sorted({*files, "README_RESULTS.md"}), ledger_doc, sample_sizes),
                                                 encoding="utf-8", newline="\n")
        findings = scan_tree(stage, extra_forbidden)
        if findings:
            raise BundleError("secret scan failed, nothing published:\n  " + "\n  ".join(findings[:20]))
        total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
        if total > max_bytes:
            raise BundleError(f"bundle is {total / 1e6:.1f} MB, over the {max_bytes / 1e6:.1f} MB cap; nothing published")
        if out.exists():
            shutil.rmtree(out)
        stage.replace(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"out": str(out), "bytes": total, "files": sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()),
            "sample_sizes": sample_sizes}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhg.analysis.bundle", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results_public"))
    ap.add_argument("--analysis", type=Path, default=Path("results/analysis"))
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--ledger", type=Path, default=Path("results/ledger.jsonl"))
    ap.add_argument("--validation", type=Path, default=None, help="validation.md (default: <analysis>/validation.md if present)")
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--max-mb", type=float, default=MAX_BYTES / 1024 / 1024)
    ap.add_argument("--forbid", action="append", default=[], help="extra literal that must not appear in the bundle (repeatable)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        res = build_bundle(args.out, analysis_dir=args.analysis, runs_dir=args.runs, repo_root=args.repo_root, ledger=args.ledger,
                           validation=args.validation, seed=args.seed, max_bytes=int(args.max_mb * 1024 * 1024), extra_forbidden=args.forbid)
    except BundleInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except BundleError as e:
        print(f"refused: {e}", file=sys.stderr)
        return e.exit_code
    print(f"wrote {res['out']} ({res['bytes'] / 1e6:.2f} MB, {len(res['files'])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
