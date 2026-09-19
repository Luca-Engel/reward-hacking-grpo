"""Near-duplicate clusters of problems (DESIGN §2.2 "keep near-duplicate clusters within one split").

Two problems are linked when

* the Jaccard similarity of their 5-word shingle sets is >= 0.8, or
* their normalised titles are identical.

Text normalisation: lower-case; cut at the first ``Example`` / ``Constraints`` / ``Follow-up`` line
(that boilerplate carries the numbers and sample inputs, not the task); drop every character that is
not a letter (so numbers and punctuation vanish). The title is the problem's slug (``problem_id``)
with pure-number and Roman-numeral tokens (``ii``..``ix``) removed, so ``foo-bar`` and ``foo-bar-ii``
share a title. ``cluster_id`` is the smallest ``problem_id`` in the connected component, so it does
not depend on input order. Jaccard is compared with exact integer arithmetic.

CLI: ``python -m rhg.data.dedupe [--fixture] [--input converted.jsonl] [--threshold 0.8]``
prints the cluster-size histogram.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SHINGLE_WORDS = 5
JACCARD_THRESHOLD = "0.8"

_CUT = re.compile(r"^[ \t]*(example\s*\d*\s*:|constraints?\s*:|follow[\s-]*up\s*:)", re.IGNORECASE | re.MULTILINE)
_NON_LETTER = re.compile(r"[^a-z]+")
_ROMAN = {"ii", "iii", "iv", "vi", "vii", "viii", "ix"}


def normalize_text(description: str) -> list[str]:
    """Words of the task statement with examples, constraints, numbers and punctuation removed."""
    text = description.replace("\xa0", " ").lower()
    m = _CUT.search(text)
    if m:
        text = text[: m.start()]
    return _NON_LETTER.sub(" ", text).split()


def normalize_title(title: str) -> str:
    toks = [t for t in re.split(r"[^a-z0-9]+", title.lower()) if t]
    kept = [t for t in toks if not t.isdigit() and t not in _ROMAN]
    return "-".join(kept)


def shingles(words: Sequence[str], n: int = SHINGLE_WORDS) -> frozenset[str]:
    if not words:
        return frozenset()
    if len(words) < n:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def cluster_problems(
    problems: Sequence[Mapping],
    threshold: str | float = JACCARD_THRESHOLD,
    n_words: int = SHINGLE_WORDS,
    return_stats: bool = False,
):
    """Map ``problem_id -> cluster_id`` (deterministic, order independent).

    With ``return_stats`` also returns ``{"edges_jaccard", "edges_title"}`` (linked pairs by kind).
    """
    thr = Fraction(str(threshold))
    ids = [p["problem_id"] for p in problems]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate problem_id in input")
    sh = [shingles(normalize_text(p.get("description", "")), n_words) for p in problems]
    uf = _UnionFind(len(problems))
    stats = {"edges_jaccard": 0, "edges_title": 0}

    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        for x in s:
            index[x].append(i)
    for i, s in enumerate(sh):
        counts: Counter[int] = Counter()
        for x in s:
            for j in index[x]:
                if j > i:
                    counts[j] += 1
        for j, c in counts.items():
            union = len(s) + len(sh[j]) - c
            if c * thr.denominator >= thr.numerator * union:
                stats["edges_jaccard"] += 1
                uf.union(i, j)

    by_title: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(problems):
        t = normalize_title(p.get("title") or p["problem_id"])
        if t:
            by_title[t].append(i)
    for members in by_title.values():
        for j in members[1:]:
            if uf.find(j) != uf.find(members[0]):
                stats["edges_title"] += 1
            uf.union(members[0], j)

    comp_min: dict[int, str] = {}
    for i, pid in enumerate(ids):
        r = uf.find(i)
        if r not in comp_min or pid < comp_min[r]:
            comp_min[r] = pid
    out = {pid: comp_min[uf.find(i)] for i, pid in enumerate(ids)}
    return (out, stats) if return_stats else out


def cluster_histogram(assignment: Mapping[str, str]) -> dict:
    """``{"size_histogram": {size: n_clusters}, "n_clusters", "n_problems", "n_affected", "largest"}``.

    ``n_affected`` = problems that share a cluster with at least one other problem.
    """
    sizes = Counter(assignment.values())
    hist = Counter(sizes.values())
    return {
        "size_histogram": {int(k): hist[k] for k in sorted(hist)},
        "n_clusters": len(sizes),
        "n_problems": len(assignment),
        "n_affected": sum(s for s in sizes.values() if s > 1),
        "largest": max(sizes.values(), default=0),
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rhg.data.dedupe", description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixture", action="store_true", help="use the synthetic fixture problems (no data files)")
    ap.add_argument("--input", type=Path, default=Path("data/processed/converted.jsonl"))
    ap.add_argument("--threshold", default=JACCARD_THRESHOLD)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.fixture:
        from rhg.data.fixture import fixture_candidates

        problems = fixture_candidates()
    else:
        if not args.input.is_file():
            print(f"error: {args.input} not found (run `python -m rhg.data.build --stage tests` first)", file=sys.stderr)
            return 2
        problems = [json.loads(line) for line in args.input.read_text(encoding="utf-8").split("\n") if line.strip()]
    assign, stats = cluster_problems(problems, args.threshold, return_stats=True)
    rep = cluster_histogram(assign)
    print(json.dumps({**rep, **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
