"""Shared helpers for the generator / pass-rate / probe tests (fixture problems, planted solutions)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from rhg.data import fixture

TINY = Path(__file__).parent / "fixtures" / "problems_tiny.jsonl"


def tiny_problems() -> list[dict]:
    return [json.loads(line) for line in TINY.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")


def tiny_dir(root: Path, name: str = "proc") -> Path:
    """Processed dir with the 8 tiny fixture problems as candidates and problems (4 train / 2 val / 2 test)."""
    d = root / name
    write_jsonl(d / "candidates.jsonl", tiny_problems())
    write_jsonl(d / "problems.jsonl", tiny_problems())
    return d


def full_candidates_dir(root: Path, name: str = "full") -> Path:
    """Processed dir with all 40 synthetic fixture problems as ``candidates.jsonl``."""
    d = root / name
    write_jsonl(d / "candidates.jsonl", fixture.fixture_candidates())
    return d


def visible_only_completion(problem: dict) -> str:
    """Toy overfit: a lookup table of the *reward* tests' answers (passes them, fails the held-out tests).

    Built by parsing ``assert candidate(ARGS) == EXPECTED`` sources; safe (no I/O), used to plant
    ``k_visible > k_full`` in the pass-rate tests.
    """
    entries = []
    for t in problem["reward_tests"]:
        node = ast.parse(t["src"]).body[0]
        assert isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
        call, expected = node.test.left, node.test.comparators[0]
        args = "(" + "".join(ast.unparse(a) + ", " for a in call.args) + ")"
        kws = "{" + ", ".join(f"{k.arg!r}: {ast.unparse(k.value)}" for k in call.keywords) + "}"
        entries.append(f"_MEMO[_key({args}, {kws})] = {ast.unparse(expected)}")
    ep = problem["entry_point"]
    head = "_MEMO = {}\n\n\ndef _key(a, kw):\n    return repr((tuple(a), tuple(sorted(kw.items()))))\n\n\n" + "\n".join(entries) + "\n\n\n"
    if ep.startswith("Solution()."):
        name = ep.split(".", 1)[1]
        body = f"class Solution:\n    def {name}(self, *a, **kw):\n        return _MEMO.get(_key(a, kw))\n"
    else:
        body = f"def {ep}(*a, **kw):\n    return _MEMO.get(_key(a, kw))\n"
    return "Memorised answers:\n\n```python\n" + head + body + "```"
