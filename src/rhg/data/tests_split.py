"""Turn a dataset ``check(candidate)`` function into individual ``assert`` tests and split them.

Primary source (DESIGN §2.2): the dataset's ``test`` field. Every real ``newfacade/LeetCodeDataset``
record has one top-level ``def check(candidate):`` whose body is a flat list of ``assert`` statements,
so each ``assert`` becomes one test. The ``input_output`` field is only a *fallback* (used when ``test``
cannot be parsed or yields no assert): it stores inputs/outputs as strings, and cannot express problems
whose tests build ``ListNode``/``TreeNode`` values, so it is never used for those (see
``docs/dataset_notes.md``).

A test's ``src`` is executable in a namespace that already contains ``candidate``. Non-assert
statements of the check function (imports, assignments, helper defs) and module-level statements other
than ``check`` are the *shared preamble* and are prepended to every test that follows them.

The reward/held-out split is a pure function of ``(problem_id, test index, TEST_SPLIT_SALT)``: it does
not take the training seed, so every seed and arm sees the same tests.
"""

from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Iterable

TEST_SPLIT_SALT = "rhg-test-split-v1"

# Minimum held-out tests on top of K reward tests (DESIGN §2.2 drop rule: fewer than K+5 tests).
MIN_EXTRA_TESTS = 5


class CheckParseError(ValueError):
    """The ``test`` code could not be parsed into a ``check`` function."""


class PrefixError(RuntimeError):
    """The dataset's import prefix cannot be executed in the sandbox (e.g. a third-party import)."""


@dataclass
class ParsedTests:
    tests: list[str]
    source: str  # "check" | "input_output"
    n_raw_asserts: int = 0
    n_duplicates: int = 0
    n_compound: int = 0
    preamble: str = ""
    notes: list[str] = field(default_factory=list)


def _parse(code: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dataset strings contain invalid escapes ('\:')
        try:
            return ast.parse(code)
        except (SyntaxError, ValueError) as e:
            raise CheckParseError(f"cannot parse test code: {e}") from e


def _stmt_src(node: ast.stmt, code: str) -> str:
    if isinstance(node, ast.Assert):
        seg = ast.get_source_segment(code, node)  # exact text; keeps multi-line string contents
        if seg:
            return seg
    return ast.unparse(node)


def _has_assert(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Assert) for n in ast.walk(node))


def parse_check(code: str) -> ParsedTests:
    """Parse ``code`` (module text containing ``def check(candidate)``) into per-assert tests."""
    tree = _parse(code)
    check_fn: ast.FunctionDef | None = None
    module_context: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check" and check_fn is None:
            check_fn = node
        else:
            module_context.append(ast.unparse(node))
    if check_fn is None:
        raise CheckParseError("no top-level `def check(...)` found")

    context: list[str] = list(module_context)
    tests: list[str] = []
    seen: set[str] = set()
    n_raw = n_dup = n_compound = 0
    notes: list[str] = []

    def emit(stmt_src: str) -> None:
        nonlocal n_dup
        src = "\n".join([*context, stmt_src]) if context else stmt_src
        if src in seen:
            n_dup += 1
            return
        seen.add(src)
        tests.append(src)

    for stmt in check_fn.body:
        if isinstance(stmt, ast.Assert):
            n_raw += 1
            emit(_stmt_src(stmt, code))
        elif _has_assert(stmt):
            n_raw += 1
            n_compound += 1
            emit(ast.unparse(stmt))
        elif isinstance(stmt, (ast.Pass, ast.Return)) or (
            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
        ):
            continue  # docstring / pass / return carry no test logic
        else:
            src = ast.unparse(stmt)
            if src not in context:
                context.append(src)
    preamble = "\n".join(context)
    if preamble:
        notes.append("preamble")
    return ParsedTests(tests, "check", n_raw, n_dup, n_compound, preamble, notes)


def tests_from_input_output(pairs: Iterable[dict]) -> ParsedTests:
    """Fallback: ``assert candidate(<input>) == <output>`` from ``{'input': 'a = 1, b = 2', 'output': '3'}``."""
    tests: list[str] = []
    seen: set[str] = set()
    n_dup = n_bad = 0
    for p in pairs:
        inp, out = str(p.get("input", "")), str(p.get("output", ""))
        src = f"assert candidate({inp}) == {out}"
        try:
            _parse(src)
        except CheckParseError:
            n_bad += 1
            continue
        if src in seen:
            n_dup += 1
            continue
        seen.add(src)
        tests.append(src)
    notes = [f"unparseable_pairs={n_bad}"] if n_bad else []
    return ParsedTests(tests, "input_output", len(tests) + n_dup, n_dup, 0, "", notes)


def split_order(problem_id: str, n_tests: int, salt: str = TEST_SPLIT_SALT) -> list[int]:
    """Test indices ordered by ``sha256(salt|problem_id|index)``; the first K are the reward tests."""

    def key(i: int) -> bytes:
        return hashlib.sha256(f"{salt}|{problem_id}|{i}".encode("utf-8")).digest()

    return sorted(range(n_tests), key=key)


def split_tests(
    problem_id: str,
    tests: list[str],
    k_reward: int,
    max_heldout: int,
    salt: str = TEST_SPLIT_SALT,
) -> tuple[list[dict], list[dict]]:
    """Disjoint ``(reward_tests, heldout_tests)``; each test is ``{"id", "src", "kind": "assert"}``.

    ``id`` is the index in ``tests`` (unique per problem). Requires ``len(tests) >= k_reward + MIN_EXTRA_TESTS``.
    """
    if len(tests) < k_reward + MIN_EXTRA_TESTS:
        raise ValueError(f"{problem_id}: {len(tests)} tests < K+{MIN_EXTRA_TESTS}={k_reward + MIN_EXTRA_TESTS}")
    order = split_order(problem_id, len(tests), salt)
    reward_idx = sorted(order[:k_reward])
    heldout_idx = sorted(order[k_reward : k_reward + max_heldout])
    mk = lambda i: {"id": i, "src": tests[i], "kind": "assert"}  # noqa: E731
    return [mk(i) for i in reward_idx], [mk(i) for i in heldout_idx]


# ---------------------------------------------------------------- helper-name analysis

_BUILTINS = frozenset(dir(builtins))


def free_names(src: str) -> set[str]:
    """Names read by ``src`` that are not bound anywhere inside it and are not builtins.

    Flow-insensitive on purpose: it only has to catch tests that need a helper (``ListNode``,
    ``tree_node``, ...) that the import prefix does not supply; execution of the reference is the
    authoritative check.
    """
    tree = _parse(src)
    loads: set[str] = set()
    bound: set[str] = {"candidate"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (loads if isinstance(n.ctx, ast.Load) else bound).add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return loads - bound - _BUILTINS


_NAMES_SCRIPT = r"""
import json, sys
ns = {"__name__": "__rhg_prefix__"}
try:
    exec(compile(sys.stdin.read(), "<import_prefix>", "exec"), ns)
    print("RHG_NAMES:" + json.dumps(sorted(k for k in ns if not k.startswith("__"))))
except BaseException as e:
    print("RHG_NAMES_ERROR:" + type(e).__name__ + ":" + str(e)[:120].replace("\n", " "))
"""


@functools.lru_cache(maxsize=64)
def prefix_names(prefix: str, timeout_s: float = 30.0) -> frozenset[str]:
    """Names defined by executing the dataset's import prefix in the sandbox."""
    from rhg.env.sandbox import run_python

    res = run_python(_NAMES_SCRIPT, timeout_s=timeout_s, mem_mb=2048, stdin_data=prefix)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("RHG_NAMES:"):
            return frozenset(json.loads(line[len("RHG_NAMES:") :]))
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("RHG_NAMES_ERROR:"):
            raise PrefixError(line[len("RHG_NAMES_ERROR:") :].strip())
    raise PrefixError(f"prefix run status {res.status}: {res.stdout[-120:]!r}")


def unsupported_names(tests: Iterable[str], prefix: str) -> list[str]:
    """Sorted names used by ``tests`` that neither the prefix nor builtins supply."""
    known = prefix_names(prefix)
    missing: set[str] = set()
    for t in tests:
        missing |= free_names(t) - known
    return sorted(missing)
