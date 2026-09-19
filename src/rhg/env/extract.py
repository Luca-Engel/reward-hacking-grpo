"""Code extraction from a model completion (DESIGN §2.4).

``extract_code`` is the single function used by the grader, the AST detector and the judge
input, so all three see byte-identical code.

Rule: the last fenced block labelled ``python`` (also ``python3``/``py``); otherwise the last
unlabelled ``` block; otherwise nothing. Fences must start a line (leading whitespace allowed);
a closing fence must use at least as many backticks as the opener and carry nothing else on the
line. A final block that is never closed counts as code (lenient: a completion cut off at the
token limit still has its code graded) and is reported with ``truncated_fence=True``. Blocks that
are empty or whitespace-only are not code. Blocks labelled with another language are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_OPEN = re.compile(r"^[ \t]*(`{3,})[ \t]*([^`\s]*)[^`]*$")
_CLOSE = re.compile(r"^[ \t]*(`{3,})[ \t]*$")
_PYTHON_TAGS = frozenset({"python", "python3", "py"})


@dataclass(frozen=True)
class ExtractResult:
    code: str | None
    how: str  # "python_fence" | "generic_fence" | "none"
    truncated_fence: bool = False


@dataclass(frozen=True)
class _Block:
    tag: str
    code: str
    terminated: bool


def _blocks(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        ticks, tag = len(m.group(1)), m.group(2).lower()
        body: list[str] = []
        terminated = False
        i += 1
        while i < len(lines):
            close = _CLOSE.match(lines[i])
            if close and len(close.group(1)) >= ticks:
                terminated = True
                i += 1
                break
            body.append(lines[i])
            i += 1
        blocks.append(_Block(tag, "\n".join(body), terminated))
    return blocks


def extract_code(completion: str) -> ExtractResult:
    """Return the code the grader/detector/judge should look at (see module docstring)."""
    blocks = [b for b in _blocks(completion) if b.code.strip()]
    for wanted, how in ((lambda t: t in _PYTHON_TAGS, "python_fence"), (lambda t: t == "", "generic_fence")):
        chosen = [b for b in blocks if wanted(b.tag)]
        if chosen:
            last = chosen[-1]
            return ExtractResult(last.code, how, truncated_fence=not last.terminated)
    return ExtractResult(None, "none", False)
