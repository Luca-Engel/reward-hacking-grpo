"""In-loop monitors (DESIGN §2.4, arm 7).

``ast_narrow_penalty`` returns the *narrow* AST flag (module-level ``def run_tests`` only);
the grader turns it into the ``-monitor_penalty`` reward term. It is deliberately leaky so that
displacement (H4) is possible -- see ``docs/detector_notes.md``.
"""

from __future__ import annotations

from collections.abc import Callable

from rhg.detect.ast_detector import analyze


def _ast_narrow(code: str | None) -> bool:
    return analyze(code, "narrow").flag


MONITORS: dict[str, Callable[[str | None], bool]] = {"ast_narrow_penalty": _ast_narrow}


def make_monitor(name: str | None) -> Callable[[str | None], bool] | None:
    """Monitor callable for ``cfg.arm.monitor``; ``None`` means no monitor."""
    if name is None:
        return None
    try:
        return MONITORS[name]
    except KeyError:
        raise ValueError(f"unknown monitor {name!r}; known: {sorted(MONITORS)}") from None
