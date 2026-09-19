"""Statistics helpers. Subtask 06 only adds ``wilson``; subtask 13 owns and extends this module."""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    Solves ``(p_hat - p)^2 = z^2 p (1 - p) / n`` for ``p``; the interval is always inside [0, 1] and
    non-degenerate at ``k = 0`` and ``k = n``. ``n == 0`` returns the uninformative ``(0.0, 1.0)``.
    """
    if isinstance(k, bool) or isinstance(n, bool) or int(k) != k or int(n) != n:
        raise ValueError(f"wilson needs integer counts, got k={k!r}, n={n!r}")
    k, n = int(k), int(n)
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"wilson needs 0 <= k <= n, got k={k}, n={n}")
    if z < 0 or not math.isfinite(z):
        raise ValueError(f"wilson needs a finite z >= 0, got {z!r}")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)  # exact endpoints; avoids 1e-17 rounding residue
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi
