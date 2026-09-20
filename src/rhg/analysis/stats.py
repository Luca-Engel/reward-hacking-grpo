"""Exact tests and helpers for the pre-registered analysis (DESIGN §6, PREREG §2-§5).

Everything that decides a hypothesis is *exact* (full enumeration of the null distribution) or honestly
labelled: a Monte-Carlo fallback for very large enumerations sets ``exact=False``; bootstrap intervals are
descriptive and carry a low-coverage flag at n <= 5 seeds.

* ``perm_test``           two-sample permutation test on the difference of means (unit = seed);
* ``jonckheere_terpstra`` ordered-alternatives trend test (unequal group sizes, ties, censored values);
* ``wilcoxon_signed_rank`` exact signed-rank test with tie-aware (mid-rank) statistic;
* ``spearman``            rank correlation with mid-ranks;
* ``wilson``              binomial interval;
* ``bootstrap_ci``        percentile bootstrap over seeds (descriptive only);
* ``holm``                Holm-Bonferroni with adjusted p and per-test thresholds;
* ``min_attainable_p``    the smallest p a design can ever produce;
* ``loo``                 leave-one-unit-out helper;
* ``primary_decision`` / ``h4b_decision``   the PREREG §2 and §5 decision rules.

Censored onsets are passed as the value ``T + 1`` (PREREG §3); the tests treat equal values as ties, i.e.
"censored" is a tied top rank, which is the pre-declared handling.

All null distributions are over the relabelings of *units* (seeds), so tied values never change the
number of equally likely relabelings, only how many of them reach the observed statistic.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from rhg import prereg_constants as C

EXACT_LIMIT = 2_000_000  # above this many relabelings the permutation tests fall back to Monte Carlo
MC_DEFAULT_DRAWS = 200_000
_CACHE_LIMIT = 400_000  # enumerations up to this size are cached
_CHUNK = 250_000
_REL_TOL = 1e-9  # relative tolerance when comparing floating-point statistics (ties in the statistic)
_TIE_TOL = 1e-12  # relative tolerance for "equal values" inside JT / Wilcoxon

LOW_COVERAGE_NOTE = (
    "percentile bootstrap over seeds: with n <= 5 seeds there are only C(2n-1, n) distinct resamples "
    "(126 at n=5), so nominal coverage is not attained; descriptive only, show per-seed dots beside it"
)


# ------------------------------------------------------------------ small helpers
def leq(p: float, alpha: float) -> bool:
    """``p <= alpha`` with a guard against 1/20 vs 0.05-style representation noise."""
    return p <= alpha + 1e-12


def _as_float_array(x: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _midranks(x: np.ndarray) -> np.ndarray:
    """Ascending mid-ranks (tied values get the average of their ranks)."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    return (upper - (counts - 1) / 2.0)[inverse.ravel()]


def _check_alt(alternative: str, allowed: tuple[str, ...]) -> None:
    if alternative not in allowed:
        raise ValueError(f"alternative must be one of {allowed}, got {alternative!r}")


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


# ------------------------------------------------------------------ two-sample permutation test
@dataclass(frozen=True)
class PermResult:
    p: float
    observed: float  # mean(a) - mean(b)
    exact: bool  # False = seeded Monte Carlo (p = (1 + hits) / (1 + draws))
    n_relabelings: int  # C(n, k) if exact, else the number of random draws
    alternative: str
    n_a: int
    n_b: int
    min_attainable_p: float


@lru_cache(maxsize=64)
def _combo_index_cached(n: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n), k)), dtype=np.int16).reshape(-1, k)


def _iter_combo_chunks(n: int, k: int):
    """All k-subsets of range(n) as int arrays of shape (chunk, k)."""
    total = math.comb(n, k)
    if total <= _CACHE_LIMIT:
        yield _combo_index_cached(n, k)
        return
    it = itertools.combinations(range(n), k)
    while True:
        block = list(itertools.islice(it, _CHUNK))
        if not block:
            return
        yield np.array(block, dtype=np.int16)


def _diff_from_sums(sum_a: np.ndarray, total: float, k: int, m: int) -> np.ndarray:
    return sum_a / k - (total - sum_a) / m


def _tail_count(diffs: np.ndarray, obs: float, alternative: str, tol: float) -> int:
    if alternative == "greater":
        return int(np.count_nonzero(diffs >= obs - tol))
    if alternative == "less":
        return int(np.count_nonzero(diffs <= obs + tol))
    return int(np.count_nonzero(np.abs(diffs) >= abs(obs) - tol))


def perm_test(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "greater",
    *,
    exact_limit: int = EXACT_LIMIT,
    n_draws: int = MC_DEFAULT_DRAWS,
    seed: int = 0,
) -> PermResult:
    """Permutation test of ``mean(a) - mean(b)`` over all C(n_a + n_b, n_a) relabelings of the pooled units.

    ``alternative``: ``greater`` (mean a > mean b), ``less`` or ``two-sided`` (|difference|). Full enumeration when
    C(n, k) <= ``exact_limit``; otherwise ``n_draws`` seeded random relabelings with the Phipson-Smyth p-value
    ``(1 + hits) / (1 + n_draws)`` and ``exact=False``.
    """
    _check_alt(alternative, ("greater", "less", "two-sided"))
    xa, xb = _as_float_array(a, "a"), _as_float_array(b, "b")
    k, m = xa.size, xb.size
    if k == 0 or m == 0:
        raise ValueError("perm_test needs at least one unit in each group")
    pooled = np.concatenate([xa, xb])
    n, total = k + m, float(pooled.sum())
    obs = float(xa.mean() - xb.mean())
    tol = _REL_TOL * (float(np.abs(pooled).max()) or 1.0)
    n_rel = math.comb(n, k)
    if n_rel <= exact_limit:
        hits = 0
        for idx in _iter_combo_chunks(n, k):
            diffs = _diff_from_sums(pooled[idx].sum(axis=1), total, k, m)
            hits += _tail_count(diffs, obs, alternative, tol)
        p = hits / n_rel
        return PermResult(p, obs, True, n_rel, alternative, k, m, _min_p_perm(k, m, alternative))
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_draws:
        cur = min(20_000, n_draws - done)
        idx = np.argsort(rng.random((cur, n)), axis=1)[:, :k]
        diffs = _diff_from_sums(pooled[idx].sum(axis=1), total, k, m)
        hits += _tail_count(diffs, obs, alternative, tol)
        done += cur
    return PermResult((1 + hits) / (1 + n_draws), obs, False, n_draws, alternative, k, m, 1 / (1 + n_draws))


# ------------------------------------------------------------------ Jonckheere-Terpstra
@dataclass(frozen=True)
class JTResult:
    p: float
    statistic: float  # J = sum over ordered level pairs of Mann-Whitney counts (ties count 1/2)
    n_relabelings: int
    exact: bool
    alternative: str
    sizes: tuple[int, ...]
    min_attainable_p: float


def _multinomial(sizes: Sequence[int]) -> int:
    out, left = 1, sum(sizes)
    for s in sizes:
        out *= math.comb(left, s)
        left -= s
    return out


def _iter_label_rows(sizes: tuple[int, ...]):
    """Every distinct assignment of ordered group labels to n units (multiset permutations), one row at a time."""
    n = sum(sizes)
    row = [0] * n

    def rec(remaining: tuple[int, ...], g: int):
        if g == len(sizes) - 1:
            for pos in remaining:
                row[pos] = g
            yield tuple(row)
            return
        for comb in itertools.combinations(remaining, sizes[g]):
            for pos in comb:
                row[pos] = g
            chosen = set(comb)
            yield from rec(tuple(p for p in remaining if p not in chosen), g + 1)

    yield from rec(tuple(range(n)), 0)


@lru_cache(maxsize=32)
def _label_assignments_cached(sizes: tuple[int, ...]) -> np.ndarray:
    return np.array(list(_iter_label_rows(sizes)), dtype=np.int8)


def _iter_label_chunks(sizes: tuple[int, ...]):
    if _multinomial(sizes) <= _CACHE_LIMIT:
        yield _label_assignments_cached(sizes)
        return
    it = _iter_label_rows(sizes)
    while True:
        block = list(itertools.islice(it, _CHUNK))
        if not block:
            return
        yield np.array(block, dtype=np.int8)


def _jt_scores(x: np.ndarray) -> np.ndarray:
    """2 * (1[x_v > x_u] + 1/2 1[x_v == x_u]) as integers, entry [u, v]."""
    scale = float(np.abs(x).max()) or 1.0
    diff = x[None, :] - x[:, None]
    tol = _TIE_TOL * scale
    return 2 * (diff > tol).astype(np.int64) + (np.abs(diff) <= tol).astype(np.int64)


def _jt_stat2(labels: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """2*J for each row of ``labels`` (M, n)."""
    out = np.empty(labels.shape[0], dtype=np.int64)
    step = max(1, 2_000_000 // max(1, labels.shape[1] ** 2))
    for i in range(0, labels.shape[0], step):
        lab = labels[i : i + step]
        lt = lab[:, :, None] < lab[:, None, :]
        out[i : i + step] = np.tensordot(lt, s2, axes=([1, 2], [0, 1]))
    return out


def jonckheere_terpstra(
    groups: Sequence[Sequence[float]],
    alternative: str = "increasing",
    *,
    exact_limit: int = EXACT_LIMIT,
    n_draws: int = MC_DEFAULT_DRAWS,
    seed: int = 0,
) -> JTResult:
    """Exact Jonckheere-Terpstra test for an ordered trend across ``groups`` (lowest level first).

    ``J = sum_{i<j} #{(x in group i, y in group j): y > x} + 1/2 #{y == x}``. The null distribution is
    obtained by enumerating every assignment of the pooled units to the ordered groups with the observed
    group sizes (multiset permutations of the group labels), so unequal sizes, ties and censored values
    are handled exactly. ``increasing``: p = P(J >= J_obs); ``decreasing``: P(J <= J_obs); ``two-sided``:
    min(1, 2 min(both tails)). Empty groups are ignored; at least two non-empty groups are required.
    """
    _check_alt(alternative, ("increasing", "decreasing", "two-sided"))
    arrs = [_as_float_array(g, f"group {i}") for i, g in enumerate(groups)]
    arrs = [g for g in arrs if g.size]
    if len(arrs) < 2:
        raise ValueError("jonckheere_terpstra needs at least two non-empty groups")
    sizes = tuple(int(g.size) for g in arrs)
    x = np.concatenate(arrs)
    n = x.size
    s2 = _jt_scores(x)
    obs_labels = np.repeat(np.arange(len(sizes), dtype=np.int8), sizes)[None, :]
    obs2 = int(_jt_stat2(obs_labels, s2)[0])
    n_rel = _multinomial(sizes)
    if n_rel <= exact_limit:
        hi = lo = 0
        for lab in _iter_label_chunks(sizes):
            st = _jt_stat2(lab, s2)
            hi += int(np.count_nonzero(st >= obs2))
            lo += int(np.count_nonzero(st <= obs2))
        exact, denom = True, n_rel
    else:
        rng = np.random.default_rng(seed)
        base = np.repeat(np.arange(len(sizes), dtype=np.int8), sizes)
        stats_list = []
        done = 0
        while done < n_draws:
            cur = min(5_000, n_draws - done)
            lab = base[np.argsort(rng.random((cur, n)), axis=1)]
            stats_list.append(_jt_stat2(lab, s2))
            done += cur
        stats2 = np.concatenate(stats_list)
        hi = 1 + int(np.count_nonzero(stats2 >= obs2))
        lo = 1 + int(np.count_nonzero(stats2 <= obs2))
        exact, denom = False, n_draws + 1
    p_hi, p_lo = hi / denom, lo / denom
    p = {"increasing": p_hi, "decreasing": p_lo, "two-sided": min(1.0, 2 * min(p_hi, p_lo))}[alternative]
    min_p = (1 / n_rel if exact else 1 / denom) * (2 if alternative == "two-sided" else 1)
    return JTResult(p, obs2 / 2.0, denom if exact else n_draws, exact, alternative, sizes, min(1.0, min_p))


# ------------------------------------------------------------------ Wilcoxon signed-rank
@dataclass(frozen=True)
class WilcoxonResult:
    p: float
    statistic: float  # W+ = sum of mid-ranks of |d| over positive d
    n: int  # non-zero differences used
    n_zero: int  # zero differences dropped (Wilcoxon's convention)
    alternative: str
    min_attainable_p: float


def _signed_rank_null(doubled_ranks: np.ndarray) -> np.ndarray:
    """Exact null distribution of 2*W+ under independent fair signs: P[t] = P(2 W+ = t), by dynamic programming
    over the (integer) doubled mid-ranks. Ties in |d| are therefore handled on the tied statistic itself."""
    total = int(doubled_ranks.sum())
    dist = np.zeros(total + 1)
    dist[0] = 1.0
    for r in doubled_ranks:
        r = int(r)
        shifted = np.zeros_like(dist)
        shifted[r:] = dist[: total + 1 - r]
        dist = 0.5 * (dist + shifted)
    return dist


def wilcoxon_signed_rank(d: Sequence[float], alternative: str = "less") -> WilcoxonResult:
    """Exact Wilcoxon signed-rank test of symmetry about zero.

    Zeros are dropped; |d| gets mid-ranks; the null distribution of ``W+`` is exact under independent
    fair signs (ties handled by using the mid-rank statistic itself). ``less``: p = P(W+ <= W+_obs) (the
    hypothesis d < 0, e.g. rho < 0 in H2); ``greater``: P(W+ >= W+_obs); ``two-sided``: min(1, 2 min tail).
    With no non-zero difference the p-value is 1.
    """
    _check_alt(alternative, ("less", "greater", "two-sided"))
    arr = _as_float_array(d, "d")
    scale = float(np.abs(arr).max()) if arr.size else 0.0
    zero = np.abs(arr) <= _TIE_TOL * (scale or 1.0)
    nz = arr[~zero]
    n = int(nz.size)
    if n == 0:
        return WilcoxonResult(1.0, 0.0, 0, int(arr.size), alternative, 1.0)
    mag = np.round(np.abs(nz) / (scale or 1.0), 12)
    ranks2 = np.rint(2 * _midranks(mag)).astype(np.int64)
    obs2 = int(ranks2[nz > 0].sum())
    dist = _signed_rank_null(ranks2)
    cdf = np.cumsum(dist)
    p_le = float(cdf[obs2])
    p_ge = float(1.0 - (cdf[obs2 - 1] if obs2 > 0 else 0.0))
    # the two tails are built from the same exact distribution; round off tiny DP noise so p is stable
    p_le, p_ge = _clean_prob(p_le, n), _clean_prob(p_ge, n)
    p = {"less": p_le, "greater": p_ge, "two-sided": min(1.0, 2 * min(p_le, p_ge))}[alternative]
    min_p = 2.0**-n * (2 if alternative == "two-sided" else 1)
    return WilcoxonResult(p, obs2 / 2.0, n, int(zero.sum()), alternative, min(1.0, min_p))


def _clean_prob(p: float, n: int) -> float:
    """Probabilities here are integers over 2**n; snap DP round-off (only relevant for n <= 50)."""
    if n <= 50:
        return min(1.0, max(0.0, round(p * 2.0**n) / 2.0**n))
    return min(1.0, max(0.0, p))


# ------------------------------------------------------------------ Spearman
def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson correlation of the mid-ranks (ties averaged); ``nan`` if either
    variable is constant or there are fewer than 2 pairs."""
    xa, ya = _as_float_array(x, "x"), _as_float_array(y, "y")
    if xa.size != ya.size:
        raise ValueError("spearman needs x and y of equal length")
    if xa.size < 2:
        return float("nan")
    rx, ry = _midranks(xa), _midranks(ya)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if denom == 0.0:
        return float("nan")
    return max(-1.0, min(1.0, float(rx @ ry) / denom))


# ------------------------------------------------------------------ bootstrap (descriptive)
@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lo: float
    hi: float
    level: float
    n_boot: int
    n_units: tuple[int, ...]
    n_distinct_resamples: int
    low_coverage: bool
    note: str


def _distinct_resamples(n: int) -> int:
    return math.comb(2 * n - 1, n)


def bootstrap_ci(
    *groups: Sequence[float],
    statistic: Callable[..., float] | None = None,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap over units (seeds). One group: ``statistic(sample)`` (default mean); two groups:
    ``statistic(sample_a, sample_b)`` (default difference of means). Each group is resampled independently.
    DESCRIPTIVE ONLY: flagged ``low_coverage`` when any group has n <= 5 (see ``LOW_COVERAGE_NOTE``)."""
    if not 1 <= len(groups) <= 2:
        raise ValueError("bootstrap_ci takes one or two groups")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    arrs = [_as_float_array(g, f"group {i}") for i, g in enumerate(groups)]
    if any(a.size == 0 for a in arrs):
        raise ValueError("bootstrap_ci needs non-empty groups")
    if statistic is None:
        statistic = (lambda s: float(np.mean(s))) if len(arrs) == 1 else (lambda s, t: float(np.mean(s) - np.mean(t)))
    est = float(statistic(*arrs))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = statistic(*[a[rng.integers(0, a.size, a.size)] for a in arrs])
    lo, hi = np.percentile(boots, [50 * (1 - level), 100 - 50 * (1 - level)])
    sizes = tuple(int(a.size) for a in arrs)
    distinct = math.prod(_distinct_resamples(s) for s in sizes)
    low = min(sizes) <= 5
    return BootstrapCI(est, float(lo), float(hi), level, n_boot, sizes, distinct, low, LOW_COVERAGE_NOTE if low else "")


# ------------------------------------------------------------------ Holm-Bonferroni
@dataclass(frozen=True)
class HolmRow:
    name: str
    p: float | None  # as supplied; None / nan = the test could not be run
    p_used: float  # p, or 1.0 for an untestable test (it still counts toward m)
    rank: int  # 1 = smallest p
    threshold: float  # alpha / (m - rank + 1): the level this test is compared with in the step-down
    p_adj: float
    reject: bool


def holm(pvals: Mapping[str, float | None] | Sequence[float | None], alpha: float | None = None) -> list[HolmRow]:
    """Holm-Bonferroni step-down. Returns one row per test in *input* order.

    Sorted ascending (stable), test i (1-based) is compared with ``alpha / (m - i + 1)`` and the procedure stops
    at the first non-rejection. ``p_adj_i = min(1, max_{j<=i} (m - j + 1) p_(j))``; ``reject`` <=> ``p_adj <= alpha``.
    A test that could not be run (``None``/nan) gets p = 1 but keeps counting toward m (the family is fixed).
    ``alpha`` defaults to ``prereg_constants.ALPHA``.
    """
    alpha = C.ALPHA if alpha is None else alpha
    items = list(pvals.items()) if isinstance(pvals, Mapping) else [(f"test{i + 1}", p) for i, p in enumerate(pvals)]
    m = len(items)
    used = [1.0 if (p is None or (isinstance(p, float) and math.isnan(p))) else float(p) for _, p in items]
    if any(not 0.0 <= p <= 1.0 for p in used):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(m), key=lambda i: (used[i], i))
    rows: dict[int, HolmRow] = {}
    running, stopped = 0.0, False
    for pos, i in enumerate(order):
        mult = m - pos
        running = max(running, mult * used[i])
        thr = alpha / mult
        ok = (not stopped) and leq(used[i], thr)
        stopped = stopped or not ok
        rows[i] = HolmRow(items[i][0], items[i][1], used[i], pos + 1, thr, min(1.0, running), ok)
    return [rows[i] for i in range(m)]


# ------------------------------------------------------------------ minimum attainable p
@dataclass(frozen=True)
class Design:
    """A planned test. ``kind``: ``perm`` (sizes = (n_a, n_b)), ``jt`` (sizes = ordered level sizes),
    ``signed_rank`` (sizes = (n,) = number of usable paired differences, i.e. the emerged seeds for H2)."""

    kind: str
    sizes: tuple[int, ...]
    sided: str = "one"


def _min_p_perm(k: int, m: int, alternative: str) -> float:
    base = 1.0 / math.comb(k + m, k)
    if alternative == "two-sided" and k == m:  # the complement relabeling has the same |difference|
        base *= 2
    return min(1.0, base)


def min_attainable_p(design: Design | Mapping[str, Any]) -> float:
    """Smallest p-value the exact test can produce for the given design (perfectly separated / all-same-sign
    data). One-sided: ``1 / C(n_a + n_b, n_a)`` (perm), ``1 / (n! / prod n_i!)`` (JT), ``2**-n`` (signed-rank).
    Two-sided doubles (perm: only when n_a == n_b, since only then the complementary relabeling ties)."""
    if isinstance(design, Mapping):
        design = Design(str(design["kind"]), tuple(int(s) for s in design["sizes"]), str(design.get("sided", "one")))
    if design.sided not in ("one", "two"):
        raise ValueError(f"sided must be 'one' or 'two', got {design.sided!r}")
    two = design.sided == "two"
    s = tuple(int(v) for v in design.sizes)
    if any(v < 0 for v in s):
        raise ValueError("sizes must be non-negative")
    if design.kind == "perm":
        if len(s) != 2 or min(s) < 1:
            raise ValueError("perm design needs sizes (n_a, n_b) with both >= 1")
        return _min_p_perm(s[0], s[1], "two-sided" if two else "greater")
    if design.kind == "jt":
        s = tuple(v for v in s if v > 0)
        if len(s) < 2:
            raise ValueError("jt design needs at least two non-empty levels")
        return min(1.0, _multinomial(s) ** -1 * (2 if two else 1))
    if design.kind == "signed_rank":
        if len(s) != 1:
            raise ValueError("signed_rank design needs sizes (n,)")
        n = s[0]
        if n < 1:
            return 1.0
        return min(1.0, 2.0**-n * (2 if two else 1))
    raise ValueError(f"unknown design kind {design.kind!r}")


# ------------------------------------------------------------------ leave-one-out
def loo(fn: Callable[[Any], Any], data: Any) -> list[tuple[Any, Any]]:
    """Leave-one-unit-out: ``[(unit, fn(data without that unit)), ...]``.

    ``data`` is a sequence (unit = index), a mapping of group -> sequence (unit = (group, index)), or a mapping of
    group -> mapping of unit -> value (unit = (group, key)). ``fn`` receives data of the same shape."""
    out: list[tuple[Any, Any]] = []
    if isinstance(data, Mapping):
        for g, members in data.items():
            keys = list(members.keys()) if isinstance(members, Mapping) else list(range(len(members)))
            for key in keys:
                if isinstance(members, Mapping):
                    rest: Any = {k: v for k, v in members.items() if k != key}
                else:
                    rest = [v for i, v in enumerate(members) if i != key]
                out.append(((g, key), fn({**data, g: rest})))
        return out
    seq = list(data)
    for i in range(len(seq)):
        out.append((i, fn(seq[:i] + seq[i + 1 :])))
    return out


# ------------------------------------------------------------------ decision rules
@dataclass(frozen=True)
class PrimaryDecision:
    outcome: str  # "supported" | "inconclusive" | "no_discovery"  (PREREG §2, exactly these labels)
    p: float
    delta: float
    n_emerged: int
    n_hackable: int
    test: PermResult


def primary_decision(hack_rates: Sequence[float], clean_rates: Sequence[float]) -> PrimaryDecision:
    """PREREG §2: supported iff exact one-sided p <= ALPHA and mean difference >= DELTA_MIN; otherwise
    ``inconclusive`` if >= 1 hackable seed has final rate >= EMERGED_THRESHOLD, else ``no_discovery``."""
    res = perm_test(hack_rates, clean_rates, "greater")
    h = _as_float_array(hack_rates, "hack_rates")
    emerged = int(np.count_nonzero(h >= C.EMERGED_THRESHOLD - 1e-12))
    supported = leq(res.p, C.ALPHA) and res.observed >= C.DELTA_MIN - 1e-12
    outcome = "supported" if supported else ("inconclusive" if emerged >= 1 else "no_discovery")
    return PrimaryDecision(outcome, res.p, res.observed, emerged, int(h.size), res)


def h4b_decision(hack_rates: Sequence[float], evasion: Sequence[float | None]) -> str:
    """PREREG §5. ``evasion[i]`` = P(not ast_narrow | HACK_RT) of seed i (``None``/nan if it has no hack).

    ``displacement``: >= H4B_MIN_SEEDS seeds with HACK_RT >= H4B_DISPLACEMENT_HACK_MIN and evasion >=
    H4B_EVASION_MIN; ``suppression_only``: every seed HACK_RT < H4B_SUPPRESSION_MAX; else ``mixed``."""
    if len(hack_rates) != len(evasion):
        raise ValueError("hack_rates and evasion must have the same length")
    eps = 1e-12
    n_disp = 0
    for h, e in zip(hack_rates, evasion):
        ev_ok = e is not None and not (isinstance(e, float) and math.isnan(e)) and e >= C.H4B_EVASION_MIN - eps
        if h >= C.H4B_DISPLACEMENT_HACK_MIN - eps and ev_ok:
            n_disp += 1
    if n_disp >= C.H4B_MIN_SEEDS:
        return "displacement"
    if len(hack_rates) > 0 and all(h < C.H4B_SUPPRESSION_MAX - eps for h in hack_rates):
        return "suppression_only"
    return "mixed"
