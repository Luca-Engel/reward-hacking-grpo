"""Independent reference implementations for the statistics tests.

Deliberately naive: exact rational arithmetic (``fractions.Fraction``), plain Python loops, full enumeration.
They share no code with ``rhg.analysis.stats`` (no numpy, no scipy, no cached index matrices).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product


def _frac(xs):
    return [Fraction(str(x)) if not isinstance(x, (int, Fraction)) else Fraction(x) for x in xs]


def ref_perm_p(a, b, alternative="greater") -> Fraction:
    """P over all size-|a| subsets of the pooled sample of (mean_sub - mean_rest) >= / <= / |.|>= observed."""
    a, b = _frac(a), _frac(b)
    pooled = a + b
    n, k = len(pooled), len(a)
    total = sum(pooled)

    def diff(idx):
        s = sum(pooled[i] for i in idx)
        return s / k - (total - s) / (n - k)

    obs = sum(a) / k - sum(b) / len(b)
    hits = cnt = 0
    for idx in combinations(range(n), k):
        d = diff(idx)
        cnt += 1
        if alternative == "greater":
            hits += d >= obs
        elif alternative == "less":
            hits += d <= obs
        else:
            hits += abs(d) >= abs(obs)
    return Fraction(hits, cnt)


def ref_jt_stat(groups) -> Fraction:
    j = Fraction(0)
    for i in range(len(groups)):
        for jj in range(i + 1, len(groups)):
            for x in groups[i]:
                for y in groups[jj]:
                    j += 1 if y > x else (Fraction(1, 2) if y == x else 0)
    return j


def _distinct_label_sequences(sizes):
    """All distinct sequences containing label g exactly sizes[g] times (multiset permutations), by brute force."""
    n = sum(sizes)
    seen = set()
    for labels in product(range(len(sizes)), repeat=n):
        if all(labels.count(g) == s for g, s in enumerate(sizes)) and labels not in seen:
            seen.add(labels)
    return sorted(seen)


def ref_jt(groups, alternative="increasing"):
    """(J_obs, p) by enumerating every distinct assignment of the pooled values to the ordered groups."""
    groups = [_frac(g) for g in groups if len(g)]
    sizes = [len(g) for g in groups]
    pooled = [x for g in groups for x in g]
    obs = ref_jt_stat(groups)
    hi = lo = cnt = 0
    for labels in _distinct_label_sequences(sizes):
        regroup = [[pooled[i] for i, lab in enumerate(labels) if lab == g] for g in range(len(sizes))]
        j = ref_jt_stat(regroup)
        cnt += 1
        hi += j >= obs
        lo += j <= obs
    p_hi, p_lo = Fraction(hi, cnt), Fraction(lo, cnt)
    p = {"increasing": p_hi, "decreasing": p_lo, "two-sided": min(Fraction(1), 2 * min(p_hi, p_lo))}[alternative]
    return obs, p


def ref_midranks(values):
    """Mid-ranks (average rank of tied values), ascending."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [Fraction(0)] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = Fraction(i + 1 + j + 1, 2)
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def ref_wilcoxon(d, alternative="less"):
    """(W+_obs, p, n_nonzero) by enumerating all 2**n sign patterns over the mid-ranks of |d| (zeros dropped)."""
    d = [x for x in _frac(d) if x != 0]
    n = len(d)
    if n == 0:
        return Fraction(0), Fraction(1), 0
    ranks = ref_midranks([abs(x) for x in d])
    obs = sum(r for r, x in zip(ranks, d) if x > 0)
    le = ge = 0
    for signs in product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        le += w <= obs
        ge += w >= obs
    p_le, p_ge = Fraction(le, 2**n), Fraction(ge, 2**n)
    p = {"less": p_le, "greater": p_ge, "two-sided": min(Fraction(1), 2 * min(p_le, p_ge))}[alternative]
    return obs, p, n


def ref_holm(pvals, alpha):
    """Sequential Holm: returns (reject flags in input order, adjusted p in input order)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (pvals[i], i))
    reject = [False] * m
    for pos, i in enumerate(order):
        if pvals[i] <= alpha / (m - pos):
            reject[i] = True
        else:
            break
    adj = [0.0] * m
    run = 0.0
    for pos, i in enumerate(order):
        run = max(run, (m - pos) * pvals[i])
        adj[i] = min(1.0, run)
    return reject, adj
