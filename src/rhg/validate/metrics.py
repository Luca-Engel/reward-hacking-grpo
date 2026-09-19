"""Agreement / accuracy metrics for detector, judge and human labels (DESIGN §4, §8 items 5-8).

Everything works on a 2x2 ``Confusion`` (positive class = "hack"); cells may be real numbers, which
is how the inverse-probability-weighted (IPW) variants reuse the same formulas.

Contents
* ``Confusion`` / ``confusion_counts``; precision, recall, accuracy, specificity with **Wilson**
  intervals; F1 and Cohen's kappa with **percentile-bootstrap** CIs (seeded, >= 2000 resamples).
* Cohen's kappa (2x2 and multi-class), **PABAK** (``2 p_o - 1``), positive agreement (``2 tp / (2 tp +
  fp + fn)``, identical to F1) and negative agreement (``2 tn / (2 tn + fp + fn)``): the extreme-
  prevalence companions of kappa (DESIGN §8.6).
* ``mcnemar_exact``: exact two-sided McNemar test (binomial, p = 1/2 on the discordant pairs).
* IPW: items carry ``inclusion_prob`` (the judge's Horvitz-Thompson design, ``docs/judge_notes.md``).
  Cell totals are HT totals ``sum(w_i * 1[cell])`` with ``w_i = 1 / pi_i``; rates are ratios of
  weighted totals. See ``ipw_summary`` for the variance treatment.

IPW variance treatment (documented approximation). The judged set is two independent fixed-size
simple random samples. Its exact variance is messy, so two things are reported, neither claimed exact:
1. ``se_design``: linearised (Taylor) standard error of a ratio of HT totals under **Poisson
   sampling** (independent Bernoulli(pi_i) inclusion), ``sum_i (1 - pi_i) / pi_i^2 * u_i^2 / D^2``
   with ``u_i = a_i - R d_i``. Only sampling-design noise on a fixed set of rollouts; zero for a census.
2. The reported interval is a **Wilson interval at the Kish effective sample size**
   ``n_eff = (sum w)^2 / sum w^2`` of the denominator items (treats items as draws from the process,
   with the weights' design effect). It never leaves [0, 1] and stays sensible at 0 or 1. For F1,
   kappa and the other non-linear statistics a **weighted percentile bootstrap** (items resampled with
   replacement, weights carried along, the same Monte-Carlo scheme as the unweighted case) is used.
   Both are approximations; the bootstrap ignores the without-replacement design and clustering by
   problem/run.

Wilson/bootstrap intervals here (like everywhere in this project) treat rollouts as independent; they
are per-item precision statements, not seed-level inference.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

MIN_BOOT = 2000
Z95 = 1.96  # same constant as rhg.analysis.stats.wilson, so every interval in the project is comparable


# ------------------------------------------------------------------ confusion


@dataclass(frozen=True)
class Confusion:
    """Cells of a 2x2 table; rows = truth, columns = prediction (positive = hack). Floats allowed."""

    tp: float
    fp: float
    fn: float
    tn: float

    @property
    def n(self) -> float:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def n_pos(self) -> float:
        return self.tp + self.fn

    @property
    def n_neg(self) -> float:
        return self.fp + self.tn

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


def _as_bool_array(x: Iterable, name: str) -> np.ndarray:
    a = np.asarray(list(x))
    if a.dtype != bool:
        if not np.isin(a, (0, 1)).all():
            raise ValueError(f"{name} must be boolean (or 0/1)")
        a = a.astype(bool)
    return a


def confusion_counts(y_true: Iterable, y_pred: Iterable) -> Confusion:
    t, p = _as_bool_array(y_true, "y_true"), _as_bool_array(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: {t.shape} vs {p.shape}")
    return Confusion(int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum()))


def _div(a: float, b: float) -> float:
    return a / b if b > 0 else math.nan


def precision(c: Confusion) -> float:
    return _div(c.tp, c.tp + c.fp)


def recall(c: Confusion) -> float:
    return _div(c.tp, c.tp + c.fn)


def specificity(c: Confusion) -> float:
    return _div(c.tn, c.tn + c.fp)


def false_positive_rate(c: Confusion) -> float:
    return _div(c.fp, c.fp + c.tn)


def accuracy(c: Confusion) -> float:
    return _div(c.tp + c.tn, c.n)


def f1(c: Confusion) -> float:
    return _div(2 * c.tp, 2 * c.tp + c.fp + c.fn)


def positive_agreement(c: Confusion) -> float:
    """Cicchetti-Feinstein positive agreement ``2 tp / (2 tp + fp + fn)`` (algebraically equal to F1)."""
    return f1(c)


def negative_agreement(c: Confusion) -> float:
    return _div(2 * c.tn, 2 * c.tn + c.fp + c.fn)


def observed_agreement(c: Confusion) -> float:
    return accuracy(c)


def cohen_kappa(c: Confusion) -> float:
    """Cohen's kappa of a 2x2 table; NaN when chance agreement is 1 (a constant rater) or n == 0."""
    n = c.n
    if n <= 0:
        return math.nan
    po = (c.tp + c.tn) / n
    pe = ((c.tp + c.fp) * (c.tp + c.fn) + (c.fn + c.tn) * (c.fp + c.tn)) / (n * n)
    if 1.0 - pe <= 1e-12:
        return math.nan
    return (po - pe) / (1.0 - pe)


def pabak(c: Confusion) -> float:
    """Prevalence-adjusted bias-adjusted kappa = ``2 p_o - 1`` (Byrt et al. 1993)."""
    return 2.0 * observed_agreement(c) - 1.0 if c.n > 0 else math.nan


def cohen_kappa_labels(a: Sequence, b: Sequence) -> float:
    """Multi-class Cohen's kappa from two label sequences (marginals from the data)."""
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    if n == 0:
        return math.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    return math.nan if 1.0 - pe <= 1e-12 else (po - pe) / (1.0 - pe)


# ------------------------------------------------------------------ intervals


def wilson_interval(p: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion ``p`` observed on ``n`` (possibly fractional, e.g.
    effective) trials. Equals ``rhg.analysis.stats.wilson`` for integer counts."""
    if n <= 0 or math.isnan(p):
        return (0.0, 1.0)
    p = min(max(p, 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = 0.0 if p == 0.0 else max(0.0, centre - half)
    hi = 1.0 if p == 1.0 else min(1.0, centre + half)
    return lo, hi


def proportion(k: int, n: int, z: float = Z95) -> dict[str, float]:
    """``{est, lo, hi, k, n}`` with a Wilson interval; est is NaN (interval (0, 1)) when n == 0."""
    from rhg.analysis.stats import wilson

    lo, hi = wilson(int(k), int(n), z)
    return {"est": (k / n if n else math.nan), "lo": lo, "hi": hi, "k": int(k), "n": int(n)}


# ------------------------------------------------------------------ vectorised cell statistics + bootstrap

STAT_NAMES = ("precision", "recall", "specificity", "f1", "accuracy", "kappa", "pabak", "positive_agreement",
              "negative_agreement")


def cell_stats(tp, fp, fn, tn) -> dict[str, np.ndarray]:
    """All statistics for arrays of cell values (NaN where undefined)."""
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        kappa = np.where(1.0 - pe <= 1e-12, np.nan, (po - pe) / (1.0 - pe))
        out = {
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "accuracy": po,
            "kappa": kappa,
            "pabak": 2 * po - 1,
            "positive_agreement": 2 * tp / (2 * tp + fp + fn),
            "negative_agreement": 2 * tn / (2 * tn + fp + fn),
        }
    for k, v in out.items():
        out[k] = np.where(np.isfinite(v), v, np.nan)
    return out


def _check_boot(n_boot: int) -> None:
    if n_boot < MIN_BOOT:
        raise ValueError(f"n_boot must be >= {MIN_BOOT} (got {n_boot}); percentile intervals need many resamples")


def _percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float, int]:
    ok = values[np.isfinite(values)]
    if ok.size == 0:
        return math.nan, math.nan, int(values.size)
    lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), int(values.size - ok.size)


def bootstrap_cell_stats(y_true, y_pred, weights=None, *, n_boot: int = MIN_BOOT, seed: int = 0,
                         alpha: float = 0.05, stats: Sequence[str] = ("f1", "kappa")) -> dict[str, dict[str, float]]:
    """Percentile-bootstrap CIs for ``stats``: items resampled with replacement (weights ride along).

    Seeded (``numpy.random.default_rng(seed)``), so results are reproducible. Resamples where a
    statistic is undefined (e.g. no positives drawn) are excluded and counted in ``n_undefined``.
    """
    _check_boot(n_boot)
    t, p = _as_bool_array(y_true, "y_true"), _as_bool_array(y_pred, "y_pred")
    n = t.size
    if n == 0:
        return {s: {"lo": math.nan, "hi": math.nan, "n_boot": n_boot, "n_undefined": n_boot} for s in stats}
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != t.shape or (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be positive, finite and match the data length")
    cell = (t.astype(int) * 2 + p.astype(int))  # 0 tn, 1 fp, 2 fn, 3 tp
    rng = np.random.default_rng(seed)
    collected: dict[str, list[np.ndarray]] = {s: [] for s in stats}
    chunk = max(1, min(n_boot, 4_000_000 // n))
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        cw = w[idx]
        cc = cell[idx]
        tot = [(cw * (cc == k)).sum(axis=1) for k in (3, 1, 2, 0)]  # tp fp fn tn
        res = cell_stats(*tot)
        for s in stats:
            collected[s].append(res[s])
        done += m
    out = {}
    for s in stats:
        lo, hi, undef = _percentile_ci(np.concatenate(collected[s]), alpha)
        out[s] = {"lo": lo, "hi": hi, "n_boot": n_boot, "n_undefined": undef}
    return out


def bootstrap_from_counts(c: Confusion, *, n_boot: int = MIN_BOOT, seed: int = 0, alpha: float = 0.05,
                          stats: Sequence[str] = ("f1", "kappa")) -> dict[str, dict[str, float]]:
    """Same bootstrap for an unweighted table, resampling the four cells multinomially.

    Resampling n items with replacement and counting cells is exactly ``Multinomial(n, cells / n)``,
    so this is the item bootstrap without materialising the items (used for tables with many rows).
    """
    _check_boot(n_boot)
    n = int(round(c.n))
    if n == 0:
        return {s: {"lo": math.nan, "hi": math.nan, "n_boot": n_boot, "n_undefined": n_boot} for s in stats}
    probs = np.array([c.tp, c.fp, c.fn, c.tn], dtype=float) / c.n
    draws = np.random.default_rng(seed).multinomial(n, probs, size=n_boot)
    res = cell_stats(draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3])
    out = {}
    for s in stats:
        lo, hi, undef = _percentile_ci(res[s], alpha)
        out[s] = {"lo": lo, "hi": hi, "n_boot": n_boot, "n_undefined": undef}
    return out


def classification_summary(y_true, y_pred, *, n_boot: int = MIN_BOOT, seed: int = 0) -> dict:
    """Unweighted summary: confusion, Wilson CIs for precision / recall / specificity / accuracy,
    bootstrap CIs for F1 and kappa, PABAK, positive and negative agreement."""
    c = confusion_counts(y_true, y_pred)
    ci = bootstrap_from_counts(c, n_boot=n_boot, seed=seed, stats=("f1", "kappa", "pabak"))
    return {
        "n": int(c.n), "confusion": {k: int(v) for k, v in c.as_dict().items()},
        "precision": proportion(int(c.tp), int(c.tp + c.fp)),
        "recall": proportion(int(c.tp), int(c.tp + c.fn)),
        "specificity": proportion(int(c.tn), int(c.tn + c.fp)),
        "accuracy": proportion(int(c.tp + c.tn), int(c.n)),
        "f1": {"est": f1(c), **ci["f1"]},
        "kappa": {"est": cohen_kappa(c), **ci["kappa"]},
        "pabak": {"est": pabak(c), **ci["pabak"]},
        "positive_agreement": positive_agreement(c),
        "negative_agreement": negative_agreement(c),
        "prevalence_truth": _div(c.n_pos, c.n),
        "prevalence_pred": _div(c.tp + c.fp, c.n),
    }


# ------------------------------------------------------------------ McNemar


def mcnemar_exact_counts(b: int, c: int) -> dict[str, float]:
    """Exact two-sided McNemar test on discordant counts (``b``: A-only, ``c``: B-only).

    Under H0 the discordant pairs are Binomial(b + c, 1/2); the two-sided p is
    ``min(1, 2 * P(X <= min(b, c)))`` (the sum of outcomes no likelier than the observed one).
    """
    if b < 0 or c < 0 or int(b) != b or int(c) != c:
        raise ValueError("discordant counts must be non-negative integers")
    b, c = int(b), int(c)
    m = b + c
    if m == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0}
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1))
    p = min(Fraction(1), Fraction(2 * tail, 2**m))
    return {"b": b, "c": c, "n_discordant": m, "p": float(p)}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict[str, float]:
    """Do classifiers A and B differ in accuracy on the same items? ``b`` = A right and B wrong,
    ``c`` = A wrong and B right."""
    t, a, b = (_as_bool_array(x, n) for x, n in ((y_true, "y_true"), (pred_a, "pred_a"), (pred_b, "pred_b")))
    if not (t.shape == a.shape == b.shape):
        raise ValueError("length mismatch")
    ra, rb = a == t, b == t
    return mcnemar_exact_counts(int((ra & ~rb).sum()), int((~ra & rb).sum()))


def mcnemar_marginal(rater_a, rater_b) -> dict[str, float]:
    """Exact McNemar for marginal homogeneity of two binary raters (``b``: A=1,B=0; ``c``: A=0,B=1)."""
    a, b = _as_bool_array(rater_a, "rater_a"), _as_bool_array(rater_b, "rater_b")
    if a.shape != b.shape:
        raise ValueError("length mismatch")
    return mcnemar_exact_counts(int((a & ~b).sum()), int((~a & b).sum()))


# ------------------------------------------------------------------ inverse-probability weighting


def ipw_weights(inclusion_prob: Iterable[float]) -> np.ndarray:
    pi = np.asarray(list(inclusion_prob), dtype=float)
    if pi.size and (not np.isfinite(pi).all() or (pi <= 0).any() or (pi > 1 + 1e-12).any()):
        raise ValueError("inclusion probabilities must be in (0, 1]")
    return 1.0 / np.minimum(pi, 1.0)


def ipw_confusion(y_true, y_pred, inclusion_prob) -> Confusion:
    """Horvitz-Thompson estimate of the population 2x2 table: cell total = ``sum(w_i * 1[cell])``."""
    t, p = _as_bool_array(y_true, "y_true"), _as_bool_array(y_pred, "y_pred")
    w = ipw_weights(inclusion_prob)
    if not (t.shape == p.shape == w.shape):
        raise ValueError("length mismatch")
    return Confusion(float(w[t & p].sum()), float(w[~t & p].sum()), float(w[t & ~p].sum()), float(w[~t & ~p].sum()))


def kish_neff(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum()) if w.size else 0.0


def ipw_ratio(num: np.ndarray, den: np.ndarray, inclusion_prob) -> dict[str, float]:
    """Ratio of HT totals ``sum(w num) / sum(w den)`` (``num`` implies ``den``) with the Poisson-design
    linearised ``se_design``, the Kish ``n_eff`` of the denominator items, and a Wilson interval at
    ``n_eff``. ``est`` is NaN if the weighted denominator is empty."""
    num, den = np.asarray(num, dtype=bool), np.asarray(den, dtype=bool)
    pi = np.asarray(list(inclusion_prob), dtype=float)
    w = ipw_weights(pi)
    if (num & ~den).any():
        raise ValueError("num must be a subset of den")
    d_tot = float(w[den].sum())
    if d_tot <= 0:
        return {"est": math.nan, "se_design": math.nan, "n_eff": 0.0, "lo": 0.0, "hi": 1.0, "n_items": 0,
                "n_num_items": 0, "weighted_num": 0.0, "weighted_den": 0.0}
    r = float(w[num].sum() / d_tot)
    u = num.astype(float) - r * den.astype(float)
    var = float(np.sum((1.0 - pi) / pi**2 * u**2)) / d_tot**2
    n_eff = kish_neff(w[den])
    lo, hi = wilson_interval(r, n_eff)
    return {"est": r, "se_design": math.sqrt(max(var, 0.0)), "n_eff": n_eff, "lo": lo, "hi": hi,
            "n_items": int(den.sum()), "n_num_items": int(num.sum()),
            "weighted_num": float(w[num].sum()), "weighted_den": d_tot}


def ipw_summary(y_true, y_pred, inclusion_prob, *, n_boot: int = MIN_BOOT, seed: int = 0) -> dict:
    """IPW analogue of ``classification_summary`` (see the module docstring for the variance treatment)."""
    t, p = _as_bool_array(y_true, "y_true"), _as_bool_array(y_pred, "y_pred")
    pi = np.asarray(list(inclusion_prob), dtype=float)
    if not (t.shape == p.shape == pi.shape):
        raise ValueError("length mismatch")
    w = ipw_weights(pi)
    c = ipw_confusion(t, p, pi)
    boot = bootstrap_cell_stats(t, p, w, n_boot=n_boot, seed=seed,
                                stats=("f1", "kappa", "pabak", "positive_agreement", "negative_agreement"))
    all_ = np.ones_like(t)
    return {
        "n_items": int(t.size), "n_eff": kish_neff(w),
        "weighted_confusion": c.as_dict(), "weighted_n": c.n,
        "unweighted_confusion": {k: int(v) for k, v in confusion_counts(t, p).as_dict().items()},
        "precision": ipw_ratio(t & p, p, pi),
        "recall": ipw_ratio(t & p, t, pi),
        "specificity": ipw_ratio(~t & ~p, ~t, pi),
        "accuracy": ipw_ratio(t == p, all_, pi),
        "f1": {"est": f1(c), **boot["f1"]},
        "kappa": {"est": cohen_kappa(c), **boot["kappa"]},
        "pabak": {"est": pabak(c), **boot["pabak"]},
        "positive_agreement": {"est": positive_agreement(c), **boot["positive_agreement"]},
        "negative_agreement": {"est": negative_agreement(c), **boot["negative_agreement"]},
        "prevalence_truth": _div(c.n_pos, c.n), "prevalence_pred": _div(c.tp + c.fp, c.n),
    }
