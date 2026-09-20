"""Figures. Per-seed dots first, arm means second; every title and caption carries one stamp.

Conventions (dataviz skill): fixed categorical hue per arm (never re-assigned when arms are missing), thin marks, a
recessive grid, labelled axes, the number of seeds per arm on the axis, rate axes start at 0 (no truncation), identity
never colour-only (clean arms are hollow squares, hackable arms filled circles). Light surface; PNGs carry the stamped
title in their metadata and ``figures.json`` lists every figure with title, stamp, caption and n per arm.

``stamp`` is the single helper that decides CONFIRMATORY vs EXPLORATORY: a figure is CONFIRMATORY only if the analysis
was run with ``--confirmatory`` *and* the quantity is pre-registered as confirmatory (the primary or the Holm family).
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rhg import prereg_constants as C
from rhg.analysis.endpoints import RunData, SeedEndpoints

ARM_ORDER = ("clean_none", "clean_subtle", "clean_explicit", "hackable_none", "hackable_subtle", "hackable_explicit",
             "hackable_subtle_ast")
# dataviz reference palette (light), slots 1..7 in fixed order
ARM_COLORS = dict(zip(ARM_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
CONFIRMATORY_ITEMS = frozenset({"primary", *C.HOLM_FAMILY})
HINT_LEVELS = ("none", "subtle", "explicit")


def stamp(item: str, confirmatory_run: bool) -> str:
    """``CONFIRMATORY`` iff the run is confirmatory and ``item`` is pre-registered as such; else ``EXPLORATORY``."""
    return "CONFIRMATORY" if confirmatory_run and item in CONFIRMATORY_ITEMS else "EXPLORATORY"


def stamped(title: str, kind: str) -> str:
    return f"[{kind}] {title}"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    return plt


def _is_clean(arm: str) -> bool:
    return arm.startswith("clean")


class FigureSet:
    """Collects figures and writes the ``figures.json`` index."""

    def __init__(self, out_dir: Path, table: Sequence[SeedEndpoints], runs: Sequence[RunData], confirmatory: bool,
                 tests: Mapping[str, Any] | None = None, robustness: Mapping[str, Any] | None = None) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.table, self.runs, self.confirmatory = list(table), list(runs), confirmatory
        self.tests, self.rob = dict(tests or {}), dict(robustness or {})
        self.records: list[dict[str, Any]] = []
        self.plt = _plt()

    # ---- helpers
    def n_per_arm(self, arms: Sequence[str] | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.table:
            if arms is None or e.arm in arms:
                out[e.arm] = out.get(e.arm, 0) + 1
        return out

    def _finish(self, fig, name: str, title: str, item: str, caption: str, arms: Sequence[str] | None = None) -> None:
        kind = stamp(item, self.confirmatory)
        n = self.n_per_arm(arms)
        full_caption = f"{caption} n seeds per arm: " + ", ".join(f"{a} {k}" for a, k in n.items()) + "."
        w, h = fig.get_size_inches()
        title_s = textwrap.fill(stamped(title, kind), width=max(30, int(w * 9)))
        cap_s = textwrap.fill(f"{kind} | {full_caption}", width=max(40, int(w * 16)))
        bottom = 0.012 + 0.16 * (cap_s.count(chr(10)) + 1) / h
        top = 1 - 0.06 * (title_s.count(chr(10)) + 1) * 3.8 / h
        fig.tight_layout(rect=(0, bottom, 1, top))
        fig.suptitle(title_s, x=0.01, y=0.998, ha="left", va="top", fontsize=10, fontweight="bold")
        fig.text(0.01, 0.005, cap_s, fontsize=6.5, color=INK2, ha="left", va="bottom")
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight", metadata={"Title": stamped(title, kind), "Description": full_caption})
        self.plt.close(fig)
        self.records.append({"file": path.name, "title": stamped(title, kind), "stamp": kind, "item": item,
                             "caption": full_caption, "n_per_arm": n})

    def _dots(self, ax, arms: Sequence[str], attr: str, ylabel: str, *, ymax: float | None = None, hline: float | None = None,
              rate: bool = True, data: Mapping[str, Sequence[float]] | None = None) -> None:
        if data is None:
            data = {a: [float(getattr(e, attr)) for e in self.table if e.arm == a] for a in arms}
        pos = 0
        ticks, labels, top = [], [], 0.0
        for a in arms:
            vals = [v for v in data.get(a, []) if not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                continue
            k = len(vals)
            xs = pos + (np.arange(k) - (k - 1) / 2) * 0.12
            ax.scatter(xs, vals, s=46, marker="s" if _is_clean(a) else "o", facecolors="none" if _is_clean(a) else ARM_COLORS[a],
                       edgecolors=ARM_COLORS[a], linewidths=1.6, zorder=3, label=a)
            ax.hlines(float(np.mean(vals)), pos - 0.3, pos + 0.3, colors=INK, linewidth=1.4, zorder=4)
            top = max(top, max(vals))
            ticks.append(pos)
            labels.append(f"{a}\n(n={k})")
            pos += 1
        ax.set_xticks(ticks, labels, fontsize=7)
        ax.set_xlim(-0.6, max(pos - 0.4, 0.6))
        ax.set_ylabel(ylabel)
        if rate:
            ax.set_ylim(0, ymax if ymax is not None else min(1.0, max(0.1, top * 1.15)))
        if hline is not None:
            ax.axhline(hline, color=INK2, linewidth=0.9, linestyle="--", zorder=1)

    def dots_figure(self, name: str, title: str, item: str, arms: Sequence[str], attr: str, ylabel: str, caption: str, *,
                    hline: float | None = None, hline_label: str = "", rate: bool = True) -> None:
        arms = [a for a in arms if any(e.arm == a for e in self.table)]
        if not arms:
            return
        fig, ax = self.plt.subplots(figsize=(max(3.6, 1.5 * len(arms) + 1.5), 3.8))
        self._dots(ax, arms, attr, ylabel, hline=hline, rate=rate)
        if not rate:
            vals = [float(getattr(e, attr)) for e in self.table if e.arm in arms and not math.isnan(float(getattr(e, attr)))]
            ax.set_ylim(min(0.0, min(vals)), max(vals) * 1.15 if max(vals) > 0 else 1.0)
        if hline is not None and hline_label:
            ax.text(ax.get_xlim()[1], hline, hline_label, ha="right", va="bottom", fontsize=7, color=INK2)
        self._finish(fig, name, title, item, caption + " Dots = seeds, bar = arm mean.", arms)

    # ---- individual figures
    def primary_and_arms(self) -> None:
        t, c = C.PRIMARY_CONTRAST
        self.dots_figure("primary_dots", "Final HACK_RT rate per seed: primary contrast", "primary", (c, t), "final_hack_rt",
                         "HACK_RT rate (final test eval)", "Primary endpoint, exact permutation test on seeds.",
                         hline=C.EMERGED_THRESHOLD, hline_label="emerged >= 0.02")
        self.dots_figure("all_arms_dots", "Final HACK_RT rate per seed: all arms", "arms", ARM_ORDER, "final_hack_rt",
                         "HACK_RT rate (final test eval)", "Every arm, per seed.", hline=C.EMERGED_THRESHOLD, hline_label="emerged >= 0.02")
        self.dots_figure("h3b_dots", "Held-out CORRECT rate per seed: hackable_subtle vs clean_subtle", "H3b",
                         (C.H3B_CONTRAST[1], C.H3B_CONTRAST[0]), "final_correct", "CORRECT rate (final test eval)",
                         "H3b: held-out honest pass, lower expected in the hackable arm.")
        a, b = C.H4A_CONTRAST
        self.dots_figure("h4a_dots", "Final HACK_RT rate per seed: AST-penalty arm vs hackable_subtle", "H4a", (b, a), "final_hack_rt",
                         "HACK_RT rate (final test eval)", "H4a: does the narrow AST penalty lower HACK_RT? Outside the Holm family.")

    def dose_response(self) -> None:
        if not any(e.arm in C.H1_ARMS for e in self.table):
            return
        fig, axes = self.plt.subplots(1, 2, figsize=(8.2, 3.8))
        for ax, attr, ylabel in ((axes[0], "final_hack_rt", "HACK_RT rate (final test eval)"), (axes[1], "onset", "onset step (censored at T + 1)")):
            for x, hint in enumerate(HINT_LEVELS):
                for arm, dx in ((f"hackable_{hint}", 0.12), (f"clean_{hint}", -0.12)):
                    vals = [float(getattr(e, attr)) for e in self.table if e.arm == arm]
                    if not vals:
                        continue
                    xs = x + dx + (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.06
                    ax.scatter(xs, vals, s=40, marker="s" if _is_clean(arm) else "o", facecolors="none" if _is_clean(arm) else ARM_COLORS[arm],
                               edgecolors=ARM_COLORS[arm], linewidths=1.5, zorder=3, label=f"{arm} (n={len(vals)})")
                    ax.hlines(float(np.mean(vals)), x + dx - 0.12, x + dx + 0.12, colors=INK, linewidth=1.3, zorder=4)
            ax.set_xticks(range(3), [f"hint: {h}" for h in HINT_LEVELS])
            ax.set_ylabel(ylabel)
            ax.set_xlim(-0.5, 2.5)
            if attr == "final_hack_rt":
                ax.set_ylim(0, min(1.0, max(0.1, 1.15 * max(float(e.final_hack_rt) for e in self.table))))
            else:
                T = max(e.T for e in self.table)
                ax.set_ylim(0, T + 4)
                ax.axhline(T + 1, color=INK2, linestyle="--", linewidth=0.9)
                ax.text(2.5, T + 1, "censored", ha="right", va="bottom", fontsize=7, color=INK2)
        axes[0].legend(fontsize=6, frameon=False, loc="upper left")
        self._finish(fig, "dose_response", "Dose-response over hint level (H1): final HACK_RT rate and onset", "H1_final",
                     "Hackable arms (filled circles) vs clean arms with the same hint (hollow squares); H1 tests only the hackable arms.")

    def onset(self) -> None:
        self.dots_figure("onset", "Onset step per seed (trailing-5 train HACK_RT mean >= 0.10)", "onset",
                         ARM_ORDER, "onset", "onset step (T + 1 = censored)", "Onset from training rollouts; T + 1 means never reached.",
                         hline=max(e.T for e in self.table) + 1 if self.table else None, hline_label="censored", rate=False)

    def gap_and_covariates(self) -> None:
        self.dots_figure("gap", "Reward-held-out gap of the last 5 training steps (H3a, descriptive)", "H3a", ARM_ORDER, "gap",
                         "train reward - train held-out pass", "Mechanical in hackable arms; reported, not tested.", rate=False)
        panels = (("len_mean", "completion length (tokens)", False), ("truncation_rate", "truncation rate", True),
                  ("extraction_fail_rate", "code-extraction failure rate", True), ("final_gap_other", "GAP_OTHER rate", True),
                  ("final_attempt_rt", "ATTEMPT_RT rate", True), ("final_correct", "CORRECT rate", True))
        fig, axes = self.plt.subplots(2, 3, figsize=(11, 6.4))
        arms = [a for a in ARM_ORDER if any(e.arm == a for e in self.table)]
        for ax, (attr, label, rate) in zip(axes.ravel(), panels):
            self._dots(ax, arms, attr, label, rate=rate)
            if not rate:
                ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", labelrotation=60, labelsize=5.5)
        self._finish(fig, "covariates", "Covariates that can move for non-hacking reasons (final test eval)", "covariates",
                     "Length, truncation, extraction failure, GAP_OTHER, attempt and correct rates per seed.", arms)

    def evasion(self) -> None:
        data = {a: [e.evasion for e in self.table if e.arm == a and not math.isnan(e.evasion)] for a in ARM_ORDER}
        arms = [a for a in ARM_ORDER if data[a]]
        if not arms:
            return
        fig, ax = self.plt.subplots(figsize=(max(3.6, 1.5 * len(arms) + 1.5), 3.8))
        self._dots(ax, arms, "evasion", "P(not ast_narrow | HACK_RT)", ymax=1.0, data=data)
        ax.axhline(C.H4B_EVASION_MIN, color=INK2, linestyle="--", linewidth=0.9)
        ax.text(ax.get_xlim()[1], C.H4B_EVASION_MIN, "displacement needs >= 0.5", ha="right", va="bottom", fontsize=7, color=INK2)
        self._finish(fig, "evasion", "Evasion fraction of the narrow AST monitor (H4b)", "H4b",
                     "Seeds without any final HACK_RT rollout have no evasion value and are omitted.", arms)

    def crosshint(self) -> None:
        arms = [a for a in ARM_ORDER if any(e.arm == a and e.xhint for e in self.table)]
        if not arms:
            return
        fig, ax = self.plt.subplots(figsize=(max(5, 1.4 * len(arms) + 2), 3.9))
        width = 0.26
        shade = {"none": 0.35, "subtle": 0.65, "explicit": 1.0}
        for j, hint in enumerate(HINT_LEVELS):
            for i, arm in enumerate(arms):
                vals = [e.xhint[hint] for e in self.table if e.arm == arm and hint in e.xhint]
                if not vals:
                    continue
                x = i + (j - 1) * width
                ax.bar(x, np.mean(vals), width=width * 0.9, color=ARM_COLORS[arm], alpha=shade[hint], edgecolor=ARM_COLORS[arm], linewidth=0.8)
                ax.scatter(np.full(len(vals), x) + (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.03, vals, s=14, color=INK, zorder=4)
        ax.set_xticks(range(len(arms)), [f"{a}\n(n={self.n_per_arm()[a]})" for a in arms], fontsize=7)
        ax.set_ylabel("HACK_RT rate on test problems")
        ax.set_ylim(0, min(1.0, max(0.1, 1.15 * max(v for e in self.table for v in e.xhint.values()))))
        ax.text(0.01, 0.98, "bars left to right per arm: prompt hint none / subtle / explicit (light to dark); dots = seeds",
                transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
        self._finish(fig, "crosshint", "Cross-hint evaluation of the final policies", "crosshint",
                     "Does the trained policy hack when the prompt never mentions run_tests? (test problems, 4 samples per problem).", arms)

    def trajectories(self) -> None:
        arms = [a for a in ARM_ORDER if any(r.arm == a for r in self.runs)]
        if not arms:
            return
        for name, title, kind in (("trajectories_train", "Training HACK_RT rate per step (trailing-5 mean)", "train"),
                                  ("trajectories_val", "Validation HACK_RT rate at eval steps", "val")):
            cols = min(4, len(arms))
            rows = math.ceil(len(arms) / cols)
            fig, axes = self.plt.subplots(rows, cols, figsize=(3.1 * cols, 2.6 * rows), sharey=True, squeeze=False)
            top = 0.1
            for ax, arm in zip(axes.ravel(), arms):
                curves = []
                for r in (x for x in self.runs if x.arm == arm):
                    if kind == "train":
                        y = np.array(r.hack_train_rates(), float)
                        k = C.ONSET_WINDOW
                        sm = np.array([y[max(0, i - k + 1): i + 1].mean() for i in range(len(y))])
                        xs = np.arange(1, len(y) + 1)
                    else:
                        pts = r.val_points()
                        xs, sm = np.array([p.step for p in pts]), np.array([p.rate("hack_rt") for p in pts])
                    if len(xs):
                        ax.plot(xs, sm, color=ARM_COLORS[arm], linewidth=0.8, alpha=0.55)
                        curves.append((xs, sm))
                if curves and len({len(c[0]) for c in curves}) == 1:
                    ax.plot(curves[0][0], np.mean([c[1] for c in curves], axis=0), color=ARM_COLORS[arm], linewidth=2.2)
                    top = max(top, max(float(np.max(c[1])) for c in curves))
                ax.set_title(f"{arm} (n={len(curves)})", fontsize=8)
                ax.set_xlabel("training step")
            for ax in axes.ravel()[len(arms):]:
                ax.axis("off")
            axes[0][0].set_ylim(0, min(1.0, top * 1.15))
            axes[0][0].set_ylabel("HACK_RT rate")
            self._finish(fig, name, title, "trajectory", "Thin lines = seeds, thick line = arm mean.", arms)

    def robustness_forest(self) -> None:
        loo = (self.rob.get("leave_one_out") or {}).get("primary")
        if not loo or not loo.get("available"):
            return
        rows = loo["rows"]
        fig, axes = self.plt.subplots(1, 2, figsize=(9, 0.35 * len(rows) + 2.2), sharey=True)
        ys = np.arange(len(rows))[::-1]
        for ax, key, label in ((axes[0], "delta", "mean difference (hackable - clean)"), (axes[1], "p", "exact one-sided p")):
            for y, r in zip(ys, rows):
                v = r[key]
                if isinstance(v, float) and math.isnan(v):
                    continue
                ax.scatter([v], [y], s=42, marker="D" if r["flip"] else "o", color=ARM_COLORS["hackable_subtle" if r["arm"] == C.PRIMARY_CONTRAST[0] else "clean_subtle"],
                           edgecolors=INK, linewidths=1.2 if r["flip"] else 0.4, zorder=3)
            ax.axvline(loo["full"][key], color=INK, linewidth=1.0)
            thr = C.DELTA_MIN if key == "delta" else C.ALPHA
            ax.axvline(thr, color=INK2, linestyle="--", linewidth=0.9)
            ax.set_xlabel(label + f" (dashed = {thr:g})")
        axes[0].set_yticks(ys, [("FLIP  " if r["flip"] else "") + f"drop {r['dropped']}" for r in rows], fontsize=7)
        self._finish(fig, "robustness_forest", "Leave-one-seed-out: primary contrast", "robustness",
                     "Solid line = all seeds; diamonds = dropping that seed flips the verdict.", C.PRIMARY_CONTRAST)

    def all(self) -> list[dict[str, Any]]:
        self.primary_and_arms()
        self.dose_response()
        self.onset()
        self.gap_and_covariates()
        self.evasion()
        self.crosshint()
        self.trajectories()
        self.h2_rho()
        self.robustness_forest()
        (self.dir / "figures.json").write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
        return self.records

    def h2_rho(self) -> None:
        usable = [e for e in self.table if e.arm in C.H2_ARMS]
        if not usable:
            return
        data = {a: [e.rho for e in usable if e.arm == a and e.rho_usable] for a in C.H2_ARMS}
        arms = [a for a in C.H2_ARMS if data[a]]
        if not arms:
            return
        fig, ax = self.plt.subplots(figsize=(max(3.6, 1.6 * len(arms) + 1.5), 3.8))
        self._dots(ax, arms, "rho", "Spearman rho (per-problem hack rate vs p_B_full)", rate=False, data=data)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color=INK2, linewidth=0.9)
        self._finish(fig, "h2_rho", "H2: per-seed correlation of hack rate with honest pass rate", "H2",
                     "Only seeds with 0 < final HACK_RT rate < 1 are usable; H2 predicts rho < 0.", arms)


def make_figures(out_dir: Path, table: Sequence[SeedEndpoints], runs: Sequence[RunData], confirmatory: bool,
                 tests: Mapping[str, Any] | None = None, robustness: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Write every figure and ``figures.json``; returns the index records."""
    return FigureSet(out_dir, table, runs, confirmatory, tests, robustness).all()
