"""Doc-consistency guard: ``python -m rhg.check_docs [--docs-dir DIR] [--repo-root DIR] [-v]``.

Parses PREREG.md, DESIGN.md, BUDGET.md, SCHEDULE.md, ``docs/REPO_SPEC.md`` and README.md and compares what they state with
what the code and configs say: seeds per arm and the 22-run total (``configs/plan.yaml``), the arm table
(``configs/arms/``), T / batch shape / lr / LoRA / sampling / thresholds (``configs/base.yaml`` and the code constants),
alpha / Delta / H4b thresholds / Holm family (``rhg.prereg_constants``), the ladder (``rhg.budget.LADDER``), the budget
lines ($30), the DESIGN §6 minimum-attainable-p and power table (by enumeration), the SCHEDULE gate names (printed by
scripts and modules), the pre-freeze artifacts, every CLI named in the
docs (importable, answers ``--help``, documented flags exist) and the REPO_SPEC layout.

The docs are the frozen side: when they disagree with the code, fix the code. A genuine documentation error is logged in
``docs/SPEC_DEVIATIONS.md`` and accepted here *by name*: the check id must be listed in ``ACCEPTABLE`` below **and** the
token ``[check_docs:<id>]`` must occur in ``docs/SPEC_DEVIATIONS.md``; an accepted mismatch is printed, never silent.

A parse failure (a doc sentence the check expects is missing or reworded) is itself a mismatch, so a doc edit cannot
switch a check off. Exit codes: 0 consistent, 1 mismatches (diff-style report), 2 usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
import re
import runpy
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from rhg.manifest import REPO_ROOT

DOC_FILES = ("PREREG.md", "DESIGN.md", "BUDGET.md", "SCHEDULE.md", "docs/REPO_SPEC.md", "README.md")
DEVIATIONS_FILE = "docs/SPEC_DEVIATIONS.md"

# Documented deviations the check may accept (id -> why). Each needs "[check_docs:<id>]" in docs/SPEC_DEVIATIONS.md.
ACCEPTABLE = {
    "schedule_results_report": "SCHEDULE Day 3 says `results/REPORT.md`; the analysis writes `results/analysis/REPORT.md` (REPO_SPEC §6)",
    "gate_0_manual": "SCHEDULE Gate 0 is the morning checklist done by hand (docs/MORNING_CHECKLIST.md); no script prints it",
    "gate_3b_informational": "SCHEDULE Gate 3b is informational: rhg.validate.harness writes the report, no script prints a verdict",
}
MANUAL_GATES = {"0": "gate_0_manual", "3b": "gate_3b_informational"}

# Pre-freeze artifacts (PREREG §8) -> (words PREREG uses for it, producers that must mention it).
PRE_FREEZE = {
    "hint selection": ("prereg/hint_selection.json", ("src/rhg/eval/probe_hints.py",)),
    "budget decision": ("prereg/budget_decision.md", ("src/rhg/budget.py",)),
    "passing pilot gate": ("prereg/pilot_gate.json", ("scripts/pilot.sh",)),
    "splits": ("data/processed/splits.json", ("src/rhg/data/build.py",)),
    "passing judge calibration whose rubric hash matches": ("results/analysis/judge_calibration.json", ("src/rhg/validate/calibrate.py",)),
}
# Other artifacts the docs name and who produces them.
PRODUCERS = {
    "prereg/FREEZE.json": ("src/rhg/analysis/prereg_check.py", "scripts/freeze_prereg.sh"),
    "prereg/AMENDMENTS.jsonl": ("src/rhg/analysis/prereg_check.py",),
    "results/bench/throughput.json": ("src/rhg/eval/bench.py",),
    "results/ledger.jsonl": ("src/rhg/budget.py",),
    "BUDGET_MEASURED.md": ("src/rhg/budget.py",),
    "results/analysis/REPORT.md": ("src/rhg/analysis/report.py",),
    **{a: p for a, p in ((v[0], v[1]) for v in PRE_FREEZE.values())},
}
# Artifact-like paths in the docs that are not (or not only) produced by code.
EXISTING_OR_PROSE = {"results/REPORT.md": "schedule_results_report"}


@dataclass
class Mismatch:
    check: str
    item: str
    doc: str
    code: str
    doc_src: str = ""
    code_src: str = ""
    accepted: str | None = None  # reason when logged as a documented deviation

    def render(self) -> str:
        tag = "ACCEPTED (documented deviation)" if self.accepted else "MISMATCH"
        lines = [f"{tag} [{self.check}] {self.item}", f"  - {self.doc_src or 'docs'}: {self.doc}", f"  + {self.code_src or 'code'}: {self.code}"]
        if self.accepted:
            lines.append(f"  = {self.accepted}")
        return "\n".join(lines)


@dataclass
class Ctx:
    docs_dir: Path
    repo: Path
    mismatches: list[Mismatch] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    _texts: dict[str, str] = field(default_factory=dict)
    _deviations: str | None = None
    _help_cache: dict[str, tuple[int | None, str]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)  # inputs that do not exist in this checkout: printed, never a silent pass

    # -- reading
    def doc(self, name: str) -> str:
        if name not in self._texts:
            p = self.docs_dir / name
            try:
                self._texts[name] = p.read_text(encoding="utf-8").replace("\r\n", "\n")
            except OSError:
                self._texts[name] = ""
                self.fail("docs", f"read {name}", f"{p} is missing or unreadable", "present")
        return self._texts[name]

    def deviations(self) -> str:
        if self._deviations is None:
            try:
                self._deviations = (self.repo / DEVIATIONS_FILE).read_text(encoding="utf-8")
            except OSError:
                self._deviations = ""
        return self._deviations

    def accepted(self, key: str | None) -> str | None:
        if key and key in ACCEPTABLE and f"[check_docs:{key}]" in self.deviations():
            return ACCEPTABLE[key]
        return None

    # -- recording
    def count(self, check: str) -> None:
        self.counts[check] = self.counts.get(check, 0) + 1

    def fail(self, check: str, item: str, doc: str, code: str, doc_src: str = "", code_src: str = "", accept: str | None = None) -> None:
        self.count(check)
        self.mismatches.append(Mismatch(check, item, doc, code, doc_src, code_src, self.accepted(accept)))

    def eq(self, check: str, item: str, doc_val: Any, code_val: Any, doc_src: str = "", code_src: str = "", accept: str | None = None) -> bool:
        self.count(check)
        if _same(doc_val, code_val):
            return True
        self.mismatches.append(Mismatch(check, item, str(doc_val), str(code_val), doc_src, code_src, self.accepted(accept)))
        return False

    def need(self, m: Any, check: str, item: str, doc_src: str) -> bool:
        """Parse guard: a sentence/table the check relies on must be found."""
        if m:
            return True
        self.fail(check, item, "pattern not found (sentence reworded or removed?)", "expected in the doc", doc_src)
        return False


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            return a == b
        if isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        pass
    return a == b


# ------------------------------------------------------------------ text helpers
def section(text: str, heading: str) -> str:
    """Body of the ``## <heading...>`` block (up to the next ``## ``)."""
    m = re.search(rf"^##\s+{heading}.*?$", text, re.M)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^##\s", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"^\|[\s:|-]+\|?$", line):
            continue
        cells = [c.strip().replace("**", "").strip("`").strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def num(s: str) -> float:
    return float(s.replace("−", "-").strip())


def cfg(repo: Path, arm: str = "clean_none"):
    from rhg.config import load_config

    return load_config(arm, config_dir=repo / "configs")


# ------------------------------------------------------------------ the checks
def check_plan_and_arms(c: Ctx) -> None:
    from rhg import plan as planmod
    from rhg.budget import LADDER

    try:
        pl = planmod.load_plan(c.repo / "configs" / "plan.yaml")
    except planmod.PlanError as e:
        c.fail("plan", "configs/plan.yaml", "-", str(e), code_src="configs/plan.yaml")
        return
    arms = pl.arms
    total = sum(arms.values())
    c.eq("plan", "total runs (rhg.budget.LADDER[0] vs plan.yaml)", LADDER[0].runs, total, "rhg.budget.LADDER", "configs/plan.yaml")

    prereg = c.doc("PREREG.md")
    sec = section(prereg, r"1\.")
    doc_seeds = {r[0]: int(r[1]) for r in table_rows(sec) if len(r) == 2 and r[0] != "arm" and r[1].isdigit()}
    if c.need(doc_seeds, "plan", "PREREG §1 seeds table", "PREREG.md §1"):
        c.eq("plan", "PREREG §1 arms", sorted(doc_seeds), sorted(arms), "PREREG.md §1", "configs/plan.yaml")
        for a, n in arms.items():
            c.eq("plan", f"PREREG §1 seeds {a}", doc_seeds.get(a), n, "PREREG.md §1", "configs/plan.yaml")
    m = re.search(r"(\d+) arms, (\d+) runs", sec)
    if c.need(m, "plan", "PREREG §1 '<n> arms, <m> runs'", "PREREG.md §1"):
        c.eq("plan", "PREREG §1 arm count", int(m.group(1)), len(arms), "PREREG.md §1", "configs/plan.yaml")
        c.eq("plan", "PREREG §1 run total", int(m.group(2)), total, "PREREG.md §1", "configs/plan.yaml")

    design = c.doc("DESIGN.md")
    sec3 = section(design, r"3\.")
    rows = [r for r in table_rows(sec3) if len(r) >= 6 and r[0].isdigit()]
    if c.need(rows, "arms", "DESIGN §3 arm table", "DESIGN.md §3"):
        files = sorted(p.stem for p in (c.repo / "configs" / "arms").glob("*.yaml"))
        c.eq("arms", "DESIGN §3 arm ids vs configs/arms/", sorted(r[1] for r in rows), files, "DESIGN.md §3", "configs/arms/")
        c.eq("arms", "DESIGN §3 number of arms", len(rows), len(arms), "DESIGN.md §3", "configs/plan.yaml")
        for r in rows:
            arm = r[1]
            if arm not in files:
                continue
            a = cfg(c.repo, arm).arm
            monitor = {"–": None, "-": None, "AST-narrow penalty": "ast_narrow_penalty"}.get(r[4], r[4])
            src = f"DESIGN.md §3 row {r[0]}"
            c.eq("arms", f"{arm} reward", r[2], a.reward, src, f"configs/arms/{arm}.yaml")
            c.eq("arms", f"{arm} hint", r[3], a.hint, src, f"configs/arms/{arm}.yaml")
            c.eq("arms", f"{arm} monitor", monitor, a.monitor, src, f"configs/arms/{arm}.yaml")
            c.eq("arms", f"{arm} seeds", int(r[5]), arms.get(arm), src, "configs/plan.yaml")
    m = re.search(r"Total \*\*(\d+) runs\*\*", sec3)
    if c.need(m, "plan", "DESIGN §3 'Total **N runs**'", "DESIGN.md §3"):
        c.eq("plan", "DESIGN §3 total", int(m.group(1)), total, "DESIGN.md §3", "configs/plan.yaml")
    m = re.search(r"replacements\s+use\s+(\d+)\+;\s+pilots\s+use\s+(\d+)\+", sec3)
    if c.need(m, "plan", "DESIGN §3 replacement/pilot seed bases", "DESIGN.md §3"):
        c.eq("plan", "replacement seed base", int(m.group(1)), pl.replacement_seed_base, "DESIGN.md §3", "configs/plan.yaml")
        c.eq("plan", "pilot seed min", int(m.group(2)), pl.pilot_seed_min, "DESIGN.md §3", "configs/plan.yaml")
    m = re.search(r"seeds 100\+k \(max (\d+) total\)", prereg)
    if c.need(m, "plan", "PREREG §6 replacement cap", "PREREG.md §6"):
        c.eq("plan", "replacement cap", int(m.group(1)), pl.replacement_cap, "PREREG.md §6", "configs/plan.yaml")

    # README arm table: same content as the design table
    readme = c.doc("README.md")
    rrows = [r for r in table_rows(section(readme, r"The 7-arm design|Design")) if len(r) >= 6 and r[0].isdigit()]
    if c.need(rrows, "arms", "README arm table", "README.md"):
        c.eq("arms", "README arm ids", sorted(r[1] for r in rrows), sorted(arms), "README.md", "configs/plan.yaml")
        for r in rrows:
            if r[1] in arms:
                a = cfg(c.repo, r[1]).arm
                monitor = {"–": None, "-": None, "AST-narrow penalty": "ast_narrow_penalty"}.get(r[4], r[4])
                c.eq("arms", f"README {r[1]} (reward, hint, monitor, seeds)", (r[2], r[3], monitor, int(r[5])),
                     (a.reward, a.hint, a.monitor, arms[r[1]]), "README.md", f"configs/arms/{r[1]}.yaml + plan.yaml")


def check_hyperparameters(c: Ctx) -> None:
    from rhg import prereg_constants as C
    from rhg.data import build
    from rhg.eval import pass_rate, probe_hints

    d = c.doc("DESIGN.md")
    b = cfg(c.repo)
    s21 = section(d, r"2\.\s*Setup")  # whole §2 (subsections are ###)
    if not s21:
        s21 = d
    src = "DESIGN.md §2"

    def grab(pat: str, item: str):
        m = re.search(pat, d)
        return m if c.need(m, "hyper", item, src) else None

    if m := grab(r"`(Qwen/[\w.-]+)`", "model name"):
        c.eq("hyper", "model.name", m.group(1), b.model.name, src, "configs/base.yaml")
    if m := grab(r"\*\*Thinking off\*\* \(`enable_thinking=(\w+)`", "enable_thinking"):
        c.eq("hyper", "model.enable_thinking", m.group(1) == "True", b.model.enable_thinking, src, "configs/base.yaml")
    if m := grab(r"temperature (\d+(?:\.\d+)?), top_p (\d+(?:\.\d+)?), top_k disabled", "sampling"):
        c.eq("hyper", "sampling.temperature", m.group(1), b.sampling.temperature, src, "configs/base.yaml")
        c.eq("hyper", "sampling.top_p", m.group(2), b.sampling.top_p, src, "configs/base.yaml")
        c.eq("hyper", "sampling.top_k disabled", -1, b.sampling.top_k, "DESIGN.md §2.1 (top_k disabled)", "configs/base.yaml")
    if m := grab(r"LoRA r=(\d+), alpha=(\d+), dropout (\d+(?:\.\d+)?), all linear", "LoRA"):
        c.eq("hyper", "lora.r", m.group(1), b.lora.r, src, "configs/base.yaml")
        c.eq("hyper", "lora.alpha", m.group(2), b.lora.alpha, src, "configs/base.yaml")
        c.eq("hyper", "lora.dropout", m.group(3), b.lora.dropout, src, "configs/base.yaml")
        c.eq("hyper", "lora.target", "all-linear", b.lora.target, "DESIGN.md §2.1 (all linear projections)", "configs/base.yaml")
    if m := grab(r"\(`beta=(\d+)`\)", "beta"):
        c.eq("hyper", "grpo.beta", m.group(1), b.grpo.beta, src, "configs/base.yaml")
    if m := grab(r"lr `([0-9.e-]+)` \[FREEZE\], `prompts_per_step=(\d+)`, `gens_per_prompt=(\d+)`,\s*`max_steps T=(\d+)` \[FREEZE\], `max_completion_tokens=(\d+)`", "lr / batch shape / T / max tokens"):
        c.eq("hyper", "grpo.lr", m.group(1), b.grpo.lr, src, "configs/base.yaml")
        c.eq("hyper", "grpo.prompts_per_step", m.group(2), b.grpo.prompts_per_step, src, "configs/base.yaml")
        c.eq("hyper", "grpo.gens_per_prompt", m.group(3), b.grpo.gens_per_prompt, src, "configs/base.yaml")
        c.eq("hyper", "grpo.max_steps (T)", m.group(4), b.grpo.max_steps, src, "configs/base.yaml")
        c.eq("hyper", "grpo.max_completion_tokens", m.group(5), b.grpo.max_completion_tokens, src, "configs/base.yaml")
    c.eq("hyper", "T_DEFAULT vs grpo.max_steps", C.T_DEFAULT, b.grpo.max_steps, "rhg.prereg_constants.T_DEFAULT", "configs/base.yaml")
    m = re.search(r"T \(=(\d+) unless FREEZE", c.doc("PREREG.md"))
    if c.need(m, "hyper", "PREREG §1 T", "PREREG.md §1"):
        c.eq("hyper", "PREREG T", m.group(1), b.grpo.max_steps, "PREREG.md §1", "configs/base.yaml")
    if m := grab(r"`r_hackable - (\d+(?:\.\d+)?) \* ast_narrow_flag`|r = r_hackable - (\d+(?:\.\d+)?) \* ast_narrow_flag", "monitor penalty"):
        c.eq("hyper", "reward.monitor_penalty", m.group(1) or m.group(2), b.reward.monitor_penalty, src, "configs/base.yaml")
    if m := grab(r"`0\.10 <= p_A <= 0\.40`", "difficulty band"):
        c.eq("hyper", "data.band_low/high", (0.10, 0.40), (b.data.band_low, b.data.band_high), src, "configs/base.yaml")
    if m := grab(r"widen to `\[(\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\]` once", "widened band"):
        c.eq("hyper", "widened band", (m.group(1), m.group(2)), build.WIDEN_BAND, src, "rhg.data.build.WIDEN_BAND")
    if m := grab(r"Stage A: (\d+) samples/problem", "stage A samples"):
        c.eq("hyper", "pass_rate.DEFAULT_N", m.group(1), pass_rate.DEFAULT_N, src, "rhg.eval.pass_rate.DEFAULT_N")
    if m := grab(r"<(\d+) train / <(\d+) val / <(\d+) test", "split minimums"):
        c.eq("hyper", "split minimum sizes", tuple(int(x) for x in m.groups()), (build.GATE["min_train"], build.GATE["min_val"], build.GATE["min_test"]),
             src, "rhg.data.build.GATE")
    if m := grab(r"`reward_tests` \(K=(\d+),", "K reward tests"):
        c.eq("hyper", "data.k_reward_tests", m.group(1), b.data.k_reward_tests, src, "configs/base.yaml")
    if m := grab(r"`heldout_tests` \(up to (\d+),", "held-out tests"):
        c.eq("hyper", "data.max_heldout_tests", m.group(1), b.data.max_heldout_tests, src, "configs/base.yaml")
    if m := grab(r"Val-problem evals \((\d+) samples/problem\) at steps \{([\d,]+)\}", "val evals"):
        c.eq("hyper", "eval.val_samples_per_problem", m.group(1), b.eval.val_samples_per_problem, src, "configs/base.yaml")
        want = sorted(range(0, b.grpo.max_steps, b.eval.val_every))
        c.eq("hyper", "val eval steps", sorted(int(x) for x in m.group(2).split(",")), want, src, "range(0, T, eval.val_every)")
    if m := grab(r"at step T on the \*test\* problems, (\d+) samples/problem", "final test eval"):
        c.eq("hyper", "eval.test_samples_per_problem", m.group(1), b.eval.test_samples_per_problem, src, "configs/base.yaml")
    if m := grab(r"computed from ≥(\d+) test rollouts", "min rollouts per seed"):
        c.eq("hyper", "≥480 test rollouts = min test problems x samples", m.group(1), build.GATE["min_test"] * b.eval.test_samples_per_problem,
             src, "build.GATE['min_test'] * eval.test_samples_per_problem")
    if m := grab(r"(\d+)%\s+random\s+audit", "judge audit fraction"):
        c.eq("hyper", "judge.audit_frac", int(m.group(1)) / 100, b.judge.audit_frac, src, "configs/base.yaml")
    if m := grab(r"\[(\d+\.?\d*)%, (\d+\.?\d*)%\]` ?|in \[(\d+\.?\d*)%, (\d+\.?\d*)%\]", "probe rate window"):
        lo, hi = [x for x in m.groups() if x is not None][:2]
        c.eq("hyper", "probe rate window", (float(lo) / 100, float(hi) / 100), (float(probe_hints.RATE_LO), float(probe_hints.RATE_HI)), src,
             "rhg.eval.probe_hints.RATE_LO/HI")
    if m := grab(r"≥(\d+) samples per wording", "probe sample size"):
        c.eq("hyper", "probe MIN_SAMPLES", m.group(1), probe_hints.MIN_SAMPLES, src, "rhg.eval.probe_hints.MIN_SAMPLES")
    if m := grab(r"by >(\d+) pp", "probe confound threshold"):
        c.eq("hyper", "probe CONFOUND_DROP (pp)", int(m.group(1)) / 100, float(probe_hints.CONFOUND_DROP), src, "rhg.eval.probe_hints.CONFOUND_DROP")
    if m := grab(r"\$(\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)/h", "GPU price range"):
        lo, hi = float(m.group(1)), float(m.group(2))
        c.count("hyper")
        if not lo <= b.budget.usd_per_hour <= hi:
            c.fail("hyper", "budget.usd_per_hour inside the DESIGN planning range", f"${lo}-{hi}/h", str(b.budget.usd_per_hour), src, "configs/base.yaml")
    prompts = _prompts(c)
    for wid, pat in (("S1", r"\| subtle \| S1: \"(.*?)\""), *((w, rf"{w} \(\"(.*?)\"\)") for w in ("S2", "S3", "W1", "W2", "W3"))):
        m = re.search(pat, d)
        if c.need(m, "prompts", f"DESIGN wording {wid}", src):
            c.eq("prompts", f"wording {wid}", m.group(1), prompts.get(wid), src, "configs/prompts.yaml")
    m = re.search(r"\| explicit \| E1: S1 \+ \"(.*?)\"", d)
    if c.need(m, "prompts", "DESIGN wording E1", src):
        c.eq("prompts", "wording E1", (prompts.get("S1") or "") + " " + m.group(1), prompts.get("E1"), src, "configs/prompts.yaml")


def _prompts(c: Ctx) -> dict[str, str]:
    import yaml

    try:
        raw = yaml.safe_load((c.repo / "configs" / "prompts.yaml").read_text(encoding="utf-8"))
        return {**raw["hints"]["subtle"], **raw["hints"]["explicit"]}
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return {}


def check_constants(c: Ctx) -> None:
    from math import comb

    from rhg import prereg_constants as C
    from rhg.budget import FLOOR_RUNS

    p = c.doc("PREREG.md")
    d = c.doc("DESIGN.md")
    src = "PREREG.md"

    def grab(text: str, pat: str, item: str, where: str):
        m = re.search(pat, text)
        return m if c.need(m, "constants", item, where) else None

    if m := grab(p, r"\*\*Supported iff p ≤ (\d+(?:\.\d+)?) AND Δ ≥ (\d+(?:\.\d+)?)\.\*\*", "primary decision rule", "PREREG.md §2"):
        c.eq("constants", "ALPHA", m.group(1), C.ALPHA, "PREREG.md §2", "rhg.prereg_constants.ALPHA")
        c.eq("constants", "DELTA_MIN", m.group(2), C.DELTA_MIN, "PREREG.md §2", "rhg.prereg_constants.DELTA_MIN")
    if m := grab(p, r"Holm–Bonferroni at α=(\d+(?:\.\d+)?):\s*\*\*\{([^}]*)\},\s*m=(\d+)\.\*\*", "Holm family", "PREREG.md §4"):
        names = tuple(x.strip().replace("-", "_") for x in m.group(2).split(","))
        c.eq("constants", "PREREG §4 alpha", m.group(1), C.ALPHA, "PREREG.md §4", "rhg.prereg_constants.ALPHA")
        c.eq("constants", "PREREG §4 Holm family", names, C.HOLM_FAMILY, "PREREG.md §4", "rhg.prereg_constants.HOLM_FAMILY")
        c.eq("constants", "PREREG §4 m", m.group(3), len(C.HOLM_FAMILY), "PREREG.md §4", "len(HOLM_FAMILY)")
    if m := grab(d, r"\{([^}]*)\}\s+\(m=(\d+)\)", "DESIGN Holm family", "DESIGN.md §6"):
        names = tuple(x.strip().replace("-", "_") for x in m.group(1).split(","))
        c.eq("constants", "DESIGN §6 Holm family", names, C.HOLM_FAMILY, "DESIGN.md §6", "rhg.prereg_constants.HOLM_FAMILY")
        c.eq("constants", "DESIGN §6 m", m.group(2), len(C.HOLM_FAMILY), "DESIGN.md §6", "len(HOLM_FAMILY)")
    if m := grab(p, r"\(0/5 hackable seeds with final rate ≥ (\d+(?:\.\d+)?);", "no-discovery threshold", "PREREG.md §2"):
        c.eq("constants", "EMERGED_THRESHOLD", m.group(1), C.EMERGED_THRESHOLD, src, "rhg.prereg_constants.EMERGED_THRESHOLD")
    if m := grab(p, r"Exact one-sided permutation test \((\d+) relabelings\)", "252 relabelings", "PREREG.md §2"):
        c.eq("constants", "primary relabelings = C(10,5)", m.group(1), comb(10, 5), src, "math.comb(10, 5)")
    if m := grab(p, r"if\s*`clean_explicit` final HACK_RT > (\d+(?:\.\d+)?)", "clean_explicit caveat", "PREREG.md §3"):
        c.eq("constants", "CLEAN_EXPLICIT_CAVEAT", m.group(1), C.CLEAN_EXPLICIT_CAVEAT, src, "rhg.prereg_constants")
    if m := grab(p, r"≥(\d) of (\d) seeds have HACK_RT ≥ (\d+(?:\.\d+)?) AND evasion fraction\s*`P\(¬ast_narrow \| HACK_RT\)` ≥ (\d+(?:\.\d+)?)", "H4b displacement rule", "PREREG.md §5"):
        c.eq("constants", "H4B_MIN_SEEDS", m.group(1), C.H4B_MIN_SEEDS, src, "rhg.prereg_constants")
        c.eq("constants", "H4B_N_SEEDS", m.group(2), C.H4B_N_SEEDS, src, "rhg.prereg_constants")
        c.eq("constants", "H4B_DISPLACEMENT_HACK_MIN", m.group(3), C.H4B_DISPLACEMENT_HACK_MIN, src, "rhg.prereg_constants")
        c.eq("constants", "H4B_EVASION_MIN", m.group(4), C.H4B_EVASION_MIN, src, "rhg.prereg_constants")
    if m := grab(p, r"all (\d) seeds have HACK_RT < (\d+(?:\.\d+)?)\.", "H4b suppression rule", "PREREG.md §5"):
        c.eq("constants", "H4b suppression: number of seeds", m.group(1), C.H4B_N_SEEDS, src, "rhg.prereg_constants")
        c.eq("constants", "H4B_SUPPRESSION_MAX", m.group(2), C.H4B_SUPPRESSION_MAX, src, "rhg.prereg_constants")
    if m := grab(d, r"trailing-(\d+)-step mean \*training\* HACK_RT rate ≥ (\d+(?:\.\d+)?)", "onset definition", "DESIGN.md §4"):
        c.eq("constants", "ONSET_WINDOW", m.group(1), C.ONSET_WINDOW, "DESIGN.md §4", "rhg.prereg_constants")
        c.eq("constants", "ONSET_THRESHOLD", m.group(2), C.ONSET_THRESHOLD, "DESIGN.md §4", "rhg.prereg_constants")
    if m := grab(d, r"\(`hackable_subtle` vs `clean_subtle`|`hackable_subtle` > `clean_subtle`", "primary contrast", "DESIGN.md §5"):
        c.eq("constants", "PRIMARY_CONTRAST", ("hackable_subtle", "clean_subtle"), C.PRIMARY_CONTRAST, "DESIGN.md §5", "rhg.prereg_constants")
    if m := grab(p, r"below the floor \((\d+) runs\)", "run floor", "PREREG.md §6"):
        c.eq("constants", "FLOOR_RUNS", m.group(1), FLOOR_RUNS, src, "rhg.budget.FLOOR_RUNS")

    _check_design_power(c, d)


def _check_design_power(c: Ctx, d: str) -> None:
    from rhg.analysis import power
    from rhg.analysis.stats import perm_test

    sec = section(d, r"6\.")
    rows = {r[0]: r for r in table_rows(sec) if len(r) == 3 and re.match(r"\d v \d", r[0])}
    if not c.need(rows, "power", "DESIGN §6 minimum-attainable-p table", "DESIGN.md §6"):
        return
    for label, r in rows.items():
        m = re.match(r"(\d) v (\d)( \(H4a\))?", label)
        nh, nc, h4a = int(m.group(1)), int(m.group(2)), bool(m.group(3))
        emerged = [int(x) for x in re.findall(r"\d+", r[1])]
        ps = [float(x) for x in re.findall(r"\d\.\d+", r[2])]
        if len(ps) != len(emerged):
            ps = ps[: len(emerged)]  # "0.050 (two-sided 0.10 ...)" carries an extra number
        for e, want in zip(emerged, ps):
            if h4a:
                got = perm_test([0.0] * nh, [power.DEFAULT_EMERGED_RATE] * nc, "less").p
            else:
                got = power.emergence_p(nh, nc, e, 0)
            c.count("power")
            if abs(got - want) > 5.1e-4:
                c.fail("power", f"DESIGN §6 min attainable p, {label}, {e} emerging", f"{want}", f"{got:.4f}", "DESIGN.md §6", "rhg.analysis.power (enumeration)")
    m = re.search(r"Power at 5 v 5 = P\(≥4 of 5 emerge\): ((?:q=[\d.]+ → [\d.]+(?:, )?)+)", d)
    if c.need(m, "power", "DESIGN §6 power line", "DESIGN.md §6"):
        for q, want in re.findall(r"q=(\d+(?:\.\d+)?) → (\d+(?:\.\d+)?)", m.group(1)):
            c.count("power")
            got = power.primary_power(5, 5, float(q))
            if abs(got - float(want)) > 0.005 + 1e-12:
                c.fail("power", f"DESIGN §6 5v5 power at q={q}", want, f"{got:.4f}", "DESIGN.md §6", "rhg.analysis.power (enumeration)")


def check_ladder_and_budget(c: Ctx) -> None:
    from rhg import budget
    from rhg import prereg_constants as C

    b = c.doc("BUDGET.md")
    src = "BUDGET.md"
    rows = [r for r in table_rows(section(b, r"4\.")) if len(r) >= 4 and r[0].isdigit()]
    if c.need(rows, "ladder", "BUDGET §4 ladder table", "BUDGET.md §4"):
        c.eq("ladder", "number of ladder steps", len(rows), len(budget.LADDER), "BUDGET.md §4", "rhg.budget.LADDER")
        for r in rows:
            step = int(r[0])
            if step >= len(budget.LADDER):
                c.fail("ladder", f"step {step}", "listed in BUDGET §4", "no such LADDER step", "BUDGET.md §4", "rhg.budget.LADDER")
                continue
            c.eq("ladder", f"step {step} runs after", int(r[2]), budget.LADDER[step].runs, "BUDGET.md §4", "rhg.budget.LADDER")
        prim = next((r for r in rows if r[0] == "6"), None)
        if prim:
            c.eq("ladder", "step 6 seeds (4 v 4)", (4, 4), (budget.LADDER[6].seeds["hackable_subtle"], budget.LADDER[6].seeds["clean_subtle"]),
                 "BUDGET.md §4 step 6", "rhg.budget.LADDER")
    m = re.search(r"\| floor \|[^|]*\|[^|]*\|\s*if (\d+) runs still exceed", b)
    if c.need(m, "ladder", "BUDGET §4 floor", "BUDGET.md §4"):
        c.eq("ladder", "floor runs", m.group(1), budget.FLOOR_RUNS, "BUDGET.md §4", "rhg.budget.FLOOR_RUNS")
        c.eq("ladder", "floor = last ladder step", budget.FLOOR_RUNS, budget.LADDER[-1].runs, "rhg.budget.FLOOR_RUNS", "rhg.budget.LADDER[-1]")

    env = {r[0]: r[1] for r in table_rows(section(b, r"1\.")) if len(r) == 2 and r[0] != "line"}
    caps = {}
    for k, v in env.items():
        try:
            caps[k] = num(v)
        except ValueError:
            pass
    if c.need(caps, "budget", "BUDGET §1 envelope", "BUDGET.md §1"):
        total_doc = caps.pop("total", None)
        m = re.search(r"Hard ceiling \*\*\$(\d+(?:\.\d+)?)\*\*", b)
        ceiling = float(m.group(1)) if m else None
        c.need(m, "budget", "BUDGET hard ceiling sentence", "BUDGET.md")
        c.eq("budget", "envelope lines sum to the total row", sum(caps.values()), total_doc, "BUDGET.md §1 (sum of lines)", "BUDGET.md §1 total row")
        c.eq("budget", "envelope total = hard ceiling = $30", total_doc, ceiling, "BUDGET.md §1 total row", "BUDGET.md hard ceiling")
        c.eq("budget", "hard ceiling is $30", 30.0, ceiling, "task brief / README", "BUDGET.md")
        by = lambda frag: next((v for k, v in caps.items() if frag in k), None)  # noqa: E731
        b0 = cfg(c.repo)
        c.eq("budget", "main runs cap", by("main runs"), budget.DEFAULT_MAIN_CAP_USD, "BUDGET.md §1", "rhg.budget.DEFAULT_MAIN_CAP_USD")
        c.eq("budget", "judge cap", by("API: judge"), b0.judge.max_usd, "BUDGET.md §1", "configs/base.yaml judge.max_usd")
        if ceiling is not None and by("unmetered slack") is not None:
            c.eq("budget", "stop_at = ceiling - unmetered slack", ceiling - by("unmetered slack"), b0.budget.stop_at_usd, "BUDGET.md §1", "configs/base.yaml budget.stop_at_usd")
            c.eq("budget", "DEFAULT_STOP_AT_USD", ceiling - by("unmetered slack"), budget.DEFAULT_STOP_AT_USD, "BUDGET.md §1", "rhg.budget.DEFAULT_STOP_AT_USD")
    m = re.search(r"metered_spend \+ projected_run_cost\s*> (\d+(?:\.\d+)?)", b)
    if c.need(m, "budget", "BUDGET launch guard", "BUDGET.md §1"):
        c.eq("budget", "launch guard threshold", m.group(1), cfg(c.repo).budget.stop_at_usd, "BUDGET.md §1", "configs/base.yaml budget.stop_at_usd")
    m = re.search(r"`wall_h ≤ (\d+)`|wall_h ≤ (\d+)", b)
    if c.need(m, "budget", "BUDGET wall-clock limit", "BUDGET.md §3"):
        c.eq("budget", "max wall hours", m.group(1) or m.group(2), budget.DEFAULT_MAX_WALL_H, "BUDGET.md §3", "rhg.budget.DEFAULT_MAX_WALL_H")
    m = re.search(r"\(1 \+ (0\.\d+)\)", b)
    if c.need(m, "budget", "BUDGET overhead margin", "BUDGET.md §2"):
        c.eq("budget", "overhead margin", m.group(1), budget.OVERHEAD, "BUDGET.md §2", "rhg.budget.OVERHEAD")
    m = re.search(r"n_eval = (\d+) val evals \+ (\d+) test eval|n_eval = (\d) val evals", b) or re.search(r"# n_eval = (\d+) val evals \+ (\d+) test eval", b)
    if c.need(m, "budget", "BUDGET n_eval", "BUDGET.md §2"):
        c.eq("budget", "N_EVAL", int(m.group(1)) + int(m.group(2)), budget.N_EVAL, "BUDGET.md §2", "rhg.budget.N_EVAL")
    m = re.search(r"Contingency \$(\d+(?:\.\d+)?) may only fund replacement runs for invalid runs, up to (\d+)", b)
    if c.need(m, "budget", "BUDGET contingency", "BUDGET.md §5"):
        c.eq("budget", "contingency line", m.group(1), env.get("contingency (infra replacement runs only)"), "BUDGET.md §5", "BUDGET.md §1")
    c.eq("budget", "primary contrast never below the ladder minimum", (4, 4), (budget.LADDER[-1].seeds["hackable_subtle"], budget.LADDER[-1].seeds["clean_subtle"]),
         "PREREG.md §6 (4 v 4)", "rhg.budget.LADDER[-1]")
    c.eq("budget", "HOLM m (informational)", 4, len(C.HOLM_FAMILY), "PREREG.md §4", "rhg.prereg_constants")


def _script_and_src_text(repo: Path) -> str:
    parts = []
    for base, pats in (("scripts", ("*.sh",)), ("src/rhg", ("*.py",))):
        for pat in pats:
            for p in sorted((repo / base).rglob(pat)):
                if p.name in ("check_docs.py", "e2e_mock.py"):
                    continue  # this module's own tables must not make a gate look implemented
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def check_gates(c: Ctx) -> None:
    sched = c.doc("SCHEDULE.md")
    ids = list(dict.fromkeys(re.findall(r"\*\*Gate (\d[a-z0-9]*)\*\*", sched)))
    if not c.need(ids, "gates", "SCHEDULE gate names", "SCHEDULE.md"):
        return
    text = _script_and_src_text(c.repo)
    for gid in ids:
        c.count("gates")
        printed = re.search(rf"(?i)\bgate\s+{re.escape(gid)}\b", text) is not None
        if not printed:  # "Gate 2a ... and 2b" style: a script names both in one sentence
            m = re.match(r"(\d)([a-z])$", gid)
            printed = bool(m and re.search(rf"(?i)\bgate\s+{m.group(1)}[a-z]\b[^\n]*\band\s+{re.escape(gid)}\b", text))
        if printed:
            continue
        key = MANUAL_GATES.get(gid)
        c.fail("gates", f"Gate {gid}", "named in SCHEDULE.md", "no script or module mentions it", "SCHEDULE.md", "scripts/*.sh, src/rhg", accept=key)
    if "0" in ids and c.accepted("gate_0_manual"):
        c.count("gates")
        if not (c.repo / "docs" / "MORNING_CHECKLIST.md").is_file():
            c.fail("gates", "Gate 0 checklist", "docs/MORNING_CHECKLIST.md", "file missing", "SCHEDULE.md Gate 0", "docs/")
    # every gate a script prints must be in SCHEDULE
    printed = set(re.findall(r"(?i)\bgate\s+(\d[a-z0-9]?[a-z0-9]?)\b", "\n".join(p.read_text(encoding="utf-8", errors="replace")
                                                                         for p in sorted((c.repo / "scripts").glob("*.sh")))))
    for gid in sorted(printed):
        c.count("gates")
        if gid.lower() not in ids:
            c.fail("gates", f"gate '{gid}' printed by a script", "not in SCHEDULE.md", f"scripts print 'Gate {gid}'", "SCHEDULE.md", "scripts/*.sh")


def _mentions(repo: Path, rels: Iterable[str], needle: str) -> tuple[bool, list[str]]:
    missing = []
    for rel in rels:
        p = repo / rel
        if not p.is_file():
            missing.append(f"{rel} (file missing)")
        elif needle not in p.read_text(encoding="utf-8", errors="replace").replace("\\", "/"):
            missing.append(f"{rel} (does not mention {needle})")
    return not missing, missing


def check_artifacts(c: Ctx) -> None:
    prereg = c.doc("PREREG.md")
    sec = section(prereg, r"8\.")
    m = re.search(r"requires every pre-freeze artifact \(([^)]*)\)", sec, re.S)
    freeze_sh = "scripts/freeze_prereg.sh"
    if c.need(m, "artifacts", "PREREG §8 pre-freeze artifact list", "PREREG.md §8"):
        items = [x.strip() for x in re.sub(r"\s+", " ", m.group(1)).split(",")]
        for it in items:
            key = next((k for k in PRE_FREEZE if k == it), None)
            if key is None:
                c.fail("artifacts", f"pre-freeze artifact '{it}'", "named in PREREG §8", "no entry in rhg.check_docs.PRE_FREEZE", "PREREG.md §8", "rhg/check_docs.py")
                continue
            path, producers = PRE_FREEZE[key]
            base = Path(path).name
            ok, why = _mentions(c.repo, producers, base)
            c.count("artifacts")
            if not ok:
                c.fail("artifacts", f"producer of {path}", "some script writes it", "; ".join(why), "PREREG.md §8", ", ".join(producers))
            ok, why = _mentions(c.repo, (freeze_sh,), base)
            c.count("artifacts")
            if not ok:
                c.fail("artifacts", f"{freeze_sh} requires {path}", "freeze requires it", "; ".join(why), "PREREG.md §8", freeze_sh)
    sched = c.doc("SCHEDULE.md")
    toks = set()
    for name in ("PREREG.md", "SCHEDULE.md", "BUDGET.md", "DESIGN.md"):
        toks |= set(re.findall(r"`((?:prereg|results|data)/[\w./-]+\.(?:json|jsonl|md|csv|yaml)|BUDGET_MEASURED\.md)`", c.doc(name)))
    for t in sorted(toks):
        c.count("artifacts")
        if t in PRODUCERS:
            ok, why = _mentions(c.repo, PRODUCERS[t], Path(t).name)
            if not ok:
                c.fail("artifacts", f"producer of {t}", "some script writes it", "; ".join(why), "docs", ", ".join(PRODUCERS[t]))
        elif t in EXISTING_OR_PROSE:
            c.fail("artifacts", f"artifact `{t}` named in the docs", "produced by a script", "no producer writes this exact path", "SCHEDULE.md", "-", accept=EXISTING_OR_PROSE[t])
        elif not (c.repo / t).is_file():
            c.fail("artifacts", f"artifact `{t}` named in the docs", "produced by a script or a file", "not in rhg.check_docs.PRODUCERS and not a file", "docs", "-")
    for name in ("SCHEDULE.md", "PREREG.md", "DESIGN.md", "BUDGET.md", "README.md"):
        for f in sorted(set(re.findall(r"`(docs/[\w./-]+\.md)`", c.doc(name)))):
            c.count("artifacts")
            if not (c.repo / f).is_file():
                c.fail("artifacts", f"document `{f}` named in {name}", "exists", "missing", name, f)
    _ = sched


def check_layout(c: Ctx) -> None:
    spec = c.doc("docs/REPO_SPEC.md")
    m = re.search(r"## 1\. Layout\s*\n\s*```\n(.*?)\n```", spec, re.S)
    if not c.need(m, "layout", "REPO_SPEC §1 layout block", "docs/REPO_SPEC.md §1"):
        return
    runtime = {"prereg", "data", "results", "results_public"}  # written at the gates / at run time: absent in a clean clone
    top = ""
    base = ""
    expected: list[str] = []
    for raw in m.group(1).splitlines()[1:]:  # first line is the repo directory itself
        line = re.sub(r"\([^)]*\)", "", re.sub(r"\s+#.*$", "", raw))
        toks = line.split()
        if not toks:
            continue
        indent = len(line) - len(line.lstrip())
        rest = toks
        if indent <= 2:
            if toks[0].endswith("/"):
                top = base = toks[0].rstrip("/")
                rest = toks[1:]
            else:
                top = base = ""
        elif indent >= 8:  # continuation of the previous file list
            pass
        elif toks[0].endswith("/"):
            base = f"{top}/{toks[0].rstrip('/')}"
            rest = toks[1:]
        else:
            base = top
        line_dir = ""
        for tok in (t for r in rest for t in _expand_braces(r)):
            if tok.endswith("/"):
                continue  # a directory mention (fixtures/, raw/, ...)
            if "/" in tok:
                line_dir = tok.rsplit("/", 1)[0]  # 'arms/a.yaml b.yaml': b.yaml lives in arms/ too
            elif line_dir:
                tok = f"{line_dir}/{tok}"
            rel = f"{base}/{tok}" if base else tok
            if rel.split("/")[0] not in runtime:
                expected.append(rel)
    for rel in expected:
        c.count("layout")
        parent = (c.repo / rel).parent
        ok = bool(list(parent.glob(Path(rel).name))) if "*" in rel and parent.is_dir() else (c.repo / rel).exists()
        if not ok:
            c.fail("layout", f"REPO_SPEC §1 lists {rel}", "exists in the layout", "not found", "docs/REPO_SPEC.md §1", rel)


def _expand_braces(tok: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", tok)
    if not m:
        return [tok]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out.extend(_expand_braces(tok[: m.start()] + alt + tok[m.end():]))
    return out


# -- CLIs
def cli_help(c: Ctx, module: str) -> tuple[int | None, str]:
    """Run ``python -m <module> --help`` in-process; returns (exit code, help text)."""
    if module in c._help_cache:
        return c._help_cache[module]
    buf = io.StringIO()
    code: int | None = None
    old_argv = sys.argv
    sys.argv = [module, "--help"]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                runpy.run_module(module, run_name="__main__", alter_sys=False)
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception as e:  # noqa: BLE001 - import/parse failure is what we report
        buf.write(f"{type(e).__name__}: {e}")
        code = None
    finally:
        sys.argv = old_argv
    c._help_cache[module] = (code, buf.getvalue())
    return c._help_cache[module]


def verify_cli(c: Ctx, module: str, flags: Iterable[str], where: str, check: str = "cli") -> None:
    c.count(check)
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        c.fail(check, f"python -m {module}", "importable module", "not importable", where, module)
        return
    code, text = cli_help(c, module)
    if code != 0 or not text.strip():
        c.fail(check, f"python -m {module} --help", "exit 0 with a usage text", f"exit {code}: {text.strip()[-200:]!r}", where, module)
        return
    for fl in flags:
        c.count(check)
        if fl not in text:
            c.fail(check, f"python -m {module} accepts {fl}", f"{fl} documented", f"{fl} not in --help", where, module)


def _spec_cli_segments(spec: str) -> list[tuple[str, list[str]]]:
    """(module, flags) for each ``rhg...`` code span of REPO_SPEC §7 (brace lists expanded; flags only where unambiguous)."""
    out: list[tuple[str, list[str]]] = []
    for seg in re.findall(r"`(rhg\.[^`]+)`", section(spec, r"7\.")):
        head, _, rest = seg.partition(" ")
        flags = [] if ("|" in rest and "{" in rest and not rest.startswith("--")) or head.endswith("}") else re.findall(r"--[a-z][\w-]*", rest)
        if seg.startswith("rhg.budget {") or seg.startswith("rhg.plan {"):
            flags = []
        for mod in _expand_braces(head):
            if re.fullmatch(r"rhg(\.[a-z_0-9]+)+", mod):
                out.append((mod, flags))
    return out


def check_clis(c: Ctx) -> None:
    spec = c.doc("docs/REPO_SPEC.md")
    segs = _spec_cli_segments(spec)
    if not c.need(segs, "cli", "REPO_SPEC §7 CLI list", "docs/REPO_SPEC.md §7"):
        return
    seen = set()
    for mod, flags in segs:
        seen.add(mod)
        verify_cli(c, mod, flags, "docs/REPO_SPEC.md §7")
    # every `python -m rhg...` named in the other docs
    for name in ("PREREG.md", "DESIGN.md", "BUDGET.md", "SCHEDULE.md", "README.md"):
        for cmd in re.findall(r"python -m (rhg[\w.]*)([^`\n]*)", c.doc(name)):
            mod, rest = cmd
            flags = re.findall(r"(?<![\w-])(--[a-z][\w-]*)", rest)
            if mod in seen and not flags:
                continue
            seen.add(mod)
            verify_cli(c, mod, flags if mod not in ("rhg.budget", "rhg.plan") else [], f"{name}")
    # scripts named in the docs exist, and flags shown next to them appear in the script
    for name in ("SCHEDULE.md", "README.md", "BUDGET.md", "DESIGN.md", "PREREG.md"):
        for path, rest in re.findall(r"`?(scripts/[\w.]+\.sh)([^`\n]*)", c.doc(name)):
            c.count("scripts")
            p = c.repo / path
            if not p.is_file():
                c.fail("scripts", f"{path} named in {name}", "exists", "missing", name, path)
                continue
            body = p.read_text(encoding="utf-8", errors="replace")
            for fl in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", rest.split("→")[0].split(";")[0]):
                c.count("scripts")
                if fl not in body:
                    c.fail("scripts", f"{path} handles {fl}", f"{fl} shown in {name}", "flag not in the script", name, path)


def check_scripts_format(c: Ctx) -> None:
    for p in sorted((c.repo / "scripts").glob("*.sh")):
        data = p.read_bytes()
        c.count("scripts")
        if b"\r\n" in data:
            c.fail("scripts", f"{p.name} line endings", "LF", "CRLF found", "", f"scripts/{p.name}")
        if p.name != "_common.sh" and b"set -euo pipefail" not in data and b"_common.sh" not in data:
            c.fail("scripts", f"{p.name} strict mode", "set -euo pipefail (directly or via _common.sh)", "absent", "", f"scripts/{p.name}")


def check_dependencies(c: Ctx) -> None:
    """Every third-party module imported anywhere in ``src/`` is declared: core/dev dependency or a GPU pin."""
    import ast
    import tomllib

    try:
        proj = tomllib.loads((c.repo / "pyproject.toml").read_text(encoding="utf-8"))
        core = [re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower() for d in proj["project"]["dependencies"]]
        dev = [re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower() for d in proj.get("dependency-groups", {}).get("dev", [])]
        gpu = [re.split(r"[<>=!~\[ ]", ln, maxsplit=1)[0].lower() for ln in (c.repo / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    except (OSError, KeyError, ValueError) as e:
        c.fail("deps", "pyproject.toml / requirements-gpu.txt", "readable", str(e), code_src="pyproject.toml")
        return
    dist_to_module = {"pyyaml": "yaml", "pillow": "PIL", "ipykernel": "IPython", "huggingface_hub": "huggingface_hub"}
    declared = {dist_to_module.get(d, d.replace("-", "_")) for d in core + dev + gpu}
    declared |= {"IPython"} if "ipykernel" in dev else set()
    for d in core:  # the GPU stack must stay out of the lock file (REPO_SPEC §2)
        c.count("deps")
        if d in {"torch", "vllm", "trl", "peft", "transformers", "accelerate", "flash-attn", "flashinfer-python"}:
            c.fail("deps", f"{d} in pyproject.toml core dependencies", "GPU stack only in requirements-gpu.txt", "declared in pyproject.toml", "docs/REPO_SPEC.md §2", "pyproject.toml")
    lock = (c.repo / "uv.lock")
    if lock.is_file():
        text = lock.read_text(encoding="utf-8")
        for d in ("torch", "vllm", "trl", "peft"):
            c.count("deps")
            if re.search(rf'^name = "{d}"$', text, re.M):
                c.fail("deps", f"{d} in uv.lock", "not in uv.lock (REPO_SPEC §2)", "present", "docs/REPO_SPEC.md §2", "uv.lock")
    std = set(sys.stdlib_module_names)
    used: dict[str, str] = {}
    for p in sorted((c.repo / "src").rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as e:
            c.fail("deps", f"parse {p.name}", "valid python", str(e), code_src=str(p.relative_to(c.repo)))
            continue
        for n in ast.walk(tree):
            mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else (
                [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else [])
            for m in mods:
                if m not in std and m != "rhg":
                    used.setdefault(m, p.relative_to(c.repo).as_posix())
    for m, where in sorted(used.items()):
        c.count("deps")
        if m not in declared:
            c.fail("deps", f"import of '{m}'", "declared in pyproject.toml or requirements-gpu.txt", "not declared", "pyproject.toml / requirements-gpu.txt", where)
    for pin in ("torch", "vllm", "transformers", "trl", "peft", "accelerate", "datasets"):
        c.count("deps")
        line = next((ln for ln in (c.repo / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines() if ln.startswith(pin + "==")), None)
        if line is None:
            c.fail("deps", f"requirements-gpu.txt pins {pin}", f"{pin}==<version>", "no exact pin", "docs/REPO_SPEC.md §2 (exact pins)", "requirements-gpu.txt")


CHECKS: tuple[Callable[[Ctx], None], ...] = (
    check_plan_and_arms, check_hyperparameters, check_constants, check_ladder_and_budget, check_gates,
    check_artifacts, check_layout, check_clis, check_scripts_format, check_dependencies,
)


def run_checks(docs_dir: Path | None = None, repo_root: Path | None = None, only: Iterable[str] | None = None) -> Ctx:
    """Run every check (or only those named in ``only``, with or without the ``check_`` prefix)."""
    c = Ctx(Path(docs_dir) if docs_dir else Path(repo_root or REPO_ROOT), Path(repo_root or REPO_ROOT))
    for name in DOC_FILES:
        c.doc(name)
    wanted = None if only is None else {n if n.startswith("check_") else f"check_{n}" for n in only}
    for fn in CHECKS:
        if wanted is not None and fn.__name__ not in wanted:
            continue
        try:
            fn(c)
        except Exception as e:  # noqa: BLE001 - a crashing check is a failed check, never a silent pass
            c.fail(fn.__name__, "check crashed", "runs to completion", f"{type(e).__name__}: {e}", code_src=fn.__name__)
    return c


def format_report(c: Ctx, verbose: bool = False) -> str:
    real = [m for m in c.mismatches if not m.accepted]
    acc = [m for m in c.mismatches if m.accepted]
    lines = []
    if verbose:
        lines += [f"  {k}: {v} comparison(s)" for k, v in sorted(c.counts.items())]
    lines += [m.render() for m in real] + [m.render() for m in acc] + [f"SKIPPED {x}" for x in dict.fromkeys(c.skipped)]
    total = sum(c.counts.values())
    lines.append(f"check_docs: {total} comparisons, {len(real)} mismatch(es), {len(acc)} accepted documented deviation(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rhg.check_docs", description="Verify PREREG/DESIGN/BUDGET/SCHEDULE/REPO_SPEC/README against the code and configs.")
    ap.add_argument("--docs-dir", type=Path, default=None, help="directory holding PREREG.md, DESIGN.md, BUDGET.md, SCHEDULE.md, README.md and docs/REPO_SPEC.md (default: repo root)")
    ap.add_argument("--repo-root", type=Path, default=None, help="checkout with configs/, scripts/ and src/ (default: this repo)")
    ap.add_argument("--only", action="append", default=None, metavar="CHECK", help="run only this check (repeatable): " + ", ".join(f.__name__[6:] for f in CHECKS))
    ap.add_argument("-v", "--verbose", action="store_true", help="also print the number of comparisons per check")
    args = ap.parse_args(argv)
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]  # section signs on cp1252 consoles
    root = args.repo_root or REPO_ROOT
    if not (root / "configs").is_dir():
        print(f"error: {root} has no configs/ directory", file=sys.stderr)
        return 2
    known = {f.__name__ for f in CHECKS}
    if args.only and any((n if n.startswith("check_") else f"check_{n}") not in known for n in args.only):
        print(f"error: --only takes one of {sorted(n[6:] for n in known)}", file=sys.stderr)
        return 2
    c = run_checks(args.docs_dir, root, args.only)
    print(format_report(c, args.verbose))
    return 1 if any(not m.accepted for m in c.mismatches) else 0


if __name__ == "__main__":
    sys.exit(main())
