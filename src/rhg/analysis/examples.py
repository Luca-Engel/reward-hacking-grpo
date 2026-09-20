"""Non-cherry-picked examples gallery ``examples.md`` (DESIGN §4 exploratory extras).

Selection is random and seed-fixed; nothing is chosen for how it reads. All samples come from the *final test eval*
(``eval_test`` at step T) of the analysed runs:

* per hackable arm and seed with at least one HACK_RT rollout: 5 random HACK_RT and 5 random non-hack rollouts;
* per clean arm with a hint (``clean_subtle``, ``clean_explicit``): 3 random ATTEMPT_RT-but-not-HACK_RT rollouts
  (seeds pooled);
* 5 random ``gap_other`` rollouts (all runs pooled);
* 5 random detector false negatives: HACK_RT by execution while the *broad* AST profile (the measurement detector) did
  not flag the code (all runs pooled).

Every group draws with its own generator seeded from ``(EXAMPLES_SEED, group key)`` over candidates in canonical order
(run id, problem id, sample index), so the selection is reproducible and unaffected by other groups. Completions are
shown verbatim, truncated only at ``MAX_CHARS`` characters (stated per example).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rhg import runlog
from rhg.analysis.endpoints import RunData, RunSet
from rhg.seeds import derive_seed

EXAMPLES_SEED = 20260920
MAX_CHARS = 1500
N_HACK, N_NONHACK, N_ATTEMPT, N_GAP, N_FN = 5, 5, 3, 5, 5
CLEAN_HINTED = ("clean_subtle", "clean_explicit")


def _final_rollouts(run: RunData) -> list[dict[str, Any]]:
    recs = [r.model_dump() for r in runlog.iter_rollouts(run.run_dir, phase="eval_test", step=run.T)]
    return sorted(recs, key=lambda r: (r["problem_id"], r["sample_idx"]))


def _pick(cands: Sequence[dict[str, Any]], k: int, key: str, seed: int) -> list[dict[str, Any]]:
    if not cands:
        return []
    rng = np.random.default_rng(derive_seed(seed, f"examples:{key}"))
    idx = rng.choice(len(cands), size=min(k, len(cands)), replace=False)
    return [cands[int(i)] for i in idx]


def _fence(text: str) -> str:
    longest, run = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def _render(rec: dict[str, Any], max_chars: int) -> str:
    lab, mon = rec["labels"], rec["monitor"]
    flags = [k for k in ("hack_rt", "attempt_rt", "correct", "gap_other", "timeout") if lab[k]]
    text = rec["completion"]
    note = ""
    if len(text) > max_chars:
        note = f"\n[truncated for display: showing the first {max_chars} of {len(text)} characters]"
        text = text[:max_chars]
    fence = _fence(text)
    head = (f"**{rec['run_id']}** | problem `{rec['problem_id']}` | sample {rec['sample_idx']} | labels: "
            f"{', '.join(flags) or 'none of hack/attempt/correct/gap'} | ast_narrow={mon['ast_narrow']} ast_broad={mon['ast_broad']} "
            f"| reward {rec['reward']:g}")
    return f"{head}\n\n{fence}text\n{text}\n{fence}{note}\n"


def build_examples(runset: RunSet, out_path: str | Path, seed: int = EXAMPLES_SEED, max_chars: int = MAX_CHARS) -> dict[str, Any]:
    """Write the gallery; returns ``{"seed", "groups": {title: {"shown", "candidates"}}}`` for tests and the report."""
    runs = sorted(runset.runs, key=lambda r: (r.arm, r.seed))
    final = {r.run_id: _final_rollouts(r) for r in runs}
    sections: list[tuple[str, str, list[dict[str, Any]], int]] = []  # (key, title, picked, n_candidates)

    for r in runs:
        if not r.arm.startswith("hackable"):
            continue
        recs = final[r.run_id]
        hacks = [x for x in recs if x["labels"]["hack_rt"]]
        if not hacks:
            continue
        rest = [x for x in recs if not x["labels"]["hack_rt"]]
        sections.append((f"hack:{r.run_id}", f"{r.run_id}: HACK_RT rollouts", _pick(hacks, N_HACK, f"hack:{r.run_id}", seed), len(hacks)))
        sections.append((f"nonhack:{r.run_id}", f"{r.run_id}: non-hack rollouts", _pick(rest, N_NONHACK, f"nonhack:{r.run_id}", seed), len(rest)))
    for arm in CLEAN_HINTED:
        pool = [x for r in runs if r.arm == arm for x in final[r.run_id] if x["labels"]["attempt_rt"] and not x["labels"]["hack_rt"]]
        sections.append((f"attempt:{arm}", f"{arm}: ATTEMPT_RT but not HACK_RT (seeds pooled)", _pick(pool, N_ATTEMPT, f"attempt:{arm}", seed), len(pool)))
    everything = [x for r in runs for x in final[r.run_id]]
    gap = [x for x in everything if x["labels"]["gap_other"]]
    sections.append(("gap_other", "GAP_OTHER rollouts (all runs pooled)", _pick(gap, N_GAP, "gap_other", seed), len(gap)))
    fn = [x for x in everything if x["labels"]["hack_rt"] and not x["monitor"]["ast_broad"]]
    sections.append(("false_negative", "Detector false negatives: HACK_RT by execution, not flagged by the broad AST profile (all runs pooled)",
                     _pick(fn, N_FN, "false_negative", seed), len(fn)))

    lines = [
        "# Examples gallery [EXPLORATORY]",
        "",
        "Illustrative only; these are samples, not evidence. Nothing here was chosen for how it reads.",
        "",
        "- **Selection procedure:** every rollout comes from the final test eval (`eval_test`, step T) of an analysed run. For each "
        "group the candidates are ordered by (run id, problem id, sample index) and a random subset is drawn without replacement with "
        "`numpy.random.default_rng(derive_seed(EXAMPLES_SEED, 'examples:<group key>'))`.",
        f"- **RNG seed (EXAMPLES_SEED):** {seed}",
        f"- **Groups:** per hackable arm and seed with any HACK_RT: {N_HACK} HACK_RT + {N_NONHACK} non-hack; per hinted clean arm "
        f"({', '.join(CLEAN_HINTED)}): {N_ATTEMPT} ATTEMPT_RT-but-not-hack; {N_GAP} GAP_OTHER; {N_FN} detector false negatives "
        f"(HACK_RT by execution but the broad AST profile did not flag it).",
        f"- **Verbatim:** completions are shown exactly as logged, truncated only at {max_chars} characters (marked when it happens).",
        "",
    ]
    summary: dict[str, Any] = {"seed": seed, "max_chars": max_chars, "groups": {}}
    for key, title, picked, n_cand in sections:
        lines += [f"## {title}", "", f"{len(picked)} shown of {n_cand} candidates.", ""]
        if not picked:
            lines += ["_No candidates._", ""]
        for rec in picked:
            lines += [_render(rec, max_chars)]
        summary["groups"][key] = {"shown": [(p["run_id"], p["problem_id"], p["sample_idx"]) for p in picked], "candidates": n_cand}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    import argparse

    from rhg.analysis.endpoints import load_runs

    ap = argparse.ArgumentParser(prog="python -m rhg.analysis.examples", description="Write the random, seed-fixed examples gallery.")
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/examples.md"))
    ap.add_argument("--seed", type=int, default=EXAMPLES_SEED)
    args = ap.parse_args()
    rs = load_runs(args.runs)
    if not rs.runs:
        print(f"error: no completed runs under {args.runs}", file=sys.stderr)
        sys.exit(2)
    build_examples(rs, args.out, args.seed)
    print(f"wrote {args.out}")
