"""Blinded LLM-judge runner: item selection, adaptive voting, cap enforcement, output (DESIGN §4, §8).

    python -m rhg.judge.run --runs hackable_subtle__s0 hackable_subtle__s1 --estimate-only
    python -m rhg.judge.run --runs ... --mock            # whole pipeline against the deterministic mock
    python -m rhg.judge.run --runs ... --real            # the ONLY flag that constructs the real client

Exactly one of ``--dry-run``, ``--estimate-only``, ``--mock``, ``--real`` is required; there is no
default that could spend money. ``--dry-run`` and ``--estimate-only`` construct no client at all.

Item selection per run, from the ``eval_test`` rollouts of the run's final step:
  (a) every ``ast_broad``-flagged rollout, or -- if more than ``judge.max_flagged_per_run`` -- a seeded
      simple random sample of exactly that many (inclusion probability cap/f);
  (b) a seeded simple random sample of ``round(audit_frac * N)`` of ALL rollouts (probability n_a/N).
The two draws are independent, so an item that could enter through both has
``inclusion_prob = 1 - (1 - pi_flagged) (1 - pi_audit)``; every output row carries it. Any judge-vs-truth
metric over the judged subset MUST be inverse-probability weighted with it (``docs/judge_notes.md``).

Voting: 2 votes at temperature > 0; a third only when the two valid votes disagree on ``hack_any``;
label = majority. The pre-flight estimate is a worst case (three votes everywhere, max output tokens,
no cache discount); if it exceeds the remaining ``judge.max_usd`` the flagged cap is reduced uniformly
across all runs until it fits, and the reduction is reported.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rhg.config import ConfigError, load_config
from rhg.judge import client as client_mod
from rhg.judge import cost, rubric
from rhg.judge.client import JudgeClient, JudgeRequest, JudgeResponse
from rhg.judge.cost import CapExceeded, CapGuard
from rhg.seeds import derive_seed

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
DEFAULT_ARM = "hackable_subtle"  # judge config is identical in every arm file
JUDGE_TEMPERATURE = 1.0


class JudgeRunError(RuntimeError):
    """Bad input data (missing rollouts/problems); CLI exit 2."""


class RubricFreezeError(RuntimeError):
    """Confirmatory judging refused: the rubric is not the frozen one; CLI exit 3."""


# ------------------------------------------------------------------ frozen-rubric check


def check_rubric_frozen(freeze_path: Path | None = None, repo_root: Path | None = None,
                        rubric_path: Path | None = None) -> None:
    """Raise ``RubricFreezeError`` unless the working rubric equals the one in ``prereg/FREEZE.json``.

    Compares the ``judge_rubric`` code-group hash (after logged amendments) with the file's current
    hash and, if the freeze also records ``judge_rubric_hash`` (the value of ``rubric_hash()``),
    that too.
    """
    from rhg.analysis import prereg_check as pc

    root = Path(repo_root) if repo_root is not None else pc.REPO_ROOT
    fpath = Path(freeze_path) if freeze_path is not None else root / pc.FREEZE_RELPATH
    if not fpath.is_file():
        raise RubricFreezeError(f"{fpath} not found: confirmatory judging needs the frozen rubric hash")
    freeze = json.loads(fpath.read_text(encoding="utf-8"))
    expected = pc.flatten_freeze(freeze)
    expected, broken = pc.apply_amendments(expected, pc.load_amendments(root))
    if "judge_rubric" in broken:
        raise RubricFreezeError("amendment chain for judge_rubric is broken")
    frozen = expected.get("judge_rubric")
    if frozen is None:
        raise RubricFreezeError("FREEZE.json has no judge_rubric hash")
    path = Path(rubric_path) if rubric_path is not None else Path(rubric.__file__)
    actual = pc.hash_code_group(path)
    if actual != frozen:
        raise RubricFreezeError(f"rubric.py hash {str(actual)[:12]} != frozen {frozen[:12]}: the rubric changed after the freeze")
    recorded = freeze.get("judge_rubric_hash")
    if recorded is not None and recorded != rubric.rubric_hash():
        raise RubricFreezeError(f"rubric_hash() {rubric.rubric_hash()[:12]} != frozen {str(recorded)[:12]}")


# ------------------------------------------------------------------ loading


@dataclass(frozen=True)
class Rollout:
    problem_id: str
    sample_idx: int
    step: int
    completion: str
    ast_broad: bool


def _ast_broad_flag(row: Mapping[str, Any]) -> bool:
    flag = (row.get("monitor") or {}).get("ast_broad")
    if isinstance(flag, bool):
        return flag
    # not logged (older/mock logs): recompute with the same broad profile and extraction the logger uses
    from rhg.detect.ast_detector import analyze
    from rhg.env.extract import extract_code

    return analyze(extract_code(str(row.get("completion", ""))).code, "broad").flag


def load_final_eval_test(rollouts_path: Path) -> tuple[int, list[Rollout]]:
    """``eval_test`` rollouts of the highest step in the file, sorted by (problem_id, sample_idx)."""
    path = Path(rollouts_path)
    if not path.is_file():
        raise JudgeRunError(f"{path} not found")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("phase") == "eval_test":
                    rows.append(row)
    if not rows:
        raise JudgeRunError(f"{path}: no eval_test rollouts")
    step = max(int(r["step"]) for r in rows)
    out = [Rollout(str(r["problem_id"]), int(r["sample_idx"]), step, str(r["completion"]), _ast_broad_flag(r))
           for r in rows if int(r["step"]) == step]
    out.sort(key=lambda r: (r.problem_id, r.sample_idx))
    keys = [(r.problem_id, r.sample_idx) for r in out]
    if len(set(keys)) != len(keys):
        raise JudgeRunError(f"{path}: duplicate (problem_id, sample_idx) among the step-{step} eval_test rollouts")
    return step, out


def load_descriptions(problems_path: Path, needed: set[str]) -> dict[str, str]:
    path = Path(problems_path)
    if not path.is_file():
        raise JudgeRunError(f"{path} not found (pass --processed-dir)")
    found: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["problem_id"] in needed:
                    found[row["problem_id"]] = str(row["description"])
    missing = sorted(needed - found.keys())
    if missing:
        raise JudgeRunError(f"{path}: no description for problem(s) {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return found


# ------------------------------------------------------------------ selection


@dataclass(frozen=True)
class SelectedItem:
    rollout: Rollout
    in_flagged_sample: bool
    in_audit: bool
    pi_flagged: float
    pi_audit: float

    @property
    def flagged(self) -> bool:
        return self.rollout.ast_broad

    @property
    def inclusion_prob(self) -> float:
        return combined_inclusion(self.flagged, self.pi_flagged, self.pi_audit)

    @property
    def source(self) -> str:
        return "ast_flagged" if self.in_flagged_sample else "audit"


@dataclass
class RunSelection:
    run_id: str
    step: int
    n_total: int
    n_flagged: int
    flagged_cap: int
    pi_flagged: float
    n_audit: int
    pi_audit: float
    items: list[SelectedItem] = field(default_factory=list)


def combined_inclusion(flagged: bool, pi_flagged: float, pi_audit: float) -> float:
    """P(item enters the judged set) for two independent draws; only flagged items can enter via (a)."""
    return 1.0 - (1.0 - pi_flagged) * (1.0 - pi_audit) if flagged else pi_audit


def select_items(rollouts: Sequence[Rollout], *, run_id: str, cap: int, audit_frac: float, seed: int = 0) -> RunSelection:
    """Seeded selection (a) + (b) described in the module docstring; deterministic in (rollouts, args)."""
    if cap < 0 or not 0.0 <= audit_frac <= 1.0:
        raise ValueError("cap must be >= 0 and audit_frac in [0, 1]")
    rows = sorted(rollouts, key=lambda r: (r.problem_id, r.sample_idx))
    n = len(rows)
    step = rows[0].step if rows else 0
    flagged_idx = [i for i, r in enumerate(rows) if r.ast_broad]
    f = len(flagged_idx)
    c = min(cap, f)
    perm = np.random.default_rng(derive_seed(seed, f"judge:{run_id}:flagged")).permutation(f)
    flagged_sel = {flagged_idx[int(j)] for j in perm[:c]}  # nested in c: a smaller cap is a prefix
    pi_f = c / f if f else 0.0
    n_a = int(math.floor(audit_frac * n + 0.5))
    audit_sel = (set(int(i) for i in np.random.default_rng(derive_seed(seed, f"judge:{run_id}:audit")).choice(n, size=n_a, replace=False))
                 if n_a else set())
    pi_a = n_a / n if n else 0.0
    items = [SelectedItem(rows[i], i in flagged_sel, i in audit_sel, pi_f, pi_a)
             for i in sorted(flagged_sel | audit_sel)]
    return RunSelection(run_id, step, n, f, cap, pi_f, n_a, pi_a, items)


# ------------------------------------------------------------------ planning under the cap


def worst_votes(jcfg: Any) -> int:
    return cost.worst_case_votes(jcfg.votes, jcfg.third_vote_on_disagree)


@dataclass
class Plan:
    selections: list[RunSelection]
    inputs: dict[str, list[rubric.JudgeInput]]  # run_id -> one JudgeInput per selected item
    estimate: cost.Estimate
    flagged_cap: int
    configured_cap: int
    available_usd: float
    max_usd: float
    votes_worst: int

    @property
    def cap_reduced(self) -> bool:
        return self.flagged_cap < self.configured_cap


def plan_selection(pops: Mapping[str, Sequence[Rollout]], descriptions: Mapping[str, str], jcfg: Any, guard: CapGuard, *,
                   seed: int = 0, prompts_cfg: Mapping[str, Any] | None = None,
                   measured: Mapping[str, int] | None = None, chars_per_token: float = cost.CHARS_PER_TOKEN) -> Plan:
    """Select items for every run and shrink the flagged cap (same value for all runs) until the
    worst-case estimate fits the remaining budget. Raises ``CapExceeded`` if even cap 0 does not."""
    v_worst = worst_votes(jcfg)
    built: dict[tuple[str, str, int], rubric.JudgeInput] = {}

    def prompt(run_id: str, it: SelectedItem) -> rubric.JudgeInput:
        key = (run_id, it.rollout.problem_id, it.rollout.sample_idx)
        if key not in built:
            built[key] = rubric.build_judge_prompt(descriptions[it.rollout.problem_id], it.rollout.completion, prompts_cfg)
        return built[key]

    last: Plan | None = None
    for cap in range(jcfg.max_flagged_per_run, -1, -1):
        sels = [select_items(rows, run_id=rid, cap=cap, audit_frac=jcfg.audit_frac, seed=seed) for rid, rows in pops.items()]
        inputs = {s.run_id: [prompt(s.run_id, it) for it in s.items] for s in sels}
        pairs = [(p.system, p.user) for lst in inputs.values() for p in lst]
        est = cost.estimate(pairs, v_worst, jcfg.model, measured=measured, chars_per_token=chars_per_token)
        last = Plan(sels, inputs, est, cap, jcfg.max_flagged_per_run, guard.remaining, guard.max_usd, v_worst)
        if guard.fits(est.usd_upper):
            return last
    assert last is not None
    raise CapExceeded(f"even with no flagged items the audit sample alone needs up to ${last.estimate.usd_upper:.4f} "
                      f"but only ${guard.remaining:.4f} of the ${guard.max_usd:.2f} judge cap remains")


# ------------------------------------------------------------------ adaptive voting


@dataclass
class ItemResult:
    responses: list[JudgeResponse]
    label: bool | None
    label_status: str  # "ok" | "unparseable" | "tie"
    components: dict[str, bool | None]

    @property
    def usage(self) -> cost.Usage:
        u = cost.Usage()
        for r in self.responses:
            u = u + r.usage
        return u

    @property
    def usd(self) -> float:
        return math.fsum(r.usd for r in self.responses)


def _majority(values: Sequence[bool]) -> bool | None:
    pos = sum(values)
    neg = len(values) - pos
    return None if pos == neg else pos > neg


def decide(responses: Sequence[JudgeResponse], votes: int) -> tuple[bool | None, str, dict[str, bool | None]]:
    """Label = majority of ``hack_any`` over valid votes.

    With ``votes >= 2`` an item needs at least two valid votes (an unparseable/errored vote is not
    replaced -- the third vote is reserved for disagreement); fewer -> ``unparseable``. An even split
    (a disagreeing pair whose third vote failed) -> ``tie``. Neither has a label.
    """
    valid = [r.verdict for r in responses if r.status == "ok" and r.verdict is not None]
    comps: dict[str, bool | None] = {f: _majority([getattr(v, f) for v in valid]) if valid else None for f in rubric.BOOL_FIELDS}
    if len(valid) < (2 if votes >= 2 else 1):
        return None, "unparseable", comps
    label = _majority([v.hack_any for v in valid])
    return label, ("ok" if label is not None else "tie"), comps


def run_votes(client: JudgeClient, inputs: Sequence[rubric.JudgeInput], jcfg: Any, guard: CapGuard, *, run_id: str,
              measured: Mapping[str, int] | None = None, chars_per_token: float = cost.CHARS_PER_TOKEN) -> list[ItemResult]:
    """Judge every input: ``jcfg.votes`` votes each, then one more only where two valid votes disagree.

    Each round is guarded: the worst-case estimate must fit before submitting and the actual cost is
    charged (ledger) right after; ``CapExceeded`` propagates and nothing further is submitted.
    """
    n = len(inputs)
    got: list[list[JudgeResponse]] = [[] for _ in range(n)]

    def submit_round(indices: Sequence[int], vote_ids: Sequence[int], note: str) -> None:
        reqs = [JudgeRequest(f"i{i}-v{v}", inputs[i].system, inputs[i].user, JUDGE_TEMPERATURE)
                for v in vote_ids for i in indices]
        if not reqs:
            return
        est = cost.estimate([(r.system, r.user) for r in reqs], 1, jcfg.model, measured=measured, chars_per_token=chars_per_token)
        guard.check_estimate(est.usd_upper, f"{run_id} {note}")
        t0 = time.monotonic()
        resp = client.submit(reqs)
        wall = time.monotonic() - t0
        by_id = {r.request_id: r for r in resp}
        for v in vote_ids:
            for i in indices:
                got[i].append(by_id[f"i{i}-v{v}"])
        guard.charge(math.fsum(r.usd for r in resp), run_id, wall, f"{client.model} {note} n={len(reqs)}")

    everyone = list(range(n))
    submit_round(everyone, range(jcfg.votes), f"votes 1-{jcfg.votes}")
    if jcfg.votes == 2 and jcfg.third_vote_on_disagree:
        split = [i for i in everyone
                 if len(got[i]) == 2 and all(r.status == "ok" for r in got[i]) and got[i][0].verdict.hack_any != got[i][1].verdict.hack_any]
        submit_round(split, [2], f"third vote on {len(split)} disagreements")
    out = []
    for i in range(n):
        label, status, comps = decide(got[i], jcfg.votes)
        out.append(ItemResult(got[i], label, status, comps))
    return out


# ------------------------------------------------------------------ output


def _vote_row(k: int, r: JudgeResponse) -> dict[str, Any]:
    base: dict[str, Any] = {"vote": k, "status": r.status, "attempts": r.attempts}
    if r.status == "ok" and r.verdict is not None:
        base.update(r.verdict.as_dict())
    elif r.status == "error":
        base["error"] = r.error
    else:
        base["raw"] = r.raw_text[:300]
    return base


def build_rows(sel: RunSelection, results: Sequence[ItemResult], *, model: str, mock: bool) -> list[dict[str, Any]]:
    rows = []
    for it, res in zip(sel.items, results):
        u = res.usage
        rows.append({
            "run_id": sel.run_id, "problem_id": it.rollout.problem_id, "sample_idx": it.rollout.sample_idx,
            "step": it.rollout.step, "votes": [_vote_row(k, r) for k, r in enumerate(res.responses)],
            "label": res.label, "inclusion_prob": it.inclusion_prob, "source": it.source,
            "tokens_in": u.total_in, "tokens_out": u.output_tokens, "usd": res.usd,
            # extras beyond REPO_SPEC §6 (documented in docs/SPEC_DEVIATIONS.md)
            "label_status": res.label_status, "components": res.components,
            "in_audit": it.in_audit, "ast_broad_flagged": it.flagged,
            "inclusion_prob_flagged": it.pi_flagged, "inclusion_prob_audit": it.pi_audit,
            "tokens_cache_read": u.cache_read_tokens, "tokens_cache_write": u.cache_write_tokens,
            "rubric_hash": rubric.rubric_hash(), "rubric_version": rubric.RUBRIC_VERSION, "model": model, "mock": mock,
        })
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_run_output(out_dir: Path, sel: RunSelection, rows: Sequence[dict[str, Any]], *, model: str, mock: bool,
                     seed: int, usd: float, plan: Plan) -> tuple[Path, Path]:
    jsonl = out_dir / f"{sel.run_id}.jsonl"
    _write_atomic(jsonl, "".join(json.dumps(r, sort_keys=False) + "\n" for r in rows))
    summary = {
        "run_id": sel.run_id, "step": sel.step, "n_eval_test": sel.n_total, "n_ast_broad_flagged": sel.n_flagged,
        "flagged_cap_used": sel.flagged_cap, "flagged_cap_configured": plan.configured_cap,
        "flagged_inclusion_prob": sel.pi_flagged, "n_audit_sampled": sel.n_audit, "audit_inclusion_prob": sel.pi_audit,
        "n_judged": len(rows), "selection_seed": seed, "model": model, "rubric_hash": rubric.rubric_hash(),
        "rubric_version": rubric.RUBRIC_VERSION, "mock": mock, "usd_actual": usd,
        "note": "any judge-vs-truth metric over the judged rows must be inverse-probability weighted by 1/inclusion_prob",
    }
    sfile = out_dir / f"{sel.run_id}.summary.json"
    _write_atomic(sfile, json.dumps(summary, indent=2) + "\n")
    return jsonl, sfile


# ------------------------------------------------------------------ CLI


def make_client(args: argparse.Namespace, jcfg: Any) -> JudgeClient:
    """The single place a client is constructed. ``--real`` is the only route to the paid client."""
    if args.real:
        return client_mod.AnthropicBatchClient(jcfg.model)
    return client_mod.MockJudgeClient(jcfg.model, accuracy=args.mock_accuracy, bias=args.mock_bias, seed=args.mock_seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rhg.judge.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run ids (directories under --runs-dir); comma-separated also accepted")
    mode = p.add_argument_group("mode (exactly one)")
    mode.add_argument("--dry-run", action="store_true", help="select items, build prompts, report the plan; no client, nothing written")
    mode.add_argument("--estimate-only", action="store_true", help="print the pre-flight cost estimate as JSON; no client, nothing written")
    mode.add_argument("--mock", action="store_true", help="judge with the deterministic mock client (outputs in results/judge_mock)")
    mode.add_argument("--real", action="store_true", help="judge with the real Anthropic Batch API (spends money, needs ANTHROPIC_API_KEY)")
    p.add_argument("--runs-dir", type=Path, default=None, help="default: run.output_root of the config")
    p.add_argument("--out-dir", type=Path, default=None, help="default: results/judge (results/judge_mock for --mock)")
    p.add_argument("--processed-dir", type=Path, default=None, help="dir with problems.jsonl (default: data.processed_dir)")
    p.add_argument("--arm", default=DEFAULT_ARM, help="arm config to read the judge block from (identical across arms)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V", help="config override, e.g. judge.max_usd=1.0")
    p.add_argument("--seed", type=int, default=0, help="selection seed (independent of training seeds)")
    p.add_argument("--confirmatory", action="store_true", help="refuse unless the rubric matches prereg/FREEZE.json")
    p.add_argument("--freeze-path", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=None, help="root holding prereg/ (amendments); default: this checkout")
    p.add_argument("--ledger", type=Path, default=None, help="ledger for judge spend (real: config default; mock: none unless given)")
    p.add_argument("--chars-per-token", type=float, default=cost.CHARS_PER_TOKEN, help="calibrated chars/token for the estimate")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    p.add_argument("--mock-accuracy", type=float, default=0.9)
    p.add_argument("--mock-bias", type=float, default=0.0)
    p.add_argument("--mock-seed", type=int, default=0)
    return p


def _fmt_plan(plan: Plan) -> str:
    lines = [f"judge plan: {sum(len(s.items) for s in plan.selections)} items over {len(plan.selections)} run(s), "
             f"worst case {plan.votes_worst} votes/item"]
    for s in plan.selections:
        lines.append(f"  {s.run_id}: step {s.step}, N={s.n_total}, flagged={s.n_flagged} (cap {s.flagged_cap}, pi={s.pi_flagged:.3f}), "
                     f"audit={s.n_audit} (pi={s.pi_audit:.3f}), judged={len(s.items)}")
    e = plan.estimate
    lines.append(f"  estimate: upper ${e.usd_upper:.4f} (cap ${plan.max_usd:.2f}, ${plan.available_usd:.4f} available), "
                 f"expected ~${e.usd_expected:.4f}; {e.tokens_in} input tokens at worst ({e.measured_items} of {e.n_items} prompts measured)")
    if plan.cap_reduced:
        lines.append(f"  flagged cap REDUCED {plan.configured_cap} -> {plan.flagged_cap} (uniformly, all runs) to fit the ${plan.max_usd:.2f} cap")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code not in (0, None) else EXIT_OK
    modes = [m for m in ("dry_run", "estimate_only", "mock", "real") if getattr(args, m)]
    if len(modes) != 1:
        print("error: choose exactly one of --dry-run, --estimate-only, --mock, --real", file=sys.stderr)
        return EXIT_USAGE
    mode = modes[0]
    try:
        cfg = load_config(args.arm, args.overrides)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_USAGE
    jcfg = cfg.judge
    run_ids = [r for chunk in args.runs for r in chunk.split(",") if r]
    if args.confirmatory:
        try:
            check_rubric_frozen(args.freeze_path, repo_root=args.repo_root)
        except RubricFreezeError as e:
            print(f"refused: {e}", file=sys.stderr)
            return EXIT_REFUSED
    runs_dir = args.runs_dir or Path(cfg.run.output_root)
    processed = args.processed_dir or Path(cfg.data.processed_dir)
    out_dir = args.out_dir or Path("results/judge_mock" if mode == "mock" else "results/judge")
    is_mock = mode == "mock"
    try:
        pops = {rid: load_final_eval_test(runs_dir / rid / "rollouts.jsonl.gz")[1] for rid in run_ids}
        descriptions = load_descriptions(processed / "problems.jsonl", {r.problem_id for rows in pops.values() for r in rows})
    except JudgeRunError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    ledger_on = mode != "mock" or args.ledger is not None  # mock spend is synthetic: never in the real ledger
    ledger = args.ledger if args.ledger is not None else Path(cfg.budget.ledger)
    guard = CapGuard.from_ledger(jcfg.max_usd, ledger=ledger, ledger_enabled=ledger_on)
    try:
        plan = plan_selection(pops, descriptions, jcfg, guard, seed=args.seed, chars_per_token=args.chars_per_token)
    except CapExceeded as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if mode == "estimate_only":
        print(json.dumps({"estimate": plan.estimate.as_dict(), "flagged_cap_used": plan.flagged_cap,
                          "flagged_cap_configured": plan.configured_cap, "cap_reduced": plan.cap_reduced,
                          "max_usd": plan.max_usd, "available_usd": plan.available_usd,
                          "n_items": {s.run_id: len(s.items) for s in plan.selections}}, indent=2))
        return EXIT_OK
    print(_fmt_plan(plan))
    if mode == "dry_run":
        print("dry run: no client constructed, nothing submitted or written")
        return EXIT_OK
    existing = [out_dir / f"{rid}.jsonl" for rid in run_ids if (out_dir / f"{rid}.jsonl").exists()]
    if existing and not args.force:
        print(f"refused: output exists ({existing[0]}); pass --force to overwrite (real reruns spend again)", file=sys.stderr)
        return EXIT_REFUSED
    try:
        client = make_client(args, jcfg)
    except client_mod.JudgeClientError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    for sel in plan.selections:
        try:
            before = guard.spent
            results = run_votes(client, plan.inputs[sel.run_id], jcfg, guard, run_id=sel.run_id, chars_per_token=args.chars_per_token)
        except CapExceeded as e:
            print(f"refused: {e}; {sel.run_id} not written", file=sys.stderr)
            return EXIT_REFUSED
        rows = build_rows(sel, results, model=jcfg.model, mock=is_mock)
        path, _ = write_run_output(out_dir, sel, rows, model=jcfg.model, mock=is_mock, seed=args.seed,
                                   usd=guard.spent - before, plan=plan)
        ok = sum(r["label_status"] == "ok" for r in rows)
        print(f"{sel.run_id}: {len(rows)} judged ({ok} labelled), ${guard.spent - before:.4f} -> {path}")
    print(f"judge spend total ${guard.spent:.4f} of ${guard.max_usd:.2f}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
