#!/usr/bin/env bash
# Day 3: run the blinded LLM judge over the completed runs. LOCAL machine only. Usage:
#   scripts/judge_all.sh [--yes] [--dry-run]
# Refuses on a machine with an NVIDIA driver and without ANTHROPIC_API_KEY. First runs `rhg.judge.run --estimate-only` (Gate 3a:
# pre-flight cost from measured token counts <= $4, else the code subsamples flagged items with recorded inclusion weights),
# and only with --yes runs the real judge (`--real`; `--confirmatory` iff prereg_check passes: without it labels are EXPLORATORY).
# Runs judged = every completed run of the plan (incl. granted replacements) found in results/runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes) YES=1 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

require_uv
gate_criterion 3a "pre-flight judge cost from measured token counts <= \$4 (else the judge subsamples flagged items with weights)"
require_local_machine
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "ANTHROPIC_API_KEY is not set; a real run would refuse (dry-run continues)"; else refuse "ANTHROPIC_API_KEY is not set (export it in this local shell; the key never goes on a GPU box)"; fi
fi

RUNS=()
while IFS= read -r rid; do
  if [[ -n "$rid" ]]; then RUNS+=("$rid"); fi
done < <(py -m rhg.plan list --state completed --format ids 2>/dev/null | tr -d '\r' || true)
if [[ ${#RUNS[@]} -eq 0 ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    warn "no completed run in results/runs yet; using a placeholder id"
    RUNS=(RUN_ID)
  else
    die "no completed run found in results/runs (unpack the output of scripts/package_results.sh first)"
  fi
fi
log "${#RUNS[@]} completed run(s) to judge"

CONF=()
if prereg_check_ok; then
  CONF=(--confirmatory)
  log "prereg_check passes: judging with --confirmatory"
else
  warn "prereg_check does not pass: judge labels will be EXPLORATORY"
fi

run_py -m rhg.judge.run --runs "${RUNS[@]}" --estimate-only "${CONF[@]}"
if [[ "$YES" != 1 && "$DRY_RUN" != 1 ]]; then
  refuse "estimate above; nothing was sent. Re-run with --yes to spend it (the hard cap judge.max_usd = \$4 is enforced in code)."
fi
run_py -m rhg.judge.run --runs "${RUNS[@]}" --real "${CONF[@]}"
if [[ "$DRY_RUN" != 1 ]]; then log "done. Next: human labels (python -m rhg.validate.label), then scripts/analyze.sh"; fi
