#!/usr/bin/env bash
# Gate 1b: throughput bench -> cost model. Usage:
#   scripts/bench_throughput.sh --usd-per-hour RATE [--n-boxes N] [--steps 8] [--force] [--dry-run]
# Runs `rhg.eval.bench` (writes results/bench/throughput.json), then `rhg.budget cost_model` (writes BUDGET_MEASURED.md
# and the DRAFT prereg/budget_decision.md). RATE is the rental rate of the box you are actually paying for. Also set
# `budget.usd_per_hour` to it in configs/base.yaml (and commit) BEFORE the freeze: manifests, the ledger and the
# config hash use the config value. Commit BUDGET_MEASURED.md too (the freeze needs a clean tree).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RATE=""
NBOXES=1
STEPS=8
FORCE=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --usd-per-hour) RATE="${2:?--usd-per-hour needs a value}"; shift ;;
    --n-boxes) NBOXES="${2:?--n-boxes needs a value}"; shift ;;
    --steps) STEPS="${2:?--steps needs a value}"; shift ;;
    --force) FORCE=(--force) ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done
if [[ -z "$RATE" ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then RATE="RATE"; else die_usage "--usd-per-hour is required (a wrong rate silently changes the Gate 1b decision)"; fi
fi

require_uv
drop_api_key
gate_criterion 1b "main_usd <= 16 and wall_h <= 12 at 22 runs (with --n-boxes boxes), else apply the cut ladder; NO-GO if even 11 runs do not fit"
if ! gpu_present; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "no NVIDIA GPU detected; continuing because of --dry-run"; else die "no NVIDIA GPU detected"; fi
fi
log "note: also set budget.usd_per_hour: $RATE in configs/base.yaml before the freeze (the config hash includes it)"

run_py -m rhg.eval.bench --steps "$STEPS" --set "budget.usd_per_hour=$RATE" "${FORCE[@]}"
if [[ "$DRY_RUN" == 1 ]]; then
  run_py -m rhg.budget cost_model --usd-per-hour "$RATE" --n-boxes "$NBOXES" "${FORCE[@]}"
  exit 0
fi
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT
RC=0
py -m rhg.budget cost_model --usd-per-hour "$RATE" --n-boxes "$NBOXES" "${FORCE[@]}" | tee "$OUT" || RC=${PIPESTATUS[0]}
if [[ "$RC" != 0 ]]; then
  nogo 1b "rhg.budget cost_model exited $RC"
fi
if grep -q "NO ladder step fits" "$OUT"; then
  nogo 1b "not even the 11-run floor fits"
elif grep -q "first fitting ladder step: 0 " "$OUT"; then
  go 1b "the full 22-run design fits; review prereg/budget_decision.md (DRAFT), commit it and BUDGET_MEASURED.md"
else
  go 1b "the full design does NOT fit; the ladder step above is the pre-declared cut. Review prereg/budget_decision.md (DRAFT), commit it and BUDGET_MEASURED.md, and pass --ladder N to run_all.sh"
fi
