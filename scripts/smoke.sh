#!/usr/bin/env bash
# Gate 1a: 5 real GRPO steps of hackable_subtle end to end with a tiny eval. Usage: scripts/smoke.sh [--dry-run] [--seed N]
# Output goes to results/smoke/ (never results/runs) and is billed to the ledger as kind "pilot", so it can never be
# mistaken for a main run. Prints per-phase timings and the Gate 1a verdict; exits 1 on NO-GO.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SEED=9900
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --seed) SEED="${2:?--seed needs a value}"; shift ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

require_uv
drop_api_key
gate_criterion 1a "no hang, finite reward, <= 15 min wall, per-phase timings logged (5 steps of hackable_subtle)"
if ! gpu_present; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "no NVIDIA GPU detected; continuing because of --dry-run"; else die "no NVIDIA GPU detected"; fi
fi

OUT=results/smoke
RUN_DIR="$OUT/hackable_subtle__s$SEED"
START=$SECONDS
RC=0
run_py -m rhg.train.run --arm hackable_subtle --seed "$SEED" --steps 5 --backend trl --pilot --force \
  --set run.tag=smoke --set "run.output_root=$OUT" \
  --set eval.val_samples_per_problem=1 --set eval.test_samples_per_problem=1 --set eval.xhint_samples_per_problem=1 || RC=$?
ELAPSED=$((SECONDS - START))
log "rhg.train.run exit code $RC after ${ELAPSED}s"

if [[ "$DRY_RUN" == 1 ]]; then
  run_py -m rhg.gates smoke --run-dir "$RUN_DIR" --elapsed-s "$ELAPSED"
  exit 0
fi
GATE_RC=0
py -m rhg.gates smoke --run-dir "$RUN_DIR" --elapsed-s "$ELAPSED" || GATE_RC=$?
if [[ "$RC" != 0 || "$GATE_RC" != 0 ]]; then
  nogo 1a "train exit code $RC, gate check exit code $GATE_RC (see $RUN_DIR/stdout.log)"
fi
go 1a "next: scripts/bench_throughput.sh --usd-per-hour RATE --n-boxes N"
