#!/usr/bin/env bash
# One main run. Usage: scripts/run_arm.sh ARM SEED [--exploratory] [--force] [--dry-run] [-- extra rhg.train.run args]
# Launches `rhg.train.run --arm ARM --seed SEED --backend trl --confirmatory` (clean tree, prereg-v1 an ancestor of HEAD and
# matching hashes are enforced by the trainer). --exploratory drops --confirmatory (manual runs; the results are then labelled
# EXPLORATORY). Seeds >= 9000 are pilots and may not be run confirmatorily. The exit code of the trainer is passed through:
# 0 ok, 1 failed/invalid, 2 usage, 3 guard refused, 75 stall watchdog (infrastructure).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ARMS="clean_none clean_subtle clean_explicit hackable_none hackable_subtle hackable_explicit hackable_subtle_ast"
CONFIRMATORY=(--confirmatory)
FORCE=()
POS=()
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --exploratory) CONFIRMATORY=() ;;
    --force) FORCE=(--force) ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    -*) die_usage "unknown argument: $1" ;;
    *) POS+=("$1") ;;
  esac
  shift
done
[[ ${#POS[@]} -eq 2 ]] || die_usage "expected ARM SEED (arms: $ARMS)"
ARM="${POS[0]}"
SEED="${POS[1]}"
case " $ARMS " in *" $ARM "*) ;; *) die_usage "unknown arm '$ARM' (arms: $ARMS)" ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || die_usage "seed must be a non-negative integer, got '$SEED'"
if [[ ${#CONFIRMATORY[@]} -gt 0 && "$SEED" -ge 9000 ]]; then
  die_usage "seed $SEED >= 9000 is a pilot seed: pilots are never confirmatory (use scripts/pilot.sh)"
fi

require_uv
drop_api_key
if ! gpu_present; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "no NVIDIA GPU detected; continuing because of --dry-run"; else die "no NVIDIA GPU detected"; fi
fi

RC=0
run_py -m rhg.train.run --arm "$ARM" --seed "$SEED" --backend trl "${CONFIRMATORY[@]}" "${FORCE[@]}" "${EXTRA[@]}" || RC=$?
exit "$RC"
