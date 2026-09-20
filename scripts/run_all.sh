#!/usr/bin/env bash
# Main runs on one box. Usage: scripts/run_all.sh [--shard I/N] [--ladder N] [--stop-at USD] [--stale-after S] [--dry-run]
# Takes the next unstarted run of this shard from `rhg.plan next` (priority order, resume-safe: completed runs are skipped),
# checks the budget guard (`rhg.budget check --next-run-usd`, priced from results/bench/throughput.json = the numbers behind
# BUDGET_MEASURED.md; without a bench the UNVERIFIED planning-prior ceiling), launches scripts/run_arm.sh (--confirmatory), and
# continues past failures. Exit 75 (stall watchdog), OOM and other invalid/failed runs are handled by `rhg.plan replacement`
# (seeds 100+k, hard cap 3 in total across boxes: shard i of n uses k = i-1 mod n; validity only, never outcomes).
# Refuses to launch unless the prereg-v1 tag is an ancestor of HEAD (with --dry-run this is only a warning).
# All boxes must use the same --ladder N (BUDGET §4). --stale-after 0 marks every 'running' run invalid at start (only when no
# other process trains on this box); the default treats a status not rewritten for 30 min as a killed/preempted run.
# Writes results/RUNS_HEALTH.md: status/steps/wall/usd/final TRAIN reward per run. Nobody looks at hack rate by arm.
# Gate 2a (after the first 2 primary pairs) and 2b (60% of runs or budget): read RUNS_HEALTH.md and the ledger only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SHARD="1/1"
LADDER=0
STOP_AT="28.0"
STALE=1800
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --shard) SHARD="${2:?--shard needs I/N}"; shift ;;
    --ladder) LADDER="${2:?--ladder needs a value}"; shift ;;
    --stop-at) STOP_AT="${2:?--stop-at needs a value}"; shift ;;
    --stale-after) STALE="${2:?--stale-after needs seconds}"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done
[[ "$SHARD" =~ ^[0-9]+/[0-9]+$ ]] || die_usage "--shard must look like I/N (1-based), got '$SHARD'"
[[ "$LADDER" =~ ^[0-6]$ ]] || die_usage "--ladder must be 0..6, got '$LADDER'"

require_uv
drop_api_key
PLAN_ARGS=(--shard "$SHARD" --ladder "$LADDER")
HEALTH=results/RUNS_HEALTH.md

if prereg_tag_ancestor; then
  log "prereg-v1 is an ancestor of HEAD: ok"
elif [[ "$DRY_RUN" == 1 ]]; then
  warn "prereg-v1 is not an ancestor of HEAD: a real run would REFUSE here (dry-run continues)"
else
  refuse "the git tag prereg-v1 is not an ancestor of HEAD. Freeze with scripts/freeze_prereg.sh, push the commit and the tag, and set this box up from a checkout that contains the tag."
fi
if ! gpu_present; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "no NVIDIA GPU detected; continuing because of --dry-run"; else die "no NVIDIA GPU detected"; fi
fi

if [[ "$DRY_RUN" == 1 ]]; then
  COST="$(py -m rhg.plan next-run-usd 2>/dev/null | tr -d '\r' || echo '?')"
  TOTAL="$(py -m rhg.plan list "${PLAN_ARGS[@]}" --format ids | wc -l | tr -d ' ')"
  log "shard $SHARD, ladder step $LADDER: $TOTAL run(s) in launch order (states: completed runs are skipped when run for real)"
  N=0
  while IFS=$'\t' read -r PRIO TIER ARM SEED RID STATE; do
    N=$((N + 1))
    printf '[dry-run] RUN %d/%s %s (tier %s, priority %s, state %s)\n' "$N" "$TOTAL" "$RID" "$TIER" "$PRIO" "$STATE"
    printf '[dry-run]     python -m rhg.budget check --next-run-usd %s --stop-at %s\n' "$COST" "$STOP_AT"
    printf '[dry-run]     bash scripts/run_arm.sh %s %s\n' "$ARM" "$SEED"
  done < <(py -m rhg.plan list "${PLAN_ARGS[@]}" --format tsv | tr -d '\r')
  printf '[dry-run] after each run: rhg.plan replacement --apply (infrastructure failures only), rhg.plan health --out %s\n' "$HEALTH"
  exit 0
fi

mkdir -p results/logs
LOG="results/logs/run_all_${SHARD//\//of}.log"
exec > >(tee -a "$LOG") 2>&1
log "shard $SHARD ladder step $LADDER stop-at \$$STOP_AT; log: $LOG"

declare -A LAUNCHED=()
N_OK=0
N_BAD=0
STOP=""
EXIT_CODE=0
while true; do
  py -m rhg.plan reconcile "${PLAN_ARGS[@]}" --stale-after "$STALE" || true
  py -m rhg.plan replacement "${PLAN_ARGS[@]}" --apply || true
  LINE="$(py -m rhg.plan next "${PLAN_ARGS[@]}" | tr -d '\r')"
  if [[ -z "$LINE" ]]; then
    log "queue empty: every run of this shard has started"
    break
  fi
  IFS=$'\t' read -r PRIO TIER ARM SEED RID _STATE <<<"$LINE"
  if [[ -n "${LAUNCHED[$RID]:-}" ]]; then
    STOP="$RID was launched but left no status.json; stopping to avoid a loop (inspect results/runs/$RID)"
    EXIT_CODE=1
    break
  fi
  COST="$(py -m rhg.plan next-run-usd --arm "$ARM" | tr -d '\r')" || { STOP="cannot price the next run (mock bench file?)"; EXIT_CODE=3; break; }
  if ! py -m rhg.budget check --next-run-usd "$COST" --stop-at "$STOP_AT"; then
    STOP="budget guard refused the next run ($RID, projected \$$COST)"
    EXIT_CODE=3
    break
  fi
  log "launching $RID (tier $TIER, priority $PRIO), projected \$$COST"
  LAUNCHED[$RID]=1
  START=$SECONDS
  RC=0
  bash "$_COMMON_DIR/run_arm.sh" "$ARM" "$SEED" || RC=$?
  WALL=$((SECONDS - START))
  case "$RC" in
    0)
      N_OK=$((N_OK + 1))
      log "$RID completed in ${WALL}s"
      ;;
    3)
      STOP="$RID: guard refused (budget or prereg check inside rhg.train.run, exit 3)"
      EXIT_CODE=3
      ;;
    2)
      STOP="$RID: usage/config error (exit 2); it would repeat for every run"
      EXIT_CODE=2
      ;;
    75)
      N_BAD=$((N_BAD + 1))
      warn "$RID: stall watchdog (exit 75) = infrastructure failure, eligible for replacement"
      ;;
    *)
      N_BAD=$((N_BAD + 1))
      warn "$RID: exit $RC (see results/runs/$RID/status.json; OOM and NaN runs are infrastructure failures, eligible for replacement)"
      ;;
  esac
  if [[ "$RC" != 0 ]]; then
    py -m rhg.plan record-missing --run-id "$RID" --arm "$ARM" --wall-s "$WALL" || warn "could not check the ledger entry of $RID"
  fi
  py -m rhg.plan health "${PLAN_ARGS[@]}" --out "$HEALTH" >/dev/null || warn "could not write $HEALTH"
  grep -F "| $RID" "$HEALTH" || true
  if [[ -n "$STOP" ]]; then break; fi
done

py -m rhg.plan health "${PLAN_ARGS[@]}" --out "$HEALTH" || true
py -m rhg.budget status || true
if [[ -n "$STOP" ]]; then
  warn "stopped: $STOP"
  py -m rhg.plan replacement "${PLAN_ARGS[@]}" || true
fi
log "done: $N_OK completed, $N_BAD failed/invalid this session. Dashboard: $HEALTH. Next: scripts/package_results.sh"
exit "$EXIT_CODE"
