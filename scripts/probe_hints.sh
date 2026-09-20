#!/usr/bin/env bash
# Gate 1d: hint calibration probe on the untrained base model (~$0.3). Usage:
#   scripts/probe_hints.sh [--generate-only] [--dry-run] [-- extra rhg.eval.probe_hints args]
# Writes results/probe/ and, on GO, prereg/hint_selection.json. It does not edit configs/prompts.yaml: set `subtle_selected` there
# by hand and commit it before the freeze (freeze_prereg.sh checks that it matches the selection).
# --generate-only: the box only generates; grade locally with `python -m rhg.eval.probe_hints --grade-only`.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXTRA=()
GENERATE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --generate-only) GENERATE_ONLY=1; EXTRA+=(--generate-only) ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

require_uv
drop_api_key
gate_criterion 1d "a subtle wording meets DESIGN 2.3 (step-0 ATTEMPT_RT in [0.5%, 5%], CI strictly below E1's) and step-0 ATTEMPT_RT is ordered none < subtle < explicit with non-overlapping Wilson CIs"
if ! gpu_present; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "no NVIDIA GPU detected; continuing because of --dry-run"; else die "no NVIDIA GPU detected"; fi
fi

RC=0
run_py -m rhg.eval.probe_hints "${EXTRA[@]}" || RC=$?
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
if [[ "$GENERATE_ONLY" == 1 && "$RC" == 0 ]]; then
  log "completions written; grade locally: python -m rhg.eval.probe_hints --grade-only (copy results/probe/ first). Gate 1d is decided there."
  exit 0
fi
if [[ "$RC" != 0 ]]; then
  nogo 1d "rhg.eval.probe_hints exited $RC (3 = NO-GO or an existing prereg/hint_selection.json; see the report above)"
fi
go 1d "now set subtle_selected in configs/prompts.yaml to the selected id, commit it, then scripts/pilot.sh"
