#!/usr/bin/env bash
# Gate 1c: base-model pass rates (stage A, stage B) and the problem split. Usage:
#   scripts/measure_pass_rate.sh [--generate-only] [--widen] [--dry-run]
# Default (all on the box): A -> `split --select-only` -> B -> `split --strict` (prints the Gate 1c checklist), then checks that
# the labeler agrees with the hand-built synthetic controls.
# --generate-only: the box only generates (no grading CPU work). Stage B needs the band selection made from graded stage A, so
# it takes two rounds: (1) box: this script generates A, you grade A locally, select, copy data/processed back; (2) box: this
# script generates B (it detects selected_A.json), you grade B and split locally. The exact commands are printed each time.
# --widen applies the single pre-declared band widening [0.05, 0.50] (SCHEDULE Gate 1c).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GENERATE_ONLY=0
WIDEN=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --generate-only) GENERATE_ONLY=1 ;;
    --widen) WIDEN=(--widen) ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

require_uv
drop_api_key
gate_criterion 1c ">= 150 train / >= 40 val / >= 60 test problems, >= 95% of references valid, and the labeler agrees with the synthetic controls"
if ! gpu_present; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "no NVIDIA GPU detected; continuing because of --dry-run"; else die "no NVIDIA GPU detected"; fi
fi
PROCESSED=data/processed
LABELER_TESTS='controls_agree or controls_cannot_drift'

if [[ "$GENERATE_ONLY" == 1 ]]; then
  if [[ ! -f "$PROCESSED/selected_A.json" ]]; then
    run_py -m rhg.eval.pass_rate --stage A --generate-only
    cat <<EOT

Stage A completions written. Grade and select LOCALLY (data/processed must be the one built on the box: copy it):
  rsync -av USER@BOX:$REPO_ROOT/data/processed/ data/processed/
  python -m rhg.eval.pass_rate --stage A --grade-only
  python -m rhg.data.build --stage split --select-only ${WIDEN[*]:-}
  rsync -av data/processed/selected_A.json data/processed/passrate_A.jsonl USER@BOX:$REPO_ROOT/data/processed/
Then re-run on the box:  scripts/measure_pass_rate.sh --generate-only ${WIDEN[*]:-}
EOT
  else
    run_py -m rhg.eval.pass_rate --stage B --generate-only
    cat <<EOT

Stage B completions written. Grade and split LOCALLY:
  rsync -av USER@BOX:$REPO_ROOT/data/processed/completions_B.jsonl.gz data/processed/
  python -m rhg.eval.pass_rate --stage B --grade-only
  python -m rhg.data.build --stage split --strict ${WIDEN[*]:-}     # prints the Gate 1c checklist, exit 1 = NO-GO
  python -m pytest -q tests/test_validate.py -k "$LABELER_TESTS"   # labeler vs synthetic controls
EOT
  fi
  exit 0
fi

run_py -m rhg.eval.pass_rate --stage A
run_py -m rhg.data.build --stage split --select-only "${WIDEN[@]}"
run_py -m rhg.eval.pass_rate --stage B
SPLIT_RC=0
run_py -m rhg.data.build --stage split --strict "${WIDEN[@]}" || SPLIT_RC=$?
LABELER_RC=0
run_py -m pytest -q tests/test_validate.py -k "$LABELER_TESTS" || LABELER_RC=$?

if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
if [[ "$SPLIT_RC" != 0 ]]; then
  nogo 1c "the split / counts / reference-validity checklist above failed (exit $SPLIT_RC)"
fi
if [[ "$LABELER_RC" != 0 ]]; then
  nogo 1c "the labeler does not agree with the synthetic controls (exit $LABELER_RC)"
fi
go 1c "next: scripts/probe_hints.sh"
