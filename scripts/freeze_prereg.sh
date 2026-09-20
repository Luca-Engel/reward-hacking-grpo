#!/usr/bin/env bash
# Gate 1f: freeze the pre-registration. Usage: scripts/freeze_prereg.sh [--gpu-type NAME] [--dry-run]
# Refuses unless EVERY pre-freeze gate left its artifact (Gates 1b-1e2: prereg/hint_selection.json, prereg/budget_decision.md,
# prereg/pilot_gate.json with pass true and the frozen lr, data/processed/splits.json, results/analysis/judge_calibration.json
# with passed true and the current rubric hash), unless the tag prereg-v1 exists already, and unless the tree is clean apart
# from prereg/ (commit BUDGET_MEASURED.md and the prompts/config edits first).
# Then: writes prereg/FREEZE.json via rhg.analysis.prereg_check, commits prereg/ with a fixed message, tags prereg-v1, verifies.
# It does NOT push. Publishing is a precondition of --confirmatory runs: see the commands printed at the end.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GPU_TYPE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --gpu-type) GPU_TYPE="${2:?--gpu-type needs a value}"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

require_uv
gate_criterion 1f "prereg_check passes AND the commit and tag are pushed to your public remote (git push origin HEAD; git push origin prereg-v1)"

PREREQ_RC=0
py -m rhg.gates freeze-prereqs --repo-root "$REPO_ROOT" || PREREQ_RC=$?
if [[ "$PREREQ_RC" != 0 ]]; then
  refuse "freeze prerequisites are missing (list above). Nothing was written, committed or tagged."
fi

if [[ -z "$GPU_TYPE" ]]; then
  GPU_TYPE="$(py -c "import json; v = json.load(open('prereg/pilot_gate.json')).get('values', {}); print(v.get('gpu_name') or 'unknown')" 2>/dev/null || echo unknown)"
fi
log "GPU type recorded in FREEZE.json: $GPU_TYPE"

PUSH_HELP="Publish the freeze BEFORE any main run (the public timestamp is what makes the pre-registration third-party verifiable):
  git push origin HEAD
  git push origin prereg-v1
The GPU boxes must be set up (git clone / git pull --tags) from a checkout that contains the pushed prereg-v1 tag.
scripts/run_all.sh refuses to launch unless prereg-v1 is an ancestor of HEAD, so pushing the tag first is a precondition of every --confirmatory run."

if [[ "$DRY_RUN" == 1 ]]; then
  run_py -m rhg.analysis.prereg_check --repo-root "$REPO_ROOT" --write-freeze --gpu-type "$GPU_TYPE"
  run git add prereg
  run git commit -m "Freeze pre-registration (prereg-v1)" -- prereg
  run git tag -a prereg-v1 -m "Pre-registration v1 (Gate 1f)"
  run_py -m rhg.analysis.prereg_check --repo-root "$REPO_ROOT"
  printf '\n%s\n' "$PUSH_HELP"
  exit 0
fi

py -m rhg.analysis.prereg_check --repo-root "$REPO_ROOT" --write-freeze --gpu-type "$GPU_TYPE"
git add prereg
git commit -m "Freeze pre-registration (prereg-v1)" -- prereg
git tag -a prereg-v1 -m "Pre-registration v1 (Gate 1f)"
if ! py -m rhg.analysis.prereg_check --repo-root "$REPO_ROOT"; then
  printf '\nprereg_check FAILED right after tagging. Nothing has been pushed. To undo locally:\n  git tag -d prereg-v1 && git reset --soft HEAD~1\n' >&2
  exit "$EXIT_NOGO"
fi
go 1f "locally frozen and verified; it is not final until you push"
printf '\n%s\n' "$PUSH_HELP"
