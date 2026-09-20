#!/usr/bin/env bash
# Day 3: validation harness, then the single analysis run. Usage: scripts/analyze.sh [--dry-run] [--mock-validation]
# `--confirmatory` is passed to `rhg.analysis.run` only if `python -m rhg.analysis.prereg_check` passes; otherwise the output is
# stamped EXPLORATORY. Needs the judge outputs (scripts/judge_all.sh) and the human labels (rhg.validate.label) for the full
# validation report; the harness says what it is missing. Then: figures, results/REPORT.md, `python -m rhg.analysis.bundle
# --out results_public/`, and the write-up from docs/WRITEUP_TEMPLATE.md.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HARNESS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --mock-validation) HARNESS=(--mock) ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

require_uv
drop_api_key
CONF=()
if prereg_check_ok; then
  CONF=(--confirmatory)
  log "prereg_check passes: the analysis is CONFIRMATORY"
else
  warn "prereg_check does not pass: the analysis will be stamped EXPLORATORY (python -m rhg.analysis.prereg_check shows why)"
fi
if [[ "$DRY_RUN" != 1 ]] && ! py -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('rhg.analysis.run') else 1)"; then
  die "rhg.analysis.run does not exist yet (owned by subtask 14 of the build)"
fi

run_py -m rhg.validate.harness "${HARNESS[@]}"
run_py -m rhg.analysis.run --runs results/runs --out results/analysis "${CONF[@]}"
if [[ "$DRY_RUN" != 1 ]]; then log "done: results/analysis/, results/REPORT.md. Next: python -m rhg.analysis.bundle --out results_public/"; fi
