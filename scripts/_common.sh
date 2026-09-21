#!/usr/bin/env bash
# Shared helpers for scripts/*.sh. Source it (`source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"`); do not execute it.
#
# Conventions (docs/REPO_SPEC.md §7):
#   exit 0 ok | 1 error or NO-GO | 2 usage | 3 guard refused (budget, prereg, missing key, wrong machine) | 75 stall
#   --dry-run prints the commands instead of running them (read-only checks still run, failures become warnings).
#   RHG_PYTHON=/path/to/python replaces `uv run --no-sync python` (tests, or a box without uv on PATH).
#
# After setup_box.sh never run a plain `uv sync` (it is exact and would remove the GPU stack that
# `uv pip install -r requirements-gpu.txt` put into .venv); every script uses `uv run --no-sync`.
set -euo pipefail

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_COMMON_DIR/.." && pwd)"
SCRIPT_NAME="$(basename "${0:-script}")"
cd "$REPO_ROOT"

DRY_RUN=0
EXIT_NOGO=1
EXIT_USAGE=2
EXIT_REFUSED=3

log() { printf '[%s %s] %s\n' "$SCRIPT_NAME" "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$SCRIPT_NAME" "$*" >&2; }
die() { printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2; exit 1; }
die_usage() { printf '[%s] usage error: %s\n' "$SCRIPT_NAME" "$*" >&2; exit "$EXIT_USAGE"; }
refuse() { printf '[%s] REFUSED: %s\n' "$SCRIPT_NAME" "$*" >&2; exit "$EXIT_REFUSED"; }

# Run a command, or print it with --dry-run.
run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# The project python: `uv run --no-sync python` unless RHG_PYTHON is set.
py() {
  if [[ -n "${RHG_PYTHON:-}" ]]; then
    "$RHG_PYTHON" "$@"
  else
    uv run --no-sync python "$@"
  fi
}

run_py() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run] python'
    printf ' %q' "$@"
    printf '\n'
  else
    py "$@"
  fi
}

require_uv() {
  if [[ -n "${RHG_PYTHON:-}" ]] || command -v uv >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    warn "uv not found on PATH (install: https://docs.astral.sh/uv/); continuing because of --dry-run"
    return 0
  fi
  die "uv not found on PATH. Install it (https://docs.astral.sh/uv/) or run scripts/setup_box.sh on a GPU box."
}

gpu_present() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

# The judge key must never reach a GPU box: warn and drop it from this process (the sandbox env is cleared anyway).
drop_api_key() {
  if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    warn "ANTHROPIC_API_KEY is set on a machine that runs training/eval. Unsetting it for this script; remove it from your shell/profile on GPU boxes (the judge runs locally)."
    unset ANTHROPIC_API_KEY
  fi
}

# LOCAL-machine scripts (judge): never on a box with an NVIDIA driver (the API key must not live next to the trainer).
require_local_machine() {
  if gpu_present; then
    refuse "an NVIDIA driver was detected: this script runs on your LOCAL machine only (the judge key never goes on a GPU box)."
  fi
}

# HF cache and library env shared by every script. Override HF_HOME (or RHG_HF_HOME) to point at a big volume.
setup_env() {
  export HF_HOME="${RHG_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
  export HF_HUB_DISABLE_TELEMETRY=1
  export TOKENIZERS_PARALLELISM=false
}

# git: is prereg-v1 an ancestor of HEAD?
prereg_tag_ancestor() {
  git rev-parse -q --verify "refs/tags/prereg-v1^{commit}" >/dev/null 2>&1 \
    && git merge-base --is-ancestor "refs/tags/prereg-v1^{commit}" HEAD >/dev/null 2>&1
}

# git: is the SAME prereg-v1 tag object on the `origin` remote? The public push is what makes the freeze timestamp checkable by
# a third party (local tag/commit dates are author-controlled), so every confirmatory launch requires it.
prereg_tag_pushed() {
  local local_obj remote_obj
  local_obj="$(git rev-parse -q --verify "refs/tags/prereg-v1" 2>/dev/null)" || return 1
  remote_obj="$(git ls-remote --tags origin "refs/tags/prereg-v1" 2>/dev/null | grep -v '\^{}' | awk '{print $1}' | head -n1)" || return 1
  [[ -n "$remote_obj" && "$remote_obj" == "$local_obj" ]]
}

# python -m rhg.analysis.prereg_check passes?
prereg_check_ok() {
  py -m rhg.analysis.prereg_check >/dev/null 2>&1
}

gate_criterion() { # gate id, criterion text
  printf '\n=== Gate %s ===\nCriterion: %s\n\n' "$1" "$2"
}

# Pre-declared next step on NO-GO (SCHEDULE.md).
next_step() {
  case "$1" in
    1a) echo "Follow docs/GPU_COMPAT.md fallbacks; cap debugging at \$1, then NO-GO: rethink the trainer, do not spend the main budget." ;;
    1b) echo "Apply the cut ladder (BUDGET.md §4) top to bottom until it fits and write prereg/budget_decision.md. NO-GO if even the floor (11 runs) does not fit: no confirmatory study, report pilots/probes only." ;;
    1c) echo "Single pre-declared band widening [0.05, 0.50] (rhg.data.build --stage split --widen; pass --widen to this script), then the MBPP-sanitized add-in; otherwise NO-GO." ;;
    1d) echo "NO-GO. One further round of at most 3 new candidate wordings, <= \$0.3, is allowed before freezing." ;;
    1e) echo "If there is no emergence: ONE allowed retry with lr x2 (scripts/pilot.sh --lr-retry; it is recorded in prereg/pilot_gate.json). A second failure is NO-GO." ;;
    1e2) echo "Edit the rubric (tuned on the synthetic controls only, never on real rollouts) and re-run scripts/calibrate_judge.sh --yes; the rubric hash is frozen at the next step." ;;
    1f) echo "Gate 1f is GO iff prereg_check passes AND the commit and tag are pushed to your public remote; set the boxes up from a checkout that contains the tag." ;;
    *) echo "See SCHEDULE.md." ;;
  esac
}

nogo() { # gate id, [message]
  printf '\n[%s] GATE %s: NO-GO%s\n' "$SCRIPT_NAME" "$1" "${2:+ - $2}" >&2
  printf 'Next step (pre-declared): %s\n' "$(next_step "$1")" >&2
  exit "$EXIT_NOGO"
}

go() { printf '\n[%s] GATE %s: GO%s\n' "$SCRIPT_NAME" "$1" "${2:+ - $2}"; }

physical_cores() {
  local n=""
  if command -v lscpu >/dev/null 2>&1; then
    n="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')" || n=""
  fi
  if [[ -z "$n" || "$n" == 0 ]]; then
    n="$(nproc 2>/dev/null || echo 0)"
  fi
  echo "$n"
}

setup_env
