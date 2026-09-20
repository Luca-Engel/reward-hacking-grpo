#!/usr/bin/env bash
# One-time GPU box setup (SCHEDULE Day 1, step 1). Usage: scripts/setup_box.sh [--dry-run] [--skip-model]
#   1. checks the NVIDIA GPU/driver and the physical CPU core count (warns below 8: reward execution is CPU-bound)
#   2. installs uv if missing, `uv sync --frozen`, `uv pip install -r requirements-gpu.txt`
#   3. HF cache env, prefetch of the model weights (skip with --skip-model)
#   4. `rhg.data.build --stage fetch,tests,validate`
# ANTHROPIC_API_KEY must not exist on this machine (the judge runs locally); it is warned about and unset here.
# Set the box up from a checkout that contains the pushed prereg-v1 tag once the freeze has happened (Gate 1f).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SKIP_MODEL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

drop_api_key
log "repo root: $REPO_ROOT   HF_HOME=$HF_HOME"

# 1. GPU and CPU ----------------------------------------------------------------------------------------------
GPU_NAME="none"
DRIVER="unknown"
if gpu_present; then
  info="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1)"
  GPU_NAME="${info%%,*}"
  DRIVER="${info##*, }"
  log "GPU: $info"
  major="${DRIVER%%.*}"
  if [[ "$major" =~ ^[0-9]+$ ]] && (( major < 580 )); then
    warn "driver $DRIVER < 580: the pinned PyPI torch 2.13.0 is the CUDA 13 build and will not initialise. Use the cu129 route in docs/GPU_COMPAT.md section 2 (do not edit requirements-gpu.txt after the freeze)."
  fi
elif [[ "$DRY_RUN" == 1 ]]; then
  warn "no NVIDIA GPU detected (nvidia-smi missing or failing); continuing because of --dry-run"
else
  die "no NVIDIA GPU detected (nvidia-smi missing or failing). This script is for the GPU box."
fi
CORES="$(physical_cores)"
CORE_NOTE=""
log "physical CPU cores: $CORES"
if [[ "$CORES" =~ ^[0-9]+$ ]] && (( CORES < 8 )); then
  CORE_NOTE="(BELOW 8: expect CPU-bound reward execution)"
  warn "only $CORES physical cores (< 8): t_reward (sandbox execution) will dominate the step time; the throughput bench will show it (BUDGET.md section 2)."
fi

# 2. uv and dependencies ---------------------------------------------------------------------------------------
if [[ -z "${RHG_PYTHON:-}" ]] && ! command -v uv >/dev/null 2>&1; then
  log "uv not found: installing it (https://docs.astral.sh/uv/)"
  run bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
require_uv
run uv sync --frozen
run uv pip install -r requirements-gpu.txt
run mkdir -p results/bench
run bash -c 'uv pip freeze > results/bench/pip_freeze.txt'
run_py -c "import torch, transformers, trl, peft, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('transformers', transformers.__version__, 'trl', trl.__version__, 'peft', peft.__version__, 'vllm', vllm.__version__); assert torch.cuda.is_available(), 'CUDA not available to torch'"

# 3. model weights ----------------------------------------------------------------------------------------------
if [[ "$SKIP_MODEL" == 0 ]]; then
  run_py -c "from huggingface_hub import snapshot_download; from rhg.config import load_config; n = load_config('hackable_subtle').model.name; print('prefetching', n); print(snapshot_download(n))"
fi

# 4. dataset ----------------------------------------------------------------------------------------------------
for stage in fetch tests validate; do
  run_py -m rhg.data.build --stage "$stage"
done

printf '\n================ setup summary ================\n'
printf 'GPU:            %s (driver %s)\n' "$GPU_NAME" "$DRIVER"
printf 'physical cores: %s %s\n' "$CORES" "$CORE_NOTE"
printf 'HF_HOME:        %s\n' "$HF_HOME"
printf 'pip freeze:     results/bench/pip_freeze.txt\n'
printf 'API key:        not present in this environment\n'
printf 'Next:           scripts/smoke.sh (Gate 1a), then scripts/bench_throughput.sh --usd-per-hour RATE\n'
printf "Never run a plain 'uv sync' on this box again (it would remove the GPU stack); scripts use 'uv run --no-sync'.\n"
if [[ "$DRY_RUN" == 1 ]]; then printf '(dry run: nothing above was executed)\n'; fi
