#!/usr/bin/env bash
# Pack the run directories of this box for the trip back to your local machine. Usage:
#   scripts/package_results.sh [--out FILE.tar.gz] [--dry-run]
# tar of results/runs (adapters, snapshots, weights and any file > 50 MB excluded), plus results/ledger.jsonl,
# results/replacements.jsonl, results/RUNS_HEALTH.md and results/bench/ when present. Prints the rsync/scp commands to pull it.
# Shut the box down right after (idle time is unmetered but not free).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HOST_TAG="$(hostname | tr -c 'A-Za-z0-9\n' '_')"
OUT="results/package/runs_${HOST_TAG}_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
MAX_MB=50
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --out) OUT="${2:?--out needs a file}"; shift ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) die_usage "unknown argument: $1" ;;
  esac
  shift
done

if [[ ! -d results/runs ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then warn "results/runs does not exist"; else die "results/runs does not exist: nothing to package"; fi
fi
EXTRA=()
for f in results/ledger.jsonl results/replacements.jsonl results/RUNS_HEALTH.md results/bench; do
  if [[ -e "$f" ]]; then EXTRA+=("$f"); fi
done
BIG=()
if [[ -d results/runs ]]; then
  while IFS= read -r f; do BIG+=("--exclude=$f"); done < <(find results/runs -type f -size +"${MAX_MB}"M 2>/dev/null)
fi
run mkdir -p "$(dirname "$OUT")"
run tar -czf "$OUT" \
  --exclude='adapters' --exclude='snapshots' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth' \
  "${BIG[@]}" results/runs "${EXTRA[@]}"
if [[ "$DRY_RUN" != 1 ]]; then
  log "wrote $OUT ($(du -h "$OUT" | cut -f1)); ${#BIG[@]} file(s) larger than ${MAX_MB} MB were left out"
fi
BASENAME="$(basename "$OUT")"
cat <<EOT

Pull it to your local machine (run these THERE), then unpack into the repo root:
  rsync -avP USER@BOX:$REPO_ROOT/$OUT ./
  scp USER@BOX:$REPO_ROOT/$OUT ./
  tar -xzf $BASENAME    # -> results/runs/<run_id>/, results/ledger.jsonl (with several boxes: append the ledger lines by hand)
Then shut the box down. Next (local): scripts/judge_all.sh
EOT
