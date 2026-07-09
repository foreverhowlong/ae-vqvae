#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Sync the remote ae-vqvae outputs directory from the Tailscale host `mech`.

Usage:
  scripts/sync_outputs_from_mech.sh [options]

Options:
  --host HOST           SSH host alias to read from. Default: mech
  --remote-dir DIR      Remote outputs directory. Default: ~/Developer/ae-vqvae/outputs
  --local-dir DIR       Local outputs directory. Default: <repo>/outputs
  --best-only           Sync all outputs, but for *.pt model files only sync best.pt.
  --dry-run             Print what would be transferred without copying files.
  --delete              Delete local files that no longer exist on the remote.
  -h, --help            Show this help.

Environment overrides:
  REMOTE_HOST
  REMOTE_OUTPUTS_DIR
  LOCAL_OUTPUTS_DIR
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

remote_host="${REMOTE_HOST:-mech}"
remote_outputs_dir="${REMOTE_OUTPUTS_DIR:-~/Developer/ae-vqvae/outputs}"
local_outputs_dir="${LOCAL_OUTPUTS_DIR:-${repo_root}/outputs}"
best_only=false
dry_run=false
delete=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      remote_host="${2:?Missing value for --host}"
      shift 2
      ;;
    --remote-dir)
      remote_outputs_dir="${2:?Missing value for --remote-dir}"
      shift 2
      ;;
    --local-dir)
      local_outputs_dir="${2:?Missing value for --local-dir}"
      shift 2
      ;;
    --best-only)
      best_only=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --delete)
      delete=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found in PATH." >&2
  exit 1
fi

mkdir -p "${local_outputs_dir}"

rsync_args=(
  -az
  --partial
  --progress
)

if [[ "${best_only}" == true ]]; then
  rsync_args+=(
    --include='*/'
    --include='best.pt'
    --exclude='*.pt'
    --include='*'
  )
fi

if [[ "${dry_run}" == true ]]; then
  rsync_args+=(--dry-run)
fi

if [[ "${delete}" == true ]]; then
  rsync_args+=(--delete)
fi

remote_source="${remote_host}:${remote_outputs_dir%/}/"

echo "Syncing ${remote_source} -> ${local_outputs_dir%/}/"
rsync "${rsync_args[@]}" "${remote_source}" "${local_outputs_dir%/}/"
