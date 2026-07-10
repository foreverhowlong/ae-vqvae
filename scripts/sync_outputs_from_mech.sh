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
  --no-models           Sync outputs without model/checkpoint weight files.
  --latest-only         Sync only the most recently modified run directory under outputs.
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
no_models=false
latest_only=false
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
    --no-models)
      no_models=true
      shift
      ;;
    --latest-only)
      latest_only=true
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

if [[ "${latest_only}" == true ]] && ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required for --latest-only but was not found in PATH." >&2
  exit 1
fi

mkdir -p "${local_outputs_dir}"

rsync_args=(
  -az
  --partial
  --progress
)

if [[ "${no_models}" == true ]]; then
  rsync_args+=(
    --exclude='*.pt'
    --exclude='*.pth'
    --exclude='*.ckpt'
    --exclude='*.safetensors'
  )
elif [[ "${best_only}" == true ]]; then
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

if [[ "${latest_only}" == true ]]; then
  remote_outputs_quoted="$(printf '%q' "${remote_outputs_dir%/}")"
  latest_relative_dir="$(
    ssh "${remote_host}" "remote_dir=${remote_outputs_quoted}; \
      if [[ \"\${remote_dir}\" == '~/'* ]]; then remote_dir=\"\${HOME}/\${remote_dir#~/}\"; fi; \
      cd \"\${remote_dir}\"; \
      latest=\$(find . -mindepth 2 -type f \( \
          -name summary.json -o \
          -name config.json -o \
          -name log.json -o \
          -name training_logs.json \
        \) -printf '%T@ %h\n' | sort -nr | head -n 1 | cut -d' ' -f2- | sed 's#^\./##'); \
      if [[ -z \"\${latest}\" ]]; then \
        latest=\$(find . -mindepth 1 -maxdepth 1 -type d -printf '%T@ %P\n' | sort -nr | head -n 1 | cut -d' ' -f2-); \
      fi; \
      printf '%s\n' \"\${latest}\""
  )"

  if [[ -z "${latest_relative_dir}" ]]; then
    echo "No output directories found under ${remote_host}:${remote_outputs_dir%/}" >&2
    exit 1
  fi

  latest_parent_dir="$(dirname "${latest_relative_dir}")"
  if [[ "${latest_parent_dir}" == "." ]]; then
    latest_parent_dir=""
  fi

  mkdir -p "${local_outputs_dir%/}/${latest_parent_dir}"
  remote_source="${remote_host}:${remote_outputs_dir%/}/${latest_relative_dir}/"
  local_target="${local_outputs_dir%/}/${latest_relative_dir}/"
else
  remote_source="${remote_host}:${remote_outputs_dir%/}/"
  local_target="${local_outputs_dir%/}/"
fi

echo "Syncing ${remote_source} -> ${local_target}"
rsync "${rsync_args[@]}" "${remote_source}" "${local_target}"
