#!/usr/bin/env bash
# entrypoint.sh — align container user UID/GID with host and drop privileges
#
# Features:
#  - Clean --help output
#  - Optional --debug enables set -x
#  - Safe UID/GID remapping
#  - Optional --chown PATH (can be used multiple times)
#  - Avoids unnecessary recursive chown of entire home directory

# --------------------------
# Defaults
# --------------------------
DEBUG="false"
USER_NAME="user"
CHOWN_PATHS=()

help() {
  cat <<EOF
Usage:
  entrypoint.sh [options] [--] [command...]

Options:
  -h, --help        Show this help and exit
  --debug           Enable command tracing (set -x)
  --chown PATH      Recursively chown PATH to ${USER_NAME}:${USER_NAME}
                    Can be specified multiple times.

Environment:
  HOST_UID (required)
  HOST_GID (required)

Examples:
  docker run \\
    -e HOST_UID=\$(id -u) \\
    -e HOST_GID=\$(id -g) \\
    <image> --chown /home/user/ws --chown /data
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# --------------------------
# Parse args
# --------------------------
CMD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) help; exit 0 ;;
    --debug) DEBUG="true"; shift ;;
    --chown)
      [[ $# -ge 2 ]] || die "--chown requires a path"
      CHOWN_PATHS+=("$2")
      shift 2
      ;;
    --) shift; CMD=("$@"); break ;;
    -*)
      die "Unknown option: $1 (use --help)"
      ;;
    *)
      CMD=("$@"); break ;;
  esac
done

# --------------------------
# Strict mode
# --------------------------
set -Eeuo pipefail
trap 'echo "[entrypoint.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
[[ "${DEBUG}" == "true" ]] && set -x

# --------------------------
# Validate environment
# --------------------------
[[ -n "${HOST_UID:-}" ]] || die "please set HOST_UID"
[[ -n "${HOST_GID:-}" ]] || die "please set HOST_GID"

echo "Mapping ${USER_NAME} -> ${HOST_UID}:${HOST_GID}"

if [[ "$(id -u)" -ne 0 ]]; then
  die "Entrypoint must run as root"
fi

# --------------------------
# Update group
# --------------------------
current_gid="$(id -g "${USER_NAME}")"
if [[ "${current_gid}" != "${HOST_GID}" ]]; then
  groupmod --gid "${HOST_GID}" "${USER_NAME}"
fi

# --------------------------
# Update user
# --------------------------
current_uid="$(id -u "${USER_NAME}")"
if [[ "${current_uid}" != "${HOST_UID}" ]]; then
  usermod --uid "${HOST_UID}" "${USER_NAME}"
fi

# --------------------------
# Selective chown paths
# --------------------------
for path in "${CHOWN_PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "Chowning ${path} -> ${USER_NAME}:${USER_NAME}"
    chown -R "${USER_NAME}:${USER_NAME}" "${path}" || true
  else
    echo "WARNING: ${path} does not exist, skipping"
  fi
done

# --------------------------
# Drop privileges
# --------------------------
if [[ ${#CMD[@]} -gt 0 ]]; then
  exec sudo -u "${USER_NAME}" -H -- "${CMD[@]}"
else
  exec sudo -u "${USER_NAME}" -H -- bash
fi