#!/usr/bin/env bash
# entrypoint.sh — align container user UID/GID with host, run setup, drop privileges

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
  HOST_UID          Required host user id
  HOST_GID          Required host group id
  HOST_USER_NAME    Optional host user name (for logging)
  HOST_GROUP_NAME   Optional host group name (for logging)

Examples:
  docker run \\
    -e HOST_UID=\$(id -u) \\
    -e HOST_GID=\$(id -g) \\
    <image> --chown /home/user/FIRE_flightstack_sim -- bash
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

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

set -Eeuo pipefail
trap 'echo "[entrypoint.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
[[ "${DEBUG}" == "true" ]] && set -x

[[ -n "${HOST_UID:-}" ]] || die "please set HOST_UID"
[[ -n "${HOST_GID:-}" ]] || die "please set HOST_GID"

HOST_USER_NAME="${HOST_USER_NAME:-unknown}"
HOST_GROUP_NAME="${HOST_GROUP_NAME:-unknown}"

echo "[ENTRYPOINT] Host user : ${HOST_USER_NAME}"
echo "[ENTRYPOINT] Host group: ${HOST_GROUP_NAME}"
echo "[ENTRYPOINT] Mapping ${USER_NAME} -> ${HOST_UID}:${HOST_GID}"

if [[ "$(id -u)" -ne 0 ]]; then
  die "Entrypoint must run as root"
fi

current_gid="$(id -g "${USER_NAME}")"
if [[ "${current_gid}" != "${HOST_GID}" ]]; then
  groupmod --gid "${HOST_GID}" "${USER_NAME}"
fi

current_uid="$(id -u "${USER_NAME}")"
if [[ "${current_uid}" != "${HOST_UID}" ]]; then
  usermod --uid "${HOST_UID}" "${USER_NAME}"
fi

for path in "${CHOWN_PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "[ENTRYPOINT] Chowning ${path} -> ${USER_NAME}:${USER_NAME}"
    chown -R "${USER_NAME}:${USER_NAME}" "${path}" || true
  else
    echo "[ENTRYPOINT] WARNING: ${path} does not exist, skipping"
  fi
done

SETUP_FLAG="/home/${USER_NAME}/.setup_done"

if [[ ! -f "${SETUP_FLAG}" ]]; then
  echo "[ENTRYPOINT] Running first-time setup..."

  sudo -u "${USER_NAME}" -H bash -lc "
    set -euo pipefail

    if [[ -f /install/autopilot.sh ]]; then
      echo '[SETUP] autopilot setup'
      bash /install/autopilot.sh --mode setup --with-ardupilot
    fi

    if [[ -f /install/extra.sh ]]; then
      echo '[SETUP] extra setup'
      bash /install/extra.sh --mode setup
    fi

    if [[ -f /install/usersetup.sh ]]; then
      echo '[SETUP] user setup'
      bash /install/usersetup.sh
    fi
  "

  touch "${SETUP_FLAG}"
  chown "${USER_NAME}:${USER_NAME}" "${SETUP_FLAG}"

  echo "[ENTRYPOINT] Setup completed."
else
  echo "[ENTRYPOINT] Setup already done. Skipping."
fi

if [[ ${#CMD[@]} -gt 0 ]]; then
  exec sudo -u "${USER_NAME}" -H -- "${CMD[@]}"
else
  exec sudo -u "${USER_NAME}" -H -- bash
fi