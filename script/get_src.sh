#!/usr/bin/env bash
# get_src.sh — fetch source repos for flightstack_sim
#
# What this script does (idempotent):
#  - Ensures the project workspace layout exists
#  - Clones PX4 into ap/px4 (default ref: v1.16.1) + initializes submodules
#  - Clones px4_msgs into ros/sim_ws/src/px4_msgs
#  - Clones ArduPilot into ap/ardupilot + initializes submodules
#
# Design notes:
#  - This script should be executed (bash get_src.sh), not sourced.
#  - It avoids destructive operations; it only clones when directories are missing/empty.
#
# Final layout (default):
#   <project-root>/
#     ap/px4
#     ap/ardupilot
#     ros/px4_msgs_ws/src/px4_msgs

# --------------------------
# Defaults
# --------------------------
DEBUG="false"
PROJECT_ROOT="$(pwd)"                 # assume invoked from flightstack_sim root

# Clone behavior
UPDATE_SUBMODULES="true"              # always init submodules by default

# PX4
PX4_REF="v1.16.1"
PX4_DIR_REL="ap/px4"

# px4_msgs (ROS2 message package)
PX4_MSGS_REF="main"
PX4_MSGS_DIR_REL="ros/px4_msgs_ws/src/px4_msgs"

# Optional ArduPilot (Default enabled)
WITH_ARDUPILOT="true"
ARDUPILOT_REF="Copter-4.6.2"
ARDUPILOT_DIR_REL="ap/ardupilot"

help() {
  cat <<EOF
Usage:
  bash get_src.sh [options]

Options:
  -h, --help              Show this help and exit
  --debug                 Enable command tracing (set -x)
  --project-root PATH     Project root (default: current directory)

  --px4-ref REF           PX4 ref/tag/branch (default: ${PX4_REF})
  --px4-dir PATH          PX4 directory (default: <project-root>/${PX4_DIR_REL})

  --px4-msgs-ref REF      px4_msgs ref/tag/branch (default: ${PX4_MSGS_REF})
  --px4-msgs-dir PATH     px4_msgs directory (default: <project-root>/${PX4_MSGS_DIR_REL})

  --ardupilot-ref REF     ArduPilot ref/tag/branch (default: ${ARDUPILOT_REF})
  --ardupilot-dir PATH    ArduPilot dir (default: <project-root>/${ARDUPILOT_DIR_REL})
  --no-ardupilot          Skip cloning ArduPilot

  --no-submodules         Skip 'git submodule update --init --recursive'

Examples:
  # Default (PX4 v1.16.1 + AP Copter-4.6.2 + px4_msgs)
  bash get_src.sh

  # Skip ArduPilot
  bash get_src.sh --no-ardupilot

  # Pin PX4 or ArduPilot
  bash get_src.sh --px4-ref v1.16.0 --ardupilot-ref Copter-4.6.1
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent sourcing
(return 0 2>/dev/null) && { echo "Do not source this script. Run: bash $0" >&2; return 1; }

# --------------------------
# Parse args
# --------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) help; exit 0 ;;
    --debug) DEBUG="true"; shift ;;
    --project-root) [[ $# -ge 2 ]] || die "--project-root requires a path"; PROJECT_ROOT="$2"; shift 2 ;;

    --px4-ref) [[ $# -ge 2 ]] || die "--px4-ref requires a value"; PX4_REF="$2"; shift 2 ;;
    --px4-dir) [[ $# -ge 2 ]] || die "--px4-dir requires a path"; PX4_DIR_REL=""; PX4_DIR_ABS="$2"; shift 2 ;;

    --px4-msgs-ref) [[ $# -ge 2 ]] || die "--px4-msgs-ref requires a value"; PX4_MSGS_REF="$2"; shift 2 ;;
    --px4-msgs-dir) [[ $# -ge 2 ]] || die "--px4-msgs-dir requires a path"; PX4_MSGS_DIR_REL=""; PX4_MSGS_DIR_ABS="$2"; shift 2 ;;

    --ardupilot-ref) [[ $# -ge 2 ]] || die "--ardupilot-ref requires a value"; ARDUPILOT_REF="$2"; shift 2 ;;
    --ardupilot-dir) [[ $# -ge 2 ]] || die "--ardupilot-dir requires a path"; ARDUPILOT_DIR_REL=""; ARDUPILOT_DIR_ABS="$2"; shift 2 ;;
    --no-ardupilot) WITH_ARDUPILOT="false"; shift ;;

    --no-submodules) UPDATE_SUBMODULES="false"; shift ;;

    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# Compute default absolute paths
PX4_DIR="${PX4_DIR_ABS:-${PROJECT_ROOT}/${PX4_DIR_REL}}"
PX4_MSGS_DIR="${PX4_MSGS_DIR_ABS:-${PROJECT_ROOT}/${PX4_MSGS_DIR_REL}}"
ARDUPILOT_DIR="${ARDUPILOT_DIR_ABS:-${PROJECT_ROOT}/${ARDUPILOT_DIR_REL}}"

# --------------------------
# Strict mode
# --------------------------
set -Ee
trap 'echo "[get_src.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
[[ "${DEBUG}" == "true" ]] && set -x

# --------------------------
# Helpers
# --------------------------
is_dir_empty() {
  local d="$1"
  [[ ! -d "$d" ]] && return 0
  [[ -z "$(ls -A "$d" 2>/dev/null)" ]]
}

git_clone_repo() {
  # git_clone_repo <url> <ref> <dest>
  local url="$1"
  local ref="$2"
  local dest="$3"
  if [[ "${SHALLOW}" == "true" ]]; then
    git clone -b "${ref}" --depth 1 "${url}" "${dest}"
  else
    git clone -b "${ref}" "${url}" "${dest}"
  fi
}

init_submodules() {
  local repo="$1"
  [[ "${UPDATE_SUBMODULES}" == "true" ]] || { echo "Skipping submodules (--no-submodules): ${repo}"; return 0; }
  [[ -d "${repo}/.git" ]] || return 0
  echo "Initializing submodules: ${repo}"
  ( cd "${repo}" && git submodule update --init --recursive )
}

# --------------------------
# Ensure base directories (new layout)
# --------------------------
mkdir -p "${PROJECT_ROOT}/ap" "${PROJECT_ROOT}/ros"
mkdir -p "$(dirname "${PX4_DIR}")" "$(dirname "${ARDUPILOT_DIR}")" "$(dirname "${PX4_MSGS_DIR}")"

# --------------------------
# Clone PX4
# --------------------------
if is_dir_empty "${PX4_DIR}"; then
  echo "Cloning PX4-Autopilot (${PX4_REF}) into: ${PX4_DIR}"
  git_clone_repo "https://github.com/PX4/PX4-Autopilot.git" "${PX4_REF}" "${PX4_DIR}"
  init_submodules "${PX4_DIR}"
else
  echo "PX4 directory exists and is not empty, skipping: ${PX4_DIR}"
fi

# # --------------------------
# # Clone px4_msgs
# # --------------------------
# if is_dir_empty "${PX4_MSGS_DIR}"; then
#   echo "Cloning px4_msgs (${PX4_MSGS_REF}) into: ${PX4_MSGS_DIR}"
#   git_clone_repo "https://github.com/PX4/px4_msgs.git" "${PX4_MSGS_REF}" "${PX4_MSGS_DIR}"
# else
#   echo "px4_msgs directory exists and is not empty, skipping: ${PX4_MSGS_DIR}"
# fi

# # --------------------------
# # Clone ArduPilot (default enabled)
# # --------------------------
# if [[ "${WITH_ARDUPILOT}" == "true" ]]; then
#   if is_dir_empty "${ARDUPILOT_DIR}"; then
#     echo "Cloning ArduPilot (${ARDUPILOT_REF}) into: ${ARDUPILOT_DIR}"
#     git_clone_repo "https://github.com/ArduPilot/ardupilot.git" "${ARDUPILOT_REF}" "${ARDUPILOT_DIR}"
#     init_submodules "${ARDUPILOT_DIR}"
#   else
#     echo "ArduPilot directory exists and is not empty, skipping: ${ARDUPILOT_DIR}"
#   fi
# fi

echo ""
echo "DONE. Sources prepared under: ${PROJECT_ROOT}"
echo "  PX4:        ${PX4_DIR}"
echo "  px4_msgs:   ${PX4_MSGS_DIR}"
echo "  ArduPilot:  ${WITH_ARDUPILOT} -> ${ARDUPILOT_DIR}"
echo "  Shallow:    ${SHALLOW}"
echo "  Submodules: ${UPDATE_SUBMODULES}"
echo ""