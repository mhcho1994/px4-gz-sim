#!/bin/bash
# px4_setup.sh — install PX4 deps + Micro XRCE-DDS Agent
#
# Features:
#  - Clean --help output (no tracing)
#  - Optional --debug enables `set -x`
#  - Optional --px4-ref to pin PX4 version/ref (default: stable for 1.16 line)
#  - Optional --ws-root to control where sources live
#
# Notes:
#  - PX4 uses uXRCE-DDS for ROS 2 native comms via Micro XRCE-DDS Agent.
#  - We run PX4's official ubuntu setup script with --no-sim-tools.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"
PX4_REF="v1.16.1"                # for 1.16 stable line
WS_ROOT="/home/${USER}/work"     # where to put PX4-related sources by default
PX4_DIR_REL="flightstack_sim/ap/px4"  # recommended in-project path
DDS_AGENT_REF="v2.4.3"

help() {
  cat <<EOF
Usage:
  bash px4_setup.sh [options]

Options:
  -h, --help           Show this help and exit (no command tracing)
  --debug              Enable command tracing (set -x)
  --px4-ref REF        PX4 ref/tag/branch for setup scripts (default: ${PX4_REF})
  --ws-root PATH       Workspace root (default: ${WS_ROOT})
  --px4-path PATH      Where PX4 repo should live (default: \${WS_ROOT}/${PX4_DIR_REL})
  --dds-ref REF        Micro XRCE-DDS Agent ref/tag (default: ${DDS_AGENT_REF})

Examples:
  bash px4_setup.sh
  bash px4_setup.sh --debug
  bash px4_setup.sh --px4-ref v1.16.1
  bash px4_setup.sh --ws-root /home/user/work --px4-path /home/user/work/flightstack_sim/ws/px4
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
PX4_PATH="${WS_ROOT}/${PX4_DIR_REL}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) help; exit 0 ;;
    --debug) DEBUG="true"; shift ;;
    --px4-ref) [[ $# -ge 2 ]] || die "--px4-ref requires an argument"; PX4_REF="$2"; shift 2 ;;
    --ws-root) [[ $# -ge 2 ]] || die "--ws-root requires a path"; WS_ROOT="$2"; PX4_PATH="${WS_ROOT}/${PX4_DIR_REL}"; shift 2 ;;
    --px4-path) [[ $# -ge 2 ]] || die "--px4-path requires a path"; PX4_PATH="$2"; shift 2 ;;
    --dds-ref) [[ $# -ge 2 ]] || die "--dds-ref requires a ref"; DDS_AGENT_REF="$2"; shift 2 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# --------------------------
# Strict mode
# --------------------------
set -Ee
trap 'echo "[px4_setup.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# Install PX4 dependencies (official script)
# --------------------------
# Use stable ref for 1.16 line unless user pins a tag/commit.
TMPDIR="/tmp/px4_setup"
mkdir -p "${TMPDIR}"

UBUNTU_SH_URL="https://raw.githubusercontent.com/PX4/PX4-Autopilot/${PX4_REF}/Tools/setup/ubuntu.sh"
REQ_TXT_URL="https://raw.githubusercontent.com/PX4/PX4-Autopilot/${PX4_REF}/Tools/setup/requirements.txt"

wget -q "${UBUNTU_SH_URL}" -O "${TMPDIR}/ubuntu.sh"
wget -q "${REQ_TXT_URL}" -O "${TMPDIR}/requirements.txt"

chmod +x "${TMPDIR}/ubuntu.sh"

# Install PX4 deps, but skip sim-tools because we manage Gazebo separately
bash "${TMPDIR}/ubuntu.sh" --no-sim-tools

# --------------------------
# (Optional) Ensure PX4 directory exists (clone happens elsewhere usually)
# --------------------------
mkdir -p "${PX4_PATH}"
echo "PX4 workspace directory prepared at: ${PX4_PATH}"
echo "NOTE: This script installs dependencies only; clone PX4 repo separately if desired."

# --------------------------
# Micro XRCE-DDS Agent build/install
# --------------------------
cd "${WS_ROOT}"
if [[ ! -d "Micro-XRCE-DDS-Agent" ]]; then
  git clone -b "${DDS_AGENT_REF}" --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
fi

cd Micro-XRCE-DDS-Agent
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
sudo make install
sudo ldconfig

echo "Micro XRCE-DDS Agent installed."