#!/bin/bash
# ros2.sh — install ROS 2 Humble on Ubuntu 22.04 (Jammy)
#
# Features:
#  - help
#  - debug enables `set -x`
#
# Note:
#   ROS-GZ bridges are installed in the Gazebo installation script because
#   they must match the specific Gazebo distribution (e.g., Harmonic). ROS 2 Humble
#   officially targets Gazebo Fortress by default, so we keep Gazebo integration separate.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"           # debug flat
DO_UPGRADE="false"      # upgrade flag (not recommended for reproducibility)

ROS_VERSION="humble"

help() {
  cat <<EOF
Usage:
  bash ros2.sh [options]

Options:
  -h, --help     Show this help and exit (no command tracing)
  --debug        Enable command tracing (set -x)
  --ros-distro   ROS 2 distro to install (default: ${ROS_VERSION})

Examples:
  bash ros2.sh
  bash ros2.sh --debug
  bash ros2.sh --ros-distro humble
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent sourcing (this script should be executed, not sourced)
(return 0 2>/dev/null) && { echo "Do not source this script. Run: bash $0" >&2; return 1; }

# --------------------------
# Parse args (before set -x)
# --------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      help
      exit 0
      ;;
    --debug)
      DEBUG="true"
      shift
      ;;
    --ros-distro)
      [[ $# -ge 2 ]] || die "--ros-distro requires an argument (e.g., humble)"
      ROS_VERSION="$2"
      shift 2
      ;;
    *)
      die "Unknown option: $1 (use --help)"
      ;;
  esac
done

# --------------------------
# Strict mode (after help)
# --------------------------
set -Ee
trap 'echo "[ros2.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# Repo setup
# --------------------------
# Ensure prerequisites exist (add-apt-repository, curl, etc.)
sudo apt-get -y update
sudo apt-get -y --no-install-recommends install \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  software-properties-common

# Enable Ubuntu universe repository (required for many ROS deps)
sudo add-apt-repository -y universe

# Add ROS 2 GPG key (secure package verification)
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS 2 apt repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# --------------------------
# APT update / optional upgrade
# --------------------------
sudo apt-get -y update

# Warning: Do NOT run `apt upgrade` here to if you want to strongly preserve Docker reproducibility.
# Running upgrade makes the image non-deterministic over time.
if [[ "${DO_UPGRADE}" == "true" ]]; then
  sudo apt-get -y upgrade
fi

# --------------------------
# Install ROS 2 packages
# --------------------------
# Install ROS2 Desktop + tooling + MAVLink integrations.
# CycloneDDS is selected as middleware for deterministic ROS 2 behavior.
PKGS=(
  "ros-${ROS_VERSION}-desktop"            # Core ROS2 desktop (RViz, rqt, demos)
  "ros-${ROS_VERSION}-cyclonedds"         # DDS implementation
  "ros-${ROS_VERSION}-rmw-cyclonedds-cpp" # RMW layer for CycloneDDS
  "ros-${ROS_VERSION}-gps-msgs"           # GPS message definitions
  "ros-${ROS_VERSION}-actuator-msgs"      # Actuator message types
  "ros-${ROS_VERSION}-mavlink"            # MAVLink message bindings
  "ros-${ROS_VERSION}-mavros"             # MAVLink <-> ROS bridge
  "ros-${ROS_VERSION}-mavros-extras"      # Additional MAVROS plugins
  "ros-dev-tools"                         # ROS developer utilities
  "python3-rosdep"                        # Dependency management
  "python3-rospkg"                        # ROS Python helpers
  "python3-colcon-common-extensions"      # Colcon build extensions
  "libgflags-dev"                         # Used by some Gazebo components / deps
)
sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install "${PKGS[@]}"

# --------------------------
# Notes / next steps
# --------------------------
echo "ROS 2 '${ROS_VERSION}' Installation DONE."
echo "Reminder: ROS-GZ bridges are installed in the Gazebo script to match the Gazebo distro (e.g., Harmonic)."