#!/usr/bin/env bash
# gazebo.sh — install/build Gazebo Harmonic on Ubuntu 22.04 (Jammy)
#
# Features:
#  - Clean --help output (no tracing)
#  - Optional --debug enables `set -x`
#  - binary/source install modes
#  - source mode builds Gazebo Harmonic from source (colcon + vcs)
#  - optional ros_gz build (ROS 2 Humble) against the source-built Gazebo overlay
#
# Notes:
#  - We intentionally do NOT run `apt-get upgrade` to preserve reproducibility.
#  - ROS-GZ bridges for Harmonic can be installed as binaries (ros-humble-ros-gzharmonic),
#    but if you're source-building Gazebo for custom plugins, building ros_gz from source
#    against that overlay is often the most consistent setup.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"                 # --debug
INSTALL_MODE="binary"         # --install binary|source
BUILD_ROS_GZ="true"           # --no-ros-gz disables ros_gz build in source mode
BUILD_TYPE="RelWithDebInfo"   # --build-type
ROS_DISTRO="humble"
GZ_VERSION="harmonic"
PROJECT_ROOT="/home/${USER}/ws/flightstack_sim"
GZ_WS_DIR="${PROJECT_ROOT}/gz/${GZ_VERSION}_ws"
ROS_GZ_WS_DIR="${PROJECT_ROOT}/ros/ros_gz_ws"

# Where to fetch Gazebo repos from in source mode:
#   - default: official Harmonic collection file from gazebodistro
#   - you can override with --repos-yaml URL_OR_PATH
REPOS_YAML="https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/collection-harmonic.yaml"

help() {
  cat <<EOF
Usage:
  bash gazebo.sh [options]

Options:
  -h, --help               Show this help and exit (no command tracing)
  --debug                  Enable command tracing (set -x)

  -i, --install MODE       Install mode: binary | source (default: ${INSTALL_MODE})
      --no-ros-gz          (source mode) Build Gazebo only, skip building ros_gz
      --project-root PATH      Default: ${PROJECT_ROOT}
      --build-type TYPE    CMake build type (default: ${BUILD_TYPE})
      --repos-yaml SRC     (source mode) Repos YAML URL/path (default: official Harmonic collection)

Examples:
  # Binary install (Gazebo Harmonic + ROS bridge binaries)
  bash gazebo.sh --install binary

  # Source build Gazebo Harmonic + ros_gz overlay
  bash gazebo.sh --install source --ws-root /home/user/ws --build-type RelWithDebInfo

  # Source build Gazebo only (skip ros_gz)
  bash gazebo.sh --install source --no-ros-gz

  # Source build using your custom repos file
  bash gazebo.sh --install source --repos-yaml https://raw.githubusercontent.com/<you>/<repo>/main/gz_repos.yaml
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
    -i|--install)
      [[ $# -ge 2 ]] || die "--install requires an argument (binary|source)"
      INSTALL_MODE="$2"
      shift 2
      ;;
    --no-ros-gz)
      BUILD_ROS_GZ="false"
      shift
      ;;
    --ws-root)
      [[ $# -ge 2 ]] || die "--ws-root requires a path"
      WS_ROOT="$2"
      shift 2
      ;;
    --build-type)
      [[ $# -ge 2 ]] || die "--build-type requires a value"
      BUILD_TYPE="$2"
      shift 2
      ;;
    --repos-yaml)
      [[ $# -ge 2 ]] || die "--repos-yaml requires a URL or file path"
      REPOS_YAML="$2"
      shift 2
      ;;
    *)
      die "Unknown option: $1 (use --help)"
      ;;
  esac
done

# Validate install mode
if [[ "${INSTALL_MODE}" != "binary" && "${INSTALL_MODE}" != "source" ]]; then
  die "Invalid --install mode: ${INSTALL_MODE} (expected: binary|source)"
fi

# --------------------------
# Strict mode (after help)
# --------------------------
set -Ee
trap 'echo "[gazebo.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# Helpers
# --------------------------
add_osrf_repo() {
  sudo apt-get -y update
  sudo apt-get -y --no-install-recommends install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    wget

  sudo wget -q https://packages.osrfoundation.org/gazebo.gpg \
    -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

  sudo apt-get -y update
}

fetch_repos_yaml() {
  # Accepts URL or local path; outputs local file path
  local src="$1"
  local out="$2"

  if [[ "$src" =~ ^https?:// ]]; then
    curl -L -o "$out" "$src"
  else
    [[ -f "$src" ]] || die "--repos-yaml path not found: $src"
    cp -f "$src" "$out"
  fi
}

# --------------------------
# Binary install
# --------------------------
install_binary() {
  echo ""
  echo "binary -> Installing Gazebo (${GZ_VERSION}) from binaries"
  echo "         ROS_DISTRO=${ROS_DISTRO}, BUILD_ROS_GZ=${BUILD_ROS_GZ}"
  echo ""

  add_osrf_repo

  
  # 1) Install Gazebo
  sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    "gz-${GZ_VERSION}"

  # 2) Optionally install ROS-GZ bridge binaries (pairing-aware)
  if [[ "${BUILD_ROS_GZ}" != "true" ]]; then
    echo "Skipping ROS-GZ bridge install (BUILD_ROS_GZ=false)"
    return 0
  fi

  # Pairing logic:
  # - Humble officially pairs with Fortress (recommended).
  # - Humble + Harmonic is possible via non-official OSRF binaries: ros-humble-ros-gzharmonic,
  #   which can conflict with ros-humble-ros-gz* packages.
  # --------------------------
  # Strict mode for paring logic
  # --------------------------
  if [[ "${ROS_DISTRO}:${GZ_VERSION}" == "humble:harmonic" ]]; then
  # Ensure Fortress-paired packages aren't installed (conflict risk). 
    if dpkg -l | grep -q "^ii  ros-${ROS_DISTRO}-ros-gz "; then
      sudo apt-get -y remove "ros-${ROS_DISTRO}-ros-gz"
    fi
  fi
  
  case "${ROS_DISTRO}:${GZ_VERSION}" in
    humble:fortress)
      # Official / recommended pairing for Humble
      sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        "ros-${ROS_DISTRO}-ros-gz"
      ;;

    jazzy:harmonic)
      # Official / recommended pairing for Jazzy
      sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        "ros-${ROS_DISTRO}-ros-gz"
      ;;

    humble:harmonic)
      # Non-official binary packages hosted in packages.osrfoundation.org
      # NOTE: Conflicts with ros-humble-ros-gz* (Fortress pairing). 
      echo "WARNING: ROS 2 Humble + Gazebo Harmonic is a 'use with caution' pairing."
      sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        "ros-${ROS_DISTRO}-ros-gzharmonic"
      ;;

    *)
      echo "ERROR: No supported binary ROS-GZ bridge pairing for ROS_DISTRO=${ROS_DISTRO} and GZ_VERSION=${GZ_VERSION}." >&2
      echo "       Either switch to a recommended pairing, or use --install source to build ros_gz from source." >&2
      exit 2
      ;;
  esac
}

# --------------------------
# Source build (Gazebo + optional ros_gz)
# --------------------------
install_source() {
  echo ""
  echo "source -> Building Gazebo (${GZ_VERSION}) from source"
  echo "         repos: ${REPOS_YAML}"
  echo "         ws-root: ${WS_ROOT}"
  echo ""

  # Tools required for vcs/colcon builds
  sudo apt-get -y update
  sudo apt-get -y --no-install-recommends install \
    git \
    cmake \
    ninja-build \
    pkg-config \
    python3-pip \
    python3-venv \
    python3-vcstool \
    python3-colcon-common-extensions

  # OSRF repo helps satisfy many dependencies via apt
  add_osrf_repo

  # Workspace layout
  local GZ_WS="${WS_ROOT}/gz_${GZ_VERSION}_ws"
  local GZ_SRC="${GZ_WS}/src"
  mkdir -p "${GZ_SRC}"

  # 1) Import repos
  cd "${GZ_SRC}"
  fetch_repos_yaml "${REPOS_YAML}" "${GZ_SRC}/repos.yaml"
  vcs import < "${GZ_SRC}/repos.yaml"

  # 2) Install apt dependencies declared by the source repos
  #    This follows the common Gazebo pattern: scan packages*.apt files.
  #    IMPORTANT: If you rely on apt-provided gz/sdf instead of source versions,
  #    remove the sed filter below.
  echo "==> Installing Gazebo source dependencies (apt)"
  sudo apt-get -y install \
    $(sort -u $(find . -iname "packages-$(lsb_release -cs).apt" -o -iname "packages.apt" | grep -v '/\.git/') \
      | sed '/gz\|sdf/d' | tr '\n' ' ')

  # 3) Build Gazebo
  cd "${GZ_WS}"
  echo "==> colcon build (Gazebo ${GZ_VERSION})"
  colcon build --merge-install \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

  # 4) Optional: build ros_gz from source against the overlay
  if [[ "${BUILD_ROS_GZ}" == "true" ]]; then
    echo ""
    echo "==> Building ros_gz (${ROS_DISTRO}) from source against Gazebo overlay"
    echo ""

    # rosdep setup (safe if already initialized)
    sudo apt-get -y --no-install-recommends install python3-rosdep
    sudo rosdep init 2>/dev/null || true
    rosdep update

    local ROS_GZ_WS="${WS_ROOT}/ros_gz_ws"
    local ROS_GZ_SRC="${ROS_GZ_WS}/src"
    mkdir -p "${ROS_GZ_SRC}"
    cd "${ROS_GZ_SRC}"

    # ros_gz repo (Humble branch)
    if [[ ! -d ros_gz ]]; then
      git clone -b "${ROS_DISTRO}" https://github.com/gazebosim/ros_gz.git
    fi

    # Source ROS + Gazebo overlays so dependencies resolve correctly
    # shellcheck disable=SC1091
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    # shellcheck disable=SC1090
    source "${GZ_WS}/install/setup.bash"

    cd "${ROS_GZ_WS}"
    rosdep install -r --from-paths src -i -y --rosdistro "${ROS_DISTRO}" || true

    echo "==> colcon build (ros_gz)"
    colcon build --merge-install \
      --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  else
    echo "==> Skipping ros_gz build (--no-ros-gz)"
  fi

  echo ""
  echo "DONE."
  echo "To use the source-built Gazebo overlay:"
  echo "  source ${GZ_WS}/install/setup.bash"
  if [[ "${BUILD_ROS_GZ}" == "true" ]]; then
    echo "To use the ros_gz overlay:"
    echo "  source ${WS_ROOT}/ros_gz_ws/install/setup.bash"
  fi
  echo ""
}

# --------------------------
# Main
# --------------------------
if [[ "${INSTALL_MODE}" == "binary" ]]; then
  install_binary
else
  install_source
fi