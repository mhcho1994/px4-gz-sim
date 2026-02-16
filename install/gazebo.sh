#!/usr/bin/env bash
# gazebo.sh — install/build Gazebo Harmonic on Ubuntu 22.04 (Jammy)
#
# Features:
#  - help
#  - debug enables `set -x`
#  - binary/source install modes for Gazebo (debian or colcon + vcs)
#  - binary/source ros_gz install, independently selectable
#
# Notes:
#  - We intentionally do NOT run `apt-get upgrade` to preserve reproducibility.
#  - ROS-GZ bridges for Harmonic can be installed as binaries (ros-humble-ros-gzharmonic).
#    But if you're source-building Gazebo for custom plugins, building ros_gz from source
#    against that overlay is often the most consistent setup.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"                     # --debug
INSTALL_MODE="binary"             # --install binary|source (Gazebo)
BUILD_TYPE="RelWithDebInfo"       # --build-type
ROS_DISTRO="humble"
GZ_VERSION="harmonic"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
GZ_WS_DIR="${PROJECT_ROOT}/gz/${GZ_VERSION}_ws"

# ros_gz control:
#  - ROS_GZ_MODE=auto  => Gazebo=binary -> ros_gz=binary, Gazebo=source -> ros_gz=source
#  - ROS_GZ_MODE=binary|source => force that mode regardless of Gazebo install
#  - --no-ros-gz disables ros_gz entirely
BUILD_ROS_GZ="true"               # --no-ros-gz
ROS_GZ_MODE="auto"                # --ros-gz auto|binary|source
ROS_GZ_WS_DIR="${PROJECT_ROOT}/ros2/ros_gz_ws"

# Where to fetch Gazebo repos from in source mode:
#   - default: official Harmonic collection file from gazebodistro
#   - you can override with --repos-yaml URL_OR_PATH
REPOS_YAML="https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/collection-harmonic.yaml"

help() {
  cat <<EOF
Usage:
  bash gazebo.sh [options]

Options:
  -h, --help                 Show this help and exit (no command tracing)
  --debug                    Enable command tracing (set -x)

  -i, --install MODE         Gazebo install mode: binary | source (default: ${INSTALL_MODE})

      --ros-gz MODE           ros_gz install mode: auto | binary | source (default: ${ROS_GZ_MODE})
                              auto => follows --install (binary->binary, source->source)
      --no-ros-gz             Skip installing ros_gz entirely

      --project-root PATH     Default: ${PROJECT_ROOT}
      --build-type TYPE       CMake build type (default: ${BUILD_TYPE})
      --repos-yaml SRC        (source mode) Repos YAML URL/path for Gazebo source
                              (default: official Harmonic collection)

Examples:
  # Gazebo binary + ros_gz binary (default auto)
  bash gazebo.sh --install binary

  # Gazebo source + ros_gz source (default auto)
  bash gazebo.sh --install source

  # Gazebo source, no ros_gz
  bash gazebo.sh --install source --no-ros-gz

  # Gazebo binary, but ros_gz from source
  bash gazebo.sh --install binary --ros-gz source
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
    --ros-gz)
      [[ $# -ge 2 ]] || die "--ros-gz requires an argument (auto|binary|source)"
      ROS_GZ_MODE="$2"
      shift 2
      ;;
    --no-ros-gz)
      BUILD_ROS_GZ="false"
      shift
      ;;
    --project-root)
      [[ $# -ge 2 ]] || die "--project-root requires a path"
      PROJECT_ROOT="$2"
      shift 2
      # refresh derived paths
      GZ_WS_DIR="${PROJECT_ROOT}/gz/${GZ_VERSION}_ws"
      ROS_GZ_WS_DIR="${PROJECT_ROOT}/ros/ros_gz_ws"
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

# Validate ros_gz mode
case "${ROS_GZ_MODE}" in
  auto|binary|source) ;;
  *) die "Invalid --ros-gz mode: ${ROS_GZ_MODE} (expected: auto|binary|source)" ;;
esac

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
# ros_gz helpers (binary/source)
# --------------------------
install_ros_gz_binary() {
  if [[ "${BUILD_ROS_GZ}" != "true" ]]; then
    echo "Skipping ROS-GZ bridge install (BUILD_ROS_GZ=false)"
    return 0
  fi

  echo ""
  echo "==> Installing ros_gz binaries (pairing-aware)"
  echo "    ROS_DISTRO=${ROS_DISTRO}, GZ_VERSION=${GZ_VERSION}"
  echo ""

  # Pairing logic:
  # - Humble officially pairs with Fortress (recommended).
  # - Humble + Harmonic is possible via non-official OSRF binaries: ros-humble-ros-gzharmonic,
  #   which can conflict with ros-humble-ros-gz* packages.
  if [[ "${ROS_DISTRO}:${GZ_VERSION}" == "humble:harmonic" ]]; then
    # Ensure Fortress-paired packages aren't installed (conflict risk).
    if dpkg -l | grep -q "^ii  ros-${ROS_DISTRO}-ros-gz "; then
      sudo apt-get -y remove "ros-${ROS_DISTRO}-ros-gz"
    fi
  fi

  case "${ROS_DISTRO}:${GZ_VERSION}" in
    humble:fortress)
      sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        "ros-${ROS_DISTRO}-ros-gz"
      ;;
    jazzy:harmonic)
      sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        "ros-${ROS_DISTRO}-ros-gz"
      ;;
    humble:harmonic)
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

install_ros_gz_source() {
  echo ""
  echo "==> Building ros_gz (${ROS_DISTRO}) from source"
  echo "    (works with Gazebo=${INSTALL_MODE}; uses overlay if available)"
  echo ""

  # Tooling
  sudo apt-get -y update
  sudo apt-get -y --no-install-recommends install \
    git \
    cmake \
    ninja-build \
    pkg-config \
    build-essential \
    python3-rosdep \
    python3-colcon-common-extensions

  # rosdep setup (safe if already initialized)
  sudo rosdep init 2>/dev/null || true
  rosdep update

  local ROS_GZ_WS="${ROS_GZ_WS_DIR}"
  local ROS_GZ_SRC="${ROS_GZ_WS}/src"
  mkdir -p "${ROS_GZ_SRC}"
  cd "${ROS_GZ_SRC}"

  if [[ ! -d ros_gz ]]; then
    git clone -b "${ROS_DISTRO}" https://github.com/gazebosim/ros_gz.git
  fi

  # Source ROS
  source "/opt/ros/${ROS_DISTRO}/setup.bash"

  # If Gazebo overlay exists (source install), source it; otherwise rely on system Gazebo
  if [[ -f "${GZ_WS_DIR}/install/setup.bash" ]]; then
    source "${GZ_WS_DIR}/install/setup.bash"
    echo "==> Using Gazebo overlay: ${GZ_WS_DIR}/install/setup.bash"
  else
    echo "==> No Gazebo overlay found at ${GZ_WS_DIR}/install/setup.bash"
    echo "    Building ros_gz against system Gazebo (ensure gz + dev packages are installed)."
    # Best-effort: install common bridge build deps if available.
    # (Exact Gazebo dev package names vary by distro/GZ version; users may already have them.)
    sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
      python3-vcstool || true
  fi

  cd "${ROS_GZ_WS}"
  rosdep install -r --from-paths src -i -y --rosdistro "${ROS_DISTRO}" || true

  echo "==> colcon build (ros_gz)"
  colcon build --merge-install \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
}

install_ros_gz() {
  if [[ "${BUILD_ROS_GZ}" != "true" ]]; then
    echo "==> Skipping ros_gz (--no-ros-gz)"
    return 0
  fi

  local mode="${ROS_GZ_MODE}"
  if [[ "${mode}" == "auto" ]]; then
    mode="${INSTALL_MODE}"
  fi

  case "${mode}" in
    binary) install_ros_gz_binary ;;
    source) install_ros_gz_source ;;
    *) die "Invalid resolved ros_gz mode: ${mode}" ;;
  esac
}

# --------------------------
# gz helpers (binary/source)
# --------------------------
install_gz_binary() {
  echo ""
  echo "binary -> Installing Gazebo (${GZ_VERSION}) from binaries"
  echo "         ROS_DISTRO=${ROS_DISTRO}, ROS_GZ_MODE=${ROS_GZ_MODE}, BUILD_ROS_GZ=${BUILD_ROS_GZ}"
  echo ""

  add_osrf_repo

  sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    "gz-${GZ_VERSION}"

  # ros_gz is handled separately
  install_ros_gz

  echo ""
  echo "Gazebo Binary Installation DONE."
  echo "Installed at: /usr/bin/gz"
  if [[ "${BUILD_ROS_GZ}" == "true" ]]; then
    echo "Installed at: /opt/ros/humble/share/ros_gz_bridge (if built from binary):"
    echo "To use the ros_gz overlay (if built from source):"
    echo "  source ${ROS_GZ_WS_DIR}/install/setup.bash"
  fi
  echo ""
}

install_gz_source() {
  echo ""
  echo "source -> Building Gazebo (${GZ_VERSION}) from source"
  echo "         repos: ${REPOS_YAML}"
  echo "         project-root: ${PROJECT_ROOT}"
  echo ""

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

  add_osrf_repo

  local GZ_WS="${GZ_WS_DIR}"
  local GZ_SRC="${GZ_WS}/src"
  mkdir -p "${GZ_SRC}"

  cd "${GZ_SRC}"
  fetch_repos_yaml "${REPOS_YAML}" "${GZ_SRC}/repos.yaml"
  vcs import < "${GZ_SRC}/repos.yaml"

  echo "==> Installing Gazebo source dependencies (apt)"
  sudo apt-get -y install \
    $(sort -u $(find . -iname "packages-$(lsb_release -cs).apt" -o -iname "packages.apt" | grep -v '/\.git/') \
      | sed '/gz\|sdf/d' | tr '\n' ' ')

  cd "${GZ_WS}"
  echo "==> colcon build (Gazebo ${GZ_VERSION})"
  colcon build --merge-install \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

  # ros_gz is handled separately
  install_ros_gz

  echo ""
  echo "Gazebo Source Installation DONE."
  echo "To use the source-built Gazebo overlay:"
  echo "  source ${GZ_WS}/install/setup.bash"
  if [[ "${BUILD_ROS_GZ}" == "true" ]]; then
    echo "Installed at: /opt/ros/humble/share/ros_gz_bridge (if built from binary):"
    echo "To use the ros_gz overlay (if built from source):"
    echo "  source ${ROS_GZ_WS_DIR}/install/setup.bash"
  fi
  echo ""
}

# --------------------------
# Main
# --------------------------
if [[ "${INSTALL_MODE}" == "binary" ]]; then
  install_gz_binary
else
  install_gz_source
fi