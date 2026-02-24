#!/usr/bin/env bash
# autopilot.sh — install autopilot deps (PX4 and/or ArduPilot) + optional extras
#
# Features:
#  - help
#  - debug enables `set -x`
#  - install PX4 deps via official ubuntu.sh (with --no-sim-tools)
#  - Micro XRCE-DDS Agent install modes:
#      (A) ROS 2 workspace build (colcon)  [DEFAULT]
#      (B) source build + /usr/local install (cmake + sudo make install)
#  - optional ArduPilot setup (repo dir prep) + ardupilot_gazebo plugin + SITL_Models
#
# Notes:
#  - This script assumes ROS 2 is already installed at /opt/ros/<distro>.
#  - PX4 uses uXRCE-DDS for ROS 2 native comms via Micro XRCE-DDS Agent.
#  - ArduPilot Gazebo plugin build may use either:
#      (A) Gazebo source overlay at ${PROJECT_ROOT}/gz/${GZ_VERSION}_ws/install/setup.bash, if present
#      (B) System-installed Gazebo (binary install), otherwise

# --------------------------
# Defaults
# --------------------------
DEBUG="false"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

# What to install
WITH_PX4="true"          # default: keep PX4 flow as-is
WITH_ARDUPILOT="false"   # default: off (enable with --with-ardupilot)

# PX4
PX4_REF="v1.16.1"                     # stable for 1.16 line
PX4_DIR="${PROJECT_ROOT}/ap/px4"      # in-project path

# Micro XRCE-DDS Agent
DDS_MODE="ros2"                       # ros2 | source  (DEFAULT: ros2 workspace build)
ROS_DISTRO="humble"
ROS2_WS_DIR="${PROJECT_ROOT}/ros2/px4_ros_uxrce_dds_ws"
DDS_AGENT_REF=""                      # if empty -> auto-pick by ROS_DISTRO
DDS_AGENT_DIR="${PROJECT_ROOT}/tools/Micro-XRCE-DDS-Agent"  # used only for DDS_MODE=source

# ArduPilot
ARDUPILOT_REF="Copter-4.6.2"
ARDUPILOT_DIR="${PROJECT_ROOT}/ap/ardupilot"

# ArduPilot Gazebo plugin + models
GZ_VERSION="harmonic"                 # used only to choose dev package name for plugin
ARDUPILOT_GZ_DIR="${PROJECT_ROOT}/gz/ardupilot_gazebo"
SITL_MODELS_DIR="${PROJECT_ROOT}/gz/SITL_Models"
ARDUPILOT_GZ_BUILD_TYPE="RelWithDebInfo"

# Optional: explicitly point to a Gazebo overlay setup.bash (if you built Gazebo from source)
# If empty, we auto-detect ${PROJECT_ROOT}/gz/${GZ_VERSION}_ws/install/setup.bash
GZ_OVERLAY_SETUP=""

help() {
  cat <<EOF
Usage:
  bash autopilot.sh [options]

Options:
  -h, --help                 Show this help and exit (no command tracing)
  --debug                    Enable command tracing (set -x)

  --project-root PATH        Project root (default: ${PROJECT_ROOT})

  # Select autopilots
  --no-px4                   Disable PX4 setup (default: off)
  --with-ardupilot           Enable ArduPilot setup (default: off)

  # PX4 controls
  --px4-ref REF              PX4 ref/tag/branch for setup scripts (default: ${PX4_REF})
  --px4-path PATH            Where PX4 repo should live (default: ${PX4_DIR})

  # Micro XRCE-DDS Agent controls
  --dds-mode MODE            Agent install mode: ros2 | source (default: ${DDS_MODE})
  --ros-distro DISTRO        ROS 2 distro (default: ${ROS_DISTRO})
  --ros2-ws PATH             ROS 2 workspace for agent build (default: ${ROS2_WS_DIR})
  --dds-ref REF              Micro-XRCE-DDS-Agent ref/tag (default: auto by ROS distro)
  --dds-path PATH            (dds-mode=source) Agent source dir (default: ${DDS_AGENT_DIR})

  # ArduPilot controls
  --ardupilot-ref REF        ArduPilot ref/tag/branch (default: ${ARDUPILOT_REF})
  --ardupilot-path PATH      Where ArduPilot repo should live (default: ${ARDUPILOT_DIR})

  # ArduPilot Gazebo plugin controls (installed when --with-ardupilot is enabled)
  --gz-version VER           Gazebo version name used for deps mapping (default: ${GZ_VERSION})
  --gz-overlay-setup PATH    Optional: setup.bash for Gazebo source overlay (default: auto-detect)
  --ardupilot-gz-path PATH   Where to clone ardupilot_gazebo (default: ${ARDUPILOT_GZ_DIR})
  --sitl-models-path PATH    Where to clone SITL_Models (default: ${SITL_MODELS_DIR})
  --ardupilot-gz-build-type  CMake build type for ardupilot_gazebo (default: ${ARDUPILOT_GZ_BUILD_TYPE})

Examples:
  # PX4 deps + XRCE agent (DEFAULT: ros2 workspace build)
  bash autopilot.sh

  # PX4 deps + XRCE agent via /usr/local (source install)
  bash autopilot.sh --dds-mode source

  # Change ROS distro + ws location
  bash autopilot.sh --ros-distro humble --ros2-ws ${PROJECT_ROOT}/ros2/px4_ros_uxrce_dds_ws

  # ArduPilot + plugin + SITL_Models (PX4 still on by default)
  bash autopilot.sh --with-ardupilot

  # ArduPilot-only
  bash autopilot.sh --no-px4 --with-ardupilot
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent sourcing
(return 0 2>/dev/null) && { echo "Do not source this script. Run: bash $0" >&2; return 1; }

# --------------------------
# Parse args (before set -x)
# --------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) help; exit 0 ;;
    --debug) DEBUG="true"; shift ;;

    --project-root)
      [[ $# -ge 2 ]] || die "--project-root requires a path"
      PROJECT_ROOT="$2"
      shift 2
      # refresh derived defaults
      PX4_DIR="${PROJECT_ROOT}/ap/px4"
      DDS_AGENT_DIR="${PROJECT_ROOT}/tools/Micro-XRCE-DDS-Agent"
      ROS2_WS_DIR="${PROJECT_ROOT}/ros2/px4_ros_uxrce_dds_ws"
      ARDUPILOT_DIR="${PROJECT_ROOT}/ap/ardupilot"
      ARDUPILOT_GZ_DIR="${PROJECT_ROOT}/gz/ardupilot_gazebo"
      SITL_MODELS_DIR="${PROJECT_ROOT}/gz/SITL_Models"
      ;;

    # autopilot selection
    --no-px4) WITH_PX4="false"; shift ;;
    --with-ardupilot) WITH_ARDUPILOT="true"; shift ;;

    # px4
    --px4-ref) [[ $# -ge 2 ]] || die "--px4-ref requires an argument"; PX4_REF="$2"; shift 2 ;;
    --px4-path) [[ $# -ge 2 ]] || die "--px4-path requires a path"; PX4_DIR="$2"; shift 2 ;;

    # dds / ros2 ws
    --dds-mode) [[ $# -ge 2 ]] || die "--dds-mode requires a value (ros2|source)"; DDS_MODE="$2"; shift 2 ;;
    --ros-distro) [[ $# -ge 2 ]] || die "--ros-distro requires a value"; ROS_DISTRO="$2"; shift 2 ;;
    --ros2-ws) [[ $# -ge 2 ]] || die "--ros2-ws requires a path"; ROS2_WS_DIR="$2"; shift 2 ;;
    --dds-ref) [[ $# -ge 2 ]] || die "--dds-ref requires a ref"; DDS_AGENT_REF="$2"; shift 2 ;;
    --dds-path) [[ $# -ge 2 ]] || die "--dds-path requires a path"; DDS_AGENT_DIR="$2"; shift 2 ;;

    # ardupilot
    --ardupilot-ref) [[ $# -ge 2 ]] || die "--ardupilot-ref requires a ref"; ARDUPILOT_REF="$2"; shift 2 ;;
    --ardupilot-path) [[ $# -ge 2 ]] || die "--ardupilot-path requires a path"; ARDUPILOT_DIR="$2"; shift 2 ;;

    # ardupilot gazebo plugin
    --gz-version) [[ $# -ge 2 ]] || die "--gz-version requires a value"; GZ_VERSION="$2"; shift 2 ;;
    --gz-overlay-setup) [[ $# -ge 2 ]] || die "--gz-overlay-setup requires a path"; GZ_OVERLAY_SETUP="$2"; shift 2 ;;
    --ardupilot-gz-path) [[ $# -ge 2 ]] || die "--ardupilot-gz-path requires a path"; ARDUPILOT_GZ_DIR="$2"; shift 2 ;;
    --sitl-models-path) [[ $# -ge 2 ]] || die "--sitl-models-path requires a path"; SITL_MODELS_DIR="$2"; shift 2 ;;
    --ardupilot-gz-build-type) [[ $# -ge 2 ]] || die "--ardupilot-gz-build-type requires a value"; ARDUPILOT_GZ_BUILD_TYPE="$2"; shift 2 ;;

    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# Validate DDS mode
if [[ "${DDS_MODE}" != "ros2" && "${DDS_MODE}" != "source" ]]; then
  die "Invalid --dds-mode: ${DDS_MODE} (expected: ros2|source)"
fi

# --------------------------
# Strict mode
# --------------------------
set -Ee
trap 'echo "[autopilot.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# DDS ref picker (auto by ROS distro if not set)
# --------------------------
pick_dds_ref() {
  # Rule:
  #  - dds-mode=source  => force v2.4.3
  #  - dds-mode=ros2    => if user set --dds-ref, honor it; else auto-pick by ROS_DISTRO

  if [[ "${DDS_MODE}" == "source" ]]; then
    if [[ -n "${DDS_AGENT_REF}" && "${DDS_AGENT_REF}" != "v2.4.3" ]]; then
      echo "WARNING: dds-mode=source forces DDS_AGENT_REF=v2.4.3 (ignoring --dds-ref ${DDS_AGENT_REF})." >&2
    fi
    DDS_AGENT_REF="v2.4.3"
    return 0
  fi

  # DDS_MODE == ros2
  if [[ -n "${DDS_AGENT_REF}" ]]; then
    return 0
  fi

  case "${ROS_DISTRO}" in
    humble|foxy) DDS_AGENT_REF="v2.4.2" ;;
    jazzy)       DDS_AGENT_REF="v2.4.3" ;;
    *)
      DDS_AGENT_REF="v2.4.2"
      echo "WARNING: Unknown ROS_DISTRO='${ROS_DISTRO}'. Defaulting DDS_AGENT_REF=${DDS_AGENT_REF} (override with --dds-ref)." >&2
      ;;
  esac
}

# --------------------------
# PX4: deps
# --------------------------
install_px4_deps() {
  echo ""
  echo "==> Installing PX4 dependencies (PX4_REF=${PX4_REF})"
  echo "    (using official Tools/setup/ubuntu.sh --no-sim-tools)"
  echo ""

  local tmpdir="/tmp/px4_setup"
  mkdir -p "${tmpdir}"

  local ubuntu_sh_url="https://raw.githubusercontent.com/PX4/PX4-Autopilot/${PX4_REF}/Tools/setup/ubuntu.sh"
  local req_txt_url="https://raw.githubusercontent.com/PX4/PX4-Autopilot/${PX4_REF}/Tools/setup/requirements.txt"

  wget -q "${ubuntu_sh_url}" -O "${tmpdir}/ubuntu.sh"
  wget -q "${req_txt_url}" -O "${tmpdir}/requirements.txt"
  chmod +x "${tmpdir}/ubuntu.sh"

  bash "${tmpdir}/ubuntu.sh" --no-sim-tools

  mkdir -p "${PX4_DIR}"
  echo "PX4 workspace directory prepared at: ${PX4_DIR}"
  echo "NOTE: This script installs dependencies only; clone PX4 repo separately if desired."
}

# --------------------------
# Micro XRCE-DDS Agent (MODE A): ROS 2 workspace build (colcon)  [DEFAULT]
# --------------------------
install_micro_xrce_agent_ros2_ws() {
  pick_dds_ref

  echo ""
  echo "==> Building Micro XRCE-DDS Agent in ROS 2 workspace (dds-mode=ros2)"
  echo "    ROS_DISTRO=${ROS_DISTRO}"
  echo "    ROS2_WS_DIR=${ROS2_WS_DIR}"
  echo "    DDS_AGENT_REF=${DDS_AGENT_REF}"
  echo ""

  local ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
  [[ -f "${ros_setup}" ]] || die "ROS 2 setup not found: ${ros_setup} (install ROS 2 ${ROS_DISTRO} first)"

  sudo apt-get -y update
  sudo apt-get -y --no-install-recommends install \
    git \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

  sudo rosdep init 2>/dev/null || true
  rosdep update

  local src_dir="${ROS2_WS_DIR}/src"
  mkdir -p "${src_dir}"
  cd "${src_dir}"

  if [[ ! -d "Micro-XRCE-DDS-Agent/.git" ]]; then
    git clone -b "${DDS_AGENT_REF}" https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
  else
    git -C Micro-XRCE-DDS-Agent fetch --all --tags || true
    git -C Micro-XRCE-DDS-Agent checkout "${DDS_AGENT_REF}"
    git -C Micro-XRCE-DDS-Agent pull --ff-only || true
  fi

  # shellcheck disable=SC1090
  source "${ros_setup}"
  cd "${ROS2_WS_DIR}"

  # deps resolution best-effort
  rosdep install -r --from-paths src -i -y --rosdistro "${ROS_DISTRO}" || true

  colcon build

  echo ""
  echo "Micro XRCE-DDS Agent built in workspace."
  echo "To run:"
  echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
  echo "  source ${ROS2_WS_DIR}/install/local_setup.bash"
  echo "  MicroXRCEAgent udp4 -p 8888"
  echo ""
}

# --------------------------
# Micro XRCE-DDS Agent (MODE B): source build + /usr/local install
# --------------------------
install_micro_xrce_agent_source() {
  pick_dds_ref

  echo ""
  echo "==> Building/installing Micro XRCE-DDS Agent to /usr/local (dds-mode=source)"
  echo "    ref=${DDS_AGENT_REF}"
  echo "    dir=${DDS_AGENT_DIR}"
  echo ""

  sudo apt-get -y update
  sudo apt-get -y --no-install-recommends install \
    git \
    cmake \
    build-essential

  mkdir -p "$(dirname "${DDS_AGENT_DIR}")"
  if [[ ! -d "${DDS_AGENT_DIR}/.git" ]]; then
    git clone -b "${DDS_AGENT_REF}" https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "${DDS_AGENT_DIR}"
  else
    git -C "${DDS_AGENT_DIR}" fetch --all --tags || true
    git -C "${DDS_AGENT_DIR}" checkout "${DDS_AGENT_REF}"
    git -C "${DDS_AGENT_DIR}" pull --ff-only || true
  fi

  pushd "${DDS_AGENT_DIR}" >/dev/null
  mkdir -p build
  cd build
  cmake ..
  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
  popd >/dev/null

  echo "Micro XRCE-DDS Agent installed to /usr/local."
}

install_micro_xrce_agent() {
  case "${DDS_MODE}" in
    ros2)  install_micro_xrce_agent_ros2_ws ;;
    source) install_micro_xrce_agent_source ;;
    *) die "Internal error: unknown DDS_MODE=${DDS_MODE}" ;;
  esac
}

# --------------------------
# ArduPilot: deps (SITL)
# --------------------------
install_ardupilot_deps() {
  echo ""
  echo "==> Installing ArduPilot dependencies (ARDUPILOT_REF=${ARDUPILOT_REF})"
  echo "    (using Tools/environment_install/install-prereqs-ubuntu.sh -y)"
  echo ""

  # IMPORTANT: do NOT run as root; the script uses sudo internally.
  # deps-only: disable environment modifications / completion / git submodule updates
  export DO_AP_STM_ENV=0
  export SKIP_AP_COMPLETION_ENV=1
  export SKIP_AP_GIT_CHECK=1

  local tmpdir="/tmp/ardupilot_setup_repo"
  rm -rf "${tmpdir}"
  mkdir -p "${tmpdir}"

  git clone --depth 1 --filter=blob:none --sparse --branch ${ARDUPILOT_REF} https://github.com/ArduPilot/ardupilot.git "$tmpdir"

  pushd "${tmpdir}" >/dev/null

  git sparse-checkout set Tools/environment_install Tools/completion

  bash Tools/environment_install/install-prereqs-ubuntu.sh -y

  popd >/dev/null
  rm -rf "${tmpdir}"

  mkdir -p "${ARDUPILOT_DIR}"
  echo "ArduPilot workspace directory prepared at: ${ARDUPILOT_DIR}"
  echo "NOTE: This installs deps only; clone ArduPilot separately if desired."
}

# --------------------------
# ArduPilot: ardupilot_gazebo plugin + SITL_Models
# --------------------------
detect_or_set_gz_overlay() {
  if [[ -n "${GZ_OVERLAY_SETUP}" ]]; then
    [[ -f "${GZ_OVERLAY_SETUP}" ]] || die "--gz-overlay-setup not found: ${GZ_OVERLAY_SETUP}"
    return 0
  fi

  local candidate="${PROJECT_ROOT}/gz/${GZ_VERSION}_ws/install/setup.bash"
  if [[ -f "${candidate}" ]]; then
    GZ_OVERLAY_SETUP="${candidate}"
  else
    GZ_OVERLAY_SETUP=""
  fi
}

install_ardupilot_gazebo_plugin() {
  echo ""
  echo "==> Installing ArduPilot Gazebo plugin (ArduPilot/ardupilot_gazebo)"
  echo "    GZ_VERSION=${GZ_VERSION}"
  echo "    dir=${ARDUPILOT_GZ_DIR}"
  echo ""

  local gz_sim_dev_pkg=""
  case "${GZ_VERSION}" in
    harmonic) gz_sim_dev_pkg="libgz-sim8-dev" ;;
    garden)   gz_sim_dev_pkg="libgz-sim7-dev" ;;
    ionic)
      gz_sim_dev_pkg="libgz-sim8-dev"
      echo "WARNING: GZ_VERSION=ionic detected; defaulting dev pkg to ${gz_sim_dev_pkg}."
      ;;
    *) die "Unsupported GZ_VERSION for ArduPilot plugin deps: ${GZ_VERSION}" ;;
  esac

  sudo apt-get -y update
  sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    git \
    cmake \
    ninja-build \
    pkg-config \
    build-essential \
    "${gz_sim_dev_pkg}" \
    rapidjson-dev \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl

  mkdir -p "$(dirname "${ARDUPILOT_GZ_DIR}")"
  if [[ ! -d "${ARDUPILOT_GZ_DIR}/.git" ]]; then
    git clone https://github.com/ArduPilot/ardupilot_gazebo "${ARDUPILOT_GZ_DIR}"
  else
    git -C "${ARDUPILOT_GZ_DIR}" pull --ff-only
  fi

  mkdir -p "${ARDUPILOT_GZ_DIR}/build"
  pushd "${ARDUPILOT_GZ_DIR}/build" >/dev/null

  detect_or_set_gz_overlay
  if [[ -n "${GZ_OVERLAY_SETUP}" ]]; then
    source "${GZ_OVERLAY_SETUP}"
    echo "==> Using Gazebo overlay for build: ${GZ_OVERLAY_SETUP}"
  else
    echo "==> No Gazebo overlay detected; building against system Gazebo."
  fi

  cmake .. -DCMAKE_BUILD_TYPE="${ARDUPILOT_GZ_BUILD_TYPE}"
  cmake --build . -j"$(nproc)"
  popd >/dev/null

  echo ""
  echo "==> ArduPilot Gazebo plugin built at:"
  echo "    ${ARDUPILOT_GZ_DIR}/build"
  echo ""
}

install_sitl_models() {
  echo ""
  echo "==> Installing SITL_Models (ArduPilot/SITL_Models)"
  echo "    dir=${SITL_MODELS_DIR}"
  echo ""

  mkdir -p "$(dirname "${SITL_MODELS_DIR}")"
  if [[ ! -d "${SITL_MODELS_DIR}/.git" ]]; then
    git clone https://github.com/ArduPilot/SITL_Models "${SITL_MODELS_DIR}"
  else
    git -C "${SITL_MODELS_DIR}" pull --ff-only
  fi
}

write_ardupilot_gz_env_snippet() {
  local env_file="${PROJECT_ROOT}/gz/ardupilot_gz_env.sh"
  mkdir -p "$(dirname "${env_file}")"

  cat > "${env_file}" <<EOF
# Generated by autopilot.sh
# Suggested usage: source this from usersetup.sh (do not edit by hand)

export GZ_VERSION=${GZ_VERSION}

# ArduPilot Gazebo plugin (.so) location
export GZ_SIM_SYSTEM_PLUGIN_PATH=${ARDUPILOT_GZ_DIR}/build:\${GZ_SIM_SYSTEM_PLUGIN_PATH}

# ArduPilot Gazebo resources (models/worlds)
export GZ_SIM_RESOURCE_PATH=${ARDUPILOT_GZ_DIR}/models:${ARDUPILOT_GZ_DIR}/worlds:\${GZ_SIM_RESOURCE_PATH}

# Optional: extra models from SITL_Models (if present)
if [ -d "${SITL_MODELS_DIR}/Gazebo" ]; then
  export GZ_SIM_RESOURCE_PATH=${SITL_MODELS_DIR}/Gazebo:\${GZ_SIM_RESOURCE_PATH}
fi
EOF

  echo ""
  echo "==> Wrote env snippet:"
  echo "    ${env_file}"
  echo ""
}

# --------------------------
# Main
# --------------------------
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "WITH_PX4=${WITH_PX4}"
echo "WITH_ARDUPILOT=${WITH_ARDUPILOT}"
echo "DDS_MODE=${DDS_MODE}"
echo "ROS_DISTRO=${ROS_DISTRO}"
echo "ROS2_WS_DIR=${ROS2_WS_DIR}"
echo "DDS_AGENT_REF=${DDS_AGENT_REF:-<auto>}"

if [[ "${WITH_PX4}" == "true" ]]; then
  install_px4_deps
  install_micro_xrce_agent
fi

if [[ "${WITH_ARDUPILOT}" == "true" ]]; then
  install_ardupilot_deps
  install_ardupilot_gazebo_plugin
  install_sitl_models
  write_ardupilot_gz_env_snippet
fi

echo ""
echo "Autopilot Dependencies Installation DONE."