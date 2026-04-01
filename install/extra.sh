#!/usr/bin/env bash
# extra.sh — install extra dev/tools + GeographicLib datasets + QGroundControl prereqs
#
# Modes:
#   --mode deps   : install system dependencies only
#   --mode setup  : run runtime/user setup only
#   --mode all    : run both (default)
#
# Features:
#  - help
#  - debug enables `set -x`
#  - Installs common utilities + Python deps
#  - Installs MAVROS GeographicLib datasets
#  - Installs QGroundControl prerequisites
#  - Optionally downloads QGroundControl AppImage
#  - Ensures NumPy version is compatible with MAVProxy (numpy<2)
#
# Notes:
#  - We intentionally do NOT run `apt-get upgrade` by default to preserve reproducibility.
#    Use --upgrade if you really want it.
#  - QGC may require logout/login after adding user to dialout group.
#  - In Docker image build, user/group changes may be less useful than in runtime containers.
#  - MAVProxy itself is assumed to be installed elsewhere (e.g. autopilot.sh for ArduPilot deps).
#    This script only checks/fixes NumPy compatibility for MAVProxy.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"                                 # --debug
MODE="all"                                    # deps | setup | all
DO_UPGRADE="false"                            # --upgrade
INSTALL_QGC="true"                            # --no-qgc

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

QGC_DIR="${PROJECT_ROOT}/tools/QGC"
QGC_URL="https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-x86_64.AppImage"

# MAVProxy / NumPy compatibility controls
INSTALL_MAVPROXY_NUMPY_FIX="true"             # --no-mavproxy-numpy-fix
MAVPROXY_INSTALL_SCOPE="user"                 # user | system
MAVPROXY_NUMPY_SPEC="numpy<2"                 # pin for MAVProxy compatibility

help() {
  cat <<EOF
Usage:
  bash extra.sh [options]

Options:
  -h, --help            Show this help and exit (no command tracing)
  --debug               Enable command tracing (set -x)

  --mode MODE           What to run: deps | setup | all (default: ${MODE})
                        deps  = install apt/pip/system dependencies only
                        setup = runtime/user setup only
                        all   = run both

  --upgrade             Run apt-get upgrade (default: off)
  --no-qgc              Skip QGroundControl prerequisites + AppImage download

  --project-root PATH   Project root (default: ${PROJECT_ROOT})
  --qgc-dir PATH        Where to put QGC AppImage (default: ${QGC_DIR})
  --qgc-url URL         Optional direct URL to QGroundControl AppImage (default: ${QGC_URL})

  --no-mavproxy-numpy-fix
                        Skip NumPy compatibility check/fix for MAVProxy
  --mavproxy-scope S    NumPy install scope for MAVProxy compatibility: user | system
                        (default: ${MAVPROXY_INSTALL_SCOPE})
  --mavproxy-numpy SPEC NumPy version spec for MAVProxy compatibility
                        (default: ${MAVPROXY_NUMPY_SPEC})

Examples:
  # Dockerfile: dependencies only
  bash extra.sh --mode deps

  # Runtime in container: QGC setup only
  bash extra.sh --mode setup

  # Full install
  bash extra.sh --mode all

  # Explicit project root
  bash extra.sh --project-root /home/user/ws/flightstack_sim

  # Force NumPy below 2 for MAVProxy
  bash extra.sh --mode deps --mavproxy-numpy "numpy<2"

  # System-wide NumPy fix
  bash extra.sh --mode deps --mavproxy-scope system
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
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value (deps|setup|all)"
      MODE="$2"
      shift 2
      ;;
    --upgrade) DO_UPGRADE="true"; shift ;;
    --no-qgc) INSTALL_QGC="false"; shift ;;
    --project-root)
      [[ $# -ge 2 ]] || die "--project-root requires a path"
      PROJECT_ROOT="$2"
      QGC_DIR="${PROJECT_ROOT}/tools/QGC"
      shift 2
      ;;
    --qgc-dir)
      [[ $# -ge 2 ]] || die "--qgc-dir requires a path"
      QGC_DIR="$2"
      shift 2
      ;;
    --qgc-url)
      [[ $# -ge 2 ]] || die "--qgc-url requires a URL"
      QGC_URL="$2"
      shift 2
      ;;
    --no-mavproxy-numpy-fix)
      INSTALL_MAVPROXY_NUMPY_FIX="false"
      shift
      ;;
    --mavproxy-scope)
      [[ $# -ge 2 ]] || die "--mavproxy-scope requires a value (user|system)"
      MAVPROXY_INSTALL_SCOPE="$2"
      shift 2
      ;;
    --mavproxy-numpy)
      [[ $# -ge 2 ]] || die "--mavproxy-numpy requires a version spec"
      MAVPROXY_NUMPY_SPEC="$2"
      shift 2
      ;;
    *)
      die "Unknown option: $1 (use --help)"
      ;;
  esac
done

# --------------------------
# Validate
# --------------------------
if [[ "${MODE}" != "deps" && "${MODE}" != "setup" && "${MODE}" != "all" ]]; then
  die "Invalid --mode: ${MODE} (expected: deps|setup|all)"
fi

if [[ "${MAVPROXY_INSTALL_SCOPE}" != "user" && "${MAVPROXY_INSTALL_SCOPE}" != "system" ]]; then
  die "Invalid --mavproxy-scope: ${MAVPROXY_INSTALL_SCOPE} (expected: user|system)"
fi

# --------------------------
# Strict mode
# --------------------------
set -Eeuo pipefail
trap 'echo "[extra.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# Helpers
# --------------------------
apt_update() {
  sudo apt-get -y update
  if [[ "${DO_UPGRADE}" == "true" ]]; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
  else
    echo "==> Skipping apt-get upgrade (use --upgrade to enable)"
  fi
}

python_has_numpy() {
  python3 - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
PY
}

get_numpy_version() {
  python3 - <<'PY'
import numpy
print(numpy.__version__)
PY
}

numpy_major_version() {
  python3 - <<'PY'
import numpy
v = numpy.__version__.split(".")[0]
try:
    print(int(v))
except Exception:
    print(-1)
PY
}

# --------------------------
# Deps: common packages
# --------------------------
install_common_packages() {
  echo ""
  echo "==> Installing extra tools/dependencies (apt)"
  echo ""

  apt_update

  sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    htop \
    iproute2 \
    lcov \
    mesa-utils \
    openbox \
    python3-jinja2 \
    python3-numpy \
    python3-pip \
    python3-xdg \
    python3-xmltodict \
    qt5dxcb-plugin \
    screen \
    terminator \
    vim \
    libasio-dev \
    "libqt5*-dev" \ 
    ros-${ROS_DISTRO}-rosbag2 \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap

  echo ""
  echo "==> Installing extra Python packages (pip)"
  echo ""
  python3 -m pip install --upgrade pip
  python3 -m pip install pykwalify
}

# --------------------------
# Deps: GeographicLib datasets
# --------------------------
install_geographiclib_datasets() {
  echo ""
  echo "==> Installing MAVROS GeographicLib datasets"
  echo ""

  local tmp="/tmp/install_geographiclib_datasets.sh"
  wget -q "https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh" -O "${tmp}"
  sudo bash "${tmp}"
  rm -f "${tmp}"
}

# --------------------------
# Deps: QGC system prerequisites only
# --------------------------
install_qgc_prereqs() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
    echo "==> Skipping QGroundControl prerequisites (--no-qgc)"
    return 0
  fi

  echo ""
  echo "==> Installing QGroundControl system prerequisites"
  echo ""

  sudo apt-get remove -y modemmanager || true

  apt_update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    libfuse2 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor-dev
}

# --------------------------
# Deps: ensure NumPy compatibility for MAVProxy
# --------------------------
ensure_numpy_for_mavproxy() {
  if [[ "${INSTALL_MAVPROXY_NUMPY_FIX}" != "true" ]]; then
    echo "==> Skipping MAVProxy NumPy compatibility fix (--no-mavproxy-numpy-fix)"
    return 0
  fi

  echo ""
  echo "==> Checking NumPy compatibility for MAVProxy"
  echo "    scope=${MAVPROXY_INSTALL_SCOPE}"
  echo "    required spec=${MAVPROXY_NUMPY_SPEC}"
  echo ""

  local need_fix="false"

  if python_has_numpy; then
    local current_ver
    current_ver="$(get_numpy_version)"
    local current_major
    current_major="$(numpy_major_version)"

    echo "Detected NumPy version: ${current_ver}"

    if [[ "${current_major}" -ge 2 ]]; then
      echo "NumPy major version is >= 2, which may break MAVProxy-related modules."
      need_fix="true"
    else
      echo "NumPy version is already compatible with MAVProxy."
    fi
  else
    echo "NumPy is not currently importable from python3."
    need_fix="true"
  fi

  if [[ "${need_fix}" != "true" ]]; then
    return 0
  fi

  echo ""
  echo "==> Installing compatible NumPy for MAVProxy"
  echo ""

  if [[ "${MAVPROXY_INSTALL_SCOPE}" == "system" ]]; then
    sudo python3 -m pip install --upgrade pip setuptools wheel --break-system-packages
    sudo python3 -m pip install --upgrade "${MAVPROXY_NUMPY_SPEC}" --break-system-packages
  else
    python3 -m pip install --upgrade --user pip setuptools wheel
    python3 -m pip install --upgrade --user "${MAVPROXY_NUMPY_SPEC}"
  fi
}

verify_numpy_for_mavproxy() {
  if [[ "${INSTALL_MAVPROXY_NUMPY_FIX}" != "true" ]]; then
    return 0
  fi

  echo ""
  echo "==> Verifying NumPy compatibility for MAVProxy"
  echo ""

  python3 - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Python version   :", sys.version.split()[0])

import numpy
print("NumPy version    :", numpy.__version__)

major = int(numpy.__version__.split(".")[0])
assert major < 2, f"NumPy major version must be < 2 for MAVProxy compatibility, got {numpy.__version__}"

try:
    import MAVProxy  # noqa: F401
    print("MAVProxy import  : OK")
except Exception as e:
    print(f"MAVProxy import  : NOT VERIFIED ({e})")
PY

  echo ""
  echo "NumPy compatibility check finished."
  echo ""
}

# --------------------------
# Setup: user-level QGC permissions
# --------------------------
setup_qgc_user_access() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
    echo "==> Skipping QGroundControl user setup (--no-qgc)"
    return 0
  fi

  echo ""
  echo "==> Configuring QGroundControl user access"
  echo ""

  if [[ -n "${USER:-}" ]]; then
    sudo usermod -a -G dialout "${USER}" || true
    echo "Added user '${USER}' to dialout group (best-effort)."
  else
    echo "WARNING: USER is not set; skipping dialout group update."
  fi

  echo ""
  echo "IMPORTANT: You may need logout/login (or a new shell/session) for dialout group changes to take effect."
  echo ""
}

# --------------------------
# Setup: QGC AppImage download
# --------------------------
download_qgc_appimage() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
    echo "==> Skipping QGroundControl AppImage download (--no-qgc)"
    return 0
  fi

  mkdir -p "${QGC_DIR}"

  local out="${QGC_DIR}/QGroundControl-x86_64.AppImage"

  echo ""
  echo "==> Downloading QGroundControl AppImage"
  echo "    URL: ${QGC_URL}"
  echo "    OUT: ${out}"
  echo ""

  wget -O "${out}" "${QGC_URL}"
  chmod +x "${out}"

  echo ""
  echo "QGroundControl downloaded to:"
  echo "  ${out}"
  echo ""
  echo "To run:"
  echo "  ${out}"
  echo ""
}

# --------------------------
# Main
# --------------------------
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODE=${MODE}"
echo "DO_UPGRADE=${DO_UPGRADE}"
echo "INSTALL_QGC=${INSTALL_QGC}"
echo "QGC_DIR=${QGC_DIR}"
echo "QGC_URL=${QGC_URL}"
echo "INSTALL_MAVPROXY_NUMPY_FIX=${INSTALL_MAVPROXY_NUMPY_FIX}"
echo "MAVPROXY_INSTALL_SCOPE=${MAVPROXY_INSTALL_SCOPE}"
echo "MAVPROXY_NUMPY_SPEC=${MAVPROXY_NUMPY_SPEC}"

case "${MODE}" in
  deps)
    install_common_packages
    install_geographiclib_datasets
    install_qgc_prereqs
    ensure_numpy_for_mavproxy
    verify_numpy_for_mavproxy
    ;;
  setup)
    setup_qgc_user_access
    download_qgc_appimage
    ;;
  all)
    install_common_packages
    install_geographiclib_datasets
    install_qgc_prereqs
    ensure_numpy_for_mavproxy
    verify_numpy_for_mavproxy
    setup_qgc_user_access
    download_qgc_appimage
    ;;
esac

echo ""
echo "Extra packages installation DONE. MODE=${MODE}"