#!/usr/bin/env bash
# extra.sh — install extra dev/tools + GeographicLib datasets + QGroundControl prereqs
#
# Features:
#  - help
#  - debug enables `set -x`
#  - Installs common utilities + Python deps
#  - Installs MAVROS GeographicLib datasets
#  - Installs QGroundControl prerequisites (and optionally downloads AppImage)
#
# Notes:
#  - We intentionally do NOT run `apt-get upgrade` by default to preserve reproducibility.
#    Use --upgrade if you really want it.
#  - QGC requires logout/login after adding user to dialout group.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"                              # --debug
DO_UPGRADE="false"                         # --upgrade
INSTALL_QGC="true"                         # --no-qgc
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
QGC_DIR="${PROJECT_ROOT}/tools"            # <-- updated default
QGC_URL="https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-x86_64.AppImage"
                                 # --qgc-url (optional override)

help() {
  cat <<EOF
Usage:
  bash extra.sh [options]

Options:
  -h, --help            Show this help and exit (no command tracing)
  --debug               Enable command tracing (set -x)
  --upgrade             Run apt-get upgrade (default: off)
  --no-qgc              Skip QGroundControl prerequisites + AppImage download
  --project-root PATH   Project root (default: ${PROJECT_ROOT})
  --qgc-dir PATH        Where to put QGC AppImage (default: ${QGC_DIR})
  --qgc-url URL         Optional: direct URL to QGroundControl-x86_64.AppImage (default: not set)

Examples:
  # Default install (extras + GeographicLib + QGC prereqs)
  bash extra.sh

  # Set explicit project root (so QGC goes to <root>/tools)
  bash extra.sh --project-root /home/user/ws/flightstack_sim
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
    --upgrade) DO_UPGRADE="true"; shift ;;
    --no-qgc) INSTALL_QGC="false"; shift ;;
    --project-root)
      [[ $# -ge 2 ]] || die "--project-root requires a path"
      PROJECT_ROOT="$2"
      QGC_DIR="${PROJECT_ROOT}/tools"   # refresh default when project root changes
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
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# --------------------------
# Strict mode (after help)
# --------------------------
set -Ee
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
    python3-xdg \
    python3-xmltodict \
    qt5dxcb-plugin \
    screen \
    terminator \
    vim \
    libasio-dev \
    "libqt5*-dev"

  echo ""
  echo "==> Installing extra Python packages (pip)"
  echo ""
  python3 -m pip install --upgrade pip
  python3 -m pip install pykwalify
}

install_geographiclib_datasets() {
  echo ""
  echo "==> Installing MAVROS GeographicLib datasets"
  echo ""

  local tmp="/tmp/install_geographiclib_datasets.sh"
  wget -q "https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh" -O "${tmp}"
  sudo bash "${tmp}"
  rm -f "${tmp}"
}

install_qgc_prereqs() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
    echo "==> Skipping QGroundControl prerequisites (--no-qgc)"
    return 0
  fi

  echo ""
  echo "==> Installing QGroundControl prerequisites"
  echo ""

  sudo usermod -a -G dialout "$USER" || true
  sudo apt-get remove modemmanager -y || true

  apt_update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    libfuse2 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor-dev

  echo ""
  echo "IMPORTANT: You must logout/login for dialout group changes to take effect."
  echo ""
}

download_qgc_appimage() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
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
  echo "NOTE:"
  echo "  You must logout/login for dialout group permission to take effect."
  echo ""
}

# --------------------------
# Main
# --------------------------
install_common_packages
install_geographiclib_datasets
install_qgc_prereqs
download_qgc_appimage

echo ""
echo "Extra Packages Installation DONE."
