#!/bin/bash

# exit when errors occur
# set -e 

# define a help function
help() {
  echo "Usage: source gazebo.sh [options]"
  echo ""
  echo "Options:"
  echo "  -i, --install [MODE]   Install mode: 'binary' or 'source'"
  echo "  -h, --help             Show this help message"
  return 0
}

# parse arguments with getopt
ARGS=$(getopt -o hi: --long help,install: -- "$@")
if [[ $? -ne 0 ]]; then
  help
  return 1
fi

# make sure the arguments are parsed properly
eval set -- "$ARGS"

# default install mode
INSTALL_MODE="binary"

# extract options and arguments
while true; do
  case "$1" in
    -i|--install)
      INSTALL_MODE="$2"
      if [[ "$INSTALL_MODE" != "binary" && "$INSTALL_MODE" != "source" ]]; then
        echo "Error: Invalid install mode: $INSTALL_MODE"
        help
        return 1
      fi
      shift 2
      ;;
    -h|--help)
      help
      return 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1"
      help
      return 1
      ;;
  esac
done

# build from binaries: install ignition-gazebo directly
function _install_from_binary() {

  echo ""
  echo "binary -> Gazebo installation from the binaries"
  echo ""

  GAZEBO_VERSION="harmonic"

  sudo apt-get -y update
  sudo apt-get -y install curl lsb-release gnupg

  sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
  sudo apt-get -y update
  sudo apt-get -y upgrade
  sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    gz-${GAZEBO_VERSION}
}

# build from sources: install all necessary dependencies to build ignition-gazebo from source
function _install_from_source() {

  echo ""
  echo "source -> Gazebo installation from the sources"
  echo ""

  sudo apt install libeigen3-dev
  sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

  sudo apt-get -y update
  sudo apt-get -y upgrade
  sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3-vcstool python3-colcon-common-extensions
  
  cd /tmp
  wget https://raw.githubusercontent.com/mhcho1994/px4-gz-multidrone/refs/heads/humble/install/gz_repos.yaml -O gz_repos.yaml
  vcs import < gz_repos.yaml

  sudo apt-get -y install \
    $(sort -u $(find . -iname 'packages-'`lsb_release -cs`'.apt' -o -iname 'packages.apt' | grep -v '/\.git/') | sed '/gz\|sdf/d' | tr '\n' ' ')
}

# perform installation logic
if [[ "$INSTALL_MODE" == "source" ]]; then
  echo "Installing Gazebo from source..."
  _install_from_source
else
  echo "Installing Gazebo from binary..."
  _install_from_binary
fi