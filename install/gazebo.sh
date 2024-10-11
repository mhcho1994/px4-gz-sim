#!/bin/bash

# show commands before execution and exit when errors occur
set -e -x

# install all necessary dependencies to build gazebo from source
sudo apt install libeigen3-dev
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get -y update
sudo apt-get -y upgrade
sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
	python3-vcstool python3-colcon-common-extensions libgz-transport12-dev gz-transport12-cli
cd /tmp

wget https://raw.githubusercontent.com/mhcho1994/px4-gz-multidrone/refs/heads/humble/install/gz_repos.yaml -O gz_repos.yaml
vcs import < gz_repos.yaml

sudo apt-get -y install \
  $(sort -u $(find . -iname 'packages-'`lsb_release -cs`'.apt' -o -iname 'packages.apt' | grep -v '/\.git/') | sed '/gz\|sdf/d' | tr '\n' ' ')