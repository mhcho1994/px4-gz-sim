#!/bin/bash

# show commands before execution and exit when errors occur
set -e -x

# set the gazebo version to be installed
GAZEBO_VERSION=""

# install all necessary dependencies and gazebo
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get install --no-install-recommends -y \
    gazebo11 \
    libgazebo11-dev