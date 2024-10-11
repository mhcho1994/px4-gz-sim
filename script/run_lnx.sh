#!/bin/bash

# show commands before execution and exit when errors occur
set -e -x

# install base packages to run bash scripts
sudo apt-get -y update && \
sudo apt-get -y upgrade && \
sudo apt-get -y --quiet --no-install-recommends install \
    locales

# set and generate system locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# get the host user, group information to setup user
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# add groups before we do anything that might add a new group
export GID_INPUT=107
export GID_RENDER=110
sudo groupadd -r -g $GID_INPUT input && \
sudo groupadd -r -g $GID_RENDER render

# setup user
sudo usermod -a -G sudo,plugdev,dialout,input,render,video $USER

# install required dependencies and packages using scripts
# base
sudo chown $USER:$USER ./install/base.sh
sudo chmod +x ./install/base.sh
bash ./install/base.sh

# ROS2 (Humble)
sudo chown $USER:$USER ./install/ros2.sh
sudo chmod +x ./install/ros2.sh
bash ./install/ros2.sh

# Gazebo (Garden from source)
sudo chown $USER:$USER ./install/gazebo.sh
sudo chmod +x ./install/gazebo.sh
bash ./install/gazebo.sh

# PX4 (PX4 dependencies)
sudo chown $USER:$USER ./install/px4setup.sh
sudo chmod +x ./install/px4setup.sh
bash ./install/px4setup.sh

# extra packages
sudo chown $USER:$USER ./install/extra.sh
sudo chmod +x ./install/extra.sh
bash ./install/extra.sh

# clear docker by removing unnecessary packages and emptying temporary folder
sudo bash ./install/clean.sh

# run user setup script
COPY --chown=user:user /install/usersetup.sh /tmp/install/usersetup.sh
RUN chmod +x /tmp/install/usersetup.sh
RUN bash /tmp/install/usersetup.sh
