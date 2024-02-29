#!/bin/bash

# show commands before execution and exit when errors occur
set -e -x

# set the ros2 version to be installed
ROS_VERSION="foxy"

# enable Ubuntu universe repository
sudo apt-get -y --no-install-recommends install software-properties-common
sudo add-apt-repository universe

# add the ros2 GPG key and the repository to sources list
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# update/upgrade packages and install required packages
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y --no-install-recommends install \
	ros-${ROS_VERSION}-desktop \
	ros-${ROS_VERSION}-cyclonedds \
	ros-${ROS_VERSION}-rmw-cyclonedds-cpp \
	ros-${ROS_VERSION}-gps-msgs \
    ros-${ROS_VERSION}-ros-ign-bridge \
    ros-${ROS_VERSION}-ros-ign-gazebo-demos \
    ros-${ROS_VERSION}-ros-ign-image \
    ros-${ROS_VERSION}-mavlink \
	ros-dev-tools \
	python3-rosdep \
    python3-rospkg \
	python3-colcon-common-extensions \
	libgflags-dev