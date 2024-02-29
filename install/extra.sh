#!/bin/bash

# show commands before execution and exit when errors occur
set -e -x

# set the ros2 version to be installed
ROS_VERSION="foxy"

sudo apt-get -y update
sudo apt-get -y upgrade
sudo  apt-get install --no-install-recommends -y \
	htop \
	ipe \
	iproute2 \
	lcov \
	menu \
	mesa-utils \
	openbox \
	python3-jinja2 \
	python3-numpy \
	python3-vcstool \
	python3-xdg \
	python3-xmltodict \
	qt5dxcb-plugin \
	screen \
	terminator \
	vim \
    ros-${ROS_VERSION}-plotjuggler-ros \
    libqt5*-dev

sudo pip install pykwalify NavPy