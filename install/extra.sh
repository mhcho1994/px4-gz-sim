#!/bin/bash

# exit when errors occur and print each command
set -e
set -x

# install some extra dependencies and necessary programs
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
	python3-xdg \
	python3-xmltodict \
	qt5dxcb-plugin \
	screen \
	terminator \
	vim \
    libqt5*-dev \
	libasio-dev

# install mavros geographic library
sudo wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash install_geographiclib_datasets.sh && sudo rm install_geographiclib_datasets.sh

# install necessary python libraries
pip install \
	pykwalify