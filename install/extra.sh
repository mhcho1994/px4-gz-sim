#!/bin/bash

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

# install necessary python libraries
pip install \
	pykwalify