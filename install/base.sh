#!/bin/bash

# exit when errors occur
# set -e 

# update/upgrade packages and install required packages
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y --no-install-recommends install \
    bash-completion \
	build-essential \
	cmake \
	curl \
	git \
	gnupg \
	lsb-release \
	pkg-config \
	python3-pip \
	python3-setuptools \
	python3-venv \
	python3-wheel \
	unzip \
	wget \
	zip