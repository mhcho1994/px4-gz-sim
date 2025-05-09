#!/bin/bash

# exit when errors occur and print each command
set -e
set -x

# remove unnecessary packages
sudo apt-get clean -y
sudo apt-get autoremove --purge -y

# empty temporary files
shopt -s extglob
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf $(find /tmp/* -name "*" ! -name "clean.sh")
shopt -u extglob