#!/bin/bash

# exit when errors occur
set -e

if [ ! -d ./work/px4 ] ; then
    cd ./work
    git clone -b v1.15.4 https://github.com/PX4/PX4-Autopilot.git px4
    cd ..
fi

if [ ! -d ./work/ros2_ws/src ] ; then
    mkdir -p ./work/ros2_ws/src
    cd work/ros2_ws/src
    git clone -b release/1.15 https://github.com/PX4/px4_msgs.git px4_msgs

    # to be updated
    wget https://raw.githubusercontent.com/mhcho1994/px4-gz-multidrone/refs/heads/humble/install/gz_repos.yaml -O gz_repos.yaml
    vcs import < gz_repos.yaml
    # gzrepos delete
    cd ../../..
fi

