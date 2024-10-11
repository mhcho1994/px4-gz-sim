#!/bin/bash

# show commands before execution 
set -e

if [ ! -d ./work/px4 ] ; then
    cd ./work
    git clone https://github.com/PX4/PX4-Autopilot.git px4
    cd px4
    git checkout tags/v1.15.0
    cd ../..
fi

if [ ! -d ./work/ros2_ws/src ] ; then
    mkdir -p ./work/ros2_ws/src
    cd work/ros2_ws/src
    git clone https://github.com/PX4/px4_msgs.git px4_msgs
    cd px4_msgs
    git checkout release/1.15
    cd ..
    wget https://raw.githubusercontent.com/mhcho1994/px4-gz-multidrone/refs/heads/humble/install/gz_repos.yaml -O gz_repos.yaml
    vcs import < gz_repos.yaml
    cd ../../..
fi

