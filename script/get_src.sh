#!/bin/bash

# exit when errors occur
set -e

# check whether px4 directory exists and clone sources if it is empty
if [ ! -d ./work/px4 ] || [ -z "$(ls -A ./work/px4)" ]; then
    cd ./work
    git clone -b v1.15.4 https://github.com/PX4/PX4-Autopilot.git px4
    cd ..
fi

# check whether ros2 directory exists
if [ ! -d ./work/ros2_ws/src ]; then
    mkdir -p ./work/ros2_ws/src
fi

# clone sources if ros2 directory is empty
if [ ! "$(ls -A ./work/ros2_ws/src)" ]; then
    cd ./work/ros2_ws/src
    git clone -b release/1.15 https://github.com/PX4/px4_msgs.git px4_msgs
    cd ../../..
fi

# check whether gazebo directory exists
if [ ! -d ./work/gazebo/src ]; then
    mkdir -p ./work/gazebo/src
fi

# clone sources if gazebo directory is empty
if [ ! "$(ls -A ./work/gazebo/src)" ]; then
    cd ./work/gazebo/src
    wget https://raw.githubusercontent.com/mhcho1994/px4-gz-multidrone/refs/heads/humble/install/gz_repos.yaml -O gz_repos.yaml
    vcs import < gz_repos.yaml
    rm -rf gz_repos.yaml
    cd ../../..
fi

