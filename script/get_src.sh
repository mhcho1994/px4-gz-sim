#!/bin/bash

# show commands before execution 
set -e

if [ ! -d ./work/px4 ] ; then
    cd ./work
    git clone git@github.com:PX4/PX4-Autopilot.git px4
    cd px4
    git checkout tags/v1.15.0-alpha1
    cd ../..
fi

if [ ! -d ./work/ros2_ws/src ] ; then
    mkdir -p ./work/ros2_ws/src
    cd work/ros2_ws/src
    git clone git@github.com:PX4/px4_msgs.git
    cd ../../..
fi