#!/bin/bash

# show commands before execution and exit when errors occur
set -x -e 

# set the ros2 version to be installed
ROS_VERSION="humble"

# add the alias commands to .bashrc 
echo "Add alias commands to ~/.bashrc"
cat << EOF >> /home/${USER}/.bashrc
# source ros2
source /opt/ros/${ROS_VERSION}/setup.bash

# speed up C/C++ compilation by caching temporary build files in /tmp
export CCACHE_TEMPDIR=/tmp/ccache

# select Cyclone DDS as the middleware for ROS 2 communication.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# suppresses noisy or non-critical Python warnings from setuptools
export PYTHONWARNINGS=ignore:::setuptools.installer,ignore:::setuptools.command.install

# detect if running inside docker
if grep -qE '/docker/|/lxc/' /proc/1/cgroup 2>/dev/null; then
    WORK_PATH=~/work
else
    WORK_PATH=~/ws/px4_gz_sim/work
fi

# source the setup file if it exists
SETUP_FILE="$WORK_PATH/gazebo/install/setup.sh"
if [ -f "$SETUP_FILE" ]; then
    source "$SETUP_FILE"
    echo "gazebo built, sourcing from $SETUP_FILE"
else
    echo "setup.sh not found at $SETUP_FILE"
fi

SETUP_FILE="$WORK_PATH/ros2_ws/install/setup.sh"
if [ -f "$SETUP_FILE" ]; then
    source "$SETUP_FILE"
    echo "ros2 workspace built, sourcing from $SETUP_FILE"
else
    echo "setup.sh not found at $SETUP_FILE"
fi

# enable the colcon_cd function to quickly cd into ros2 packages
source /usr/share/colcon_cd/function/colcon_cd.sh

# enable shell tab-completion for colcon commands and options
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
EOF

# initialize rosdep
sudo rosdep init
rosdep update