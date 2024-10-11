#!/bin/bash

# show commands before execution and exit when errors occur
set -e -x

# set the ros2 version to be installed
ROS_VERSION="humble"

# add the alias commands to .bashrc 
echo "Add alias commands to ~/.bashrc"
cat << EOF >> /home/${USER}/.bashrc
echo sourcing /home/${USER}/.bashrc
source /opt/ros/${ROS_VERSION}/setup.bash
export CCACHE_TEMPDIR=/tmp/ccache
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONWARNINGS=ignore:::setuptools.installer,ignore:::setuptools.command.install
if [ -f ~/work/gazebo/install/setup.sh ]; then
source ~/work/gazebo/install/setup.sh
echo "gazebo built, sourcing"
fi
if [ -f ~/work/ros2_ws/install/setup.sh ]; then
source ~/work/ros2_ws/install/setup.sh
echo "workspace built, sourcing"
fi
source /usr/share/colcon_cd/function/colcon_cd.sh
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
EOF

# initialize rosdep
sudo rosdep init
rosdep update