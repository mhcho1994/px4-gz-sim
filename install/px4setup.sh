#!/bin/bash

# exit when errors occur and show commands before execution
set -e -x

wget https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.15.4/Tools/setup/ubuntu.sh -P /tmp/
wget https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.15.4/Tools/setup/requirements.txt -P /tmp/
bash /tmp/ubuntu.sh --no-sim-tools && rm /tmp/ubuntu.sh

cd /home/${USER}
# here we need to check whether DDS already exists...
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
