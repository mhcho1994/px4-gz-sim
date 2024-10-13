#!/bin/bash

# exit when errors occur and show commands before execution
set -e # -x

wget https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.15.0/Tools/setup/ubuntu.sh -P /tmp/
wget https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.15.0/Tools/setup/requirements.txt -P /tmp/
bash /tmp/ubuntu.sh --no-sim-tools && rm /tmp/ubuntu.sh

cd /home/${USER}
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
git checkout v2.4.0
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
