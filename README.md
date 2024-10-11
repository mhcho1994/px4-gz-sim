# px4-gz-multidrone
Repository for PX4(v1.15.0)-ROS(Humble)-Gazebo(Garden) SiTL simulator \
This repository is created for personal research and for testing control algorithms. \
For low-level PX4 offboard control algorithm with safety filter, please refer to another repositories: https://github.com/mhcho1994/px4_ros_autopilot and https://github.com/Balt-AA/balt_go_pd. 
For original repository with recent versions of PX4, GZ, Gazebo-Ignition, \
please refer to the following links: https://github.com/kpant14/px4-gz-docker and https://github.com/CogniPilot/mixed_sense. \

### Installation and Launch
Two options are available for installing the simulator: installing at local Linux host or running a development container

#### 1. Local installation and running simulations at Linux host computer (WSL2)
If you try to set up PX4-ROS-Gazebo in your local environment,

```bash
source ./script/run_lnx.sh
```

Build the dockerfile and launch simulation environment
Note that this is not the finalized version of the container for SITL.
The current version uses the classic Gazebo, it will be replaced to the ignition Gazebo soon.

1. To get the sources required (run only once in the first time),

```bash
source ./script/get_src.sh
```

2. To build the dockerfile as an image or run the container (the command will build the image if the container does not exist), 

```bash
source run_dev.sh
```

3. To enter into the container,

```bash
docker exec -u user -it drones bash
```
you can use terminator instead of the bash terminal.

4. Build PX4-Autopilot inside the container,
(Check you are inside the container as user id 'user')

```bash
cd px4 && make px4_sitl
```

Run command below to launch a single drone simulation.

```bash
./Tools/simulation/gazebo-classic/sitl_multiple_run.sh -n 1 -m iris
```

