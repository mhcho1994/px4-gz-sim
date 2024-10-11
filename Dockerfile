FROM ubuntu:22.04
LABEL maintainer="Minhyun Cho <cho515@purdue.edu>"

# environment variables to set pacakge manager
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# environment variable to designate the directory where temporary files and sockets are stored and accessible
ENV XDG_RUNTIME_DIR=/tmp/runtime-docker

# environment variables to enable the use of NVIDIA GPU
# ref. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# add bin folder as environmental variable
ENV PATH="/home/user/bin:${PATH}"

# protobuff related environmental variable
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# set default shell during Docker image build to bash
SHELL ["/bin/bash", "-l", "-c"]

# install base packages to run bash scripts
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y --quiet --no-install-recommends install \
    sudo \
    locales

# clear docker by removing unnecessary packages and emptying temporary folder
COPY /install/clean.sh /tmp/clean.sh
RUN chmod +x /tmp/clean.sh
RUN bash /tmp/clean.sh

# set and generate system locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
RUN locale-gen en_US.UTF-8 && \
	update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
	export LANG=en_US.UTF-8

# get the host user, group information to setup user
ARG HOST_USER_NAME HOST_USER_ID HOST_GROUP_NAME HOST_GROUP_ID

# add groups before we do anything that might add a new group
ARG GID_INPUT=107
ARG GID_RENDER=110
RUN sudo groupadd -r -g $GID_INPUT input \
 && sudo groupadd -r -g $GID_RENDER render

# setup user
ARG UID_USER=1000
ARG GID_USER=1000
RUN groupadd --gid $GID_USER user \
 && adduser --disabled-password --gecos '' user --uid $UID_USER --gid $UID_USER\
 && usermod -a -G sudo,plugdev,dialout,input,render,video user \
 && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user

# install required dependencies and packages using scripts (docker build cache)
# base
COPY --chown=user:user /install/base.sh /tmp/install/base.sh
RUN chmod +x /tmp/install/base.sh
RUN bash /tmp/install/base.sh

# ROS2 (Foxy)
COPY --chown=user:user /install/ros2.sh /tmp/install/ros2.sh
RUN chmod +x /tmp/install/ros2.sh
RUN bash /tmp/install/ros2.sh

# Gazebo garden (from binary)
COPY --chown=user:user /install/gazebo.sh /tmp/install/gazebo.sh
RUN chmod +x /tmp/install/gazebo.sh
RUN bash /tmp/install/gazebo.sh

# PX4 (PX4 dependencies)
COPY --chown=user:user /install/px4setup.sh /tmp/install/px4setup.sh
RUN chmod +x /tmp/install/px4setup.sh
RUN bash /tmp/install/px4setup.sh

# extra packages
COPY --chown=user:user /install/extra.sh /tmp/install/extra.sh
RUN chmod +x /tmp/install/extra.sh
RUN bash /tmp/install/extra.sh

# clear docker by removing unnecessary packages and emptying temporary folder
RUN bash /tmp/clean.sh

# enable apt auto-completion by deleting autoclean task (speed up Dockerfile)
RUN sudo rm /etc/apt/apt.conf.d/docker-clean

# create XDG runtime dir
RUN mkdir /tmp/runtime-docker && sudo chmod 700 /tmp/runtime-docker

# run user setup script
COPY --chown=user:user /install/usersetup.sh /tmp/install/usersetup.sh
RUN chmod +x /tmp/install/usersetup.sh
RUN bash /tmp/install/usersetup.sh

# setup entry point
COPY /install/entrypoint.sh /tmp/install/entrypoint.sh
RUN sudo chmod +x /tmp/install/entrypoint.sh
RUN sudo chsh -s /bin/bash user

# create workspace
RUN mkdir -p /home/user/work/px4
RUN mkdir -p /home/user/work/ros2_ws
WORKDIR /home/user/work

# change user to root and run entrypoint script
USER root
ENTRYPOINT ["/tmp/install/entrypoint.sh"]
