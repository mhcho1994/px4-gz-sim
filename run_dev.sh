#!/bin/bash

# show commands before execution 
set -e

# set the current user name/group/uid/gid as environment variables
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# enable the communication between containers and X windows in the host
xhost +local:docker
docker compose up