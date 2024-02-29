#!/bin/bash

# show commands before execution 
set -e
# get user/group id to run the container
if [[ -z "$HOST_UID" ]]; then
    echo "ERROR: please set HOST_UID" >&2
    exit 1
fi

if [[ -z "$HOST_GID" ]]; then
    echo "ERROR: please set HOST_GID" >&2
    exit 1
fi

echo running as user: $HOST_UID:$HOST_GID

# use this code if you want to modify an existing user account
groupmod --gid "$HOST_GID" user
usermod --uid "$HOST_UID" user

# use this code to change the ownership of the user directory
chown user:user -R /home/user

# drop privileges and execute next container command, or 'bash' if not specified
if [[ $# -gt 0 ]]; then
    exec sudo -u user -H -- "$@"
else
    echo 'here'
    exec sudo -u user -H -- bash
fi