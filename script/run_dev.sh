#!/usr/bin/env bash
set -euo pipefail
set -x

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export HOST_USER_NAME="$(id -un)"
export HOST_GROUP_NAME="$(id -gn)"
export DISPLAY="${DISPLAY:-:0}"

cleanup() {
    xhost -local:docker || true
}

trap cleanup EXIT

xhost +local:docker
docker compose up --build