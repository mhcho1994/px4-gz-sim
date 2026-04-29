#!/usr/bin/env bash
set -uo pipefail

# -----------------------------
# Pass host user info into container
# (used by entrypoint for UID/GID mapping)
# -----------------------------
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export HOST_USER_NAME="$(id -un)"
export HOST_GROUP_NAME="$(id -gn)"

# X11 forwarding for GUI apps (QGC, Gazebo)
export DISPLAY="${DISPLAY:-:0}"

RECREATE=false
USE_EXEC=false

# -----------------------------
# Optional flags
#   --rebuild : force container recreate
#   --exec    : enter container with bash shell
# -----------------------------
for arg in "$@"; do
    case "$arg" in
        --rebuild) RECREATE=true ;;
        --exec)    USE_EXEC=true ;;
    esac
done

SERVICE="fire_flightstack_sim"

# Setup sentinel used by entrypoint
SETUP_DONE=".docker_home/.setup_done"

cleanup() {
    xhost -local:docker || true
}

trap cleanup EXIT
xhost +local:docker


# -----------------------------
# Enter container either by:
# 1) attach to main process
# 2) exec into interactive shell
# -----------------------------
enter_container() {
    if $USE_EXEC; then
        echo "[INFO] Entering container via exec..."
        docker compose exec "${SERVICE}" bash
    else
        echo "[INFO] Attaching to container..."
        docker compose attach "${SERVICE}"
    fi
}


if $RECREATE; then
    # -----------------------------
    # Force container recreate only
    # Remove setup sentinel so entrypoint
    # reruns project setup
    # -----------------------------
    if [[ -f "${SETUP_DONE}" ]]; then
        echo "[INFO] Removing ${SETUP_DONE}"
        rm -f "${SETUP_DONE}"
    fi

    echo "[INFO] Recreating container..."
    docker compose up --force-recreate -d
    enter_container

else
    if docker compose ps -a --services --filter status=running \
        | grep -q "^${SERVICE}$"; then

        # -----------------------------
        # Container already running:
        # reuse it (no recreate)
        # -----------------------------
        echo "[INFO] Existing container already running."
        enter_container

    elif docker compose ps -a --services \
        | grep -q "^${SERVICE}$"; then

        # -----------------------------
        # Container exists but stopped:
        # restart existing one only
        # -----------------------------
        echo "[INFO] Starting existing container..."
        docker compose start "${SERVICE}"
        enter_container

    else
        # -----------------------------
        # First run:
        # build image and create container
        # -----------------------------
        echo "[INFO] No container found. Building first time..."
        docker compose up --build -d
        enter_container
    fi
fi