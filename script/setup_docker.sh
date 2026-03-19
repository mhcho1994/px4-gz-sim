#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# setup_docker.sh
# Build (and optionally run) the Docker image for FIRE_flightstack_sim.
#
# Main roles:
#   - build Docker image with host UID/GID mapping
#   - optionally run container with bind mounts
#   - optionally enable GPU and X11 forwarding
#
# Example:
#   ./setup_docker.sh
#   ./setup_docker.sh --tag dev
#   ./setup_docker.sh --run
#   ./setup_docker.sh --run --gpu --x11 --mount-src
# -----------------------------------------------------------------------------

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
CURRENT_USER="${USER:-$(id -un)}"

GID_INPUT="${GID_INPUT:-107}"
GID_RENDER="${GID_RENDER:-110}"

IMAGE_NAME="${IMAGE_NAME:-fire_flightstack_sim}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-fire_flightstack_sim}"

DO_BUILD=1
DO_RUN=0
USE_GPU=0
USE_X11=0
MOUNT_SRC=0
USE_HOST_NET=1
REMOVE_ON_EXIT=0
DEBUG=0

log() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARN] $*" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build and optionally run the Docker image for FIRE_flightstack_sim.

Options:
  --image NAME           Docker image name (default: ${IMAGE_NAME})
  --tag TAG              Docker image tag (default: ${IMAGE_TAG})
  --container NAME       Container name (default: ${CONTAINER_NAME})

  --build-only           Build image only (default behavior)
  --run                  Build and run container
  --no-build             Skip build step, only run container

  --gpu                  Enable NVIDIA GPU support at runtime
  --x11                  Enable X11 forwarding for GUI apps
  --mount-src            Bind-mount project source into container workspace

  --no-host-net          Do not use host networking
  --rm                   Remove container automatically on exit

  --debug                Enable bash debug mode
  -h, --help             Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") --tag dev
  $(basename "$0") --run
  $(basename "$0") --run --gpu --x11 --mount-src
EOF
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

require_file() {
    local file="$1"
    [[ -f "$file" ]] || die "Required file not found: $file"
}

ensure_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log "Creating directory: $dir"
        mkdir -p "$dir"
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --container)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --build-only)
                DO_BUILD=1
                DO_RUN=0
                shift
                ;;
            --run)
                DO_RUN=1
                shift
                ;;
            --no-build)
                DO_BUILD=0
                shift
                ;;
            --gpu)
                USE_GPU=1
                shift
                ;;
            --x11)
                USE_X11=1
                shift
                ;;
            --mount-src)
                MOUNT_SRC=1
                shift
                ;;
            --no-host-net)
                USE_HOST_NET=0
                shift
                ;;
            --rm)
                REMOVE_ON_EXIT=1
                shift
                ;;
            --debug)
                DEBUG=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
    done
}

check_prereqs() {
    require_cmd docker
    require_file "$PROJECT_ROOT/Dockerfile"
    require_file "$PROJECT_ROOT/install/base.sh"
    require_file "$PROJECT_ROOT/install/ros2.sh"
    require_file "$PROJECT_ROOT/install/gazebo.sh"
    require_file "$PROJECT_ROOT/install/autopilot.sh"
    require_file "$PROJECT_ROOT/install/extra.sh"
    require_file "$PROJECT_ROOT/install/entrypoint.sh"

    if [[ ! -f "$PROJECT_ROOT/install/clean.sh" ]]; then
        warn "install/clean.sh not found. Your Dockerfile currently runs /tmp/clean.sh."
        warn "You may want to add: COPY install/clean.sh /tmp/clean.sh"
    fi
}

prepare_host_dirs() {
    log "Preparing host directories under project root"
    ensure_dir "$PROJECT_ROOT/data"
    ensure_dir "$PROJECT_ROOT/ws"
    ensure_dir "$PROJECT_ROOT/ros2"
    ensure_dir "$PROJECT_ROOT/gz"
    ensure_dir "$PROJECT_ROOT/ap"
    ensure_dir "$PROJECT_ROOT/.docker_home"
}

build_image() {
    local image_ref="${IMAGE_NAME}:${IMAGE_TAG}"

    log "Building Docker image: ${image_ref}"
    log "Project root: ${PROJECT_ROOT}"
    log "Host UID:GID = ${HOST_UID}:${HOST_GID}"
    log "GID_INPUT=${GID_INPUT}, GID_RENDER=${GID_RENDER}"

    docker build \
        --build-arg UID_USER="${HOST_UID}" \
        --build-arg GID_USER="${HOST_GID}" \
        --build-arg GID_INPUT="${GID_INPUT}" \
        --build-arg GID_RENDER="${GID_RENDER}" \
        -t "${image_ref}" \
        -f "${PROJECT_ROOT}/Dockerfile" \
        "${PROJECT_ROOT}"

    log "Docker image built successfully: ${image_ref}"
}

build_run_cmd() {
    local image_ref="${IMAGE_NAME}:${IMAGE_TAG}"
    local -a cmd
    cmd=(docker run -it)

    if [[ "${REMOVE_ON_EXIT}" -eq 1 ]]; then
        cmd+=(--rm)
    fi

    cmd+=(--name "${CONTAINER_NAME}")

    if [[ "${USE_HOST_NET}" -eq 1 ]]; then
        cmd+=(--network host)
    fi

    if [[ "${USE_GPU}" -eq 1 ]]; then
        cmd+=(--gpus all)
        cmd+=(-e NVIDIA_VISIBLE_DEVICES=all)
        cmd+=(-e NVIDIA_DRIVER_CAPABILITIES=all)
    fi

    cmd+=(-e TZ=America/New_York)
    cmd+=(-e XDG_RUNTIME_DIR=/tmp/runtime-docker)
    cmd+=(-e DISPLAY="${DISPLAY:-:0}")
    cmd+=(-e QT_X11_NO_MITSHM=1)

    # Common device access
    [[ -e /dev/dri ]] && cmd+=(--device /dev/dri)
    [[ -e /dev/input ]] && cmd+=(-v /dev/input:/dev/input:ro)

    # Serial devices if needed
    [[ -e /dev/ttyUSB0 ]] && cmd+=(--device /dev/ttyUSB0)
    [[ -e /dev/ttyACM0 ]] && cmd+=(--device /dev/ttyACM0)

    if [[ "${USE_X11}" -eq 1 ]]; then
        if [[ -n "${DISPLAY:-}" ]]; then
            cmd+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
        else
            warn "DISPLAY is not set. X11 apps may not work."
        fi

        if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
            cmd+=(-e XAUTHORITY=/tmp/.docker.xauth)
            cmd+=(-v "${XAUTHORITY}:/tmp/.docker.xauth:ro")
        else
            warn "XAUTHORITY is not set or file does not exist. GUI auth may fail."
        fi
    fi

    # Persistent home-ish area inside project root if needed later
    cmd+=(-v "${PROJECT_ROOT}/.docker_home:/home/user/.host_persist")

    # Common working data mounts
    [[ -d "${PROJECT_ROOT}/data" ]] && cmd+=(-v "${PROJECT_ROOT}/data:/home/user/FIRE_flightstack_sim/data")
    [[ -d "${PROJECT_ROOT}/ws"   ]] && cmd+=(-v "${PROJECT_ROOT}/ws:/home/user/FIRE_flightstack_sim/ws")
    [[ -d "${PROJECT_ROOT}/ros2" ]] && cmd+=(-v "${PROJECT_ROOT}/ros2:/home/user/FIRE_flightstack_sim/ros2")
    [[ -d "${PROJECT_ROOT}/gz"   ]] && cmd+=(-v "${PROJECT_ROOT}/gz:/home/user/FIRE_flightstack_sim/gz")
    [[ -d "${PROJECT_ROOT}/ap"   ]] && cmd+=(-v "${PROJECT_ROOT}/ap:/home/user/FIRE_flightstack_sim/ap")

    # Optionally mount the whole source tree
    if [[ "${MOUNT_SRC}" -eq 1 ]]; then
        cmd+=(-v "${PROJECT_ROOT}:/home/user/FIRE_flightstack_sim")
    fi

    cmd+=(-w /home/user/FIRE_flightstack_sim)
    cmd+=("${image_ref}")

    printf '%q ' "${cmd[@]}"
    echo
}

run_container() {
    local run_cmd
    run_cmd="$(build_run_cmd)"

    log "Running container: ${CONTAINER_NAME}"

    if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
        warn "Container '${CONTAINER_NAME}' already exists."
        warn "Removing existing container before creating a new one."
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    if [[ "${USE_X11}" -eq 1 ]]; then
        if command -v xhost >/dev/null 2>&1; then
            xhost +local:docker >/dev/null 2>&1 || true
        else
            warn "xhost command not found. GUI permission may fail."
        fi
    fi

    eval "${run_cmd}"
}

main() {
    parse_args "$@"

    if [[ "${DEBUG}" -eq 1 ]]; then
        set -x
    fi

    log "Starting Docker setup from: ${PROJECT_ROOT}"
    check_prereqs
    prepare_host_dirs

    if [[ "${DO_BUILD}" -eq 1 ]]; then
        build_image
    else
        log "Skipping Docker build step"
    fi

    if [[ "${DO_RUN}" -eq 1 ]]; then
        run_container
    else
        log "Docker setup completed successfully."
        log "Image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
    fi
}

main "$@"