#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE=0,1

docker run \
    -it --rm \
    --name "stufootball" \
    --runtime=nvidia \
    --gpus all \
    --privileged \
    --shm-size 8g \
    -v "${CWD}/..":/workspace \
    -v "/media/disk1/data/fiit/football360":/data \
    -e CUDA_VISIBLE_DEVICES="$DEVICE" \
    ${IMAGE_TAG} \
    "$@" || exit $?
