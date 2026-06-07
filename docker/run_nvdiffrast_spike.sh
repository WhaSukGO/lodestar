#!/usr/bin/env bash
# Run the EGL-free GPU-rasterization proof-of-concept (docker/nvdiffrast_spike.py) in Docker.
#
# Hang-safe by construction (GPU containers on WSL2 can hang on context creation):
#   --rm            container is removed on exit
#   --name          stable name so a stuck run is easy to `docker rm -f`
#   timeout 900     hard wall-clock cap; the run can never hang forever
# If you background this, poll `docker ps` and `docker rm -f lodestar_nvdiff` to kill a hang.
#
# Requires: NVIDIA Container Toolkit (so `--gpus all` works) and a CUDA+PyTorch *devel* image
# (devel = has nvcc; nvdiffrast compiles CUDA kernels). Verify GPU reaches the container first:
#   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"
NAME="lodestar_nvdiff"

docker rm -f "$NAME" 2>/dev/null || true
timeout 900 docker run --rm --name "$NAME" --gpus all -v "$HERE":/work "$IMAGE" bash -c '
  apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y git build-essential >/dev/null 2>&1
  pip install -q ninja imageio >/dev/null 2>&1
  pip install -q --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
  python /work/nvdiffrast_spike.py
'
status=$?
docker rm -f "$NAME" 2>/dev/null || true     # belt-and-suspenders: never leave a container behind
exit $status
