#!/usr/bin/env bash
# Container entry point: the render path of this host, a writable HOME, then the command.
set -e
if [ -e /dev/dxg ]; then
    # Windows + WSL2 (docker/compose.wsl.yaml): no NVIDIA EGL; CUDA and Mesa's D3D12 driver use /usr/lib/wsl/lib,
    # Mesa must be told the adapter or it takes the first one (INSTALL_LOG.md, entry 1).
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
    export KZNAV_GPU_DEVICE_ID="${KZNAV_GPU_DEVICE_ID:--1}"
else
    # Linux + NVIDIA Container Toolkit (docker/compose.nvidia.yaml): NVIDIA EGL matched to CUDA device 0.
    export KZNAV_GPU_DEVICE_ID="${KZNAV_GPU_DEVICE_ID:-0}"
fi
if [ ! -w "${HOME:-/nonexistent}" ]; then
    export HOME=/tmp/kznav-home
    mkdir -p "$HOME"
fi
exec "$@"
