#!/usr/bin/env bash
# Run a command inside WSL's habitat_env with the project root as CWD and the GPU render settings.
#   From Windows Git Bash:  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash tools/wsl.sh pytest
#   Inside WSL:             bash tools/wsl.sh python -m kznav....
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${KZNAV_CONDA_ENV:-habitat_env}"
cd "$(dirname "$0")/.."
# Habitat in WSL2: there is no NVIDIA EGL; Mesa renders through D3D12. Without the adapter name
# Mesa picks the first adapter (on this machine the AMD iGPU). See INSTALL_LOG.md, entry 1.
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
export KZNAV_DATA_ROOT="${KZNAV_DATA_ROOT:-$HOME/kznav_data}"
exec "$@"
