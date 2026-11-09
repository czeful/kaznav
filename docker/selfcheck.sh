#!/usr/bin/env bash
# First check on a new machine, inside the container: the GPU in both environments, the GL renderer, the pinned
# models and the unit tests (about 6 min, whole GPU for a moment).
#   bash docker/kz.sh bash docker/selfcheck.sh
# Then the reproducibility check against the dev machine: docker/check/run.sh and compare.py (docker/README.md).
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
step() { echo; echo "=== $1"; }

step "GPU in habitat_env and qolda-env"
python -c "import torch; print('habitat_env: torch', torch.__version__, torch.cuda.get_device_name(0))" || fail=1
"${KZNAV_QOLDA_VENV:-/opt/qolda-env}/bin/python" -c \
    "import torch; print('qolda-env:   torch', torch.__version__, torch.cuda.get_device_name(0))" || fail=1

step "Renderer (must be the NVIDIA GPU, not llvmpipe or another adapter)"
r="$(python tools/check_render.py --gpu-device-id "${KZNAV_GPU_DEVICE_ID:--1}" --out /tmp/kznav_render.png \
    --scene "$KZNAV_DATA_ROOT/scene_datasets/habitat-test-scenes/van-gogh-room.glb" 2>&1 | grep -E "^(Renderer|render )")"
echo "$r"
echo "$r" | grep -q "NVIDIA" || { echo "the renderer is not the NVIDIA GPU"; fail=1; }

step "Pinned models (tools/fetch_models.py)"
python tools/fetch_models.py --check --groups core || fail=1

step "Unit tests"
python -m pytest -q || fail=1

echo
if [ "$fail" = 0 ]; then echo "SELFCHECK OK"; else echo "SELFCHECK FAILED"; fi
exit "$fail"
