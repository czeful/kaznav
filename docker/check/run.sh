#!/usr/bin/env bash
# Run the reproducibility check matrix (docker/check/repro.yaml) into runs/docker_check/<tag>/ (D-089):
#   bash tools/wsl.sh bash docker/check/run.sh host       # the host environments of the dev machine
#   bash docker/kz.sh bash docker/check/run.sh docker     # the container
#   python docker/check/compare.py runs/docker_check/host runs/docker_check/docker
# Whole GPU (one VLM server + one run), about 15 min. Also writes the GL renderer and a test frame (render.txt/png).
set -euo pipefail
cd "$(dirname "$0")/../.."
tag="${1:?usage: run.sh <tag>}"
out="runs/docker_check/$tag"
mkdir -p "$out"
sed "s#^out_root: runs/docker_check/TAG#out_root: $out#" docker/check/repro.yaml > "$out/repro.yaml"
smi="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"     # in a WSL container it is not on PATH
{ "$smi" --query-gpu=name,driver_version --format=csv,noheader || true
  python tools/check_render.py --scene "${KZNAV_DATA_ROOT:-$HOME/kznav_data}/scene_datasets/habitat-test-scenes/van-gogh-room.glb" \
      --gpu-device-id "${KZNAV_GPU_DEVICE_ID:--1}" --out "$out/render.png" 2>&1 \
      | grep -E "^(Renderer|OpenGL version|render |rgb mean)" || true; } > "$out/render.txt"
cat "$out/render.txt"
python tools/run_matrix.py run "$out/repro.yaml"
