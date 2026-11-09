#!/usr/bin/env bash
# Run a command in the stand's container from the repository root, on Windows + WSL2 or on Linux + NVIDIA:
#   bash docker/kz.sh python -m pytest -q
#   bash docker/kz.sh python tools/run_matrix.py run paper/matrices/<m>.yaml
#   bash docker/kz.sh --speech /opt/nemo-env/bin/python tools/transcribe_kk_asr.py ...
#   bash docker/kz.sh --build [sim|speech]          # build an image (default sim)
#   bash docker/kz.sh                               # a shell in the container
# Mounts the repository, $KZNAV_DATA_ROOT (default ~/kznav_data) and $HF_HOME (default ~/.cache/huggingface) at their
# host paths and runs as the calling user. HF_HUB_OFFLINE=1 unless set: models come from the pinned cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export KZNAV_REPO="$PWD" KZNAV_UID="$(id -u)" KZNAV_GID="$(id -g)"
export KZNAV_DATA_ROOT="${KZNAV_DATA_ROOT:-$HOME/kznav_data}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
files=(-f docker/compose.yaml)
if [ -e /dev/dxg ]; then files+=(-f docker/compose.wsl.yaml); else files+=(-f docker/compose.nvidia.yaml); fi
case "${1:-}" in
    --build) exec docker compose "${files[@]}" --profile speech build "${2:-sim}" ;;
    --speech) svc=speech; shift ;;
    *) svc=sim ;;
esac
[ -d "$KZNAV_DATA_ROOT" ] || { echo "no data root $KZNAV_DATA_ROOT (set KZNAV_DATA_ROOT)" >&2; exit 1; }
tty=(); [ -t 0 ] && [ -t 1 ] || tty=(-T)
exec docker compose "${files[@]}" --profile speech run --rm "${tty[@]}" "$svc" "$@"
