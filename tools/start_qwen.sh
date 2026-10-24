#!/usr/bin/env bash
# Start the Qwen3-VL-4B server (bf16) in the Qolda venv (transformers >= 4.57), in the background;
# log to runs/servers/qwen.log. Stop the Qolda server first: both do not fit in 12 GB.
#   bash tools/start_qwen.sh [--max-pixels 802816]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/servers
source "${KZNAV_QWEN_VENV:-$HOME/qolda-env}/bin/activate"
nohup python servers/qwen_vl_server.py --model Qwen/Qwen3-VL-4B-Instruct --served-name qwen3-vl-4b --port 23334 "$@" \
    > runs/servers/qwen.log 2>&1 &
echo $! > runs/servers/qwen.pid
echo "started pid $(cat runs/servers/qwen.pid); waiting for /v1/models ..."
for i in $(seq 1 300); do
    if curl -sf http://127.0.0.1:23334/v1/models > /dev/null; then echo "ready after ${i}s"; exit 0; fi
    sleep 1
done
echo "server did not come up; see runs/servers/qwen.log"; exit 1
