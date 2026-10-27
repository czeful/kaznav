#!/usr/bin/env bash
# Start the Gemma 3 4B server (bf16, D-083) in the Qolda venv, in the background; log to runs/servers/gemma.log.
# Stop the other model servers first: two do not fit in 12 GB. The model is gated on Hugging Face: the user's
# account must have accepted Google's license and be logged in (huggingface-cli login) before the first start.
#   bash tools/start_gemma.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/servers
source "${KZNAV_GEMMA_VENV:-$HOME/qolda-env}/bin/activate"
export HF_HUB_DISABLE_XET=1
nohup python servers/gemma_server.py --model google/gemma-3-4b-it --served-name gemma-3-4b --port 23335 "$@" \
    > runs/servers/gemma.log 2>&1 &
echo $! > runs/servers/gemma.pid
echo "started pid $(cat runs/servers/gemma.pid); waiting for /v1/models ..."
for i in $(seq 1 600); do
    if curl -sf http://127.0.0.1:23335/v1/models > /dev/null; then echo "ready after ${i}s"; exit 0; fi
    sleep 1
done
echo "server did not come up; see runs/servers/gemma.log"; exit 1
