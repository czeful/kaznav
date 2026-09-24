#!/usr/bin/env bash
# Start the Qolda server (bf16) in the Qolda venv, in the background; log to runs/servers/qolda.log.
#   bash tools/start_qolda.sh [--max-tiles 4]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/servers
source "${KZNAV_QOLDA_VENV:-$HOME/qolda-env}/bin/activate"
nohup python servers/qolda_server.py --model issai/Qolda --served-name qolda-4b --port 23333 "$@" \
    > runs/servers/qolda.log 2>&1 &
echo $! > runs/servers/qolda.pid
echo "started pid $(cat runs/servers/qolda.pid); waiting for /v1/models ..."
for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:23333/v1/models > /dev/null; then echo "ready after ${i}s"; exit 0; fi
    sleep 1
done
echo "server did not come up; see runs/servers/qolda.log"; exit 1
