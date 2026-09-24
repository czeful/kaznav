#!/usr/bin/env bash
# Stop the running Qolda server (if any) and start it again: bash tools/restart_qolda.sh [--max-tiles 4]
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -f runs/servers/qolda.pid ]; then
    kill "$(cat runs/servers/qolda.pid)" 2>/dev/null
    for i in $(seq 1 30); do
        curl -sf http://127.0.0.1:23333/v1/models > /dev/null || break
        sleep 1
    done
fi
exec bash tools/start_qolda.sh "$@"
