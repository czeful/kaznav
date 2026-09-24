#!/usr/bin/env bash
# Run a command while logging GPU memory / utilisation every second, plus the Qolda server's peak:
#   bash tools/with_gpu_log.sh runs/servers/gpu_<name>.csv <command ...>
# The Qolda peak window is reset before the command and read after it (servers/qolda_server.py).
set -uo pipefail
cd "$(dirname "$0")/.."
log="$1"; shift
mkdir -p "$(dirname "$log")"
curl -sf -X POST http://127.0.0.1:23333/kznav/reset_peak > /dev/null 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 > "$log" &
mon=$!
"$@"
rc=$?
kill "$mon" 2>/dev/null
peak=$(awk -F', ' 'BEGIN{m=0} {if ($2+0 > m) m=$2+0} END{print m}' "$log")
echo "GPU memory used, peak over the run: ${peak} MiB (nvidia-smi, whole GPU incl. other processes)"
echo "Qolda server: $(curl -sf http://127.0.0.1:23333/kznav/stats || echo 'not running')"
exit $rc
