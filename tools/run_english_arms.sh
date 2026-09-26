#!/usr/bin/env bash
# English-command control on the matrix episodes (D-037): same episodes and order as run_debug_matrix.sh,
# commands are literal translations of the Kazakh templates. Needs the Qolda server.
#   bash tools/wsl.sh bash tools/run_english_arms.sh
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -u +%Y%m%dT%H%M%SZ)
list="runs/english_arms_${stamp}.txt"
: > "$list"
arm() {
    local label="$1" videos="$2"; shift 2
    echo "=== arm ${label}: $*"
    local before; before=$(ls -1 runs/debug | sort | tail -1)
    bash tools/with_gpu_log.sh "runs/servers/gpu_${stamp}_${label}.csv" \
        python tools/run_debug_episodes.py --videos "$videos" "$@" 2>&1 | grep -E "^apartment|summary:|GPU memory|Error|Traceback"
    local after; after=$(ls -1 runs/debug | sort | tail -1)
    [ "$after" != "$before" ] && echo "${label}=runs/debug/${after}" >> "$list"
}
GD="detector=grounding_dino_tiny value_encoder=siglip2_base"
arm qolda_en          20 --lang en $GD
arm qolda_en_noverif  1  --lang en --no-verify $GD
echo "arms: $(tr '\n' ' ' < "$list")"
