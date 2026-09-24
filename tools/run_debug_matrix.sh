#!/usr/bin/env bash
# Debug comparison matrix on debug_ycb (same episodes in every arm), then tools/compare_runs.py.
#   bash tools/wsl.sh bash tools/run_debug_matrix.sh [extra hydra overrides, e.g. embodiment=habitat_agent]
# Needs the Qolda server for the language arms (bash tools/start_qolda.sh). Role: debug, not paper numbers.
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -u +%Y%m%dT%H%M%SZ)
list="runs/matrix_${stamp}.txt"
: > "$list"
arm() {   # arm <label> <args...>: run one arm, record label=run_dir
    local label="$1"; shift
    echo "=== arm ${label}: $*"
    local before; before=$(ls -1 runs/debug 2>/dev/null | sort | tail -1)
    bash tools/with_gpu_log.sh "runs/servers/gpu_${stamp}_${label}.csv" \
        python tools/run_debug_episodes.py --videos 1 "$@" 2>&1 | grep -vE "^\[|Warning|warn\(" | grep -E "^apartment|summary:|GPU memory|Qolda server|Error|Traceback"
    local after; after=$(ls -1 runs/debug | sort | tail -1)
    if [ "$after" != "$before" ]; then echo "${label}=runs/debug/${after}" >> "$list"; fi
}
arm oracle          detector=oracle_semantic "$@"
arm gd_nearest      detector=grounding_dino_tiny "$@"
arm gd_value        detector=grounding_dino_tiny value_encoder=siglip2_base "$@"
arm qolda_kk_noverif --lang kk --no-verify detector=grounding_dino_tiny value_encoder=siglip2_base "$@"
arm qolda_kk        --lang kk detector=grounding_dino_tiny value_encoder=siglip2_base "$@"
arm qolda_kkru      --lang kk_ru detector=grounding_dino_tiny value_encoder=siglip2_base "$@"
echo "arms: $(cat "$list" | tr '\n' ' ')"
python tools/compare_runs.py --ref gd_value --title "Debug matrix ${stamp} ($*)" \
    --out "runs/compare/matrix_${stamp}" $(cat "$list")
