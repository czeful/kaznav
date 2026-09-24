#!/usr/bin/env bash
# Two follow-up arms on the same debug episodes as run_debug_matrix.sh (labelled separately):
#   gd_value_thr060 - GroundingDINO + value map, candidates need detector score >= 0.6 (no VLM): is a
#                     stricter detector threshold enough without verification?
#   qolda_kk_minbox - qolda_kk with D-032 (no verification call on boxes < 32 px) - rule found on these episodes
#   bash tools/wsl.sh bash tools/run_debug_extra_arms.sh
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -u +%Y%m%dT%H%M%SZ)
list="runs/extra_arms_${stamp}.txt"
: > "$list"
arm() {
    local label="$1"; shift
    echo "=== arm ${label}: $*"
    local before; before=$(ls -1 runs/debug 2>/dev/null | sort | tail -1)
    bash tools/with_gpu_log.sh "runs/servers/gpu_${stamp}_${label}.csv" \
        python tools/run_debug_episodes.py --videos 1 "$@" 2>&1 | grep -vE "^\[|Warning|warn\(" | grep -E "^apartment|summary:|GPU memory|Qolda server|Error|Traceback"
    local after; after=$(ls -1 runs/debug | sort | tail -1)
    if [ "$after" != "$before" ]; then echo "${label}=runs/debug/${after}" >> "$list"; fi
}
arm gd_value_thr060 detector=grounding_dino_tiny value_encoder=siglip2_base perception.detection.score_threshold=0.6 "$@"
arm qolda_kk_minbox --lang kk detector=grounding_dino_tiny value_encoder=siglip2_base "$@"
echo "arms: $(cat "$list" | tr '\n' ' ')"
