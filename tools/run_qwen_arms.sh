#!/usr/bin/env bash
# Qwen3-VL-4B in place of Qolda (D-040): the pilot language checks, then the matrix episodes in the same order as
# run_debug_matrix.sh / run_english_arms.sh, so every arm pairs with its Qolda arm. Needs the Qwen server
# (bash tools/start_qwen.sh; the Qolda server stopped).
#   bash tools/wsl.sh bash tools/run_qwen_arms.sh
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -u +%Y%m%dT%H%M%SZ)
list="runs/qwen_arms_${stamp}.txt"
: > "$list"
Q="backend=qwen3vl_4b"
echo "=== language: parse (30 pilot commands)"
python tools/eval_language.py parse $Q 2>&1 | grep -E "^\| (all|cultural|universal) |^out:|Error|Traceback"
echo "=== language: verify (32 pilot crops)"
python tools/eval_language.py verify $Q 2>&1 | grep -E "^\| - |^out:|Error|Traceback"
arm() {
    local label="$1"; shift
    echo "=== arm ${label}: $*"
    local before; before=$(ls -1 runs/debug | sort | tail -1)
    bash tools/with_gpu_log.sh "runs/servers/gpu_${stamp}_${label}.csv" \
        python tools/run_debug_episodes.py --videos 1 "$@" 2>&1 | grep -E "^apartment|summary:|GPU memory|WARNING|Error|Traceback"
    local after; after=$(ls -1 runs/debug | sort | tail -1)
    [ "$after" != "$before" ] && echo "${label}=runs/debug/${after}" >> "$list"
}
GD="detector=grounding_dino_tiny value_encoder=siglip2_base"
arm qwen_kk          --lang kk $Q $GD
arm qwen_kkru        --lang kk_ru $Q $GD
arm qwen_en          --lang en $Q $GD
arm qwen_kk_noverif  --lang kk --no-verify $Q $GD
echo "arms: $(tr '\n' ' ' < "$list")"
