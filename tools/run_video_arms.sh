#!/usr/bin/env bash
# Re-run whole matrix arms (same episodes, same order) with a video for every episode. Doubles as the
# reproducibility check of the matrix: compare the new episodes.csv with the matrix arm (D-036).
#   bash tools/wsl.sh bash tools/run_video_arms.sh
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -u +%Y%m%dT%H%M%SZ)
list="runs/video_arms_${stamp}.txt"
: > "$list"
arm() {
    local label="$1"; shift
    echo "=== arm ${label}: $*"
    local before; before=$(ls -1 runs/debug | sort | tail -1)
    python tools/run_debug_episodes.py --videos 20 "$@" 2>&1 | grep -E "^apartment|summary:|Error|Traceback"
    local after; after=$(ls -1 runs/debug | sort | tail -1)
    [ "$after" != "$before" ] && echo "${label}=runs/debug/${after}" >> "$list"
}
GD="detector=grounding_dino_tiny value_encoder=siglip2_base"
arm gd_value   $GD
arm qolda_kk   --lang kk $GD
arm qolda_kkru --lang kk_ru $GD
echo "arms: $(tr '\n' ' ' < "$list")"
