#!/usr/bin/env bash
# Re-run selected debug episodes with video for the demo reel (same code and episodes as the matrix arms).
#   bash tools/wsl.sh bash tools/run_showcase.sh
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -u +%Y%m%dT%H%M%SZ)
list="runs/showcase_${stamp}.txt"
: > "$list"
one() {   # one <label> <episode_id> <args...>
    local label="$1" ep="$2"; shift 2
    local before; before=$(ls -1 runs/debug | sort | tail -1)
    python tools/run_debug_episodes.py --videos 1 --only "$ep" "$@" 2>&1 | grep -E "^apartment|Error|Traceback"
    local after; after=$(ls -1 runs/debug | sort | tail -1)
    [ "$after" != "$before" ] && echo "${label}=runs/debug/${after}/video_${ep}.mp4" >> "$list"
}
GD="detector=grounding_dino_tiny value_encoder=siglip2_base"
one gd_banana002      apartment_1_banana_002          $GD
one kk_banana002      apartment_1_banana_002          --lang kk $GD
one kk_mustard004     apartment_1_mustard-bottle_004  --lang kk $GD
one kkru_mustard000   apartment_1_mustard-bottle_000  --lang kk_ru $GD
cat "$list"
