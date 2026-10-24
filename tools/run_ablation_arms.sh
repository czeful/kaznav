#!/usr/bin/env bash
# Goal ablations on the matrix episodes (D-038 / D-039): same episodes and order as run_debug_matrix.sh, so each
# arm pairs with qolda_kk / gd_value of the matrix. Needs the Qolda server; NLLB runs on the CPU.
#   bash tools/wsl.sh bash tools/run_ablation_arms.sh
# Then: python tools/compare_runs.py --ref qolda_kk qolda_kk=<matrix run> $(cat runs/ablation_arms_<stamp>.txt)
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -u +%Y%m%dT%H%M%SZ)
list="runs/ablation_arms_${stamp}.txt"
: > "$list"
arm() {
    local label="$1"; shift
    echo "=== arm ${label}: $*"
    local before; before=$(ls -1 runs/debug | sort | tail -1)
    bash tools/with_gpu_log.sh "runs/servers/gpu_${stamp}_${label}.csv" \
        python tools/run_debug_episodes.py --videos 1 "$@" 2>&1 | grep -E "^apartment|summary:|GPU memory|Error|Traceback"
    local after; after=$(ls -1 runs/debug | sort | tail -1)
    [ "$after" != "$before" ] && echo "${label}=runs/debug/${after}" >> "$list"
}
GD="detector=grounding_dino_tiny value_encoder=siglip2_base"
arm qolda_kk_top1       --lang kk goal=top1 $GD                 # one hypothesis instead of 3-5 (D-038)
arm mt_word             --lang kk goal=mt_word $GD              # NLLB(Kazakh name) -> one query, Qolda verifies
arm mt_word_noverif     --lang kk --no-verify goal=mt_word $GD  # the same without verification: VLFM + NLLB
arm mt_command          --lang kk goal=mt_command $GD           # NLLB(command) -> English pipeline (D-039)
echo "arms: $(tr '\n' ' ' < "$list")"
