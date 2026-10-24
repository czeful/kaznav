#!/usr/bin/env bash
# HM3D v0.1 scenes of one split, only those that have ObjectNav v1 episodes (D-052). The habitat tar of train holds
# all 800 train scenes (26 GB); ObjectNav uses the 80 annotated ones (~3 GB). The tar is streamed through tar's
# filter, so it never lands on the disk - the full download filled drive C: once (the WSL disk grows on it).
#   bash tools/wsl.sh bash tools/download_hm3d_subset.sh train
# Needs the episodes (tools/download_hm3d.sh val fetched them) and the Matterport token (see download_hm3d.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
split="${1:?usage: download_hm3d_subset.sh train|val}"
data="${KZNAV_DATA_ROOT:-$HOME/kznav_data}"
creds="$HOME/.config/kznav/matterport.env"
if [ -z "${MATTERPORT_TOKEN_ID:-}" ] && [ -f "$creds" ]; then set -a; source "$creds"; set +a; fi
: "${MATTERPORT_TOKEN_ID:?no Matterport token id}"
: "${MATTERPORT_TOKEN_SECRET:?no Matterport token secret}"
content="$data/datasets/objectnav/hm3d/v1/$split/content"
[ -d "$content" ] || { echo "no episodes in $content (run tools/download_hm3d.sh val first)"; exit 1; }
dest="$data/versioned_data/hm3d-0.1/hm3d/$split"
mkdir -p "$dest"

# 1. small parts through habitat-sim's downloader: scene configs, semantic annotations and their configs
for part in configs semantic_annots semantic_configs; do
    echo "=== hm3d_${split}_${part}_v0.1"
    python -m habitat_sim.utils.datasets_download --uids "hm3d_${split}_${part}_v0.1" --data-path "$data/" --no-replace \
        --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET"
done

# 2. the scenes with episodes, streamed out of the habitat tar
patterns=()
for f in "$content"/*.json.gz; do
    stem=$(basename "$f" .json.gz)
    [ -f "$(ls -d "$dest"/*-"$stem" 2>/dev/null | head -1)/$stem.basis.glb" ] && continue
    patterns+=("*-$stem/*")
done
echo "=== ${#patterns[@]} scenes to fetch (of $(ls "$content" | wc -l) with episodes)"
if [ "${#patterns[@]}" -gt 0 ]; then
    curl --fail --location --silent --show-error --user "$MATTERPORT_TOKEN_ID:$MATTERPORT_TOKEN_SECRET" \
        "https://api.matterport.com/resources/habitat/hm3d-$split-habitat.tar" \
        | tar -x -C "$dest" --wildcards "${patterns[@]}"
fi
n_glb=$(find "$dest" -name '*.basis.glb' | wc -l)
n_sem=$(find "$dest" -name '*.semantic.txt' | wc -l)
echo "=== $split: $n_glb scenes, $n_sem with semantics, $(du -sh "$dest" | cut -f1)"
ln -sfn "$data/versioned_data/hm3d-0.1/hm3d" "$data/scene_datasets/hm3d"
echo "=== done"
