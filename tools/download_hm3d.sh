#!/usr/bin/env bash
# HM3D scenes (Matterport API token) and the ObjectNav episodes, into $KZNAV_DATA_ROOT (default ~/kznav_data).
#   bash tools/wsl.sh bash tools/download_hm3d.sh minival   # 10 small scenes: smoke test of the whole chain
#   bash tools/wsl.sh bash tools/download_hm3d.sh val       # HM3D val scenes + semantics, ObjectNav v1 episodes
#   bash tools/wsl.sh bash tools/download_hm3d.sh train     # train scenes (the dev split comes from them)
# HM3D version: ObjectNav HM3D v1 (the split VLFM reports on) was made on HM3D v1.0 + HM3D-Semantics v0.1, so the
# default is v0.1 (HM3D_VERSION=0.2 for HM3D-OVON). habitat-sim links scene_datasets/hm3d to the version downloaded
# last; versioned_data/hm3d-<version> keeps each version.
# Credentials (never in the repository): MATTERPORT_TOKEN_ID / MATTERPORT_TOKEN_SECRET from the environment, or
# from ~/.config/kznav/matterport.env (chmod 600).
set -euo pipefail
cd "$(dirname "$0")/.."
split="${1:?usage: download_hm3d.sh minival|val|train}"
version="${HM3D_VERSION:-0.1}"
data="${KZNAV_DATA_ROOT:-$HOME/kznav_data}"
creds="$HOME/.config/kznav/matterport.env"
if [ -z "${MATTERPORT_TOKEN_ID:-}" ] && [ -f "$creds" ]; then
    # shellcheck disable=SC1090
    set -a; source "$creds"; set +a
fi
: "${MATTERPORT_TOKEN_ID:?no Matterport token id (environment or $creds)}"
: "${MATTERPORT_TOKEN_SECRET:?no Matterport token secret}"
mkdir -p "$data"
echo "=== HM3D v$version $split -> $data"
python -m habitat_sim.utils.datasets_download --uids "hm3d_${split}_v${version}" --data-path "$data/" --no-replace \
    --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET"

# ObjectNav HM3D v1 episodes (public, no token): train / val / val_mini
ep_dir="$data/datasets/objectnav/hm3d/v1"
if [ "$version" = "0.1" ] && [ ! -f "$ep_dir/val/val.json.gz" ]; then
    echo "=== ObjectNav HM3D v1 episodes -> $ep_dir"
    tmp="$data/objectnav_hm3d_v1.zip"
    curl --fail --location --continue-at - https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip -o "$tmp"
    mkdir -p "$ep_dir"
    python -m zipfile -e "$tmp" "$data/datasets/objectnav/hm3d/_unzip"        # no unzip in a bare WSL
    src=$(dirname "$(find "$data/datasets/objectnav/hm3d/_unzip" -name val.json.gz -path '*/val/*' | head -1)")/..
    cp -r "$src"/. "$ep_dir"/
    rm -rf "$data/datasets/objectnav/hm3d/_unzip" "$tmp"
fi
echo "=== done"
du -sh "$data/versioned_data/hm3d-$version" "$ep_dir" 2>/dev/null || true
ls "$data/scene_datasets/hm3d" 2>/dev/null | head
