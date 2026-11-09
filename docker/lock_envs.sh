#!/usr/bin/env bash
# Write the exact package lists of the local environments into env/; the Docker images are built from them.
# Run in WSL on the machine the dev results come from, after any change to an environment:
#   bash docker/lock_envs.sh
#
#   env/habitat_env.conda.txt   conda packages with URL and md5 (conda list --explicit --md5)
#   env/habitat_env.pip.txt     what pip installed into habitat_env (incl. pip over a conda package, e.g. imageio-ffmpeg)
#   env/<venv>.pip.txt          every package of the venvs qolda-env, avl-env, nemo-env
# Pip lists are installed with --no-deps, so the resolver changes nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh

# name==version for every distribution pip installed (INSTALLER file); a git install keeps its commit.
read -r -d '' LIST <<'EOF' || true
import importlib.metadata as md, json, sys
rows = {}
for d in md.distributions():
    name = d.metadata["Name"]
    if not name or name.lower() == "pip":
        continue
    if (d.read_text("INSTALLER") or "").strip() != "pip":
        continue                              # conda's own packages and eggs come with the conda lock
    url = json.loads(d.read_text("direct_url.json") or "{}")
    vcs = url.get("vcs_info")
    if vcs:
        rows[name.lower()] = f"{name} @ git+{url['url']}@{vcs['commit_id']}"
    elif url:
        sys.exit(f"{name}: installed from {url.get('url')}, not from an index - pin it by hand")
    else:
        rows[name.lower()] = f"{name}=={d.version}"
print("\n".join(rows[k] for k in sorted(rows)))
EOF

conda activate habitat_env
{ echo "# conda list --explicit --md5 of habitat_env ($(date +%F))"; conda list --explicit --md5; } > env/habitat_env.conda.txt
python -c "$LIST" > env/habitat_env.pip.txt
conda deactivate
for v in qolda-env avl-env nemo-env; do
    "$HOME/$v/bin/python" -c "$LIST" > "env/$v.pip.txt"
    echo "$v: python $("$HOME/$v/bin/python" -c 'import platform;print(platform.python_version())')"
done

# HF models: the commit each cached model's refs/main points at and the files of that snapshot
# (tools/fetch_models.py downloads exactly these). Groups: core = the test matrices, speech = the spoken-command
# step, dev = detector and verifier studies on dev only.
"$HOME/miniconda3/envs/habitat_env/bin/python" - > env/models.yaml <<'EOF'
import os, datetime
from pathlib import Path
import yaml
CORE = {"issai/Qolda", "Qwen/Qwen3-VL-4B-Instruct", "google/gemma-3-4b-it", "google/siglip2-base-patch16-224",
        "facebook/sam2.1-hiera-tiny", "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
        "bert-base-uncased", "PekingU/rtdetr_r101vd_coco_o365", "IDEA-Research/grounding-dino-tiny",
        "facebook/nllb-200-distilled-600M"}
SPEECH = {"issai/Qolda-AVL-5B", "facebook/mms-tts-kaz", "facebook/mms-tts-eng"}
hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
models = {}
for d in sorted(hub.glob("models--*")):
    repo = d.name[len("models--"):].replace("--", "/", 1)
    rev = (d / "refs/main").read_text().strip()
    snap = d / "snapshots" / rev
    models[repo] = {"revision": rev, "group": "core" if repo in CORE else "speech" if repo in SPEECH else "dev",
                    "files": sorted(p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file())}
print(f"# HF models of the stand at the revisions of the dev machine ({datetime.date.today()}); "
      "written by docker/lock_envs.sh, read by tools/fetch_models.py")
print(yaml.safe_dump(models, sort_keys=False, width=120), end="")
EOF
wc -l env/*.txt env/models.yaml
