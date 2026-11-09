"""Download the stand's Hugging Face models at their pinned revisions (env/models.yaml) into the HF cache.

    bash docker/kz.sh python tools/fetch_models.py                 # groups core + speech
    bash docker/kz.sh python tools/fetch_models.py --groups core   # only what the test matrices load
    bash docker/kz.sh python tools/fetch_models.py --check         # report only, downloads nothing

Each model is downloaded at the commit and with the files listed in env/models.yaml (docker/lock_envs.sh writes it
from the dev machine's cache), and the cache's ``refs/main`` is pointed at that commit: ``from_pretrained(repo)``
under HF_HUB_OFFLINE=1, the default in the containers, then loads exactly the weights the dev results came from.
google/gemma-3-4b-it is gated: accept its licence on huggingface.co and pass HF_TOKEN (or log in) before fetching.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "env/models.yaml"


def model_dir(cache: Path, repo: str) -> Path:
    """The HF cache folder of a model repository."""
    return cache / ("models--" + repo.replace("/", "--"))


def state(cache: Path, repo: str, spec: dict) -> str:
    """``ok``, ``missing`` (no complete snapshot of the pinned commit) or ``main elsewhere`` (refs/main differs)."""
    d = model_dir(cache, repo)
    snap = d / "snapshots" / spec["revision"]
    if not all((snap / f).is_file() for f in spec["files"]):
        return "missing"
    ref = d / "refs/main"
    if not ref.is_file() or ref.read_text().strip() != spec["revision"]:
        return "main elsewhere"
    return "ok"


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["core", "speech"], help="core, speech, dev")
    ap.add_argument("--check", action="store_true", help="only report the state of each pinned model")
    args = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"                    # read by huggingface_hub at import
    from huggingface_hub import constants, snapshot_download

    cache = Path(constants.HF_HUB_CACHE)
    models = {r: s for r, s in yaml.safe_load(LOCK.read_text(encoding="utf-8")).items() if s["group"] in args.groups}
    bad = 0
    for repo, spec in models.items():
        st = state(cache, repo, spec)
        if st != "ok" and not args.check:
            print(f"[fetch] {repo} @ {spec['revision'][:12]} ({len(spec['files'])} files) ...", flush=True)
            snapshot_download(repo_id=repo, revision=spec["revision"], allow_patterns=spec["files"])
            (model_dir(cache, repo) / "refs/main").write_text(spec["revision"])
            st = state(cache, repo, spec)
        bad += st != "ok"
        print(f"{st:15s} {spec['group']:7s} {repo} @ {spec['revision'][:12]}", flush=True)
    print(f"{len(models) - bad} of {len(models)} models ok in {cache}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
