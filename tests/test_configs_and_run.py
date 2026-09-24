"""Every config combination composes, resolves and builds; manifests and logs are well-formed."""
import json
import math

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from kznav.eval.records import EpisodeResult, column_names
from kznav.language.base import BackendConfigError
from kznav.language.registry import create_backend
from kznav.perception.base import create_detector, create_segmenter, create_value_encoder
from kznav.run.logging import ImageStore, JsonlWriter
from kznav.run.manifest import build_manifest, config_hash

BACKENDS = ["mock", "qolda", "qolda_think_control", "qwen3vl_4b", "replay"]
EMBODIMENTS = ["wheeled_vacuum", "habitat_agent"]
DATASETS = ["hm3d_objectnav_v1_val", "hm3d_ovon_val_unseen", "custom_room"]


def composed(repo_root, *overrides):
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_compose_all(repo_root, backend, embodiment, dataset):
    cfg = composed(repo_root, f"backend={backend}", f"embodiment={embodiment}", f"dataset={dataset}")
    assert cfg["run"]["name"] == f"{cfg['backend']['name']}__{cfg['dataset']['name']}__{embodiment}"
    assert cfg["dataset"]["role"] in ("benchmark", "debug")


def test_custom_room_is_debug_only(repo_root):
    assert composed(repo_root, "dataset=custom_room")["dataset"]["role"] == "debug"


@pytest.mark.parametrize("backend", ["mock", "qolda", "qolda_think_control", "qwen3vl_4b"])
def test_backends_build_from_composed_config(repo_root, prompt_lib, backend):
    cfg = composed(repo_root, f"backend={backend}")
    b = create_backend(cfg["backend"], prompts=prompt_lib)
    info = b.describe_config()
    assert info["prompts"]["parse_goal"]["sha256"]
    if backend.startswith("qolda"):
        assert cfg["backend"]["extra_body"]["enable_thinking"] is (backend == "qolda_think_control")


def test_replay_requires_sources(repo_root, prompt_lib):
    cfg = composed(repo_root, "backend=replay")
    with pytest.raises(BackendConfigError):
        create_backend(cfg["backend"], prompts=prompt_lib)


def test_think_control_inherits_qolda(repo_root):
    cfg = composed(repo_root, "backend=qolda_think_control")["backend"]
    assert cfg["type"] == "qolda" and cfg["max_tokens"]["parse_goal"] == 4096
    assert cfg["max_tokens"]["verify_candidate"] == 4


def test_mock_perception_components():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert create_detector({"type": "none"}).detect(img, ["chair"]) == []
    assert create_segmenter({"type": "box"}).segment(img, (2, 2, 5, 6)).sum() == 12
    s = create_value_encoder({"type": "hash"}).score(img, ["a", "b"])
    assert s.shape == (2,) and np.all((0 <= s) & (s <= 1))


class TestManifestAndLogs:
    def test_config_hash_is_order_independent(self):
        assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash({"b": {"d": 3, "c": 2}, "a": 1})
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_manifest_fields(self, repo_root):
        m = build_manifest(run_id="r", config={"x": 1}, seed=3, repo=repo_root)
        for key in ("run_id", "created_utc", "seed", "config_sha256", "code", "packages", "gpu", "python"):
            assert key in m
        assert m["packages"]["numpy"]

    def test_jsonl_is_strict_json(self, tmp_path):
        w = JsonlWriter(tmp_path / "x.jsonl")
        w.write({"a": math.nan, "b": np.float32(1.5), "c": [math.inf], "d": np.int64(3)})
        w.close()
        line = (tmp_path / "x.jsonl").read_text(encoding="utf-8").strip()
        assert json.loads(line) == {"a": "NaN", "b": 1.5, "c": ["inf"], "d": 3}

    def test_image_store_dedupes(self, tmp_path):
        s = ImageStore(tmp_path / "img")
        assert s.put(b"abc") == s.put(b"abc")
        assert len(list((tmp_path / "img").iterdir())) == 1

    def test_episode_columns_unique(self):
        cols = column_names()
        assert len(cols) == len(set(cols))
        assert {"success", "spl", "soft_spl", "dts", "outcome", "vlm_calls", "stuck_time_s"} <= set(cols)
        assert EpisodeResult.__dataclass_fields__["schema_version"].default == 1


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_noise_models_default_to_identity(repo_root, embodiment):
    from kznav.sim.noise import build_noise_models

    cfg = composed(repo_root, f"embodiment={embodiment}")
    depth, act, odo = build_noise_models(cfg["embodiment"]["noise"], seed=0)
    d = np.random.default_rng(0).uniform(0.2, 5.0, size=(4, 4))
    assert np.array_equal(depth.apply(d), d)
    assert act.apply(0.3, 0.1, 0.1) == (0.3, 0.1)
    assert odo.apply(0.1, 0.0, 0.02) == (0.1, 0.0, 0.02)


def test_unknown_noise_model_rejected():
    from kznav.sim.noise import build_noise_models

    with pytest.raises(KeyError):
        build_noise_models({"depth": {"model": "redwood"}}, seed=0)
