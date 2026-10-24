"""Matrix runner (D-050): cells, completion state, server grouping, paper spec."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import run_matrix as rm  # noqa: E402


def matrix(tmp_path, **over):
    m = {"name": "t", "title": "T", "out_root": str(tmp_path.relative_to(REPO)) if tmp_path.is_relative_to(REPO)
         else str(tmp_path), "seeds": [0, 1], "common": ["dataset=hm3d_objectnav_v1_val"],
         "servers": {"q": {"start": "true", "pid_file": "x.pid", "health": "http://127.0.0.1:9/none"}},
         "arms": [{"key": "base", "label": "Base", "lang": "none"},
                  {"key": "q_kk", "label": "Q kk", "lang": "kk", "server": "q", "accent": True,
                   "overrides": ["backend=qolda"]},
                  {"key": "q_kk_nv", "label": "Q kk no verify", "lang": "kk", "verify": False, "server": "q"}],
         "contrasts": [["q_kk", "base"], ["q_kk", "q_kk_nv"]], "decompositions": [], "breakdowns": ["goal_category"]}
    m.update(over)
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(m), encoding="utf-8")
    return rm.load_matrix(p)


def fake_cell(c, n_selected, done_ids, failed=()):
    c.path.mkdir(parents=True, exist_ok=True)
    (c.path / "manifest.json").write_text(json.dumps({"n_selected": n_selected,
                                                      "config": {"run": {"max_episode_attempts": 2}}}))
    pd.DataFrame({"episode_id": list(done_ids)}).to_csv(c.path / "episodes.csv", index=False)
    if failed:
        pd.DataFrame({"episode_id": [e for e in failed for _ in range(2)]}).to_csv(c.path / "failures.csv", index=False)


def test_cells_arguments_and_states(tmp_path):
    m = matrix(tmp_path)
    cs = rm.cells(m)
    assert [c.key for c in cs] == ["base/seed0", "base/seed1", "q_kk/seed0", "q_kk/seed1", "q_kk_nv/seed0",
                                   "q_kk_nv/seed1"]
    assert rm.cell_args(m, cs[2]) == ["--lang", "kk", "dataset=hm3d_objectnav_v1_val", "backend=qolda", "run.seed=0"]
    assert "--no-verify" in rm.cell_args(m, cs[5]) and rm.cell_args(m, cs[0])[:2] == ["--lang", "none"]
    assert rm.status(cs[0])["state"] == "new"
    fake_cell(cs[0], 3, ["a", "b"])
    fake_cell(cs[1], 3, ["a", "b"], failed=["c"])                           # c given up after 2 attempts
    assert rm.status(cs[0])["state"] == "partial" and rm.status(cs[1]) == {"state": "complete", "done": 3,
                                                                           "selected": 3, "failed": 1}


def test_dry_run_groups_by_server_and_spec_needs_complete_cells(tmp_path, capsys):
    m = matrix(tmp_path)
    assert rm.run(m, None, dry=True) == 0
    out = capsys.readouterr().out
    assert out.index("base/seed0") < out.index("starting server q") < out.index("q_kk/seed0")
    assert out.count("starting server q") == 1 and "stopping server q" in out
    with pytest.raises(SystemExit, match="not complete"):
        rm.write_spec(m, tmp_path / "spec.yaml", allow_partial=False)
    for c in rm.cells(m):
        fake_cell(c, 1, ["a"])
    spec = rm.write_spec(m, tmp_path / "spec.yaml", allow_partial=False)
    assert [a["key"] for a in spec["arms"]] == ["base", "q_kk", "q_kk_nv"] and len(spec["arms"][1]["runs"]) == 2
    assert spec["arms"][1]["accent"] and spec["contrasts"] == [["q_kk", "base"], ["q_kk", "q_kk_nv"]]
    assert yaml.safe_load((tmp_path / "spec.yaml").read_text(encoding="utf-8"))["title"] == "T"


def test_matrix_validation(tmp_path):
    with pytest.raises(SystemExit, match="unknown server"):
        matrix(tmp_path, arms=[{"key": "a", "label": "A", "lang": "kk", "server": "nope"}], contrasts=[])
    with pytest.raises(SystemExit, match="unknown arm"):
        matrix(tmp_path, contrasts=[["q_kk", "ghost"]])
