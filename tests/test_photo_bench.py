"""Cultural photo benchmark (option C, H2; D-055): table checks, AUROC, the pair interaction, the tool on mock."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from kznav.eval.photo_bench import PHOTO_COLUMNS, auroc, load_photos, pair_interaction, verify_metrics
from kznav.language.commands import FORMAT, NaturalCommands
from kznav.language.lexicon import Lexicon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import eval_photo_benchmark as tool  # noqa: E402

LEX = Lexicon.load(REPO / "lexicon" / "cultural_pairs.yaml")


def photos_csv(tmp_path, rows):
    d = tmp_path / "photos"
    d.mkdir(exist_ok=True)
    for r in rows:
        Image.fromarray(np.full((16, 16, 3), hash(r["photo_id"]) % 255, np.uint8)).save(d / r["file"])
    p = d / "photos.csv"
    pd.DataFrame(rows, columns=PHOTO_COLUMNS).to_csv(p, index=False)
    return p


def row(pid, cat, pair, wt, split="test", **over):
    return {"photo_id": pid, "file": f"{pid}.png", "category": cat, "pair": pair, "word_type": wt, "split": split,
            "viewpoint": "low", "instance": pid, "source": "own", "license": "own", "author": "A01", **over}


GOOD = [row("k1", "қазан", "cooking_pot", "cultural"), row("k2", "қазан", "cooking_pot", "cultural"),
        row("p1", "cooking_pot", "қазан", "universal"), row("p2", "cooking_pot", "қазан", "universal")]


def test_photo_table_checks(tmp_path):
    assert len(load_photos(photos_csv(tmp_path, GOOD), LEX.categories)) == 4
    with pytest.raises(ValueError, match="not paired back"):
        load_photos(photos_csv(tmp_path, GOOD[:2]), LEX.categories)
    with pytest.raises(ValueError, match="without a license"):
        load_photos(photos_csv(tmp_path, GOOD[:3] + [row("p2", "cooking_pot", "қазан", "universal", license="")]),
                    LEX.categories)
    with pytest.raises(ValueError, match="same word_type"):
        load_photos(photos_csv(tmp_path, GOOD[:2] + [row("p1", "cooking_pot", "қазан", "cultural")]), LEX.categories)
    with pytest.raises(ValueError, match="not in the lexicon"):
        load_photos(photos_csv(tmp_path, GOOD + [row("x", "spaceship", "қазан", "universal")]), LEX.categories)


def test_auroc_and_verify_metrics():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auroc([0.5], [0.5]) == 0.5 and np.isnan(auroc([0.5], []))
    rows = pd.DataFrame({"lang": "kk", "target": "қазан", "gold": [True, True, False, False],
                         "p_yes": [0.9, 0.4, 0.6, 0.1]})
    m = verify_metrics(rows).iloc[0]
    assert m.accuracy == pytest.approx(0.5) and m.auroc == pytest.approx(0.75)


def test_pair_interaction_is_positive_when_the_cultural_gap_is_larger():
    pairs, wt, rows = {}, {}, []
    for i in range(6):
        c, u = f"c{i}", f"u{i}"
        pairs[c], pairs[u] = u, c
        wt[c], wt[u] = "cultural", "universal"
        rows += [{"lang": "en", "target": c, "m": 0.9}, {"lang": "kk", "target": c, "m": 0.4 + 0.02 * i},
                 {"lang": "en", "target": u, "m": 0.9}, {"lang": "kk", "target": u, "m": 0.85}]
    r = pair_interaction(pd.DataFrame(rows), "m", pairs, wt, "kk", n_boot=2000)
    assert r["pairs"] == 6 and r["interaction"] == pytest.approx(0.45 - 0.02 * 2.5)   # (0.5 - 0.02 i) - 0.05
    assert r["p"] == pytest.approx(2 / 2 ** 6) and r["ci_low"] > 0


def test_tool_runs_end_to_end_on_the_mock_backend(tmp_path, backend_spec, prompt_lib):
    from kznav.language.base import ListSink, TeeSink
    from kznav.language.goal import make_goal_cache
    from kznav.language.registry import create_backend
    from kznav.run.config import compose_config

    bank_file = tmp_path / "bank.yaml"
    cmds = [{"id": f"{c}-1", "category": c, "author": "A01", "split": "test", "style": "direct",
             "texts": {"kk": "Қазанды тап.", "kk_ru": "Казанды тап.", "en": "Find the cauldron."},
             "reviews": [{"reviewer": "R01", "verdict": "ok"}], "status": "approved"} for c in ("қазан", "cooking_pot")]
    bank_file.write_text(yaml.safe_dump({"format": FORMAT, "dataset": "cultural", "lexicon": "lexicon/cultural_pairs.yaml",
                                         "languages": ["kk", "kk_ru", "en"], "commands": cmds}, allow_unicode=True),
                         encoding="utf-8")
    bank = NaturalCommands(bank_file, REPO, role="debug")
    photos = load_photos(photos_csv(tmp_path, GOOD), bank.lexicon.categories)
    calls = ListSink()
    backend = create_backend(backend_spec("mock"), prompts=prompt_lib, sink=TeeSink(ListSink(), calls))
    goals = make_goal_cache(compose_config([])["goal"], backend, calls, bank.lexicon)
    parse = tool.run_parse(bank, ["kk", "en"], "test", goals, backend)
    assert len(parse) == 4 and set(parse.lang) == {"kk", "en"}
    ver = tool.run_verify(photos, tmp_path / "photos", bank.lexicon, ["kk", "kk_ru", "en"], backend)
    assert len(ver) == 4 * 2 * 3 and ver.gold.sum() == 4 * 3
    assert set(ver.loc[ver.lang == "kk_ru", "word"]) == {"казан", "кастрюля"}           # Russian names in kk-ru
    vm = verify_metrics(ver)
    assert set(vm.target) == {"қазан", "cooking_pot"} and vm.auroc.between(0, 1).all()
