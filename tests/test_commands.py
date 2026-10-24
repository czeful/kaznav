"""Natural command bank (D-048): selection by role, parallel texts, validation, sheet import."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from kznav.language.commands import FORMAT, NaturalCommands, TemplateCommands, load_commands

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import commands as commands_tool  # noqa: E402


def cmd(id, category="chair", split="test", status="approved", author="A01", reviewer="R01", verdict="ok",
        kk="Маған орындықты тауып берші.", kk_ru="Стулды тауып берші.", en="Find me a chair."):
    return {"id": id, "category": category, "author": author, "split": split, "style": "direct",
            "texts": {"kk": kk, "kk_ru": kk_ru, "en": en},
            "reviews": [{"reviewer": reviewer, "verdict": verdict}] if reviewer else [], "status": status}


def bank(tmp_path, commands, **over):
    data = {"format": FORMAT, "dataset": "hm3d_objectnav_v1", "lexicon": "lexicon/hm3d_objectnav.yaml",
            "languages": ["kk", "kk_ru", "en"], "min_reviews": 1, "commands": commands, **over}
    p = tmp_path / "bank.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


GOOD = [cmd("chair-1"), cmd("chair-2", kk="Отыратын орындық қайда?", kk_ru="Стул қайда тұр?", en="Where is a chair?"),
        cmd("chair-3", status="draft", reviewer=""), cmd("chair-dev", split="dev", status="draft", reviewer=""),
        cmd("bed-1", category="bed", kk="Төсекке апар.", kk_ru="Кроватқа апар.", en="Take me to the bed.")]


def test_benchmark_role_uses_approved_test_commands_in_every_language(tmp_path):
    nc = load_commands(bank(tmp_path, GOOD), REPO, role="benchmark")
    assert isinstance(nc, NaturalCommands) and nc.review_status == "approved"
    assert {c.id for c in nc.candidates("chair")} == {"chair-1", "chair-2"}
    picks = {nc.command(f"scene:{i}", "kk", "chair")[0] for i in range(40)}
    assert picks == {"chair-1", "chair-2"}                                   # both are used
    for i in range(10):
        ids = {nc.command(f"scene:{i}", lang, "chair")[0] for lang in ("kk", "kk_ru", "en")}
        assert len(ids) == 1                                                 # same intent in every language
    assert nc.command("scene:0", "en", "bed") == ("bed-1", "Take me to the bed.")
    with pytest.raises(KeyError, match="toilet"):
        nc.command("scene:0", "kk", "toilet")
    shuffled = load_commands(bank(tmp_path, GOOD[::-1]), REPO, role="benchmark")
    assert [shuffled.command(f"s:{i}", "kk", "chair") for i in range(20)] == \
        [nc.command(f"s:{i}", "kk", "chair") for i in range(20)]               # file order does not matter


def test_dev_role_uses_dev_commands_and_templates_still_load(tmp_path):
    nc = load_commands(bank(tmp_path, GOOD), REPO, role="dev")
    assert [c.id for c in nc.candidates("chair")] == ["chair-dev"] and nc.review_status == "draft_needs_native_review"
    t = load_commands(REPO / "commands" / "hm3d_objectnav_v1.yaml", REPO)
    assert isinstance(t, TemplateCommands) and t.object_name("kk_ru", "chair") == ("стул", "ru")


def test_validation_catches_review_and_text_problems(tmp_path):
    bad = [cmd("x-1", reviewer="A01"),                                       # self-review, approved
           cmd("x-2", en="Найди стул"),                                       # Cyrillic in English
           cmd("x-3", kk=""),                                                 # missing Kazakh
           cmd("x-4", category="spaceship"),
           cmd("x-5", kk_ru="Маған орындықты тауып берші."),                   # kk_ru == kk
           cmd("x-1", status="draft", reviewer="")]                           # duplicate id
    issues = NaturalCommands(bank(tmp_path, bad), REPO, role="debug").validate(categories=["chair", "bed"],
                                                                                min_test_per_category=2)
    msgs = {(i.command_id, i.level, i.message.split(" ")[0]) for i in issues}
    assert ("x-1", "error", "an") in msgs and ("x-1", "error", "duplicate") in msgs
    assert ("x-2", "error", "en") in msgs and ("x-3", "error", "no") in msgs and ("x-4", "error", "category") in msgs
    assert ("x-5", "warning", "kk") in msgs and ("[bed]", "error", "0") in msgs
    assert not [i for i in NaturalCommands(bank(tmp_path, GOOD), REPO, role="debug").validate() if i.level == "error"]


def test_coverage_counts_dictionary_names_with_case_endings(tmp_path):
    cov = {r["category"]: r for r in NaturalCommands(bank(tmp_path, GOOD), REPO, role="debug").coverage()}
    assert cov["chair"]["kk names object"] == "4/4"                         # орындықты, орындық
    assert cov["chair"]["test approved"] == 2 and cov["chair"]["dev draft"] == 1 and cov["bed"]["authors"] == 1


def test_sheet_import_round_trip(tmp_path):
    sheet = tmp_path / "sheet.csv"
    n = commands_tool.sheet(Path("lexicon/hm3d_objectnav.yaml"), ["A01", "A02"], 3, 1, sheet)
    assert n == 18
    df = pd.read_csv(sheet, dtype=str, encoding="utf-8-sig").fillna("")
    chair = df.category == "chair"
    df.loc[chair, ["kk", "kk_ru", "en"]] = ["Орындықты тап.", "Стулды тап.", "Find the chair."]
    df.loc[chair, ["reviewer", "verdict"]] = ["R09", "ok"]
    df.loc[df.index[chair][-1], "verdict"] = "fix"
    df.to_csv(sheet, index=False, encoding="utf-8-sig")
    out = tmp_path / "bank.yaml"
    assert commands_tool.import_sheet(sheet, "hm3d_objectnav_v1", "lexicon/hm3d_objectnav.yaml", out, 1) == 3
    nc = NaturalCommands(out, REPO, role="benchmark")
    assert [c.status for c in nc.commands] == ["approved", "approved", "draft"]    # the last one needs a fix
    assert [c.split for c in nc.commands] == ["dev", "test", "test"]
    assert nc.command("s:1", "en", "chair") == (nc.commands[1].id, "Find the chair.")
    assert commands_tool.validate(out, min_test=1) == 1                     # other categories have no commands yet
