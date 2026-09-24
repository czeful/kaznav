import json

import pytest

from kznav.language.jsonx import _find_balanced_braces
from kznav.language.lexicon import Category, Lexicon, normalize
from kznav.language.prompts import (
    PromptLibrary,
    build_parse_messages,
    build_parse_retry_messages,
    build_verify_messages,
    fewshot_targets,
)
from kznav.language.schemas import GoalHypotheses


class TestPromptFiles:
    def test_all_prompts_load(self, prompt_lib):
        refs = prompt_lib.all_refs()
        assert {"default/parse_goal/v1", "default/verify_candidate/v1", "default/describe/v1"} <= set(refs)
        for r in refs:
            t = prompt_lib.load(r)
            assert len(t.sha256) == 64

    def test_hash_ignores_line_endings(self, tmp_path, repo_root):
        src = repo_root / "prompts" / "default" / "describe" / "v1.yaml"
        for sub, eol in (("lf", b"\n"), ("crlf", b"\r\n")):
            d = tmp_path / sub / "default" / "describe"
            d.mkdir(parents=True)
            (d / "v1.yaml").write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", eol))
        a = PromptLibrary(tmp_path / "lf").load("default/describe/v1").sha256
        b = PromptLibrary(tmp_path / "crlf").load("default/describe/v1").sha256
        assert a == b

    def test_version_must_match_filename(self, tmp_path, repo_root):
        d = tmp_path / "default" / "describe"
        d.mkdir(parents=True)
        text = (repo_root / "prompts/default/describe/v1.yaml").read_text(encoding="utf-8")
        (d / "v2.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            PromptLibrary(tmp_path).load("default/describe/v2")

    def test_parse_prompt_schema_matches_code(self, prompt_lib):
        """The JSON Schema pasted into the prompt must describe the pydantic model the code validates.
        A schema change therefore forces a new prompt version."""
        system = prompt_lib.load("default/parse_goal/v1").data["system"]
        block = _find_balanced_braces(system[system.index("Схема"):])
        embedded = json.loads(block)
        current = GoalHypotheses.model_json_schema()

        structural = {"type", "required", "$ref", "items", "minItems", "maxItems", "minimum", "maximum"}

        def shape(s):
            """Schema without titles/descriptions: only what constrains the JSON."""
            if isinstance(s, dict):
                out = {}
                for k, v in s.items():
                    if k in ("properties", "$defs"):
                        out[k] = {name: shape(sub) for name, sub in v.items()}
                    elif k in structural:
                        out[k] = shape(v)
                return out
            if isinstance(s, list):
                return [shape(x) for x in s]
            return s

        assert shape(embedded) == shape(current)

    def test_fewshot_is_valid_and_not_an_evaluated_object(self, prompt_lib, repo_root):
        tpl = prompt_lib.load("default/parse_goal/v1")
        for ex in tpl.data["fewshot"]:
            GoalHypotheses.model_validate_json(ex["assistant"])
        forms = set()
        for lex_file in (repo_root / "lexicon").glob("*.yaml"):
            lex = Lexicon.load(lex_file)
            for c in lex.categories.values():
                for names in c.names.values():
                    forms |= {normalize(n) for n in names}
        for t in fewshot_targets(tpl):
            assert normalize(t) not in forms, f"few-shot target {t!r} is an evaluated object"

    def test_parse_messages_structure(self, prompt_lib):
        tpl = prompt_lib.load("default/parse_goal/v1")
        msgs = build_parse_messages(tpl, "Диванды тап {with braces}")
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[-1]["content"] == "Диванды тап {with braces}"
        retry = build_parse_retry_messages(tpl, "x", "bad", "target: missing {x}")
        assert retry[-2] == {"role": "assistant", "content": "bad"}
        assert "target: missing {x}" in retry[-1]["content"]

    @pytest.mark.parametrize("lang,verify_lang", [("kk", "kk"), ("kk_ru", "kk"), ("ru", "ru"), ("en", "en")])
    def test_verify_languages(self, prompt_lib, lang, verify_lang):
        tpl = prompt_lib.load("default/verify_candidate/v1")
        msgs, options, vl = build_verify_messages(tpl, "қазан", lang)
        assert vl == verify_lang
        assert "қазан" in msgs[-1]["content"]
        assert set(options) == {"yes", "no"}

    def test_verify_unknown_language(self, prompt_lib):
        with pytest.raises(KeyError):
            build_verify_messages(prompt_lib.load("default/verify_candidate/v1"), "x", "de")


class TestLexicon:
    @pytest.fixture(scope="class")
    def lex(self, repo_root):
        return Lexicon.load(repo_root / "lexicon" / "hm3d_objectnav.yaml")

    def test_normalize(self):
        assert normalize("Chairs", english=True) == "chair"
        assert normalize("couches", english=True) == "couch"
        assert normalize("glass", english=True) == "glass"
        assert normalize("TV_monitor") == "tv monitor"
        assert normalize("  Үлкен   ҚАЗАН! ") == "үлкен қазан"

    @pytest.mark.parametrize("cat,hyp,ok", [
        ("chair", "wooden armchair", True),
        ("chair", "chairs by the table", True),
        ("plant", "planter box", False),
        ("tv_monitor", "flat screen TV", True),
        ("sofa", "leather couch", True),
        ("bed", "bedside table", False),
    ])
    def test_match_en(self, lex, cat, hyp, ok):
        assert lex.match_en(cat, hyp) is ok

    def test_match_target(self, lex):
        assert lex.match_target("chair", "орындық", "kk") is True
        assert lex.match_target("chair", "орындықты", "kk") is False   # inflected forms must be listed
        assert lex.match_target("chair", "стул", "kk_ru") is True
        assert lex.match_target("chair", "Chairs", "en") is True        # English: plural-normalised
        assert lex.match_target("chair", "x", "de") is None

    def test_duplicate_category_rejected(self):
        c = Category("a", "universal", {"en": ["a"]})
        with pytest.raises(ValueError):
            Lexicon([c, c])

    def test_bad_word_type_rejected(self):
        with pytest.raises(ValueError):
            Lexicon([Category("a", "exotic", {})])

    def test_all_lexicons_load(self, repo_root):
        for f in (repo_root / "lexicon").glob("*.yaml"):
            assert Lexicon.load(f).categories
