"""Grammar rule selection in the Python binding: the default set, experimental rules on
request, and `rules`/`enable`/`disable` on `check_grammar`, `correct_grammar` and
`GrammarChecker`."""

import pytest

import akana


def test_rules_are_listed_with_their_status():
    rules = akana.grammar_rules()
    ids = [r["id"] for r in rules]
    assert len(ids) == len(set(ids))
    assert {"de-da-after-verb", "mi-merged", "quantity-plural"} <= set(ids)
    defaults = {r["id"] for r in rules if r["default"]}
    assert "quantity-plural" not in defaults
    assert "de-da-merged-fixed" in defaults
    assert set(akana.GrammarChecker().enabled_rules()) == defaults
    for r in rules:
        assert 0.0 < r["confidence"] <= 1.0
        assert r["summary"]


def test_experimental_rules_are_opt_in():
    text = "Pazardan üç elmalar aldım."
    fixed = "Pazardan üç elma aldım."
    assert akana.correct_grammar(text) == text
    assert akana.correct_grammar(text, experimental=True) == fixed
    assert akana.correct_grammar(text, enable=["quantity-plural"]) == fixed
    assert akana.GrammarChecker(experimental=True).correct(text) == fixed
    checker = akana.GrammarChecker()
    checker.enable_experimental()
    assert checker.correct(text) == fixed


def test_rules_can_be_chosen():
    text = "Ali de geldi, Veli te geldi."
    fixed = "Ali de geldi, Veli de geldi."
    assert akana.correct_grammar(text) == fixed
    assert akana.correct_grammar(text, disable=["de-da-hardened"]) == text
    assert akana.correct_grammar(text, rules=["capitalization"]) == text
    assert akana.correct_grammar(text, rules=["de-da-hardened"]) == fixed

    [finding] = akana.check_grammar(text).findings
    assert finding.rule == "de-da-hardened"
    assert finding.category == "CliticDeDa"

    checker = akana.GrammarChecker()
    checker.disable_rule("de-da-hardened")
    assert checker.correct(text) == text
    assert "de-da-hardened" not in checker.enabled_rules()
    checker.enable_rule("de-da-hardened")
    assert checker.correct(text) == fixed
    assert akana.GrammarChecker(rules=["capitalization"]).enabled_rules() == ["capitalization"]


def test_unknown_rule_raises_value_error():
    with pytest.raises(ValueError, match="nope"):
        akana.check_grammar("Bugün hava güzel.", rules=["nope"])
    with pytest.raises(ValueError):
        akana.correct_grammar("Bugün hava güzel.", enable=["nope"])
    with pytest.raises(ValueError):
        akana.GrammarChecker(disable=["nope"])
    with pytest.raises(ValueError):
        akana.GrammarChecker().enable_rule("nope")


@pytest.mark.parametrize(
    "text",
    [
        "Mahkeme önceki kararda direnmiştir.",
        "Cumhuriyet 1923'te kuruldu.",
        "Filmi yirmi kişi izledi.",
        "Bu karar alınan önlemlere dayanır.",
        "Ali'nin evi Bolu'da.",
        "Bunun için de çalıştık.",
    ],
)
def test_default_rules_leave_correct_text_alone(text):
    assert akana.correct_grammar(text) == text
    assert akana.check_grammar(text).findings == []
