"""Transliteration and script detection.

The ordering assertion at the bottom is the important one: it pins the
invariant that the rest of the screening engine depends on.
"""

from __future__ import annotations

import pytest

from src.exceptions import SanctionsMatchingError, TransliterationError
from src.sanctions.phonetics import phonetic_keys
from src.sanctions.transliterate import Method, Script, detect_script, to_latin, transliterate


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mohammed Hassan", Script.LATIN),
        ("محمد حسن", Script.ARABIC),
        ("Владимир Иванов", Script.CYRILLIC),
        ("张伟", Script.HAN),
        ("José Müller", Script.LATIN),
        ("", Script.LATIN),
        ("12345 !!", Script.LATIN),
        # Mixed strings resolve to the majority script, which is how OFAC
        # renders a Latin name with a native-script parenthetical.
        ("ALI Hassan علي", Script.LATIN),
    ],
)
def test_detect_script(text: str, expected: Script) -> None:
    assert detect_script(text) is expected


@pytest.mark.parametrize(
    ("cyrillic", "expected"),
    [
        ("Иванов", "ivanov"),
        ("Владимир", "vladimir"),
        ("Александр", "aleksandr"),
        ("Шевченко", "shevchenko"),
        ("Жуков", "zhukov"),
        ("Хабаров", "khabarov"),
    ],
)
def test_cyrillic_uses_digraph_romanization(cyrillic: str, expected: str) -> None:
    assert to_latin(cyrillic).lower() == expected


@pytest.mark.parametrize(
    ("arabic", "expected"),
    [("محمد", "muhammad"), ("أحمد", "ahmad"), ("حسين", "husayn"), ("عبدالله", "abdullah")],
)
def test_arabic_lexicon_restores_short_vowels(arabic: str, expected: str) -> None:
    result = transliterate(arabic)
    assert result.text == expected
    assert result.method is Method.LEXICON


def test_arabic_definite_article_is_stripped() -> None:
    # "al-" fuses onto the noun and carries no discriminating power; leaving it
    # in would give every Arabic surname a shared prefix.
    assert to_latin("الحربي") == "harbi"


def test_arabic_falls_back_to_character_table_for_unknown_tokens() -> None:
    result = transliterate("بترول")
    assert result.method is Method.TABLE
    assert result.text == "btrwl"
    assert result.lossy is True


def test_han_surname_kept_as_single_token() -> None:
    assert to_latin("张伟") == "zhang wei"
    assert to_latin("王小明").startswith("wang")


def test_latin_with_diacritics_folds_without_being_marked_lossy() -> None:
    result = transliterate("José Müller")
    assert result.text == "Jose Muller"
    assert result.lossy is False
    assert result.source_script is Script.LATIN


def test_empty_input_is_not_an_error() -> None:
    assert transliterate("   ").text == ""


def test_none_input_raises() -> None:
    with pytest.raises(TransliterationError):
        transliterate(None)  # type: ignore[arg-type]


def test_phonetics_refuse_non_latin_input() -> None:
    """The guard that makes the pipeline ordering enforceable rather than advisory."""
    with pytest.raises(SanctionsMatchingError, match="transliterate to Latin"):
        phonetic_keys("Иванов")
    with pytest.raises(SanctionsMatchingError):
        phonetic_keys("محمد")


def test_transliteration_before_phonetics_changes_the_key() -> None:
    """Concrete evidence for the ordering requirement.

    The Arabic consonant skeleton and the conventional romanization produce
    different Double Metaphone keys, so a pipeline that skipped the lexicon
    would block محمد away from every spelling of Mohammed.
    """
    from unidecode import unidecode

    skeleton = unidecode("محمد")
    assert phonetic_keys(skeleton).primary != phonetic_keys("Mohammed").primary
    assert phonetic_keys(to_latin("محمد")).primary == phonetic_keys("Mohammed").primary
