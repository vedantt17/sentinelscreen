"""Sanctions screening: normalization, transliteration, phonetics, blocking, scoring.

Pipeline order is load-bearing and is enforced by `engine.ScreeningEngine`:

    raw name -> transliterate to Latin -> normalize -> phonetic key -> block -> score

Transliteration must precede phonetics. Double Metaphone encodes *English*
orthography-to-sound rules; applied to a Cyrillic or Han codepoint it produces a
key that is syntactically valid and semantically meaningless, which silently
destroys recall instead of raising an error.
"""

from src.sanctions.engine import MatchDecision, MatchResult, ScreeningEngine
from src.sanctions.normalize import NormalizedName, normalize_name
from src.sanctions.phonetics import phonetic_keys
from src.sanctions.transliterate import Script, detect_script, transliterate

__all__ = [
    "MatchDecision",
    "MatchResult",
    "NormalizedName",
    "Script",
    "ScreeningEngine",
    "detect_script",
    "normalize_name",
    "phonetic_keys",
    "transliterate",
]
