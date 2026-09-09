"""Double Metaphone keys — computed on transliterated Latin text ONLY.

The guard in `phonetic_keys` is the point of this module. Double Metaphone will
happily accept "Иванов" and return a key; that key is an artefact of the
algorithm skipping codepoints it has no rule for, not a phonetic encoding of
anything. Rather than let that propagate into the blocking index, a non-Latin
token raises. Callers are expected to have run `transliterate` first, and the
engine does.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from metaphone import doublemetaphone

from src.exceptions import SanctionsMatchingError
from src.sanctions.transliterate import Script, detect_script


@dataclass(frozen=True)
class PhoneticKey:
    primary: str
    alternate: str

    def keys(self) -> tuple[str, ...]:
        """Both encodings, empties dropped.

        Double Metaphone emits an alternate for names whose pronunciation forks
        by language of origin — 'Wang' yields ANK and FNK because the initial W
        is a glide in English and a fricative in German/Slavic readings. Keeping
        both is what lets a Han-origin and a Slavic-origin spelling collide in
        the same block.
        """
        return tuple(k for k in (self.primary, self.alternate) if k)

    def __bool__(self) -> bool:
        return bool(self.primary or self.alternate)


@lru_cache(maxsize=200_000)
def phonetic_keys(token: str) -> PhoneticKey:
    """Double Metaphone key pair for a single Latin token.

    Cached because blocking recomputes keys for the same high-frequency
    surnames tens of thousands of times across a 50k x 5k screening run.
    """
    if not token:
        return PhoneticKey("", "")
    script = detect_script(token)
    if script is not Script.LATIN:
        raise SanctionsMatchingError(
            f"phonetic encoding attempted on {script.value} text {token!r}; "
            "transliterate to Latin before computing phonetic keys"
        )
    primary, alternate = doublemetaphone(token)
    return PhoneticKey(primary or "", alternate or "")


def key_set(tokens: Iterable[str]) -> frozenset[str]:
    keys: set[str] = set()
    for token in tokens:
        keys.update(phonetic_keys(token).keys())
    return frozenset(keys)


def phonetic_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Fraction of name tokens that have a phonetic counterpart on the other side.

    Token-level rather than set-level: 'Mohammed Ali Hassan' vs 'Ali Hassan'
    should not score 1.0 simply because every key of the shorter name is
    present. Dividing by the longer token count penalises the missing token,
    which is the behaviour an analyst expects when a middle name is absent.
    """
    if not left or not right:
        return 0.0
    right_keys = [phonetic_keys(t).keys() for t in right]
    matched = 0
    consumed: set[int] = set()
    for token in left:
        candidate = phonetic_keys(token).keys()
        if not candidate:
            continue
        for index, other in enumerate(right_keys):
            if index in consumed:
                continue
            if set(candidate) & set(other):
                matched += 1
                consumed.add(index)
                break
    return matched / max(len(left), len(right))


def surname_phonetic(tokens: Sequence[str]) -> str:
    """Primary key of the trailing token, used as the main blocking key."""
    if not tokens:
        return ""
    return phonetic_keys(tokens[-1]).primary
