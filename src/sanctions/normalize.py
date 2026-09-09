"""Name normalization: the second stage, run on the Latin form only.

Everything here is deliberately reversible-in-audit: `NormalizedName` keeps the
raw input and records exactly which honorifics and particles were stripped, so
an analyst reviewing an alert can see what the engine actually compared rather
than having to trust it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

from src.config import ScreeningConfig
from src.sanctions.transliterate import Script, transliterate

# Nobiliary and patronymic particles. These carry almost no discriminating
# power (a meaningful share of Arabic SDN entries contain "al", most Dutch
# entries contain "van") while contributing a shared prefix that inflates
# Jaro-Winkler, which is prefix-weighted by construction. They are removed from
# the comparison tokens and retained on the record for explainability.
NAME_PARTICLES: Final[frozenset[str]] = frozenset(
    {
        "al", "el", "ul", "bin", "ibn", "bint", "binti", "ben", "abu", "umm",
        "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos",
        "das", "du", "la", "le", "les", "of", "the", "and", "y", "e",
    }
)

_PUNCTUATION_TO_SPACE: Final[re.Pattern[str]] = re.compile(r"[^\w\s]", re.UNICODE)
_DIGITS: Final[re.Pattern[str]] = re.compile(r"\d+")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedName:
    raw: str
    latin: str
    normalized: str
    tokens: tuple[str, ...]
    sorted_key: str
    initials: str
    script: Script
    is_organization: bool
    removed_honorifics: tuple[str, ...]
    removed_particles: tuple[str, ...]
    lossy_transliteration: bool

    @property
    def surname_token(self) -> str:
        """Best available surname proxy.

        Western order puts the surname last; Han and Hungarian order put it
        first. Rather than guess the culture from the name, the engine compares
        against the order-invariant `sorted_key` as well, so this is only a
        blocking heuristic and a wrong guess costs recall in one strategy, not
        in the match itself.
        """
        return self.tokens[-1] if self.tokens else ""

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def is_empty(self) -> bool:
        return not self.tokens


def _fold(text: str) -> str:
    """NFKD fold: decompose, drop combining marks, collapse compatibility forms.

    NFKD rather than NFD because watchlist feeds contain fullwidth Latin and
    ligatures (ﬁ, Ｍ) that must collapse onto their ASCII equivalents before
    tokenization, or they become their own tokens.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_marks.casefold()


def _strip_legal_form(
    tokens: list[str], suffixes: frozenset[str]
) -> tuple[list[str], list[str]]:
    """Remove legal-form tokens from the edges of a name.

    Position is what distinguishes a legal form from a name fragment. "Ltd" at
    the end of "Crescent Shipping Ltd" is a company suffix; "Co" in the middle
    of "James Co Anderson" is part of a person's name, and stripping it would
    quietly mangle an individual's record. Russian and Polish legal forms lead
    rather than trail ("OOO Vostok"), so both edges are trimmed.
    """
    start, end = 0, len(tokens)
    removed: list[str] = []
    while end > start and tokens[end - 1] in suffixes:
        end -= 1
        removed.append(tokens[end])
    while start < end and tokens[start] in suffixes:
        removed.append(tokens[start])
        start += 1
    return tokens[start:end], removed


def normalize_name(raw: str, config: ScreeningConfig) -> NormalizedName:
    """Transliterate, fold, strip noise tokens, and derive comparison keys."""
    source = raw or ""
    result = transliterate(source)

    text = _fold(result.text)
    if config.strip_punctuation:
        # Hyphens and apostrophes become boundaries, not deletions: "Al-Hassan"
        # must become two tokens so that the surname can block independently,
        # whereas deleting the hyphen would produce the unmatched "alhassan".
        text = _PUNCTUATION_TO_SPACE.sub(" ", text)
    text = _DIGITS.sub(" ", text)
    if config.collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    text = text.strip()

    candidate_tokens = [t for t in text.split(" ") if t]

    core_tokens, legal_forms = _strip_legal_form(candidate_tokens, config.corporate_suffixes)
    is_org = bool(legal_forms)

    kept: list[str] = []
    removed_honorifics: list[str] = []
    removed_particles: list[str] = list(legal_forms)
    for token in core_tokens:
        if token in config.honorifics:
            removed_honorifics.append(token)
            continue
        if token in NAME_PARTICLES:
            removed_particles.append(token)
            continue
        if config.drop_single_char_tokens and len(token) == 1:
            # Off by default: single-letter tokens are usually middle initials,
            # and an initial that agrees with the watchlist is weak corroboration
            # worth keeping rather than discarding.
            continue
        kept.append(token)

    if not kept and candidate_tokens:
        # A name made entirely of particles ("Al Abu") is unusual but real.
        # Falling back keeps the record screenable instead of silently dropping
        # it from the population, which would be a false negative by omission.
        kept = candidate_tokens
        removed_honorifics = []
        removed_particles = []

    tokens = tuple(kept)
    return NormalizedName(
        raw=source,
        latin=result.text,
        normalized=" ".join(tokens),
        tokens=tokens,
        sorted_key=" ".join(sorted(tokens)),
        initials="".join(t[0] for t in tokens if t),
        script=result.source_script,
        is_organization=is_org,
        removed_honorifics=tuple(removed_honorifics),
        removed_particles=tuple(removed_particles),
        lossy_transliteration=result.lossy,
    )
