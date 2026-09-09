"""String-similarity signals used by the multi-signal scorer.

Each function returns a value on [0, 1] and is independently interpretable, so
an alert can be explained as "0.94 orthographic, 1.00 phonetic, 0.67 token
overlap" rather than as a single opaque number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import token_set_ratio

from src.sanctions.normalize import NormalizedName
from src.sanctions.phonetics import phonetic_similarity


@dataclass(frozen=True)
class SimilarityBreakdown:
    jaro_winkler: float
    token_set: float
    phonetic: float
    initials: float
    order_swapped: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "jaro_winkler": round(self.jaro_winkler, 6),
            "token_set": round(self.token_set, 6),
            "phonetic": round(self.phonetic, 6),
            "initials": round(self.initials, 6),
            "order_swapped": self.order_swapped,
        }


def jaro_winkler(left: str, right: str, prefix_weight: float = 0.1) -> float:
    if not left or not right:
        return 0.0
    return float(JaroWinkler.similarity(left, right, prefix_weight=prefix_weight))


def token_set_similarity(left: str, right: str) -> float:
    """Order- and duplicate-insensitive overlap.

    This is the signal that survives the two most common watchlist defects:
    additional middle names on one side, and a name recorded surname-first.
    """
    if not left or not right:
        return 0.0
    return float(token_set_ratio(left, right)) / 100.0


def initials_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Agreement between the sets of leading characters.

    Weak on its own, but it is the one signal that still fires when a name has
    been abbreviated to initials in a wire message ('M. A. HASSAN'), which is
    common in MT103 originator fields.
    """
    if not left or not right:
        return 0.0
    left_initials = {t[0] for t in left if t}
    right_initials = {t[0] for t in right if t}
    if not left_initials or not right_initials:
        return 0.0
    return len(left_initials & right_initials) / max(len(left_initials), len(right_initials))


def compare_names(left: NormalizedName, right: NormalizedName, prefix_weight: float = 0.1) -> SimilarityBreakdown:
    """Compute every orthographic and phonetic signal for a candidate pair.

    Jaro-Winkler is evaluated twice: once in the recorded token order and once
    on the alphabetically sorted tokens. Taking the maximum is what neutralises
    name-order swaps, which are endemic because OFAC records Arabic and Han
    names surname-first while most KYC systems record them surname-last. The
    `order_swapped` flag records when the sorted form is what carried the match,
    because that is material context for an analyst.
    """
    direct = jaro_winkler(left.normalized, right.normalized, prefix_weight)
    sorted_form = jaro_winkler(left.sorted_key, right.sorted_key, prefix_weight)
    best = max(direct, sorted_form)
    return SimilarityBreakdown(
        jaro_winkler=best,
        token_set=token_set_similarity(left.normalized, right.normalized),
        phonetic=phonetic_similarity(left.tokens, right.tokens),
        initials=initials_similarity(left.tokens, right.tokens),
        # A tolerance rather than a strict inequality: floating-point equality
        # between two Jaro-Winkler values would flag identical names as swapped.
        order_swapped=sorted_form > direct + 1e-9,
    )
