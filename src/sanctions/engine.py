"""Multi-signal sanctions screening engine.

Scoring is deliberately two-stage: an orthographic/phonetic *name score* on
[0, 1], then additive corroboration from identifiers. Keeping them separate is
what makes an alert defensible — "the names are a 0.91 match but the dates of
birth are eleven years apart" is a disposition an analyst can write down,
whereas a single blended number is not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist

from src.config import ScreeningConfig
from src.exceptions import SanctionsMatchingError
from src.logging_setup import get_logger
from src.sanctions.blocking import BlockingIndex, BlockingStats
from src.sanctions.normalize import NormalizedName, normalize_name
from src.sanctions.similarity import SimilarityBreakdown, compare_names

logger = get_logger(__name__)


class MatchDecision(str, Enum):
    ESCALATE = "ESCALATE"   # straight to L2 / sanctions officer
    REVIEW = "REVIEW"       # L1 analyst queue
    DISCARD = "DISCARD"     # below threshold, retained only in the audit trail


@dataclass(frozen=True)
class ScreeningSubject:
    subject_id: str
    name: str
    dob: str | None = None
    nationality: str | None = None
    national_id: str | None = None


@dataclass(frozen=True)
class WatchlistEntry:
    uid: str
    name: str
    program: str
    entity_type: str
    dob: str | None = None
    nationality: str | None = None
    national_id: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchResult:
    subject_id: str
    subject_name: str
    watchlist_uid: str
    watchlist_name: str
    matched_on: str
    program: str
    name_score: float
    corroboration_delta: float
    final_score: float
    decision: MatchDecision
    signals: SimilarityBreakdown
    corroboration: tuple[tuple[str, float], ...]
    rule_version: str

    def scoring_breakdown(self) -> dict[str, object]:
        """Flat, JSON-serialisable explanation written into the audit record."""
        breakdown: dict[str, object] = {
            "name_score": round(self.name_score, 6),
            "corroboration_delta": round(self.corroboration_delta, 6),
            "final_score": round(self.final_score, 6),
            "decision": self.decision.value,
            "rule_version": self.rule_version,
            "matched_on": self.matched_on,
        }
        breakdown.update({f"signal_{k}": v for k, v in self.signals.as_dict().items()})
        breakdown.update({f"corr_{k}": round(v, 6) for k, v in self.corroboration})
        return breakdown


@dataclass(frozen=True)
class ScreeningRun:
    matches: tuple[MatchResult, ...]
    blocking: BlockingStats
    subjects_screened: int
    scored_pairs: int
    elapsed_seconds: float


def _dob_year(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    return text[:4] if len(text) >= 4 and text[:4].isdigit() else None


@dataclass
class ScreeningEngine:
    """Stateful screener: index the watchlist once, then probe it many times."""

    config: ScreeningConfig
    _entries: tuple[WatchlistEntry, ...] = ()
    _reference_names: list[NormalizedName] = field(default_factory=list)
    _reference_owner: list[int] = field(default_factory=list)
    _reference_source: list[str] = field(default_factory=list)
    _reference_sorted_keys: list[str] = field(default_factory=list)
    _index: BlockingIndex | None = None
    _prefiltered_pairs: int = 0

    # -------------------------------------------------------------- indexing
    def index_watchlist(self, entries: Sequence[WatchlistEntry]) -> None:
        """Normalize primary names and every a.k.a., then build the block index.

        Aliases are indexed as first-class reference names rather than as a
        secondary pass. On the real SDN list the a.k.a. is frequently the form a
        payment message actually carries, so treating it as a lesser signal is
        how institutions miss hits.
        """
        if not entries:
            raise SanctionsMatchingError("cannot index an empty watchlist")
        self._entries = tuple(entries)
        self._reference_names = []
        self._reference_owner = []
        self._reference_source = []
        for position, entry in enumerate(self._entries):
            for label, raw in (("primary", entry.name), *(("alias", a) for a in entry.aliases)):
                normalized = normalize_name(raw, self.config)
                if normalized.is_empty():
                    continue
                self._reference_names.append(normalized)
                self._reference_owner.append(position)
                self._reference_source.append(label)
                self._reference_sorted_keys.append(normalized.sorted_key)
        if not self._reference_names:
            raise SanctionsMatchingError("watchlist produced no usable reference names")
        index = BlockingIndex(config=self.config)
        index.build(self._reference_names)
        self._index = index
        logger.info(
            "watchlist indexed",
            extra={
                "entities": len(self._entries),
                "reference_names": len(self._reference_names),
                "alias_names": sum(1 for s in self._reference_source if s == "alias"),
            },
        )

    @property
    def reference_names(self) -> Sequence[NormalizedName]:
        return self._reference_names

    @property
    def blocking_index(self) -> BlockingIndex:
        if self._index is None:
            raise SanctionsMatchingError("watchlist has not been indexed")
        return self._index

    def reference_owner(self, position: int) -> int:
        return self._reference_owner[position]


    # -------------------------------------------------------------- cascade
    def _shortlist(self, subject_name: NormalizedName, positions: Sequence[int]) -> list[int]:
        """Cheap first pass over the blocked candidates.

        Blocking prunes the pair space by ~99%, but the survivors still number
        in the hundreds per subject and the full multi-signal comparison is
        expensive in Python. Jaro-Winkler on the order-invariant sorted key is
        computed for the whole candidate list in one vectorised call, and only
        the plausible tail is scored properly.

        This is a recall/latency trade like blocking itself, so it is measured
        the same way: the cutoff sits below the weakest planted true match, and
        the cascade's own recall is reported alongside the blocking figure.
        """
        cutoff = self.config.prefilter_jaro_winkler
        limit = self.config.prefilter_max_candidates
        if cutoff <= 0.0 or not positions:
            return list(positions)

        ordered = list(positions)
        subset = [self._reference_sorted_keys[p] for p in ordered]
        scores = cdist(
            [subject_name.sorted_key],
            subset,
            scorer=JaroWinkler.similarity,
            processor=None,
            workers=1,
        )[0]
        surviving = np.flatnonzero(scores >= cutoff)
        if surviving.size > limit:
            # Keep the strongest `limit` candidates. A subject that blocks into
            # hundreds of near-identical names is a common-name case; scoring
            # the top slice is what a production screener does, and the discard
            # is counted rather than hidden.
            surviving = surviving[np.argsort(-scores[surviving], kind="stable")[:limit]]
        self._prefiltered_pairs += len(ordered) - int(surviving.size)
        return [ordered[i] for i in surviving]

    @property
    def prefiltered_pairs(self) -> int:
        return self._prefiltered_pairs

    # -------------------------------------------------------------- scoring
    def _corroborate(
        self, subject: ScreeningSubject, entry: WatchlistEntry, name: NormalizedName
    ) -> tuple[float, list[tuple[str, float]]]:
        corr = self.config.corroboration
        applied: list[tuple[str, float]] = []
        delta = 0.0

        if subject.dob and entry.dob:
            if subject.dob == entry.dob:
                delta += corr["dob_exact_boost"]
                applied.append(("dob_exact", corr["dob_exact_boost"]))
            elif _dob_year(subject.dob) == _dob_year(entry.dob):
                # OFAC publishes year-only DOBs for a large share of entries;
                # a year agreement is real but weak evidence, roughly a 1-in-70
                # coincidence across a plausible adult age range.
                delta += corr["dob_year_only_boost"]
                applied.append(("dob_year_only", corr["dob_year_only_boost"]))
            else:
                delta += corr["dob_mismatch_penalty"]
                applied.append(("dob_mismatch", corr["dob_mismatch_penalty"]))

        if subject.nationality and entry.nationality:
            if subject.nationality == entry.nationality:
                delta += corr["nationality_match_boost"]
                applied.append(("nationality_match", corr["nationality_match_boost"]))
            else:
                # A softer penalty than DOB: dual nationality and stale KYC data
                # make nationality disagreement common among genuine hits.
                delta += corr["nationality_mismatch_penalty"]
                applied.append(("nationality_mismatch", corr["nationality_mismatch_penalty"]))

        if subject.national_id and entry.national_id and subject.national_id == entry.national_id:
            delta += corr["id_exact_boost"]
            applied.append(("id_exact", corr["id_exact_boost"]))

        if name.token_count <= 1:
            delta += self.config.single_token_penalty
            applied.append(("single_token", self.config.single_token_penalty))

        return delta, applied

    def _decide(self, score: float) -> MatchDecision:
        if score >= self.config.auto_escalate:
            return MatchDecision.ESCALATE
        if score >= self.config.review:
            return MatchDecision.REVIEW
        return MatchDecision.DISCARD

    def score_pair(
        self,
        subject: ScreeningSubject,
        subject_name: NormalizedName,
        entry: WatchlistEntry,
        reference_name: NormalizedName,
        matched_on: str,
    ) -> MatchResult:
        signals = compare_names(
            subject_name, reference_name, self.config.jaro_winkler_prefix_weight
        )
        weights = self.config.weights
        name_score = (
            weights.jaro_winkler * signals.jaro_winkler
            + weights.token_set * signals.token_set
            + weights.phonetic * signals.phonetic
            + weights.initials * signals.initials
        )
        delta, applied = self._corroborate(subject, entry, subject_name)
        final = min(1.0, max(0.0, name_score + delta))
        if name_score >= self.config.strong_name_floor:
            # Corroboration de-prioritises; it does not auto-clear. Letting a
            # DOB disagreement drop an exact name match below the review line
            # would be the system deciding, unsupervised, that a designated
            # party is a different person.
            floored = max(final, self.config.strong_name_floor_score)
            if floored > final:
                applied.append(("strong_name_floor", floored - final))
                final = floored
        return MatchResult(
            subject_id=subject.subject_id,
            subject_name=subject.name,
            watchlist_uid=entry.uid,
            watchlist_name=entry.name,
            matched_on=matched_on,
            program=entry.program,
            name_score=name_score,
            corroboration_delta=delta,
            final_score=final,
            decision=self._decide(final),
            signals=signals,
            corroboration=tuple(applied),
            rule_version=self.config.rule_version,
        )

    # -------------------------------------------------------------- screening
    def screen(
        self, subject: ScreeningSubject, top_n: int = 5, min_score: float | None = None
    ) -> list[MatchResult]:
        """Screen one subject and return its surviving candidates, best first."""
        if self._index is None:
            raise SanctionsMatchingError("screen() called before index_watchlist()")
        threshold = self.config.review if min_score is None else min_score
        subject_name = normalize_name(subject.name, self.config)
        if subject_name.is_empty():
            return []

        best_per_entity: dict[int, MatchResult] = {}
        candidates = self._shortlist(subject_name, sorted(self._index.candidates(subject_name)))
        for position in candidates:
            owner = self._reference_owner[position]
            entry = self._entries[owner]
            result = self.score_pair(
                subject,
                subject_name,
                entry,
                self._reference_names[position],
                self._reference_source[position],
            )
            if result.final_score < threshold:
                continue
            # One alert per sanctioned entity, not one per alias: an entity with
            # six a.k.a.s must not generate six queue items for the same hit.
            incumbent = best_per_entity.get(owner)
            if incumbent is None or result.final_score > incumbent.final_score:
                best_per_entity[owner] = result

        ranked = sorted(
            best_per_entity.values(),
            key=lambda r: (-r.final_score, r.watchlist_uid),
        )
        return ranked[:top_n]

    def screen_many(
        self,
        subjects: Iterable[ScreeningSubject],
        top_n: int = 5,
        min_score: float | None = None,
    ) -> ScreeningRun:
        if self._index is None:
            raise SanctionsMatchingError("screen_many() called before index_watchlist()")
        start = time.perf_counter()
        subject_list = list(subjects)
        normalized = [normalize_name(s.name, self.config) for s in subject_list]

        matches: list[MatchResult] = []
        scored_pairs = 0
        candidates_per_subject: list[int] = []
        threshold = self.config.review if min_score is None else min_score
        for subject, subject_name in zip(subject_list, normalized):
            if subject_name.is_empty():
                candidates_per_subject.append(0)
                continue
            best_per_entity: dict[int, MatchResult] = {}
            blocked = sorted(self._index.candidates(subject_name))
            candidates_per_subject.append(len(blocked))
            for position in self._shortlist(subject_name, blocked):
                scored_pairs += 1
                owner = self._reference_owner[position]
                result = self.score_pair(
                    subject,
                    subject_name,
                    self._entries[owner],
                    self._reference_names[position],
                    self._reference_source[position],
                )
                if result.final_score < threshold:
                    continue
                incumbent = best_per_entity.get(owner)
                if incumbent is None or result.final_score > incumbent.final_score:
                    best_per_entity[owner] = result
            ranked = sorted(
                best_per_entity.values(), key=lambda r: (-r.final_score, r.watchlist_uid)
            )
            matches.extend(ranked[:top_n])

        elapsed = time.perf_counter() - start
        blocking_stats = self._index.stats_from_counts(candidates_per_subject, elapsed)
        logger.info(
            "screening batch complete",
            extra={
                "subjects": len(subject_list),
                "scored_pairs": scored_pairs,
                "prefiltered_pairs": self._prefiltered_pairs,
                "matches": len(matches),
                "reduction_pct": round(blocking_stats.reduction_ratio * 100, 4),
                "elapsed_seconds": round(elapsed, 2),
            },
        )
        return ScreeningRun(
            matches=tuple(matches),
            blocking=blocking_stats,
            subjects_screened=len(subject_list),
            scored_pairs=scored_pairs,
            elapsed_seconds=elapsed,
        )
