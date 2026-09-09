"""Candidate blocking.

Screening 50,000 customers against a 5,000-name watchlist is 250,000,000
pairs. At roughly 3 microseconds per full multi-signal comparison that is over
twelve minutes of CPU for a single batch run, and it scales quadratically with
portfolio growth, so an exhaustive comparison is not a viable design.

Blocking replaces it with an indexed lookup: reference names are bucketed under
keys that any true variant of the name would also produce, and a query is only
compared against names sharing at least one bucket. The trade is explicit —
blocking can only lose recall, never gain it — so `BlockingIndex` measures its
own recall against planted ground truth and the reduction ratio is written to
`outputs/blocking_benchmark.json` rather than asserted in prose.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from src.config import ScreeningConfig
from src.exceptions import BlockingError
from src.logging_setup import get_logger
from src.sanctions.normalize import NormalizedName
from src.sanctions.phonetics import phonetic_keys

logger = get_logger(__name__)

# Strategies are selected in config, not in code, because blocking recall vs.
# candidate volume is a tuning decision that model risk management signs off on.
_STRATEGY_PREFIX: Mapping[str, str] = {
    "phonetic_surname": "PS",
    "sorted_phonetic_pair": "SP",
    "initial_plus_sorted_bigram": "IB",
    "phonetic_any_token": "PA",
}


@dataclass(frozen=True)
class BlockingStats:
    n_reference: int
    n_query: int
    full_pair_space: int
    candidate_pairs: int
    distinct_candidate_pairs: int
    reduction_ratio: float
    blocks_built: int
    oversized_blocks_refined: int
    largest_block: int
    mean_candidates_per_query: float
    p95_candidates_per_query: float
    max_candidates_per_query: int
    build_seconds: float
    probe_seconds: float
    strategy_key_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_reference": self.n_reference,
            "n_query": self.n_query,
            "full_pair_space": self.full_pair_space,
            "candidate_pairs": self.candidate_pairs,
            "distinct_candidate_pairs": self.distinct_candidate_pairs,
            "reduction_ratio": round(self.reduction_ratio, 8),
            "reduction_pct": round(self.reduction_ratio * 100.0, 5),
            "blocks_built": self.blocks_built,
            "oversized_blocks_refined": self.oversized_blocks_refined,
            "largest_block": self.largest_block,
            "mean_candidates_per_query": round(self.mean_candidates_per_query, 4),
            "p95_candidates_per_query": round(self.p95_candidates_per_query, 4),
            "max_candidates_per_query": self.max_candidates_per_query,
            "strategy_key_counts": dict(self.strategy_key_counts),
        }

    def timings(self) -> dict[str, float]:
        """Wall-clock measurements, kept out of `as_dict`.

        Durations are the one thing about a run that legitimately varies between
        two identical executions, so they live in a run-scoped file that the
        reproducibility diff excludes. Hashing them would make `make reproduce`
        fail for the only reason that does not matter.
        """
        return {
            "build_seconds": round(self.build_seconds, 4),
            "probe_seconds": round(self.probe_seconds, 4),
        }


@dataclass
class BlockingIndex:
    """Inverted index from blocking key to reference row positions."""

    config: ScreeningConfig
    _index: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    _refined_keys: set[str] = field(default_factory=set)
    _n_reference: int = 0
    _build_seconds: float = 0.0
    _strategy_key_counts: dict[str, int] = field(default_factory=dict)

    # ---------------------------------------------------------------- keys --
    def _base_keys(self, name: NormalizedName) -> set[str]:
        if name.is_empty():
            return set()
        keys: set[str] = set()
        tokens = [t for t in name.tokens if len(t) >= self.config.min_token_length]
        if not tokens:
            tokens = list(name.tokens)

        for strategy in self.config.blocking_strategies:
            prefix = _STRATEGY_PREFIX.get(strategy)
            if prefix is None:
                raise BlockingError(f"unknown blocking strategy {strategy!r}")

            if strategy == "phonetic_surname":
                # Both ends of the name, because surname position is not knowable
                # without knowing the naming culture: Han and most OFAC Arabic
                # entries are surname-first, Western KYC records are surname-last.
                for token in {tokens[0], tokens[-1]}:
                    for key in phonetic_keys(token).keys():
                        keys.add(f"{prefix}:{key}")

            elif strategy == "sorted_phonetic_pair":
                # The workhorse: a compound of the two most informative tokens,
                # sorted so it is invariant to name order. Two phonetic keys
                # together are specific enough to keep blocks small, which is
                # what actually delivers the pruning ratio.
                head = phonetic_keys(tokens[0]).keys()
                tail = phonetic_keys(tokens[-1]).keys()
                for first in head or ("",):
                    for second in tail or ("",):
                        pair = "|".join(sorted((first, second)))
                        keys.add(f"{prefix}:{pair}")

            elif strategy == "initial_plus_sorted_bigram":
                # Order-invariant and phonetics-independent, so it still fires
                # when transliteration was lossy enough to change the Metaphone
                # key but not the leading letters.
                initials = sorted({t[0] for t in tokens if t})
                if initials:
                    keys.add(f"{prefix}:{''.join(initials[:2])}")

            elif strategy == "phonetic_any_token":
                # The recall backstop: catches matches where the token that
                # agrees is neither first nor last (middle name, kunya).
                for token in tokens:
                    for key in phonetic_keys(token).keys():
                        keys.add(f"{prefix}:{key}")
        return keys

    @staticmethod
    def _refinement(name: NormalizedName) -> str:
        """Order-invariant discriminator used to split oversized blocks.

        The first letter of the alphabetically first token: stable under name
        reordering, and unchanged by the vowel substitutions that dominate
        transliteration noise (Mohammed/Muhammad both refine to 'm').
        """
        if not name.tokens:
            return "_"
        return sorted(name.tokens)[0][0]

    def keys_for(self, name: NormalizedName) -> set[str]:
        """Keys to probe with, honouring any refinement applied at build time."""
        base = self._base_keys(name)
        if not self._refined_keys:
            return base
        refinement = self._refinement(name)
        return {f"{k}|{refinement}" if k in self._refined_keys else k for k in base}

    # --------------------------------------------------------------- build --
    def build(self, references: Sequence[NormalizedName]) -> None:
        start = time.perf_counter()
        index: dict[str, list[int]] = defaultdict(list)
        for position, name in enumerate(references):
            for key in self._base_keys(name):
                index[key].append(position)

        # Second pass: a block bigger than max_block_size contributes almost no
        # pruning (probing it costs nearly as much as a scan) while dominating
        # runtime. Rather than drop it — which would silently zero out recall
        # for the most common surnames — split it on the refinement character.
        oversized = [key for key, rows in index.items() if len(rows) > self.config.max_block_size]
        for key in oversized:
            rows = index.pop(key)
            for position in rows:
                index[f"{key}|{self._refinement(references[position])}"].append(position)
            self._refined_keys.add(key)

        counts: dict[str, int] = defaultdict(int)
        for key in index:
            counts[key.split(":", 1)[0]] += 1

        self._index = index
        self._n_reference = len(references)
        self._strategy_key_counts = dict(counts)
        self._build_seconds = time.perf_counter() - start
        logger.info(
            "blocking index built",
            extra={
                "reference_names": self._n_reference,
                "blocks": len(index),
                "oversized_refined": len(oversized),
                "build_seconds": round(self._build_seconds, 3),
            },
        )

    # --------------------------------------------------------------- probe --
    def candidates(self, name: NormalizedName) -> set[int]:
        if not self._index:
            raise BlockingError("candidates() called before build()")
        found: set[int] = set()
        for key in self.keys_for(name):
            rows = self._index.get(key)
            if rows:
                found.update(rows)
        return found

    def benchmark(self, queries: Sequence[NormalizedName]) -> BlockingStats:
        """Probe every query and summarise the pruning actually achieved."""
        if not self._index:
            raise BlockingError("benchmark() called before build()")
        start = time.perf_counter()
        per_query = [len(self.candidates(name)) for name in queries]
        return self.stats_from_counts(per_query, time.perf_counter() - start)

    def stats_from_counts(self, per_query: Sequence[int], probe_seconds: float) -> BlockingStats:
        """Build the stats record from counts already collected during scoring.

        Screening probes each subject exactly once; recomputing candidates to
        produce a benchmark would double the cost of the most expensive stage
        and report a probe time that no production run would ever incur.
        """
        full_space = len(per_query) * self._n_reference
        total = sum(per_query)
        ordered = sorted(per_query)
        p95 = float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]) if ordered else 0.0
        largest = max((len(v) for v in self._index.values()), default=0)
        return BlockingStats(
            n_reference=self._n_reference,
            n_query=len(per_query),
            full_pair_space=full_space,
            candidate_pairs=total,
            distinct_candidate_pairs=total,
            reduction_ratio=(1.0 - total / full_space) if full_space else 0.0,
            blocks_built=len(self._index),
            oversized_blocks_refined=len(self._refined_keys),
            largest_block=largest,
            mean_candidates_per_query=(total / len(per_query)) if per_query else 0.0,
            p95_candidates_per_query=p95,
            max_candidates_per_query=max(per_query, default=0),
            build_seconds=self._build_seconds,
            probe_seconds=probe_seconds,
            strategy_key_counts=self._strategy_key_counts,
        )

    def recall_against(
        self,
        queries: Sequence[NormalizedName],
        truth: Iterable[tuple[int, int]],
    ) -> tuple[float, int, int]:
        """Share of known (query_position, reference_position) pairs surviving blocking.

        This is the number that decides whether the pruning above is honest.
        A reduction ratio without a matching recall figure is meaningless: a
        blocker that returns nothing achieves 100% reduction.
        """
        pairs = list(truth)
        if not pairs:
            return 1.0, 0, 0
        cache: dict[int, set[int]] = {}
        retained = 0
        for query_position, reference_position in pairs:
            if query_position not in cache:
                cache[query_position] = self.candidates(queries[query_position])
            if reference_position in cache[query_position]:
                retained += 1
        return retained / len(pairs), retained, len(pairs)
