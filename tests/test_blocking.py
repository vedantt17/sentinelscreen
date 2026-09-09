"""Candidate blocking: pruning, recall, and the oversized-block guard."""

from __future__ import annotations

import dataclasses

import pytest

from src.config import ScreeningConfig
from src.exceptions import BlockingError
from src.sanctions.blocking import BlockingIndex
from src.sanctions.normalize import normalize_name

REFERENCE_NAMES = [
    "Muhammad Al-Hassan", "Mohammed Hassan", "Sergei Ivanov", "Anna Ivanova",
    "Zhang Wei", "Chang Wai", "James Anderson", "Maria Garcia", "Hans Muller",
    "Yusuf Al-Baghdadi", "Vladimir Petrov", "Ekaterina Smirnova",
]


def _index(config: ScreeningConfig, names: list[str]) -> tuple[BlockingIndex, list]:
    normalized = [normalize_name(n, config) for n in names]
    index = BlockingIndex(config=config)
    index.build(normalized)
    return index, normalized


def test_probe_before_build_raises(screening_config: ScreeningConfig) -> None:
    with pytest.raises(BlockingError, match="before build"):
        BlockingIndex(config=screening_config).candidates(
            normalize_name("Anyone", screening_config)
        )


def test_unknown_strategy_raises(screening_config: ScreeningConfig) -> None:
    broken = dataclasses.replace(screening_config, blocking_strategies=("not_a_strategy",))
    with pytest.raises(BlockingError, match="unknown blocking strategy"):
        BlockingIndex(config=broken).build([normalize_name("Sergei Ivanov", broken)])


def test_variant_spelling_lands_in_the_same_block(screening_config: ScreeningConfig) -> None:
    index, _ = _index(screening_config, REFERENCE_NAMES)
    candidates = index.candidates(normalize_name("Mohamed Al Hassan", screening_config))
    assert 0 in candidates or 1 in candidates


def test_name_order_swap_survives_blocking(screening_config: ScreeningConfig) -> None:
    index, _ = _index(screening_config, REFERENCE_NAMES)
    assert 2 in index.candidates(normalize_name("Ivanov Sergei", screening_config))


def test_cyrillic_query_reaches_latin_reference(screening_config: ScreeningConfig) -> None:
    index, _ = _index(screening_config, REFERENCE_NAMES)
    assert 2 in index.candidates(normalize_name("Сергей Иванов", screening_config))


def test_unrelated_name_is_pruned(screening_config: ScreeningConfig) -> None:
    index, _ = _index(screening_config, REFERENCE_NAMES)
    candidates = index.candidates(normalize_name("Kwame Boateng", screening_config))
    assert len(candidates) < len(REFERENCE_NAMES)


def test_oversized_blocks_are_refined_not_dropped(screening_config: ScreeningConfig) -> None:
    """A common surname must stay screenable after the block is split."""
    tight = dataclasses.replace(screening_config, max_block_size=4)
    names = [f"{given} Ivanov" for given in
             ("Sergei", "Ivan", "Pavel", "Boris", "Anton", "Dmitri", "Oleg", "Roman")]
    index, _ = _index(tight, names)
    assert index._refined_keys, "expected at least one block to exceed the cap"
    # Every original name must still be retrievable by itself.
    for position, name in enumerate(names):
        assert position in index.candidates(normalize_name(name, tight))


def test_reduction_and_recall_are_reported_together(
    screening_config: ScreeningConfig, small_dataset
) -> None:
    """Pruning without a recall figure is meaningless.

    A blocker that returns nothing scores 100% reduction, so the two numbers
    are only interpretable as a pair — which is why both go into the artefact.
    """
    watchlist = small_dataset.watchlist
    customers = small_dataset.customers
    reference = [normalize_name(n, screening_config) for n in watchlist["primary_name"]]
    index = BlockingIndex(config=screening_config)
    index.build(reference)

    queries = [normalize_name(n, screening_config) for n in customers["full_name"]]
    stats = index.benchmark(queries)
    assert stats.full_pair_space == len(queries) * len(reference)
    assert stats.reduction_ratio > 0.90

    uid_by_position = {uid: i for i, uid in enumerate(watchlist["sdn_uid"])}
    truth: list[tuple[int, int]] = []
    for query_position, uid in enumerate(customers["watchlist_true_match_uid"]):
        if uid is not None and uid in uid_by_position:
            truth.append((query_position, uid_by_position[uid]))
    assert truth, "fixture should plant screening ground truth"

    recall, retained, total = index.recall_against(queries, truth)
    assert retained <= total
    # Primary names only (no aliases) in this index, so the bar is the blocking
    # stage doing its job on the harder half of the planted hits.
    assert recall > 0.80
