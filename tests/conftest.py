"""Shared fixtures.

The full feed takes ~25 seconds to generate, which is too slow to pay for in
every unit test. `small_dataset` shrinks the volumes but keeps every structural
property that matters — the same seed plumbing, the same typology injection,
the same planted screening ground truth — so a test that passes here is testing
the real code path, not a mock of it.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

import duckdb
import pytest

from src.config import AppConfig, GenerationConfig, ScreeningConfig, load_app_config
from src.data.generator import GeneratedDataset, generate_dataset
from src.data.loaders import connect, create_run_window, load_frames
from src.monitoring.rules import RuleEngine


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return load_app_config()


@pytest.fixture(scope="session")
def screening_config(app_config: AppConfig) -> ScreeningConfig:
    return app_config.screening


@pytest.fixture(scope="session")
def small_generation_config(app_config: AppConfig) -> GenerationConfig:
    base = app_config.generation
    return dataclasses.replace(
        base,
        n_watchlist=400,
        n_customers=2_000,
        n_transactions=30_000,
        screening_ground_truth={"true_match_customers": 60, "near_miss_customers": 120},
    )


@pytest.fixture(scope="session")
def small_dataset(small_generation_config: GenerationConfig) -> GeneratedDataset:
    return generate_dataset(small_generation_config)


@pytest.fixture()
def loaded_db(
    small_dataset: GeneratedDataset,
    small_generation_config: GenerationConfig,
    app_config: AppConfig,
) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = connect()
    load_frames(
        connection,
        {
            "watchlist": small_dataset.watchlist,
            "customers": small_dataset.customers,
            "transactions": small_dataset.transactions,
            "labels": small_dataset.labels,
        },
    )
    create_run_window(
        connection,
        small_generation_config.start_date.isoformat(),
        (
            small_generation_config.start_date.replace(
                year=small_generation_config.start_date.year + 1
            )
        ).isoformat(),
    )
    # The scenario SQL and the feature query both join reference tables that are
    # populated from config rather than generated, so the fixture installs them.
    # Without this every feature-level test fails on a missing risk_countries.
    engine = RuleEngine(connection, app_config.monitoring)
    engine.install_risk_countries(app_config.generation.corridors)
    engine.install_params()
    try:
        yield connection
    finally:
        connection.close()
