"""End-to-end determinism.

The repository's central claim is that the seed determines every byte. These
tests check it at three levels — the generator, the artefact writers, and the
whole pipeline — because a failure at each level looks different and needs a
different fix.

The full-pipeline test is marked `slow`; CI runs `make reproduce`, which is the
same check at production volume.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from src.config import AppConfig, GenerationConfig
from src.data.generator import generate_dataset
from src.data.loaders import frame_fingerprint, write_parquet
from src.pipeline.artifacts import build_manifest, diff_manifests
from src.pipeline.run_pipeline import run_pipeline


@pytest.mark.determinism
def test_generator_is_byte_identical_across_runs(
    small_generation_config: GenerationConfig,
) -> None:
    first = generate_dataset(small_generation_config)
    second = generate_dataset(small_generation_config)
    for table in ("watchlist", "customers", "transactions", "labels"):
        pd.testing.assert_frame_equal(getattr(first, table), getattr(second, table))


@pytest.mark.determinism
def test_fingerprints_match_across_runs(small_generation_config: GenerationConfig) -> None:
    first = generate_dataset(small_generation_config)
    second = generate_dataset(small_generation_config)
    for table in ("watchlist", "customers", "transactions", "labels"):
        assert frame_fingerprint(getattr(first, table)) == frame_fingerprint(
            getattr(second, table)
        )


@pytest.mark.determinism
def test_parquet_bytes_match_across_runs(
    small_generation_config: GenerationConfig, tmp_path: Path
) -> None:
    """Compression codec and column order are pinned, so the files match exactly."""
    manifests = []
    for label in ("a", "b"):
        dataset = generate_dataset(small_generation_config)
        directory = tmp_path / label
        write_parquet(
            {
                "watchlist": dataset.watchlist,
                "customers": dataset.customers,
                "transactions": dataset.transactions,
                "labels": dataset.labels,
            },
            directory,
        )
        manifests.append(build_manifest(sorted(directory.glob("*.parquet")), directory))
    assert diff_manifests(manifests[0], manifests[1]) == []


@pytest.mark.determinism
def test_changing_the_seed_changes_the_data(
    small_generation_config: GenerationConfig,
) -> None:
    """Guards against the opposite failure: a generator that ignores its seed."""
    baseline = generate_dataset(small_generation_config)
    other = generate_dataset(dataclasses.replace(small_generation_config, seed=987_654_321))
    assert frame_fingerprint(baseline.transactions) != frame_fingerprint(other.transactions)
    assert frame_fingerprint(baseline.customers) != frame_fingerprint(other.customers)


@pytest.mark.slow
@pytest.mark.determinism
def test_full_pipeline_is_reproducible(
    app_config: AppConfig, small_generation_config: GenerationConfig, tmp_path: Path
) -> None:
    """Two independent runs, every artefact hashed and compared.

    Scaled-down volumes so the suite stays usable; `make reproduce` runs the
    identical check at full volume and CI enforces it on every push.
    """
    scaled = dataclasses.replace(app_config, generation=small_generation_config)
    manifests = []
    for label in ("a", "b"):
        output_dir = tmp_path / label / "outputs"
        data_dir = tmp_path / label / "data"
        result = run_pipeline(scaled, output_dir=output_dir, data_dir=data_dir)
        manifests.append(dict(result.manifest))

    problems = diff_manifests(manifests[0], manifests[1])
    assert not problems, "artefacts differ between two seeded runs:\n" + "\n".join(problems)
    assert manifests[0]["artefact_count"] > 15


@pytest.mark.slow
def test_pipeline_metrics_are_stable(
    app_config: AppConfig, small_generation_config: GenerationConfig, tmp_path: Path
) -> None:
    scaled = dataclasses.replace(app_config, generation=small_generation_config)
    first = run_pipeline(scaled, output_dir=tmp_path / "a", data_dir=tmp_path / "da")
    second = run_pipeline(scaled, output_dir=tmp_path / "b", data_dir=tmp_path / "db")
    # run_id is derived from the seed and rule versions, never from a clock.
    assert first.metrics["run_id"] == second.metrics["run_id"]
    assert first.metrics["screening"] == second.metrics["screening"]
    assert first.metrics["models"]["models"] == second.metrics["models"]["models"]
