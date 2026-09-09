"""Determinism harness: run the pipeline twice from a cold start and diff every artefact.

Two *independent* runs into separate directories, each regenerating the dataset
from the seed. Re-hashing one run's outputs twice would prove nothing; the claim
under test is that the seed alone determines every byte.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config
from src.exceptions import DeterminismError
from src.logging_setup import configure_logging, get_logger
from src.paths import PROJECT_ROOT
from src.pipeline.artifacts import diff_manifests
from src.pipeline.run_pipeline import run_pipeline

logger = get_logger("sentinelscreen.reproduce")


def _run(label: str, root: Path) -> dict[str, object]:
    output_dir = root / f"_repro_{label}" / "outputs"
    data_dir = root / f"_repro_{label}" / "data"
    for directory in (output_dir, data_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(load_app_config(), output_dir=output_dir, data_dir=data_dir)
    logger.info(
        "run complete",
        extra={
            "label": label,
            "artefacts": result.manifest["artefact_count"],
            "elapsed_seconds": round(result.elapsed_seconds, 1),
        },
    )
    return dict(result.manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove the pipeline is deterministic.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--keep", action="store_true", help="keep both run directories")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    args.root.mkdir(parents=True, exist_ok=True)

    first = _run("a", args.root)
    second = _run("b", args.root)

    problems = diff_manifests(first, second)
    report = {
        "artefacts_compared": first["artefact_count"],
        "differences": problems,
        "deterministic": not problems,
    }
    (args.root / "reproducibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    if not args.keep:
        for label in ("a", "b"):
            shutil.rmtree(args.root / f"_repro_{label}", ignore_errors=True)

    if problems:
        for problem in problems:
            logger.error("artefact differs between runs", extra={"detail": problem})
        raise DeterminismError(
            f"{len(problems)} of {first['artefact_count']} artefacts differ across two seeded runs"
        )

    logger.info(
        "determinism verified",
        extra={"artefacts_compared": first["artefact_count"]},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
