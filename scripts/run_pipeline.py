"""CLI entry point: `python scripts/run_pipeline.py [--output DIR] [--data DIR]`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config
from src.exceptions import SentinelScreenError
from src.logging_setup import configure_logging, get_logger
from src.paths import DATA_DIR, OUTPUT_DIR
from src.pipeline.run_pipeline import run_pipeline

logger = get_logger("sentinelscreen.cli")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SentinelScreen pipeline end to end.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(
        logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.output / "pipeline.log",
    )
    try:
        result = run_pipeline(load_app_config(), output_dir=args.output, data_dir=args.data)
    except SentinelScreenError:
        # Control failures (a blocking DQ check, a leakage tripwire) are the
        # pipeline working as designed. They must land in the log file, not only
        # on a stderr stream that a `> /dev/null` would swallow.
        logger.exception("pipeline aborted by a control")
        return 1
    logger.info(
        "run finished",
        extra={
            "elapsed_seconds": round(result.elapsed_seconds, 1),
            "artefacts": result.manifest["artefact_count"],
            "output_dir": str(result.output_dir),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
