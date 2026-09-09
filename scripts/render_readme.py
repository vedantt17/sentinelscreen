"""Regenerate README.md from the artefacts of the last pipeline run."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging_setup import configure_logging, get_logger
from src.pipeline.report import render_readme

logger = get_logger("sentinelscreen.readme")

if __name__ == "__main__":
    configure_logging(logging.INFO)
    path = render_readme()
    logger.info("README written", extra={"path": str(path)})
