"""Filesystem layout resolved from the repository root, not the CWD."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
SQL_DIR: Path = PROJECT_ROOT / "src" / "monitoring" / "sql"


def ensure_dirs(*extra: Path) -> None:
    for directory in (DATA_DIR, OUTPUT_DIR, *extra):
        directory.mkdir(parents=True, exist_ok=True)
