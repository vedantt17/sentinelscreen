"""DuckDB persistence layer.

DuckDB is used rather than SQLite or pandas-only because the monitoring
scenarios are genuinely analytical — window functions over a customer's
transaction history, self-joins for pass-through detection — and expressing
those in SQL keeps them reviewable by a compliance analyst who does not read
Python. The same reason drives keeping the scenario SQL in `.sql` files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import duckdb
import pandas as pd

from src.data.schemas import ALL_SCHEMAS, SCHEMA_BY_NAME, TableSchema
from src.data.validation import coerce_frame, validate_frame
from src.exceptions import SchemaValidationError
from src.logging_setup import get_logger
from src.paths import DATA_DIR

logger = get_logger(__name__)

PARQUET_FILES: Mapping[str, str] = {
    "watchlist": "watchlist.parquet",
    "customers": "customers.parquet",
    "transactions": "transactions.parquet",
    "labels": "labels.parquet",
}


def write_parquet(frames: Mapping[str, pd.DataFrame], directory: Path | None = None) -> dict[str, Path]:
    """Validate then persist. Validation before write, never after."""
    target = directory or DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, filename in PARQUET_FILES.items():
        if name not in frames:
            raise SchemaValidationError(name, ["frame not supplied to write_parquet"])
        schema = SCHEMA_BY_NAME[name]
        frame = coerce_frame(frames[name], schema)
        validate_frame(frame, schema)
        path = target / filename
        # Compression is fixed rather than defaulted: parquet codec choice
        # changes the bytes, and `make reproduce` hashes bytes.
        frame.to_parquet(path, index=False, compression="snappy")
        written[name] = path
        logger.info("parquet written", extra={"table": name, "rows": len(frame), "path": str(path)})
    return written


def read_parquet(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    source = directory or DATA_DIR
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in PARQUET_FILES.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"expected generated dataset at {path}; run the pipeline first")
        frames[name] = pd.read_parquet(path)
    return frames


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(database)
    # Single-threaded execution is a determinism requirement, not a performance
    # choice: DuckDB's parallel aggregation can reorder floating-point sums,
    # which is enough to change a threshold-sweep boundary between runs.
    connection.execute("SET threads TO 1;")
    connection.execute("PRAGMA enable_progress_bar=false;")
    return connection


def load_frames(
    connection: duckdb.DuckDBPyConnection,
    frames: Mapping[str, pd.DataFrame],
    validate: bool = True,
) -> None:
    for schema in ALL_SCHEMAS:
        if schema.name not in frames:
            continue
        frame = coerce_frame(frames[schema.name], schema)
        if validate:
            validate_frame(frame, schema)
        connection.execute(schema.create_table_sql())
        connection.register(f"_stage_{schema.name}", frame)
        columns = ", ".join(schema.column_names)
        connection.execute(
            f"INSERT INTO {schema.name} SELECT {columns} FROM _stage_{schema.name};"
        )
        connection.unregister(f"_stage_{schema.name}")
        logger.info("table loaded", extra={"table": schema.name, "rows": len(frame)})


def create_run_window(
    connection: duckdb.DuckDBPyConnection, window_start: str, window_end: str
) -> None:
    """Observation window as a table so DQ SQL can reference it declaratively."""
    connection.execute("CREATE OR REPLACE TABLE run_window (window_start TIMESTAMP, window_end TIMESTAMP);")
    connection.execute(
        "INSERT INTO run_window VALUES (?::TIMESTAMP, ?::TIMESTAMP);", [window_start, window_end]
    )


def table_schema(name: str) -> TableSchema:
    return SCHEMA_BY_NAME[name]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Content hash independent of row order and index.

    Used by the determinism test so that an incidental reordering is reported
    as what it is, rather than being hidden by a sort or mistaken for a value
    change.
    """
    ordered = frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    ).hexdigest()
