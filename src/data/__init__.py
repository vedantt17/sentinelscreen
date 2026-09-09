"""Data plane: synthetic generation, typology injection, schemas, validation, DuckDB."""

from src.data.generator import GeneratedDataset, SyntheticDataGenerator, generate_dataset
from src.data.loaders import connect, load_frames, read_parquet, write_parquet
from src.data.schemas import ALL_SCHEMAS, SCHEMA_BY_NAME, TableSchema, erd_mermaid
from src.data.typologies import InjectionCell, PatternRenderer, plan_injections
from src.data.validation import coerce_frame, validate_frame

__all__ = [
    "ALL_SCHEMAS",
    "SCHEMA_BY_NAME",
    "GeneratedDataset",
    "InjectionCell",
    "PatternRenderer",
    "SyntheticDataGenerator",
    "TableSchema",
    "coerce_frame",
    "connect",
    "erd_mermaid",
    "generate_dataset",
    "load_frames",
    "plan_injections",
    "read_parquet",
    "validate_frame",
    "write_parquet",
]
