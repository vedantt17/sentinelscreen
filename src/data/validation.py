"""Schema validation for generated and ingested frames.

Validation runs before anything is written or loaded. A malformed frame that
reaches DuckDB becomes a malformed alert population, and by then the failure
looks like a modelling problem rather than a data problem.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from src.data.schemas import ColumnSpec, TableSchema
from src.exceptions import SchemaValidationError
from src.logging_setup import get_logger

logger = get_logger(__name__)


def _dtype_ok(series: pd.Series, spec: ColumnSpec) -> bool:
    declared = spec.pandas_dtype
    if declared == "string":
        return bool(pdt.is_object_dtype(series) or pdt.is_string_dtype(series))
    if declared == "bool":
        return bool(pdt.is_bool_dtype(series))
    if declared == "float64":
        return bool(pdt.is_float_dtype(series))
    if declared.startswith("datetime64"):
        return bool(pdt.is_datetime64_any_dtype(series))
    if declared == "int64":
        return bool(pdt.is_integer_dtype(series))
    return str(series.dtype) == declared


def coerce_frame(frame: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Cast to the declared dtypes and column order.

    Column order matters here: parquet round-trips preserve it, and the
    determinism check hashes bytes, so an unstable column order would report a
    reproducibility failure that has nothing to do with the pipeline.
    """
    missing = [c for c in schema.column_names if c not in frame.columns]
    if missing:
        raise SchemaValidationError(schema.name, [f"missing column: {c}" for c in missing])

    out = frame.loc[:, list(schema.column_names)].copy()
    for spec in schema.columns:
        series = out[spec.name]
        if spec.pandas_dtype == "string":
            coerced = series.astype(object)
            out[spec.name] = coerced.where(coerced.notna(), None)
        elif spec.pandas_dtype == "bool":
            out[spec.name] = series.astype(bool)
        elif spec.pandas_dtype == "float64":
            out[spec.name] = series.astype("float64")
        elif spec.pandas_dtype.startswith("datetime64"):
            out[spec.name] = pd.to_datetime(series).astype("datetime64[ns]")
        elif spec.pandas_dtype == "int64":
            out[spec.name] = series.astype("int64")
    return out.reset_index(drop=True)


def validate_frame(frame: pd.DataFrame, schema: TableSchema, sample_domain: bool = True) -> None:
    violations: list[str] = []

    declared = set(schema.column_names)
    present = set(frame.columns)
    for column in sorted(declared - present):
        violations.append(f"missing column: {column}")
    for column in sorted(present - declared):
        violations.append(f"undeclared column: {column}")
    if violations:
        raise SchemaValidationError(schema.name, violations)

    for spec in schema.columns:
        series = frame[spec.name]
        if not _dtype_ok(series, spec):
            violations.append(
                f"{spec.name}: expected {spec.pandas_dtype}, found {series.dtype}"
            )
        null_count = int(series.isna().sum())
        if not spec.nullable and null_count:
            violations.append(f"{spec.name}: {null_count} nulls in a NOT NULL column")
        if spec.allowed is not None and sample_domain:
            observed = set(series.dropna().unique().tolist())
            unexpected = observed - set(spec.allowed)
            if unexpected:
                shown = sorted(str(v) for v in unexpected)[:5]
                violations.append(f"{spec.name}: values outside domain {shown}")
        if spec.minimum is not None and len(series.dropna()):
            below = int((series.dropna() < spec.minimum).sum())
            if below:
                violations.append(f"{spec.name}: {below} values below minimum {spec.minimum}")
        if spec.maximum is not None and len(series.dropna()):
            above = int((series.dropna() > spec.maximum).sum())
            if above:
                violations.append(f"{spec.name}: {above} values above maximum {spec.maximum}")

    if schema.primary_key:
        key_columns = list(schema.primary_key)
        duplicates = int(frame.duplicated(subset=key_columns).sum())
        if duplicates:
            violations.append(f"primary key {key_columns}: {duplicates} duplicate rows")

    if violations:
        raise SchemaValidationError(schema.name, violations)
    logger.info(
        "schema validated", extra={"table": schema.name, "rows": len(frame),
                                   "columns": len(schema.columns)}
    )
