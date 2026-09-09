"""Table contracts: pandas dtypes, nullability, domains, and DuckDB DDL.

The contract is declared once and used three times — to validate the generator
output, to create the DuckDB tables, and to document the ERD in the README —
so a column cannot drift between the generator and the warehouse.

Validation is column-level rather than row-level by design. Running a pydantic
model over 500,000 rows costs minutes and buys nothing that a dtype check plus a
domain check does not already give; pydantic is used at the API boundary, where
inputs are untrusted and volumes are one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    pandas_dtype: str
    duckdb_type: str
    nullable: bool = False
    allowed: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...]
    foreign_keys: Mapping[str, str] = field(default_factory=dict)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> ColumnSpec:
        for spec in self.columns:
            if spec.name == name:
                return spec
        raise KeyError(f"{self.name} has no column {name!r}")

    def create_table_sql(self) -> str:
        lines = [
            f"    {c.name} {c.duckdb_type}{'' if c.nullable else ' NOT NULL'}"
            for c in self.columns
        ]
        if self.primary_key:
            lines.append(f"    PRIMARY KEY ({', '.join(self.primary_key)})")
        body = ",\n".join(lines)
        return f"CREATE OR REPLACE TABLE {self.name} (\n{body}\n);"


WATCHLIST_SCHEMA: Final[TableSchema] = TableSchema(
    name="watchlist",
    primary_key=("sdn_uid",),
    columns=(
        ColumnSpec("sdn_uid", "string", "VARCHAR", description="OFAC-style unique identifier"),
        ColumnSpec("primary_name", "string", "VARCHAR", description="name as published"),
        ColumnSpec("name_script", "string", "VARCHAR",
                   allowed=("LATIN", "ARABIC", "CYRILLIC", "HAN")),
        ColumnSpec("entity_type", "string", "VARCHAR", allowed=("INDIVIDUAL", "ENTITY")),
        ColumnSpec("program", "string", "VARCHAR", description="sanctions program code"),
        ColumnSpec("aliases", "string", "VARCHAR",
                   description="pipe-delimited a.k.a. list; empty string when none"),
        ColumnSpec("dob", "string", "VARCHAR", nullable=True,
                   description="ISO date or year-only; frequently absent, as on the real SDN list"),
        ColumnSpec("nationality", "string", "VARCHAR", nullable=True),
        ColumnSpec("national_id", "string", "VARCHAR", nullable=True),
        ColumnSpec("address_country", "string", "VARCHAR", nullable=True),
        ColumnSpec("address_line", "string", "VARCHAR", nullable=True,
                   description="often partial: city only, or country only"),
        ColumnSpec("listed_date", "datetime64[ns]", "TIMESTAMP"),
        ColumnSpec("remarks", "string", "VARCHAR", nullable=True),
    ),
)

CUSTOMERS_SCHEMA: Final[TableSchema] = TableSchema(
    name="customers",
    primary_key=("customer_id",),
    columns=(
        ColumnSpec("customer_id", "string", "VARCHAR"),
        ColumnSpec("full_name", "string", "VARCHAR"),
        ColumnSpec("name_script", "string", "VARCHAR",
                   allowed=("LATIN", "ARABIC", "CYRILLIC", "HAN")),
        ColumnSpec("dob", "string", "VARCHAR", nullable=True),
        ColumnSpec("nationality", "string", "VARCHAR"),
        ColumnSpec("residence_country", "string", "VARCHAR"),
        ColumnSpec("national_id", "string", "VARCHAR", nullable=True),
        ColumnSpec("occupation", "string", "VARCHAR"),
        ColumnSpec("behaviour_segment", "string", "VARCHAR",
                   allowed=("DORMANT_RETAIL", "SALARIED", "SMALL_BUSINESS",
                            "CASH_INTENSIVE", "TRADE_CORRIDOR")),
        ColumnSpec("kyc_risk_rating", "string", "VARCHAR", allowed=("LOW", "MEDIUM", "HIGH")),
        ColumnSpec("account_open_date", "datetime64[ns]", "TIMESTAMP"),
        ColumnSpec("is_pep", "bool", "BOOLEAN"),
        ColumnSpec("watchlist_true_match_uid", "string", "VARCHAR", nullable=True,
                   description="screening ground truth; never exposed to the matcher"),
        ColumnSpec("is_screening_near_miss", "bool", "BOOLEAN",
                   description="planted confusable; a true negative that should be hard"),
    ),
)

TRANSACTIONS_SCHEMA: Final[TableSchema] = TableSchema(
    name="transactions",
    primary_key=("txn_id",),
    foreign_keys={"customer_id": "customers.customer_id"},
    columns=(
        ColumnSpec("txn_id", "string", "VARCHAR"),
        ColumnSpec("customer_id", "string", "VARCHAR"),
        ColumnSpec("txn_ts", "datetime64[ns]", "TIMESTAMP"),
        ColumnSpec("amount", "float64", "DOUBLE", minimum=0.01, maximum=50_000_000.0),
        ColumnSpec("currency", "string", "VARCHAR"),
        ColumnSpec("channel", "string", "VARCHAR", allowed=("WIRE", "ACH", "CARD", "CASH")),
        ColumnSpec("direction", "string", "VARCHAR", allowed=("CREDIT", "DEBIT")),
        ColumnSpec("counterparty_name", "string", "VARCHAR", nullable=True),
        ColumnSpec("counterparty_country", "string", "VARCHAR", nullable=True),
        ColumnSpec("counterparty_bank", "string", "VARCHAR", nullable=True),
        ColumnSpec("is_cross_border", "bool", "BOOLEAN"),
    ),
)

# Ground truth lives in its own table so that it is structurally impossible for
# a feature query to pick it up with SELECT *. Joining it in is a deliberate act.
LABELS_SCHEMA: Final[TableSchema] = TableSchema(
    name="labels",
    primary_key=("customer_id", "period"),
    foreign_keys={"customer_id": "customers.customer_id"},
    columns=(
        ColumnSpec("customer_id", "string", "VARCHAR"),
        ColumnSpec("period", "string", "VARCHAR", description="YYYY-MM of the scored cell"),
        ColumnSpec("is_true_anomaly", "bool", "BOOLEAN"),
        ColumnSpec("typology", "string", "VARCHAR", nullable=True),
        ColumnSpec("intensity", "string", "VARCHAR", nullable=True,
                   allowed=("LOUD", "ATTENUATED", "INVISIBLE"),
                   description="INVISIBLE = labelled positive with no rendered typology"),
        ColumnSpec("is_near_miss", "bool", "BOOLEAN"),
    ),
)

ALL_SCHEMAS: Final[tuple[TableSchema, ...]] = (
    WATCHLIST_SCHEMA,
    CUSTOMERS_SCHEMA,
    TRANSACTIONS_SCHEMA,
    LABELS_SCHEMA,
)

SCHEMA_BY_NAME: Final[Mapping[str, TableSchema]] = {s.name: s for s in ALL_SCHEMAS}


def erd_mermaid() -> str:
    """Mermaid ER diagram derived from the declared schemas, not hand-drawn."""
    lines = ["erDiagram"]
    for schema in ALL_SCHEMAS:
        lines.append(f"    {schema.name.upper()} {{")
        for column in schema.columns:
            marker = "PK" if column.name in schema.primary_key else ""
            lines.append(f"        {column.duckdb_type.lower()} {column.name} {marker}".rstrip())
        lines.append("    }")
    for schema in ALL_SCHEMAS:
        for foreign_key, target in schema.foreign_keys.items():
            parent = target.split(".")[0].upper()
            lines.append(f"    {parent} ||--o{{ {schema.name.upper()} : {foreign_key}")
    return "\n".join(lines)


def ddl_statements() -> Sequence[str]:
    return tuple(schema.create_table_sql() for schema in ALL_SCHEMAS)
