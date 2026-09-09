"""Deterministic artefact writing.

`make reproduce` compares artefacts byte for byte across two seeded runs, so
every writer here pins the things that are otherwise platform- or
version-dependent: newline style, float formatting, key order, and the absence
of any embedded timestamp.

Wall-clock time is recorded exactly once, in `run_info.json`, which is
explicitly excluded from the reproducibility comparison. Putting it anywhere
else would make the determinism check fail for the one reason that does not
matter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.logging_setup import get_logger

logger = get_logger(__name__)

# Excluded from the reproducibility diff: these legitimately differ run to run.
# `run_timings.json` is the important one — durations are the only thing about a
# deterministic run that is genuinely allowed to vary, so every wall-clock
# measurement is funnelled there and nowhere else.
VOLATILE_ARTIFACTS: frozenset[str] = frozenset(
    {"run_info.json", "run_manifest.json", "run_timings.json"}
)

_FLOAT_FORMAT = "%.6f"


def _round_floats(value: Any, places: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: _round_floats(v, places) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, places) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_round_floats(dict(payload)), indent=2, sort_keys=True, default=str)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    logger.info("artefact written", extra={"artefact": path.name, "kind": "json"})
    return path


def write_csv(path: Path, frame: pd.DataFrame, sort_by: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame
    if sort_by:
        columns = [c for c in sort_by if c in out.columns]
        if columns:
            out = out.sort_values(columns, kind="mergesort").reset_index(drop=True)
    out.to_csv(
        path,
        index=False,
        float_format=_FLOAT_FORMAT,
        lineterminator="\n",
        encoding="utf-8",
    )
    logger.info(
        "artefact written",
        extra={"artefact": path.name, "kind": "csv", "rows": len(out)},
    )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    """Hash every artefact so two runs can be compared without re-reading them."""
    entries: dict[str, Any] = {}
    for path in sorted(paths, key=lambda p: str(p)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name in VOLATILE_ARTIFACTS:
            continue
        entries[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"artefact_count": len(entries), "artefacts": entries}


def diff_manifests(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Human-readable differences between two run manifests."""
    left_files = left.get("artefacts", {})
    right_files = right.get("artefacts", {})
    problems: list[str] = []
    for name in sorted(set(left_files) - set(right_files)):
        problems.append(f"only in first run: {name}")
    for name in sorted(set(right_files) - set(left_files)):
        problems.append(f"only in second run: {name}")
    for name in sorted(set(left_files) & set(right_files)):
        if left_files[name]["sha256"] != right_files[name]["sha256"]:
            problems.append(
                f"differs: {name} "
                f"({left_files[name]['sha256'][:12]}... vs {right_files[name]['sha256'][:12]}...)"
            )
    return problems
