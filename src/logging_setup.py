"""Structured logging.

There are no `print` calls anywhere in this codebase. Anything a reviewer or an
auditor might need to see later goes through `logging`, so that it carries a
timestamp, a module, and a level, and can be routed to a file without changing
call sites.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line; extra=... keys are promoted to top-level fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    json_console: bool = False,
) -> None:
    """Idempotent root-logger configuration.

    Console stays human-readable by default; the file sink is always JSON lines
    so that pipeline runs can be replayed by a log processor.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return

    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(
        JsonLineFormatter()
        if json_console
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")
    )
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonLineFormatter())
        root.addHandler(file_handler)

    # These libraries are chatty at INFO and say nothing a reviewer needs.
    for noisy in ("matplotlib", "PIL", "numexpr", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
