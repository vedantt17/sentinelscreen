"""Domain exceptions.

Every failure mode a compliance reviewer would ask about gets its own type, so
that a stack trace answers "which control failed" without reading the code.
"""

from __future__ import annotations


class SentinelScreenError(Exception):
    """Root of the hierarchy; catch this to isolate our failures from library ones."""


class ConfigurationError(SentinelScreenError):
    """A config file is missing, malformed, or internally inconsistent."""


class SchemaValidationError(SentinelScreenError):
    """A dataframe does not conform to its declared contract."""

    def __init__(self, dataset: str, violations: list[str]) -> None:
        self.dataset = dataset
        self.violations = violations
        detail = "; ".join(violations[:8])
        suffix = f" (+{len(violations) - 8} more)" if len(violations) > 8 else ""
        super().__init__(f"schema validation failed for '{dataset}': {detail}{suffix}")


class DataQualityError(SentinelScreenError):
    """A BLOCKER-severity data-quality control failed."""

    def __init__(self, failed_checks: list[str]) -> None:
        self.failed_checks = failed_checks
        super().__init__(f"blocking data-quality failures: {', '.join(failed_checks)}")


class SanctionsMatchingError(SentinelScreenError):
    """The screening engine was asked to do something it cannot defend."""


class TransliterationError(SanctionsMatchingError):
    """A name could not be reduced to a Latin representation.

    Raised rather than silently returning the original string: feeding raw
    Cyrillic or Arabic into an English phonetic algorithm produces keys that
    look plausible and are meaningless, which is worse than failing loudly.
    """


class BlockingError(SanctionsMatchingError):
    """The candidate-generation stage could not build a usable block index."""


class RuleExecutionError(SentinelScreenError):
    """A monitoring scenario failed to execute or returned an unexpected shape."""

    def __init__(self, rule_id: str, reason: str) -> None:
        self.rule_id = rule_id
        super().__init__(f"scenario {rule_id} failed: {reason}")


class ModelTrainingError(SentinelScreenError):
    """Training aborted — typically the leakage tripwire firing."""


class DeterminismError(SentinelScreenError):
    """Two seeded runs produced different artefacts."""


class AuditIntegrityError(SentinelScreenError):
    """An audit record was mutated after being sealed."""
