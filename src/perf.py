"""
Performance Instrumentation

DEBUG-only timing helpers. Nothing here logs in production: every measurement
is emitted through logger.debug so normal INFO/WARNING output is untouched.
"""

import logging
import time

from typing import Optional


def log_duration(logger: logging.Logger, operation: str,
                 start: float) -> None:
    """Log how long an operation took (DEBUG level only)."""
    logger.debug(
        "%s completed in %.3f seconds", operation,
        time.perf_counter() - start)


class _TimedBlock:
    """Context manager that logs a DEBUG-level duration on exit."""

    __slots__ = ("logger", "operation", "start")

    def __init__(self, logger: logging.Logger, operation: str) -> None:
        self.logger = logger
        self.operation = operation
        self.start = time.perf_counter()

    def __enter__(self) -> "_TimedBlock":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        log_duration(self.logger, self.operation, self.start)
        return False


def timed(logger: logging.Logger, operation: str,
          condition: Optional[bool] = None) -> _TimedBlock:
    """Start a DEBUG-only timing block.

    When ``condition`` is provided and falsy, timing is skipped entirely so
    the block behaves as a plain no-op context manager.
    """
    if condition is not None and not condition:
        return _TimedBlock(logging.getLogger("__disabled__"), "")
    return _TimedBlock(logger, operation)
