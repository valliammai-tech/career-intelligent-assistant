"""
observability/logger.py — Structured JSON logging.

Every log line is a valid JSON object with consistent fields.
This means CloudWatch Insights / grep / jq can query logs programmatically.

Example output:
{"ts": "2024-02-21T10:00:00Z", "level": "INFO", "event": "chat_request",
 "session_id": "abc", "intent": "gap_analysis", "latency_ms": 3241, "tokens": 412}
"""
import json
import logging
import time
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via extra={...} in the log call
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                log_obj[key] = value

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Get a structured logger.
    Usage:
        logger = get_logger(__name__)
        logger.info("ingestion_complete", extra={"chunks": 12, "doc_type": "resume"})
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class Timer:
    """Context manager for measuring latency."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
