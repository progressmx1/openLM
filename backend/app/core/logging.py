"""Structured logging with sensitive-value redaction and file output."""
from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler

from .paths import LOGS_DIR, ensure_runtime_dirs

_configured = False

# Redact common secret-bearing patterns (API keys, bearer tokens, passwords).
_REDACT_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s\"',]+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s\"',]+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(password\s*[:=]\s*)([^\s\"',]+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(token\s*[:=]\s*)([A-Za-z0-9_\-\.]{12,})"), r"\1[REDACTED]"),
]


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        for pattern, replacement in _REDACT_PATTERNS:
            message = pattern.sub(replacement, message)
        return message


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)


def configure() -> None:
    global _configured
    if _configured:
        return
    ensure_runtime_dirs()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = RedactingFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    try:
        file_handler = RotatingFileHandler(
            LOGS_DIR / "backend.log", maxBytes=5_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Logging to disk is best-effort; never crash because of it.
        pass

    _configured = True
