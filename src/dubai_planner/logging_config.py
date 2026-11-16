"""Central logging configuration.

Logging is for diagnostics only; agent observability goes through ``tracing.bus``
(TraceEvents), not log lines. A redaction filter masks anything that looks like an
API key so secrets cannot leak through log messages.
"""

from __future__ import annotations

import logging
import re

_SECRET_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|secret|token|authorization)\s*[=:]\s*)(['\"]?)[^\s'\",]+"
)
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_configured = False


class RedactSecretsFilter(logging.Filter):
    """Replace ``api_key=...``-style values in log messages with ``***``."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _SECRET_PATTERN.sub(r"\1\2***", message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once (idempotent)."""
    global _configured
    root = logging.getLogger()
    root.setLevel(level)
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(RedactSecretsFilter())
    root.addHandler(handler)
    # Quieten chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
