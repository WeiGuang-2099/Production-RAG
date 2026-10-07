"""Structured logging with request correlation IDs."""
import logging
import sys
from contextvars import ContextVar
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Client libraries that log every outbound HTTP call at INFO (httpx2/httpcore2
# are what newer OpenAI SDKs resolve to; opensearch-py logs each request).
_NOISY_LOGGERS = ("httpx", "httpcore", "httpx2", "httpcore2", "opensearch")
_HANDLER_NAME = "app-json"


class _StdoutHandler(logging.StreamHandler):
    """Write to whatever sys.stdout is at emit time, so redirection is honored."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def setup_logging(log_level: str = "INFO") -> None:
    """Render structlog and stdlib ``logging`` records as one JSON line each.

    App modules log through ``logging.getLogger(__name__)``, so the root logger
    gets a handler whose ProcessorFormatter gives those records the same JSON
    shape plus the ``request_id`` the middleware binds. Calling this again
    replaces that handler instead of stacking a second one.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = _StdoutHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structured logger bound with the given name."""
    return structlog.get_logger(logger_name=name)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that injects X-Request-ID header and sets context var."""

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID", str(uuid4()))
        request_id_var.set(rid)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
