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


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
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
