"""FastAPI app for fair-football-fee: `uvicorn src.api.main:app` (or `make api`).

`create_app` takes optional components so tests can pass fakes (fakeredis, an in-memory runtime);
by default it builds the real ones from `settings`.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from src.api.cache import RateLimiter, RedisCache
from src.api.predlog import PredictionLogWriter
from src.api.routes import router
from src.api.runtime import Runtime
from src.api.service import ApiError
from src.config import settings
from src.db.session import SessionLocal

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _busy() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": {"code": "busy", "message": "the API is at capacity; retry in a moment"}},
        headers={"Retry-After": "1"},
    )


def create_app(
    runtime: Runtime | None = None,
    cache: RedisCache | None = None,
    limiter: RateLimiter | None = None,
    log_writer: Callable[[dict[str, Any]], None] | None = None,
    max_concurrent: int | None = None,
) -> FastAPI:
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime or Runtime(SessionLocal, settings.metadata_refresh_seconds)
        app.state.cache = cache or RedisCache.from_url(
            settings.redis_url, settings.redis_cache_ttl_seconds, settings.redis_lock_timeout_seconds,
            settings.redis_socket_timeout_seconds, settings.redis_retry_after_seconds,
        )
        app.state.limiter = limiter or RateLimiter(app.state.cache, settings.rate_limit_per_minute)
        app.state.log_writer = log_writer or PredictionLogWriter(SessionLocal)
        yield
        app.state.cache.close()

    app = FastAPI(
        title="fair-football-fee API",
        version="1.0.0",
        description="Estimates of a Premier League player's Transfermarkt crowd valuation (not a transfer fee), "
        "with prediction intervals, SHAP drivers and what-if scenarios.",
        lifespan=lifespan,
    )
    limit = settings.max_concurrent_requests if max_concurrent is None else max_concurrent
    in_flight = 0

    # Added before CORS so CORS wraps it: a browser must be able to read the 503
    @app.middleware("http")
    async def shed_load(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Bounded concurrency per worker. Past the limit, queueing more requests would only make every
        one slower and hold database connections longer (found by the Phase 7 load test), so the excess
        gets an immediate 503 with Retry-After. Uvicorn's --limit-concurrency also counts idle keep-alive
        connections, which browsers calling the API directly keep open."""
        nonlocal in_flight
        if in_flight >= limit:
            return _busy()
        in_flight += 1
        try:
            return await call_next(request)
        finally:
            in_flight -= 1

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Cache", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request.state.request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(PoolTimeoutError)
    async def database_busy(request: Request, exc: PoolTimeoutError) -> JSONResponse:
        # No database connection freed up within db_pool_timeout_seconds: overloaded, not broken
        return _busy()

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": {"code": exc.code, "message": exc.message}})

    app.include_router(router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
