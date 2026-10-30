"""FastAPI dependencies. Shared components live on `app.state` (built in `create_app`), so tests
can pass fakes to `create_app` and override `get_repo` instead of patching modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.api.cache import RateLimiter, RedisCache
from src.api.repository import Repository
from src.api.runtime import Runtime, ServingState
from src.api.service import require_servable
from src.db.session import get_db


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def get_state(runtime: Annotated[Runtime, Depends(get_runtime)]) -> ServingState:
    return runtime.state()


StateDep = Annotated[ServingState, Depends(get_state)]


def get_servable_state(state: StateDep) -> ServingState:
    require_servable(state)
    return state


def get_cache(request: Request) -> RedisCache:
    return request.app.state.cache


def get_log_writer(request: Request) -> Callable[[dict[str, Any]], None]:
    return request.app.state.log_writer


def get_repo(db: Annotated[Session, Depends(get_db)]) -> Repository:
    return Repository(db)


ServableStateDep = Annotated[ServingState, Depends(get_servable_state)]
CacheDep = Annotated[RedisCache, Depends(get_cache)]
LogWriterDep = Annotated[Callable[[dict[str, Any]], None], Depends(get_log_writer)]
RepoDep = Annotated[Repository, Depends(get_repo)]


def rate_limit(request: Request) -> None:
    limiter: RateLimiter = request.app.state.limiter
    # The connecting address. Uvicorn replaces it with the client named in X-Forwarded-For only when the
    # connection comes from FORWARDED_ALLOW_IPS (a trusted proxy), so a client can't pick its own bucket
    allowed, retry_after = limiter.hit(request.client.host if request.client else "unknown")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": f"at most {limiter.limit} what-if requests per minute"},
            headers={"Retry-After": str(retry_after)},
        )
