"""Redis cache-aside with a singleflight lock, a circuit breaker and a fixed-window rate limiter.

Redis is an optimization, never a dependency. Every Redis error is caught; the breaker then skips
Redis for `retry_after_seconds`, so a dead Redis costs at most one short timeout per window instead
of one per request. The cache falls back to computing the answer (`X-Cache: BYPASS`) and the rate
limiter lets requests through (fail-open).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from enum import StrEnum

import redis
from redis.exceptions import RedisError, WatchError

log = logging.getLogger(__name__)


class CacheStatus(StrEnum):
    HIT = "HIT"
    MISS = "MISS"
    BYPASS = "BYPASS"


class RedisCache:
    """`get_or_compute` runs `compute` at most once per key at a time, across all API workers."""

    def __init__(
        self,
        client: redis.Redis,
        ttl_seconds: int,
        lock_seconds: int,
        retry_after_seconds: float,
        poll_seconds: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.lock_seconds = lock_seconds
        self.retry_after_seconds = retry_after_seconds
        self.poll_seconds = poll_seconds
        self.clock = clock
        self._open_until = 0.0
        self._breaker_lock = threading.Lock()

    @classmethod
    def from_url(cls, url: str, ttl_seconds: int, lock_seconds: int, socket_timeout: float, retry_after_seconds: float) -> RedisCache:
        client = redis.Redis.from_url(url, socket_timeout=socket_timeout, socket_connect_timeout=socket_timeout)
        return cls(client, ttl_seconds, lock_seconds, retry_after_seconds)

    # --- circuit breaker
    def available(self) -> bool:
        return self.clock() >= self._open_until

    def trip(self, exc: Exception) -> None:
        with self._breaker_lock:
            if self.available():
                log.warning("Redis unavailable (%s); bypassing it for %.0fs", exc, self.retry_after_seconds)
            self._open_until = self.clock() + self.retry_after_seconds

    def ping(self) -> bool:
        if not self.available():
            return False
        try:
            return bool(self.client.ping())
        except RedisError as exc:
            self.trip(exc)
            return False

    # --- cache-aside
    def get_or_compute(self, key: str, compute: Callable[[], str]) -> tuple[str, CacheStatus]:
        """Return the cached string for `key`, or compute, store and return it."""
        if not self.available():
            return compute(), CacheStatus.BYPASS
        lock_key, token = f"lock:{key}", uuid.uuid4().hex
        try:
            cached = self.client.get(key)
            if cached is not None:
                return cached.decode(), CacheStatus.HIT
            owner = self.client.set(lock_key, token, nx=True, ex=self.lock_seconds)
        except RedisError as exc:
            self.trip(exc)
            return compute(), CacheStatus.BYPASS

        if owner:
            try:
                value = compute()
                try:
                    self.client.set(key, value, ex=self.ttl_seconds)
                except RedisError as exc:
                    self.trip(exc)
                return value, CacheStatus.MISS
            finally:
                self._release(lock_key, token)

        # Another request is computing this key: wait for its result, at most one lock lifetime
        deadline = self.clock() + self.lock_seconds
        while self.clock() < deadline:
            time.sleep(self.poll_seconds)
            try:
                cached = self.client.get(key)
            except RedisError as exc:
                self.trip(exc)
                return compute(), CacheStatus.BYPASS
            if cached is not None:
                return cached.decode(), CacheStatus.HIT
        return compute(), CacheStatus.MISS  # the owner failed or is slow: answer anyway, uncached

    def _release(self, lock_key: str, token: str) -> None:
        """Delete the lock only while it still holds our token (it may have expired and been retaken)."""
        try:
            with self.client.pipeline() as pipe:
                pipe.watch(lock_key)
                if pipe.get(lock_key) == token.encode():
                    pipe.multi()
                    pipe.delete(lock_key)
                    pipe.execute()
                else:
                    pipe.unwatch()
        except WatchError:
            pass  # the lock changed between GET and DELETE, so it's no longer ours
        except RedisError as exc:
            self.trip(exc)

    def close(self) -> None:
        try:
            self.client.close()
        except RedisError:
            pass


class RateLimiter:
    """Fixed one-minute window per identity, stored in the same Redis; fails open."""

    def __init__(self, cache: RedisCache, limit_per_minute: int, clock: Callable[[], float] = time.time):
        self.cache = cache
        self.limit = limit_per_minute
        self.clock = clock

    def hit(self, identity: str) -> tuple[bool, int]:
        """Count one request; return (allowed, seconds until the window resets)."""
        now = self.clock()
        retry_after = 60 - int(now % 60)
        if self.limit <= 0 or not self.cache.available():
            return True, retry_after
        key = f"ratelimit:{identity}:{int(now // 60)}"
        try:
            pipe = self.cache.client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 61)
            count = int(pipe.execute()[0])
        except RedisError as exc:
            self.cache.trip(exc)
            return True, retry_after
        return count <= self.limit, retry_after
