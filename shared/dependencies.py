"""Shared infrastructure: Redis state store, HTTP connection pooling,
distributed rate limiting, and structured logging.

Safe across multiple uvicorn workers because all mutable state lives
in Redis, not in process memory.

Usage:
    from shared.dependencies import (
        get_redis, get_http_client, get_state_store,
        get_github_rate_limiter, logger,
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

# ── Structured Logging ──────────────────────────────────────────────────────
_logger = logging.getLogger("multi_agent")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

logger = _logger

# ── Redis ───────────────────────────────────────────────────────────────────
_redis_pool: Any | None = None


def _get_redis_pool():
    global _redis_pool
    if _redis_pool is None:
        import redis.asyncio as redis_lib
        host = os.environ.get("REDIS_HOST", "redis")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        db = int(os.environ.get("REDIS_DB", "0"))
        _redis_pool = redis_lib.Redis(
            host=host, port=port, db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
        logger.info("Redis pool created for %s:%s/%s", host, port, db)
    return _redis_pool


async def get_redis():
    return _get_redis_pool()


# ── State Store (Redis-backed) ──────────────────────────────────────────────

class StateStore:
    """Durable task state using Redis. Falls back to in-memory dict
    if Redis is unavailable (logs a loud warning).
    """

    _TTL_SECONDS = 86400
    _FALLBACK: dict[str, dict] = {}

    def __init__(self, redis_client=None):
        self._r = redis_client
        self._use_redis = redis_client is not None

    @classmethod
    async def create(cls) -> StateStore:
        try:
            r = await get_redis()
            await r.ping()
            logger.info("StateStore using Redis (multi-worker safe)")
            return cls(redis_client=r)
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s). StateStore falling back to in-memory "
                "DICT — NOT SAFE ACROSS MULTIPLE WORKERS. "
                "Run with a single uvicorn worker or fix Redis.",
                exc,
            )
            return cls(redis_client=None)

    async def get(self, task_id: str) -> dict | None:
        if self._use_redis:
            raw = await self._r.get(f"task:{task_id}")
            return json.loads(raw) if raw else None
        return self._FALLBACK.get(task_id)

    async def set(self, task_id: str, data: dict) -> None:
        if self._use_redis:
            await self._r.setex(f"task:{task_id}", self._TTL_SECONDS, json.dumps(data))
        else:
            self._FALLBACK[task_id] = data

    async def cleanup_expired(self) -> int:
        if self._use_redis:
            return 0
        now = time.time()
        expired = [
            tid for tid, t in self._FALLBACK.items()
            if t.get("completed_at") and now - t["completed_at"] > self._TTL_SECONDS
        ]
        for tid in expired:
            del self._FALLBACK[tid]
        if expired:
            logger.info("Cleaned up %s expired in-memory tasks", len(expired))
        return len(expired)


# ── Shared HTTP Client ──────────────────────────────────────────────────────

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            http2=False,
        )
        logger.debug("Created shared httpx.AsyncClient")
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
        logger.debug("Closed shared httpx.AsyncClient")


# ── Rate Limiter (Redis-backed) ─────────────────────────────────────────────

class RateLimiter:
    def __init__(self, redis_client=None, resource: str = "github"):
        self._r = redis_client
        self._use_redis = redis_client is not None
        self._resource = resource
        self._remaining = 5000
        self._reset_at = 0

    @classmethod
    async def create(cls, resource: str = "github") -> RateLimiter:
        try:
            r = await get_redis()
            await r.ping()
            return cls(redis_client=r, resource=resource)
        except Exception:
            logger.warning("RateLimiter(%s) using per-process fallback", resource)
            return cls(redis_client=None, resource=resource)

    async def update(self, remaining: int, reset_timestamp: int) -> None:
        if self._use_redis:
            pipe = self._r.pipeline()
            pipe.set(f"ratelimit:{self._resource}:remaining", remaining, ex=300)
            pipe.set(f"ratelimit:{self._resource}:reset", reset_timestamp, ex=300)
            await pipe.execute()
        else:
            self._remaining = remaining
            self._reset_at = reset_timestamp

    async def check(self) -> tuple[int, int]:
        if self._use_redis:
            pipe = self._r.pipeline()
            pipe.get(f"ratelimit:{self._resource}:remaining")
            pipe.get(f"ratelimit:{self._resource}:reset")
            rem, rst = await pipe.execute()
            return int(rem or 5000), int(rst or 0)
        return self._remaining, self._reset_at


# ── Retry Helper with Wall-Clock Budget ─────────────────────────────────────

class RetryBudget:
    def __init__(
        self,
        max_retries: int = 3,
        max_total_seconds: float = 120.0,
        base_delay: float = 1.0,
    ):
        self.max_retries = max_retries
        self.max_total_seconds = max_total_seconds
        self.base_delay = base_delay
        self.deadline = time.monotonic() + max_total_seconds

    def should_retry(self, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if time.monotonic() >= self.deadline:
            logger.warning(
                "Retry budget exhausted (%.1fs / %.1fs)",
                self.max_total_seconds - (self.deadline - time.monotonic()),
                self.max_total_seconds,
            )
            return False
        return True

    def delay_for(self, attempt: int, retry_after: int | None = None) -> float:
        if retry_after is not None and retry_after > 0:
            remaining = self.deadline - time.monotonic()
            return min(retry_after, max(remaining, 0))
        return min(self.base_delay * (2 ** attempt), 60.0)


# ── Module-level singletons (initialized in lifespan) ───────────────────────

_store: StateStore | None = None
_github_limiter: RateLimiter | None = None
_slack_limiter: RateLimiter | None = None


async def init_shared():
    global _store, _github_limiter, _slack_limiter
    _store = await StateStore.create()
    _github_limiter = await RateLimiter.create("github")
    _slack_limiter = await RateLimiter.create("slack")
    logger.info("Shared infrastructure initialized")


def get_state_store() -> StateStore:
    if _store is None:
        raise RuntimeError("Shared infrastructure not initialized — call init_shared() in lifespan")
    return _store


def get_github_rate_limiter() -> RateLimiter:
    if _github_limiter is None:
        raise RuntimeError("Shared infrastructure not initialized")
    return _github_limiter


def get_slack_rate_limiter() -> RateLimiter:
    if _slack_limiter is None:
        raise RuntimeError("Shared infrastructure not initialized")
    return _slack_limiter
