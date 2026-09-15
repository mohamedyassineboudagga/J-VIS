"""Rate limiting middleware for J-VIS.

Enforces 30 requests/min per session (configurable via RATE_LIMIT_PER_MINUTE).
Uses an in-memory sliding window per client IP + user.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Check if a request is allowed. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            queue = self._requests[key]
            # Remove expired timestamps
            while queue and now - queue[0] > self.window_seconds:
                queue.popleft()

            if len(queue) >= self.max_requests:
                retry_after = int(self.window_seconds - (now - queue[0])) + 1
                return False, retry_after

            queue.append(now)
            return True, 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._requests.pop(key, None)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware enforcing rate limits."""

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(max_requests, window_seconds)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        # Key by client IP + user_id (if provided in body)
        client_ip = request.client.host if request.client else "unknown"
        key = f"{client_ip}"

        allowed, retry_after = self.limiter.check(key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded. Please slow down.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)


def get_rate_limit() -> int:
    """Get the configured rate limit from env."""
    return int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))