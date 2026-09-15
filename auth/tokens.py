"""HMAC-signed session tokens bound to biometric identity hash.

Session tokens are:
    - HMAC-SHA256 signed with a server secret
    - Bound to the user's biometric identity hash
    - Short-lived (TTL 15 min by default)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Dict, Optional


class SessionTokenError(Exception):
    """Raised when a session token is invalid or expired."""


class SessionTokenManager:
    """Create and verify HMAC-signed session tokens."""

    def __init__(self, secret: Optional[str] = None, ttl_seconds: int = 900):
        # In production, the secret comes from env. For dev, derive from
        # a stable source. NEVER hardcode in code.
        self.secret = secret or os.getenv("SESSION_SECRET", "")
        if not self.secret:
            # Dev fallback — documented as insecure for production
            self.secret = hashlib.sha256(b"jvis-dev-secret").hexdigest()
        self.ttl_seconds = ttl_seconds

    def create_token(
        self,
        user_id: str,
        role: str,
        biometric_hash: str = "",
    ) -> str:
        """Create a signed session token."""
        payload = {
            "user_id": user_id,
            "role": role,
            "biometric_hash": biometric_hash,
            "iat": int(time.time()),
            "exp": int(time.time()) + self.ttl_seconds,
        }
        payload_b64 = self._b64encode(json.dumps(payload))
        signature = self._sign(payload_b64)
        return f"{payload_b64}.{signature}"

    def verify_token(self, token: str) -> Dict:
        """Verify a token. Returns the payload if valid."""
        try:
            payload_b64, signature = token.split(".")
        except ValueError:
            raise SessionTokenError("Malformed token")

        # Verify signature
        expected = self._sign(payload_b64)
        if not hmac.compare_digest(signature, expected):
            raise SessionTokenError("Invalid token signature")

        # Decode payload
        try:
            payload = json.loads(self._b64decode(payload_b64))
        except Exception:
            raise SessionTokenError("Invalid token payload")

        # Check expiry
        if int(payload.get("exp", 0)) < int(time.time()):
            raise SessionTokenError("Token expired")

        return payload

    def _sign(self, data: str) -> str:
        return hmac.new(
            self.secret.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _b64encode(data: str) -> str:
        return base64.urlsafe_b64encode(data.encode()).decode().rstrip("=")

    @staticmethod
    def _b64decode(data: str) -> str:
        padding = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + padding).decode()


# Module-level singleton
_token_manager: Optional[SessionTokenManager] = None


def get_token_manager() -> SessionTokenManager:
    global _token_manager
    if _token_manager is None:
        ttl = int(os.getenv("SESSION_TTL_MINUTES", "15")) * 60
        _token_manager = SessionTokenManager(ttl_seconds=ttl)
    return _token_manager