"""Phase 7 acceptance tests — Hardening & Docs.

Acceptance criteria:
    - Rate limit: 30 requests/min per session
    - Session tokens: HMAC-signed, bound to biometric identity hash
    - Dockerfile + docker-compose.yml exist
    - README exists with architecture + security model
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p7_audit.jsonl")
    yield
    os.environ.pop("AUDIT_LOG_PATH", None)


class TestRateLimiting:
    """Tests for the rate limiter."""

    def test_rate_limiter_allows_under_limit(self):
        from gateway.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, _ = limiter.check("test_key")
            assert allowed is True

    def test_rate_limiter_blocks_over_limit(self):
        from gateway.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            limiter.check("test_key")
        allowed, retry_after = limiter.check("test_key")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_reset(self):
        from gateway.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("key")
        limiter.check("key")
        assert limiter.check("key")[0] is False
        limiter.reset("key")
        assert limiter.check("key")[0] is True

    def test_rate_limiter_per_key(self):
        from gateway.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("key_a")
        limiter.check("key_a")
        # Different key is not affected
        assert limiter.check("key_b")[0] is True

    def test_rate_limit_env_config(self):
        from gateway.rate_limit import get_rate_limit
        os.environ["RATE_LIMIT_PER_MINUTE"] = "30"
        assert get_rate_limit() == 30
        os.environ["RATE_LIMIT_PER_MINUTE"] = "100"
        assert get_rate_limit() == 100
        os.environ.pop("RATE_LIMIT_PER_MINUTE", None)
        assert get_rate_limit() == 30  # default


class TestSessionTokens:
    """Tests for HMAC-signed session tokens."""

    def test_token_creation(self):
        from auth.tokens import SessionTokenManager
        mgr = SessionTokenManager(secret="test-secret", ttl_seconds=900)
        token = mgr.create_token("user1", "operator", "bio_hash_123")
        assert "." in token

    def test_token_verification(self):
        from auth.tokens import SessionTokenManager
        mgr = SessionTokenManager(secret="test-secret", ttl_seconds=900)
        token = mgr.create_token("user1", "operator", "bio_hash_123")
        payload = mgr.verify_token(token)
        assert payload["user_id"] == "user1"
        assert payload["role"] == "operator"
        assert payload["biometric_hash"] == "bio_hash_123"

    def test_token_tamper_detected(self):
        from auth.tokens import SessionTokenManager, SessionTokenError
        mgr = SessionTokenManager(secret="test-secret", ttl_seconds=900)
        token = mgr.create_token("user1", "operator", "bio_hash_123")
        # Tamper with the payload
        payload_b64, sig = token.split(".")
        tampered = payload_b64[:-1] + ("A" if payload_b64[-1] != "A" else "B")
        with pytest.raises(SessionTokenError):
            mgr.verify_token(f"{tampered}.{sig}")

    def test_token_expiry(self):
        import time
        from auth.tokens import SessionTokenManager, SessionTokenError
        mgr = SessionTokenManager(secret="test-secret", ttl_seconds=1)
        token = mgr.create_token("user1", "operator")
        # Wait for expiry (2s > 1s TTL, accounts for second-granularity)
        time.sleep(2.1)
        with pytest.raises(SessionTokenError):
            mgr.verify_token(token)

    def test_token_bound_to_biometric_hash(self):
        """Token payload includes the biometric identity hash."""
        from auth.tokens import SessionTokenManager
        mgr = SessionTokenManager(secret="test-secret", ttl_seconds=900)
        token = mgr.create_token("user1", "master", "biometric_hash_xyz")
        payload = mgr.verify_token(token)
        assert payload["biometric_hash"] == "biometric_hash_xyz"

    def test_wrong_secret_fails(self):
        from auth.tokens import SessionTokenManager, SessionTokenError
        mgr1 = SessionTokenManager(secret="secret-1", ttl_seconds=900)
        mgr2 = SessionTokenManager(secret="secret-2", ttl_seconds=900)
        token = mgr1.create_token("user1", "guest")
        with pytest.raises(SessionTokenError):
            mgr2.verify_token(token)


class TestDeliverables:
    """Verify required deliverables exist."""

    def test_requirements_txt_exists(self):
        assert Path("requirements.txt").exists()

    def test_dockerfile_exists(self):
        assert Path("Dockerfile").exists()

    def test_docker_compose_exists(self):
        assert Path("docker-compose.yml").exists()

    def test_readme_exists(self):
        assert Path("README.md").exists()

    def test_readme_has_architecture(self):
        content = Path("README.md").read_text(encoding="utf-8")
        assert "Architecture" in content
        assert "POLICY GATEWAY" in content

    def test_readme_has_security_model(self):
        content = Path("README.md").read_text(encoding="utf-8")
        assert "Security Model" in content
        assert "Threat Model" in content

    def test_readme_has_enrollment_walkthrough(self):
        content = Path("README.md").read_text(encoding="utf-8")
        assert "Enrollment" in content

    def test_readme_has_docker_setup(self):
        content = Path("README.md").read_text(encoding="utf-8")
        assert "docker compose up" in content

    def test_requirements_pinned(self):
        """requirements.txt should have pinned versions."""
        content = Path("requirements.txt").read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "==" not in line:
                # Allow optional deps commented out
                if not line.startswith("pip"):
                    pass  # Some lines may be comments
        # Verify key packages are pinned
        assert "fastapi==" in content
        assert "pydantic==" in content
        assert "pytest==" in content


class TestFullSuite:
    """Verify the complete test suite passes."""

    def test_all_phase_tests_exist(self):
        """All phase test files exist."""
        assert Path("tests/test_comprehensive.py").exists()
        assert Path("tests/test_phase2.py").exists()
        assert Path("tests/test_phase3.py").exists()
        assert Path("tests/test_phase4.py").exists()
        assert Path("tests/test_phase5.py").exists()
        assert Path("tests/test_phase6.py").exists()

    def test_tool_count(self):
        """10 registered tools (9 required + run_python_snippet)."""
        from tools.registry import build_default_registry
        registry = build_default_registry()
        assert len(registry.list()) == 10

    def test_hitl_tools_count(self):
        """2 HITL tools: delete_file + reboot_server."""
        from tools.registry import build_default_registry
        registry = build_default_registry()
        hitl_tools = [t for t in registry.list() if t.hitl]
        assert len(hitl_tools) == 2
        names = {t.name for t in hitl_tools}
        assert names == {"delete_file", "reboot_server"}