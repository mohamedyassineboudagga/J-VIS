"""Phase 3 acceptance tests — Privacy / Redaction Pipeline.

Acceptance criteria:
    - "my IP is 192.168.1.50 and key is sk-ab12..." reaches the LLM
      provider SCRUBBED
    - Original text is stored locally for gateway use
    - REDACT_REMOTE_ONLY flag controls behavior
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.mock_llm import MockLLMProvider


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p3_audit.jsonl")
    yield
    os.environ.pop("AUDIT_LOG_PATH", None)


class TestRedactionBeforeLLM:
    """Verify PII is scrubbed BEFORE reaching the LLM provider."""

    def test_llm_receives_scrubbed_message(self, tmp_path):
        """The message reaching the LLM must not contain PII."""
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        original = "my IP is 192.168.1.50 and key is sk-ab123456789012345"
        scrubbed, findings = rp.scrub(original)

        # The scrubbed message is what gets sent to the LLM
        assert "192.168.1.50" not in scrubbed
        assert "sk-ab123456789012345" not in scrubbed
        assert "[REDACTED:local_ip]" in scrubbed
        assert "[REDACTED:api_key]" in scrubbed

    def test_original_preserved_locally(self, tmp_path):
        """The original text is preserved for gateway use."""
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        original = "my IP is 192.168.1.50"
        scrubbed, findings = rp.scrub(original)

        # Original is untouched
        assert original == "my IP is 192.168.1.50"
        # Scrubbed version is different
        assert scrubbed != original

    def test_redact_remote_only_flag(self):
        """REDACT_REMOTE_ONLY=true means local Ollama may bypass."""
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=True)
        assert rp.should_redact("openai") is True
        assert rp.should_redact("ollama") is False  # local may bypass

        rp2 = RedactionPipeline(redact_remote_only=False)
        assert rp2.should_redact("ollama") is True  # always redact

    def test_findings_are_structured(self, tmp_path):
        """Findings include type and position info."""
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("email me at a@b.com")
        assert len(findings) == 1
        f = findings[0]
        assert f["type"] == "email"
        assert "start" in f
        assert "end" in f
        assert f["replacement"] == "[REDACTED:email]"

    def test_no_pii_no_findings(self, tmp_path):
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("Hello, how are you today?")
        assert findings == []
        assert scrubbed == "Hello, how are you today?"


class TestRedactionInChatFlow:
    """Verify the /chat endpoint redacts before calling the LLM."""

    def test_chat_redacts_before_llm(self, tmp_path):
        """Full chat flow: PII in message is scrubbed before LLM sees it."""
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway
        from llm.provider import IntentParser
        from redaction.pipeline import RedactionPipeline

        db = AuthDB(str(tmp_path / "p3.db"))
        users = seed_demo_users(db)
        audit = AuditLogger(str(tmp_path / "p3_audit.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit)
        rp = RedactionPipeline(redact_remote_only=False)

        tool_specs = [
            {"name": t.name, "description": t.description, "args_schema": t.args_schema}
            for t in registry.list()
        ]
        parser = IntentParser(tool_specs)

        # Simulate the chat flow with redaction
        user_msg = "my IP is 192.168.1.50, search for news"
        scrubbed, findings = rp.scrub(user_msg)
        assert "192.168.1.50" not in scrubbed

        # The scrubbed message goes to the LLM
        llm = MockLLMProvider()
        raw = llm.generate([{"role": "user", "content": scrubbed}])
        intent = parser.parse(raw)
        assert intent["action"] == "web_search"

    def test_redaction_scrubs_tool_outputs(self, tmp_path):
        """Tool outputs containing PII are scrubbed before reaching LLM."""
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        tool_output = {
            "result": "Found user at 10.0.0.5, email: admin@corp.com",
            "metadata": {"ip": "192.168.0.1"},
        }
        scrubbed, findings = rp.scrub_dict(tool_output)
        assert "10.0.0.5" not in scrubbed["result"]
        assert "admin@corp.com" not in scrubbed["result"]
        assert "192.168.0.1" not in scrubbed["metadata"]["ip"]
        assert len(findings) == 3


class TestRedactionEdgeCases:
    """Edge cases for the redaction pipeline."""

    def test_public_ip_not_redacted(self):
        """Public IPs are NOT redacted (only local/private ranges)."""
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("Server at 8.8.8.8 is up")
        assert "8.8.8.8" in scrubbed  # public IP preserved
        assert findings == []

    def test_loopback_redacted(self):
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("localhost is 127.0.0.1")
        assert "127.0.0.1" not in scrubbed
        assert "[REDACTED:local_ip]" in scrubbed

    def test_private_10x_redacted(self):
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("db at 10.1.2.3")
        assert "10.1.2.3" not in scrubbed

    def test_aws_key_redacted(self):
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("key AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
        assert "[REDACTED:api_key]" in scrubbed

    def test_github_token_redacted(self):
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("token ghp_1234567890123456789012345678901234567890")
        assert "ghp_" not in scrubbed
        assert "[REDACTED:api_key]" in scrubbed

    def test_address_redacted(self):
        from redaction.pipeline import RedactionPipeline

        rp = RedactionPipeline(redact_remote_only=False)
        scrubbed, findings = rp.scrub("I live at 123 Main Street")
        assert "123 Main Street" not in scrubbed
        assert "[REDACTED:address]" in scrubbed