"""Phase 2 acceptance tests — Conversational Core.

Acceptance criteria:
    - "delete all my files" as guest/user returns a refusal
    - as operator it triggers the HITL flow, not immediate execution
    - LLM output is parsed into a strict intent schema
    - Unknown actions are rejected
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.mock_llm import MockLLMProvider


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p2_audit.jsonl")
    yield
    os.environ.pop("AUDIT_LOG_PATH", None)


def _build_stack(tmp_path):
    """Build the full application stack for testing."""
    from auth.models import AuthDB, seed_demo_users
    from audit.logger import AuditLogger
    from tools.registry import build_default_registry
    from gateway.policy_gateway import PolicyGateway
    from llm.provider import IntentParser

    db = AuthDB(str(tmp_path / "p2.db"))
    users = seed_demo_users(db)
    audit = AuditLogger(str(tmp_path / "p2_audit.jsonl"))
    registry = build_default_registry()
    gw = PolicyGateway(registry, db, audit)

    tool_specs = [
        {"name": t.name, "description": t.description, "args_schema": t.args_schema}
        for t in registry.list()
    ]
    parser = IntentParser(tool_specs)
    return db, users, audit, registry, gw, parser


class TestIntentParsing:
    """Tests for strict intent schema parsing."""

    def test_valid_intent_parsed(self, tmp_path):
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        intent = parser.parse(
            '{"action": "get_weather", "args": {"location": "Tokyo"}, "justification": "weather"}'
        )
        assert intent["action"] == "get_weather"
        assert intent["args"]["location"] == "Tokyo"

    def test_markdown_wrapped_json(self, tmp_path):
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        intent = parser.parse(
            '```json\n{"action": "web_search", "args": {"query": "news"}, "justification": "search"}\n```'
        )
        assert intent["action"] == "web_search"

    def test_unknown_action_rejected(self, tmp_path):
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        intent = parser.parse(
            '{"action": "hack_the_planet", "args": {}, "justification": "hack"}'
        )
        assert intent["action"] == "unknown"

    def test_non_json_response(self, tmp_path):
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        intent = parser.parse("I don't know what you mean")
        assert intent["action"] == "unknown"

    def test_invalid_args_type(self, tmp_path):
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        intent = parser.parse(
            '{"action": "get_weather", "args": "not_a_dict", "justification": "x"}'
        )
        assert intent["action"] == "get_weather"
        assert intent["args"] == {}


class TestChatFlow:
    """Tests for the full chat pipeline with mock LLM."""

    def test_weather_chat_flow(self, tmp_path):
        """Guest asks weather -> allowed -> tool runs."""
        from gateway.policy_gateway import PolicyDenied
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        guest = [u for u in users if u.role == "guest"][0]

        llm = MockLLMProvider()
        raw = llm.generate([{"role": "user", "content": "What's the weather in Tokyo?"}])
        intent = parser.parse(raw)
        assert intent["action"] == "get_weather"

        decision = gw.authorize(guest, intent["action"], intent["args"])
        assert decision.allowed is True
        result = decision.tool.execute(intent["args"])
        assert result["location"] == "Unknown"

    def test_delete_as_guest_refused(self, tmp_path):
        """Guest says 'delete all my files' -> refusal."""
        from gateway.policy_gateway import PolicyDenied
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        guest = [u for u in users if u.role == "guest"][0]

        llm = MockLLMProvider()
        raw = llm.generate([{"role": "user", "content": "delete all my files"}])
        intent = parser.parse(raw)
        assert intent["action"] == "delete_file"

        with pytest.raises(PolicyDenied) as exc_info:
            gw.authorize(guest, intent["action"], intent["args"])
        assert "role" in str(exc_info.value).lower() or "permission" in str(exc_info.value).lower()

    def test_delete_as_user_refused(self, tmp_path):
        """User says 'delete all my files' -> refusal (needs operator)."""
        from gateway.policy_gateway import PolicyDenied
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        user = [u for u in users if u.role == "user"][0]

        llm = MockLLMProvider()
        raw = llm.generate([{"role": "user", "content": "delete all my files"}])
        intent = parser.parse(raw)
        assert intent["action"] == "delete_file"

        with pytest.raises(PolicyDenied):
            gw.authorize(user, intent["action"], intent["args"])

    def test_delete_as_operator_triggers_hitl(self, tmp_path):
        """Operator says 'delete all my files' -> HITL flow, NOT immediate execution."""
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        op = [u for u in users if u.role == "operator"][0]

        llm = MockLLMProvider()
        raw = llm.generate([{"role": "user", "content": "delete all my files"}])
        intent = parser.parse(raw)
        assert intent["action"] == "delete_file"

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, intent["action"], intent["args"])
        assert exc_info.value.approval_id is not None

    def test_reboot_as_master_triggers_hitl(self, tmp_path):
        """Even master cannot reboot without HITL confirmation."""
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        master = [u for u in users if u.role == "master"][0]

        llm = MockLLMProvider()
        raw = llm.generate([{"role": "user", "content": "reboot the server"}])
        intent = parser.parse(raw)
        assert intent["action"] == "reboot_server"

        with pytest.raises(HITLRequired):
            gw.authorize(master, intent["action"], intent["args"])

    def test_unknown_intent_denied(self, tmp_path):
        """Unknown intent -> default deny."""
        from gateway.policy_gateway import PolicyDenied
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        master = [u for u in users if u.role == "master"][0]

        intent = {"action": "unknown", "args": {}, "justification": "no match"}
        with pytest.raises(PolicyDenied):
            gw.authorize(master, intent["action"], intent["args"])

    def test_audit_logs_chat_decisions(self, tmp_path):
        """Every chat decision lands in the audit log."""
        from gateway.policy_gateway import PolicyDenied, HITLRequired
        db, users, audit, registry, gw, parser = _build_stack(tmp_path)
        guest = [u for u in users if u.role == "guest"][0]
        op = [u for u in users if u.role == "operator"][0]

        llm = MockLLMProvider()

        # Allowed action
        raw = llm.generate([{"role": "user", "content": "weather"}])
        intent = parser.parse(raw)
        gw.authorize(guest, intent["action"], intent["args"])

        # Denied action
        raw = llm.generate([{"role": "user", "content": "delete"}])
        intent = parser.parse(raw)
        try:
            gw.authorize(guest, intent["action"], intent["args"])
        except PolicyDenied:
            pass

        # HITL action
        raw = llm.generate([{"role": "user", "content": "delete"}])
        intent = parser.parse(raw)
        try:
            gw.authorize(op, intent["action"], intent["args"])
        except HITLRequired:
            pass

        assert audit.count() == 3
        assert audit.verify() is True


class TestChatEndpoint:
    """Tests for the /chat HTTP endpoint."""

    def test_chat_endpoint_weather(self, tmp_path):
        """POST /chat without LLM uses the local classifier end-to-end."""
        os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p2e_audit.jsonl")
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            # Get a guest user id
            users = client.get("/users").json()
            guest = [u for u in users if u["role"] == "guest"][0]

            resp = client.post("/chat", json={
                "user_id": guest["id"],
                "message": "weather in Tokyo",
                "use_llm": False,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["action"] == "get_weather"  # local classifier matches weather

    def test_chat_endpoint_greeting(self, tmp_path):
        """Greetings get a natural-language response without a tool."""
        os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p2e3_audit.jsonl")
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            users = client.get("/users").json()
            guest = [u for u in users if u["role"] == "guest"][0]

            resp = client.post("/chat", json={
                "user_id": guest["id"],
                "message": "hello",
                "use_llm": False,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["action"] == "greeting"
            assert "response" in data["result"]

    def test_chat_endpoint_unknown_user(self, tmp_path):
        os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p2e2_audit.jsonl")
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            resp = client.post("/chat", json={
                "user_id": "nonexistent",
                "message": "hello",
                "use_llm": False,
            })
            assert resp.status_code == 404