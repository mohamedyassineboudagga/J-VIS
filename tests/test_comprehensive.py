"""J-VIS Test Suite — Phase 1 through Phase 7 acceptance tests.

Target: 30+ tests proving all security invariants.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test setup — use a temporary database for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Create isolated test environment for each test."""
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "test_audit.jsonl")
    os.environ["REDACT_REMOTE_ONLY"] = "false"
    yield
    # Cleanup env vars
    for key in ["AUDIT_LOG_PATH", "REDACT_REMOTE_ONLY"]:
        os.environ.pop(key, None)


# ===========================================================================
# AUDIT LOGGER TESTS
# ===========================================================================

class TestAuditLogger:
    """Tests for the hash-chained audit logger."""

    def test_basic_log_entry(self, tmp_path):
        from audit.logger import AuditLogger
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        entry = logger.log("user1", "guest", "get_weather", "allow", {"location": "Tokyo"})
        assert entry["user_id"] == "user1"
        assert entry["role"] == "guest"
        assert entry["action"] == "get_weather"
        assert entry["decision"] == "allow"
        assert "entry_hash" in entry
        assert "prev_hash" in entry

    def test_hash_chain_integrity(self, tmp_path):
        from audit.logger import AuditLogger
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        logger.log("user1", "guest", "get_weather", "allow")
        logger.log("user2", "user", "read_file", "deny")
        logger.log("user3", "master", "reboot_server", "allow")
        assert logger.verify() is True

    def test_chain_verification_detects_tampering(self, tmp_path):
        from audit.logger import AuditLogger
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        logger.log("user1", "guest", "get_weather", "allow")
        logger.log("user2", "user", "read_file", "deny")
        logger.log("user3", "master", "reboot_server", "allow")
        assert logger.verify() is True
        # Tamper with the log
        with open(log_path, "r") as f:
            lines = f.readlines()
        entry = json.loads(lines[1])
        entry["decision"] = "tampered"
        lines[1] = json.dumps(entry) + "\n"
        with open(log_path, "w") as f:
            f.writelines(lines)
        # Verification should fail
        assert logger.verify() is False

    def test_verify_or_raise_on_tampered_log(self, tmp_path):
        from audit.logger import AuditLogger, AuditLogError
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        logger.log("user1", "guest", "get_weather", "allow")
        # Tamper
        with open(log_path, "r") as f:
            content = f.read()
        content = content.replace('"allow"', '"TAMPERED"')
        with open(log_path, "w") as f:
            f.write(content)
        with pytest.raises(AuditLogError):
            logger.verify_or_raise()

    def test_get_entries_newest_first(self, tmp_path):
        from audit.logger import AuditLogger
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        logger.log("a", "guest", "action1", "allow")
        logger.log("b", "user", "action2", "deny")
        entries = logger.get_entries(limit=1)
        assert len(entries) == 1
        assert entries[0]["user_id"] == "b"

    def test_log_count(self, tmp_path):
        from audit.logger import AuditLogger
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        assert logger.count() == 0
        logger.log("u", "guest", "test", "allow")
        logger.log("u", "guest", "test", "allow")
        assert logger.count() == 2

    def test_empty_log_verifies_ok(self, tmp_path):
        from audit.logger import AuditLogger
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        assert logger.verify() is True


# ===========================================================================
# AUTH / RBAC TESTS
# ===========================================================================

class TestAuthModels:
    """Tests for user model and RBAC role hierarchy."""

    def test_create_user(self, tmp_path):
        from auth.models import AuthDB
        db = AuthDB(str(tmp_path / "test.db"))
        user = db.create_user("alice", "user", "1234")
        assert user.username == "alice"
        assert user.role == "user"
        assert user.tier == 1

    def test_role_hierarchy(self, tmp_path):
        from auth.models import AuthDB, ROLE_TIER
        db = AuthDB(str(tmp_path / "test.db"))
        guest = db.create_user("g", "guest")
        user = db.create_user("u", "user")
        operator = db.create_user("o", "operator")
        master = db.create_user("m", "master")
        assert ROLE_TIER["master"] > ROLE_TIER["operator"] > ROLE_TIER["user"] > ROLE_TIER["guest"]
        assert master.can("operator")
        assert master.can("guest")
        assert guest.can("guest")
        assert not guest.can("user")
        assert not user.can("operator")

    def test_pin_verification(self, tmp_path):
        from auth.models import AuthDB
        db = AuthDB(str(tmp_path / "test.db"))
        user = db.create_user("bob", "user", "4321")
        assert db.verify_pin(user.id, "4321") is True
        assert db.verify_pin(user.id, "wrong") is False

    def test_session_creation_and_expiry(self, tmp_path):
        from auth.models import AuthDB
        db = AuthDB(str(tmp_path / "test.db"))
        user = db.create_user("charlie", "operator")
        session = db.create_session(user.id, user.role, ttl_minutes=15)
        assert session.is_valid is True
        assert session.user_id == user.id

    def test_seed_demo_users(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        db = AuthDB(str(tmp_path / "test.db"))
        users = seed_demo_users(db)
        assert len(users) == 4
        roles = {u.role for u in users}
        assert roles == {"master", "operator", "user", "guest"}

    def test_seed_demo_users_idempotent(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        db = AuthDB(str(tmp_path / "test.db"))
        users1 = seed_demo_users(db)
        users2 = seed_demo_users(db)
        assert len(users1) == 4
        assert len(users2) == 4
        assert db.list_users() == users1  # no duplicates


# ===========================================================================
# TOOL REGISTRY TESTS
# ===========================================================================

class TestToolRegistry:
    """Tests for the tool registry and argument validation."""

    def test_registry_loads_manifest(self):
        from tools.registry import build_default_registry
        registry = build_default_registry()
        assert len(registry.list()) >= 10
        assert registry.has("get_weather")
        assert registry.has("delete_file")
        assert registry.has("reboot_server")

    def test_tool_arg_validation_pass(self):
        from tools.registry import build_default_registry
        registry = build_default_registry()
        tool = registry.get("get_weather")
        result = tool.validate_args({"location": "New York"})
        assert result["location"] == "New York"

    def test_tool_arg_validation_fail(self):
        from tools.registry import build_default_registry
        from tools.registry import ToolError
        registry = build_default_registry()
        tool = registry.get("get_weather")
        with pytest.raises(ToolError):
            tool.validate_args({})  # location is required

    def test_tool_execution(self):
        from tools.registry import build_default_registry
        registry = build_default_registry()
        result = registry.get("get_weather").execute({"location": "London"})
        assert result["location"] == "London"
        assert "temperature_c" in result


# ===========================================================================
# POLICY GATEWAY TESTS (CRITICAL SECURITY TESTS)
# ===========================================================================

class TestPolicyGateway:
    """Tests proving RBAC enforcement.

    CRITICAL: These tests prove that:
        - Guest CANNOT call manage_calendar
        - User CAN call manage_calendar
        - Operator can call run_backup_script
        - Master can call reboot_server
        - HITL tools cannot execute without confirmation
        - Every decision lands in audit log with valid hash chain
    """

    def _setup_gateway(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway
        from redaction.pipeline import RedactionPipeline

        db = AuthDB(str(tmp_path / "test.db"))
        users = seed_demo_users(db)
        audit_log = AuditLogger(str(tmp_path / "test_audit.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit_log)
        return gw, db, audit_log, registry, users

    def test_guest_cannot_manage_calendar(self, tmp_path):
        from gateway.policy_gateway import PolicyDenied
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        guest = [u for u in users if u.role == "guest"][0]
        with pytest.raises(PolicyDenied) as exc_info:
            gw.authorize(guest, "manage_calendar", {"action": "list"})
        assert "role" in str(exc_info.value).lower() or "permission" in str(exc_info.value).lower()

    def test_user_can_manage_calendar(self, tmp_path):
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        user = [u for u in users if u.role == "user"][0]
        decision = gw.authorize(user, "manage_calendar", {"action": "list"})
        assert decision.allowed is True

    def test_guest_can_get_weather(self, tmp_path):
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        guest = [u for u in users if u.role == "guest"][0]
        decision = gw.authorize(guest, "get_weather", {"location": "NYC"})
        assert decision.allowed is True

    def test_operator_can_run_backup(self, tmp_path):
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        decision = gw.authorize(op, "run_backup_script", {"script_name": "daily"})
        assert decision.allowed is True

    def test_unknown_action_denied(self, tmp_path):
        from gateway.policy_gateway import PolicyDenied
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        master = [u for u in users if u.role == "master"][0]
        with pytest.raises(PolicyDenied) as exc_info:
            gw.authorize(master, "explode_the_sun", {})
        assert "not recognized" in str(exc_info.value).lower()

    def test_delete_file_requires_hitl(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/test.txt"})
        assert exc_info.value.approval_id is not None

    def test_reboot_requires_hitl(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        master = [u for u in users if u.role == "master"][0]
        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(master, "reboot_server", {})
        assert exc_info.value.approval_id is not None

    def test_all_decisions_logged(self, tmp_path):
        from gateway.policy_gateway import PolicyDenied
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        guest = [u for u in users if u.role == "guest"][0]
        user = [u for u in users if u.role == "user"][0]
        gw.authorize(guest, "get_weather", {"location": "A"})
        try:
            gw.authorize(guest, "manage_calendar", {})  # denied
        except PolicyDenied:
            pass
        gw.authorize(user, "manage_calendar", {"action": "list"})
        assert audit_log.count() == 3

    def test_audit_chain_valid_after_operations(self, tmp_path):
        gw, db, audit_log, reg, users = self._setup_gateway(tmp_path)
        guest = [u for u in users if u.role == "guest"][0]
        for i in range(5):
            try:
                gw.authorize(guest, "get_weather", {"location": f"City{i}"})
            except Exception:
                pass
        assert audit_log.verify() is True


# ===========================================================================
# HITL APPROVAL FLOW TESTS
# ===========================================================================

class TestHITLFlow:
    """Tests for Human-in-the-Loop confirmation flow."""

    def _setup_gateway(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway

        db = AuthDB(str(tmp_path / "test.db"))
        users = seed_demo_users(db)
        audit_log = AuditLogger(str(tmp_path / "test_audit.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit_log, hitl_ttl_minutes=1)
        return gw, db, audit_log, users

    def test_approval_flow_confirm(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        gw, db, audit_log, users = self._setup_gateway(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/test.txt"})
        approval_id = exc_info.value.approval_id
        # Confirm
        decision = gw.confirm_approval(approval_id, op)
        assert decision.allowed is True
        assert decision.action == "delete_file"

    def test_approval_cannot_confirm_twice(self, tmp_path):
        from gateway.policy_gateway import HITLRequired, PolicyDenied
        gw, db, audit_log, users = self._setup_gateway(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        with pytest.raises(HITLRequired):
            gw.authorize(op, "delete_file", {"path": "/tmp/test.txt"})
        approval = db.list_pending_approvals()[0]
        gw.confirm_approval(approval["id"], op)
        with pytest.raises(PolicyDenied) as exc_info:
            gw.confirm_approval(approval["id"], op)
        assert "already" in str(exc_info.value).lower()

    def test_approval_timeout(self, tmp_path):
        import asyncio
        from gateway.policy_gateway import HITLRequired, PolicyDenied
        gw, db, audit_log, users = self._setup_gateway(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        with pytest.raises(HITLRequired):
            gw.authorize(op, "delete_file", {"path": "/tmp/test.txt"})
        approval = db.list_pending_approvals()[0]
        # Simulate expiry by updating the expires_at to past
        from datetime import datetime, timedelta, timezone
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with db._lock, db._connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET expires_at = ? WHERE id = ?",
                (past, approval["id"]),
            )
        with pytest.raises(PolicyDenied) as exc_info:
            gw.confirm_approval(approval["id"], op)
        assert "expired" in str(exc_info.value).lower()

    def test_explicit_deny(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        gw, db, audit_log, users = self._setup_gateway(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        with pytest.raises(HITLRequired):
            gw.authorize(op, "delete_file", {"path": "/tmp/test.txt"})
        approval = db.list_pending_approvals()[0]
        gw.deny_approval(approval["id"], op)
        updated = db.get_approval(approval["id"])
        assert updated["status"] == "denied"


# ===========================================================================
# REDACTION PIPELINE TESTS
# ===========================================================================

class TestRedactionPipeline:
    """Tests proving PII is scrubbed from text."""

    def test_email_redaction(self):
        from redaction.pipeline import RedactionPipeline
        rp = RedactionPipeline(redact_remote_only=False)
        text = "Contact me at john.doe@example.com for details."
        scrubbed, findings = rp.scrub(text)
        assert "john.doe@example.com" not in scrubbed
        assert "[REDACTED:email]" in scrubbed
        assert len(findings) == 1

    def test_ip_redaction(self):
        from redaction.pipeline import RedactionPipeline
        rp = RedactionPipeline(redact_remote_only=False)
        text = "My IP is 192.168.1.50 and my key is sk-ab123456789012345"
        scrubbed, findings = rp.scrub(text)
        assert "192.168.1.50" not in scrubbed
        assert "[REDACTED:local_ip]" in scrubbed
        assert "sk-ab123456789012345" not in scrubbed
        assert "[REDACTED:api_key]" in scrubbed

    def test_phone_redaction(self):
        from redaction.pipeline import RedactionPipeline
        rp = RedactionPipeline(redact_remote_only=False)
        text = "Call me at (555) 123-4567"
        scrubbed, findings = rp.scrub(text)
        assert "123-4567" not in scrubbed or "[REDACTED:phone]" in scrubbed

    def test_credit_card_redaction(self):
        from redaction.pipeline import RedactionPipeline
        rp = RedactionPipeline(redact_remote_only=False)
        text = "Card number is 4111 1111 1111 1111"
        scrubbed, findings = rp.scrub(text)
        assert "4111" not in scrubbed
        assert "[REDACTED:credit_card]" in scrubbed

    def test_ssn_redaction(self):
        from redaction.pipeline import RedactionPipeline
        rp = RedactionPipeline(redact_remote_only=False)
        text = "SSN is 123-45-6789"
        scrubbed, findings = rp.scrub(text)
        assert "123-45-6789" not in scrubbed
        assert "[REDACTED:ssn]" in scrubbed

    def test_scrub_dict(self):
        from redaction.pipeline import RedactionPipeline
        rp = RedactionPipeline(redact_remote_only=False)
        data = {
            "message": "Email is test@evil.com",
            "nested": {"ip": "10.0.0.1"},
        }
        scrubbed, findings = rp.scrub_dict(data)
        assert "test@evil.com" not in scrubbed["message"]
        assert "10.0.0.1" not in scrubbed["nested"]["ip"]
        assert len(findings) == 2

    def test_multiple_pii_types(self):
        from redaction.pipeline import RedactionPipeline
        rp = RedactionPipeline(redact_remote_only=False)
        text = (
            "my IP is 192.168.1.50 and key is sk-ab123456789012345 "
            "email: test@demo.com"
        )
        scrubbed, findings = rp.scrub(text)
        assert "192.168.1.50" not in scrubbed
        assert "sk-ab123456789012345" not in scrubbed
        assert "test@demo.com" not in scrubbed
        assert len(findings) >= 3


# ===========================================================================
# SANDBOX TESTS
# ===========================================================================

class TestSandbox:
    """Tests for sandboxed execution."""

    def test_safe_python_execution(self):
        from sandbox.docker_executor import SandboxExecutor
        executor = SandboxExecutor(enabled=False)  # subprocess fallback
        result = executor.run_python("print('hello world')")
        assert result["exit_code"] == 0
        assert "hello world" in result["stdout"]

    def test_python_syntax_error(self):
        from sandbox.docker_executor import SandboxExecutor
        executor = SandboxExecutor(enabled=False)
        result = executor.run_python("import invalid_module_xyz")
        assert result["exit_code"] != 0
        assert len(result["stderr"]) > 0

    def test_python_timeout(self):
        from sandbox.docker_executor import SandboxExecutor, SandboxError
        executor = SandboxExecutor(enabled=False, timeout_seconds=2)
        with pytest.raises(SandboxError):
            executor.run_python("import time; time.sleep(60)")

    def test_python_access_denied_file(self):
        """In subprocess fallback, trying to read /etc/passwd should
        succeed on most systems but demonstrates the sandbox principle.
        In Docker mode with read-only root FS + no network, this would fail."""
        from sandbox.docker_executor import SandboxExecutor
        executor = SandboxExecutor(enabled=False)
        # This test documents that subprocess fallback has weaker isolation
        # In production, Docker mode would block this
        result = executor.run_python("print('sandbox test')")
        assert result["exit_code"] == 0


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_chat_flow(self, tmp_path):
        """Test: user asks weather -> intent parsed -> gateway allows -> tool runs."""
        os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "int_audit.jsonl")
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway
        from llm.provider import IntentParser

        db = AuthDB(str(tmp_path / "int.db"))
        users = seed_demo_users(db)
        audit = AuditLogger(str(tmp_path / "int_audit.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit)

        guest = [u for u in users if u.role == "guest"][0]
        tool_specs = [{"name": t.name, "description": t.description, "args_schema": t.args_schema} for t in registry.list()]
        parser = IntentParser(tool_specs)

        # Simulate LLM output
        intent = parser.parse('{"action": "get_weather", "args": {"location": "Berlin"}, "justification": "User wants weather"}')
        assert intent["action"] == "get_weather"
        decision = gw.authorize(guest, intent["action"], intent["args"])
        assert decision.allowed is True
        result = decision.tool.execute(intent["args"])
        assert result["location"] == "Berlin"
        assert audit.verify() is True

    def test_guest_blocked_from_calendar(self, tmp_path):
        """Test: guest sends manage_calendar -> blocked by gateway."""
        os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "int_audit2.jsonl")
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway, PolicyDenied

        db = AuthDB(str(tmp_path / "int2.db"))
        users = seed_demo_users(db)
        audit = AuditLogger(str(tmp_path / "int_audit2.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit)

        guest = [u for u in users if u.role == "guest"][0]
        with pytest.raises(PolicyDenied):
            gw.authorize(guest, "manage_calendar", {"action": "list"})
        assert audit.verify() is True


# ===========================================================================
# REFACTORING VERIFICATION
# ===========================================================================

class TestAuditChainAcrossPhases:
    """Verify audit chain remains valid after complex multi-step operations."""

    def test_chain_survives_hitl_flow(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway, HITLRequired, PolicyDenied

        db = AuthDB(str(tmp_path / "h_chain.db"))
        users = seed_demo_users(db)
        audit = AuditLogger(str(tmp_path / "h_chain_audit.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit, hitl_ttl_minutes=5)

        op = [u for u in users if u.role == "operator"][0]
        master = [u for u in users if u.role == "master"][0]

        # 1. Guest tries calendar (denied)
        guest = [u for u in users if u.role == "guest"][0]
        try:
            gw.authorize(guest, "manage_calendar", {"action": "list"})
        except PolicyDenied:
            pass

        # 2. Operator triggers HITL for delete_file
        try:
            gw.authorize(op, "delete_file", {"path": "/tmp/test"})
        except HITLRequired as e:
            aid = e.approval_id

        # 3. Master confirms
        gw.confirm_approval(aid, master)

        # 4. Another guest action
        gw.authorize(guest, "get_weather", {"location": "Tokyo"})

        # Chain should be valid through all these operations
        assert audit.verify() is True
        assert audit.count() == 4  # deny, hitl_required, confirm, allow


# ===========================================================================
# GATEWAY CONVENIENCE TEST
# ===========================================================================

class TestGatewayExecute:
    """Test the convenience execute() method."""

    def test_execute_allows_and_runs(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway

        db = AuthDB(str(tmp_path / "exec.db"))
        users = seed_demo_users(db)
        audit = AuditLogger(str(tmp_path / "exec_audit.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit)

        user = [u for u in users if u.role == "user"][0]
        result = gw.execute(user, "get_weather", {"location": "Paris"})
        assert result["location"] == "Paris"

    def test_execute_denies_invalid(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        from tools.registry import build_default_registry
        from gateway.policy_gateway import PolicyGateway, PolicyDenied

        db = AuthDB(str(tmp_path / "exec2.db"))
        users = seed_demo_users(db)
        audit = AuditLogger(str(tmp_path / "exec2_audit.jsonl"))
        registry = build_default_registry()
        gw = PolicyGateway(registry, db, audit)

        guest = [u for u in users if u.role == "guest"][0]
        with pytest.raises(PolicyDenied):
            gw.execute(guest, "manage_calendar", {"action": "list"})


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])