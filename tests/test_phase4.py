"""Phase 4 acceptance tests — HITL (Human-in-the-Loop) Triggers.

Acceptance criteria:
    - delete_file and reboot_server CANNOT execute without confirmation
    - Both timeout and confirm paths are tested
    - Full flow is logged to audit
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p4_audit.jsonl")
    yield
    os.environ.pop("AUDIT_LOG_PATH", None)


def _build_hitl_stack(tmp_path, ttl_minutes=1):
    from auth.models import AuthDB, seed_demo_users
    from audit.logger import AuditLogger
    from tools.registry import build_default_registry
    from gateway.policy_gateway import PolicyGateway

    db = AuthDB(str(tmp_path / "p4.db"))
    users = seed_demo_users(db)
    audit = AuditLogger(str(tmp_path / "p4_audit.jsonl"))
    registry = build_default_registry()
    gw = PolicyGateway(registry, db, audit, hitl_ttl_minutes=ttl_minutes)
    return db, users, audit, registry, gw


class TestDeleteFileHITL:
    """delete_file requires HITL confirmation."""

    def test_delete_blocked_without_confirmation(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/important.txt"})
        assert exc_info.value.action == "delete_file"
        assert exc_info.value.approval_id is not None

    def test_delete_confirm_then_execute(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        op = [u for u in users if u.role == "operator"][0]

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/mock_file.txt"})
        aid = exc_info.value.approval_id

        decision = gw.confirm_approval(aid, op)
        assert decision.allowed is True
        assert decision.action == "delete_file"
        assert decision.tool is not None

    def test_delete_timeout_denies_execution(self, tmp_path):
        from gateway.policy_gateway import HITLRequired, PolicyDenied
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        op = [u for u in users if u.role == "operator"][0]

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/expiring.txt"})
        aid = exc_info.value.approval_id

        # Simulate TTL expiry
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with db._lock, db._connect() as conn:
            conn.execute("UPDATE pending_approvals SET expires_at = ? WHERE id = ?", (past, aid))

        with pytest.raises(PolicyDenied) as exc_info:
            gw.confirm_approval(aid, op)
        assert "expired" in str(exc_info.value).lower()

    def test_delete_only_requester_can_confirm(self, tmp_path):
        from gateway.policy_gateway import HITLRequired, PolicyDenied
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        op = [u for u in users if u.role == "operator"][0]
        other_op = db.create_user("other_op", "operator", "9999")

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/test"})
        aid = exc_info.value.approval_id

        # Different user tries to confirm -> denied
        with pytest.raises(PolicyDenied) as exc_info:
            gw.confirm_approval(aid, other_op)
        assert "Only the user" in str(exc_info.value)

        # Original requester can still confirm
        decision = gw.confirm_approval(aid, op)
        assert decision.allowed is True

    def test_deny_approval(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        op = [u for u in users if u.role == "operator"][0]

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/deny_test.txt"})
        aid = exc_info.value.approval_id

        gw.deny_approval(aid, op)
        approval = db.get_approval(aid)
        assert approval["status"] == "denied"


class TestRebootServerHITL:
    """reboot_server requires HITL confirmation — even for master."""

    def test_master_must_confirm_reboot(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        master = [u for u in users if u.role == "master"][0]

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(master, "reboot_server", {})
        assert exc_info.value.action == "reboot_server"

    def test_reboot_confirm_flow(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        master = [u for u in users if u.role == "master"][0]

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(master, "reboot_server", {})
        aid = exc_info.value.approval_id

        decision = gw.confirm_approval(aid, master)
        assert decision.allowed is True
        assert decision.action == "reboot_server"

    def test_reboot_timeout(self, tmp_path):
        from gateway.policy_gateway import HITLRequired, PolicyDenied
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        master = [u for u in users if u.role == "master"][0]

        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(master, "reboot_server", {})
        aid = exc_info.value.approval_id

        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with db._lock, db._connect() as conn:
            conn.execute("UPDATE pending_approvals SET expires_at = ? WHERE id = ?", (past, aid))

        with pytest.raises(PolicyDenied) as exc_info:
            gw.confirm_approval(aid, master)
        assert "expired" in str(exc_info.value).lower()


class TestHITLAuditLogging:
    """Every HITL decision is logged to the audit log."""

    def test_hitl_flow_logged(self, tmp_path):
        from gateway.policy_gateway import HITLRequired
        db, users, audit, registry, gw = _build_hitl_stack(tmp_path)
        op = [u for u in users if u.role == "operator"][0]

        # 1. HITL required
        with pytest.raises(HITLRequired) as exc_info:
            gw.authorize(op, "delete_file", {"path": "/tmp/test"})
        aid = exc_info.value.approval_id

        # 2. Confirm
        gw.confirm_approval(aid, op)

        # Verify audit entries
        entries = audit.get_entries()
        actions = [e["action"] for e in entries]
        decisions = [e["decision"] for e in entries]

        assert "delete_file" in actions
        assert "hitl_required" in decisions
        assert "approve" in actions
        assert "allow" in decisions
        assert audit.verify() is True


class TestHITLEndpoint:
    """Tests for the /approve HTTP endpoint."""

    def test_approve_endpoint_confirm(self, tmp_path):
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            # Get operator user from the app's own DB
            users = client.get("/users").json()
            op = [u for u in users if u["role"] == "operator"][0]

            # Trigger HITL via the app's gateway
            resp = client.post("/chat", json={
                "user_id": op["id"],
                "message": "delete",
                "use_llm": False,
            })
            # Direct mode returns unknown action, so trigger HITL via approvals
            # Instead, create an approval through the app's gateway by calling
            # the gateway directly with the app's DB
            from main import gateway as app_gateway
            from auth.models import AuthDB
            import os as _os
            db = AuthDB(_os.getenv("AUTH_DB_PATH", "jvis.db"))
            user = db.get_user(op["id"])
            from gateway.policy_gateway import HITLRequired
            # Create a real temp file to delete
            target = tmp_path / "ep_test.txt"
            target.write_text("test content")
            try:
                app_gateway.authorize(user, "delete_file", {"path": str(target)})
            except HITLRequired as e:
                aid = e.approval_id

            resp = client.post("/approve", json={
                "user_id": op["id"],
                "approval_id": aid,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "executed"