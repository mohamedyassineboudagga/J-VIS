"""Policy Gateway — the deterministic enforcement layer.

SECURITY PRIME DIRECTIVE: The LLM NEVER executes anything directly.
Every action passes through this gateway. No exceptions.

The gateway:
    1. Resolves the user's role from the session.
    2. Looks up the requested action in the tool registry.
    3. Checks role tier >= required role tier.
    4. Checks permission type against the RBAC policy.
    5. Enforces HITL for dangerous tools.
    6. Logs every decision to the audit log.
    7. Default-deny: unknown actions are denied and logged.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from auth.models import ROLE_TIER, AuthDB, Session, User
from audit.logger import AuditLogger
from tools.registry import Tool, ToolRegistry


class PolicyDenied(Exception):
    """Raised when the gateway denies an action."""

    def __init__(self, message: str, reason: str = "denied"):
        super().__init__(message)
        self.reason = reason
        self.message = message


class HITLRequired(Exception):
    """Raised when an action requires human confirmation."""

    def __init__(self, approval_id: str, action: str, message: str):
        super().__init__(message)
        self.approval_id = approval_id
        self.action = action
        self.message = message


@dataclass
class GatewayDecision:
    allowed: bool
    action: str
    user_id: str
    role: str
    reason: str = ""
    hitl_required: bool = False
    approval_id: Optional[str] = None
    tool: Optional[Tool] = None


class PolicyGateway:
    """Deterministic policy enforcement for all tool actions."""

    def __init__(
        self,
        registry: ToolRegistry,
        auth_db: AuthDB,
        audit: AuditLogger,
        hitl_ttl_minutes: int = 5,
    ):
        self.registry = registry
        self.auth_db = auth_db
        self.audit = audit
        self.hitl_ttl_minutes = hitl_ttl_minutes

    # ------------------------------------------------------------------
    # Core authorization
    # ------------------------------------------------------------------
    def authorize(
        self,
        user: User,
        action: str,
        args: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> GatewayDecision:
        """Authorize an action for a user. Returns a decision.

        Raises PolicyDenied on denial (after logging).
        Raises HITLRequired when confirmation is needed.
        """
        args = args or {}

        # 1. Default deny: unknown action
        tool = self.registry.get(action)
        if tool is None:
            self._log(user, action, "deny", {"reason": "unknown_action", "args": args})
            raise PolicyDenied(
                f"Action '{action}' is not recognized. I can only perform "
                "actions from my approved tool list.",
                reason="unknown_action",
            )

        # 2. Role tier check
        if user.tier < ROLE_TIER.get(tool.required_role, 0):
            self._log(user, action, "deny", {
                "reason": "insufficient_role",
                "required_role": tool.required_role,
                "user_role": user.role,
                "args": args,
            })
            raise PolicyDenied(
                f"I'm sorry, but your role ({user.role}) doesn't have permission "
                f"to {action}. This action requires the '{tool.required_role}' "
                "tier or higher.",
                reason="insufficient_role",
            )

        # 3. Permission check (read/write/execute/backup)
        if not self._check_permission(user.role, tool.required_permission):
            self._log(user, action, "deny", {
                "reason": "insufficient_permission",
                "required_permission": tool.required_permission,
                "user_role": user.role,
                "args": args,
            })
            raise PolicyDenied(
                f"Your role ({user.role}) lacks the '{tool.required_permission}' "
                f"permission required for {action}.",
                reason="insufficient_permission",
            )

        # 4. HITL check for dangerous tools
        if tool.hitl:
            approval_id = self._create_approval(user, action, args)
            self._log(user, action, "hitl_required", {
                "approval_id": approval_id,
                "args": args,
            })
            raise HITLRequired(
                approval_id=approval_id,
                action=action,
                message=(
                    f"Action '{action}' is dangerous and requires human "
                    f"confirmation. Approval ID: {approval_id}. "
                    f"Reply CONFIRM {approval_id} within {self.hitl_ttl_minutes} "
                    "minutes to proceed."
                ),
            )

        # 5. Allowed
        self._log(user, action, "allow", {"args": args})
        return GatewayDecision(
            allowed=True,
            action=action,
            user_id=user.id,
            role=user.role,
            reason="allowed",
            tool=tool,
        )

    # ------------------------------------------------------------------
    # HITL approval flow
    # ------------------------------------------------------------------
    def _create_approval(self, user: User, action: str, args: Dict) -> str:
        approval_id = uuid.uuid4().hex[:12].upper()
        self.auth_db.create_approval(
            approval_id=approval_id,
            user_id=user.id,
            action=action,
            args=args,
            ttl_minutes=self.hitl_ttl_minutes,
        )
        return approval_id

    def confirm_approval(self, approval_id: str, user: User) -> GatewayDecision:
        """Confirm a pending approval. Returns the decision to execute."""
        approval = self.auth_db.get_approval(approval_id)
        if approval is None:
            self._log(user, "approve", "deny", {"reason": "unknown_approval", "approval_id": approval_id})
            raise PolicyDenied(
                f"No pending approval found with ID {approval_id}.",
                reason="unknown_approval",
            )

        if approval["status"] != "pending":
            self._log(user, "approve", "deny", {
                "reason": "already_decided",
                "approval_id": approval_id,
                "status": approval["status"],
            })
            raise PolicyDenied(
                f"Approval {approval_id} has already been {approval['status']}.",
                reason="already_decided",
            )

        # TTL check
        expires_at = datetime.fromisoformat(approval["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            self.auth_db.decide_approval(approval_id, "expired")
            self._log(user, "approve", "deny", {
                "reason": "expired",
                "approval_id": approval_id,
            })
            raise PolicyDenied(
                f"Approval {approval_id} has expired (TTL exceeded). "
                "Please request the action again.",
                reason="expired",
            )

        # Only the requesting user (or higher tier) may confirm
        if approval["user_id"] != user.id and user.role != "master":
            self._log(user, "approve", "deny", {
                "reason": "not_requester",
                "approval_id": approval_id,
            })
            raise PolicyDenied(
                "Only the user who requested this action (or a master) "
                "may confirm it.",
                reason="not_requester",
            )

        self.auth_db.decide_approval(approval_id, "confirmed")
        self._log(user, "approve", "allow", {"approval_id": approval_id})

        tool = self.registry.get(approval["action"])
        return GatewayDecision(
            allowed=True,
            action=approval["action"],
            user_id=user.id,
            role=user.role,
            reason="hitl_confirmed",
            tool=tool,
        )

    def deny_approval(self, approval_id: str, user: User) -> None:
        """Explicitly deny a pending approval."""
        approval = self.auth_db.get_approval(approval_id)
        if approval is None:
            raise PolicyDenied(f"No pending approval found with ID {approval_id}.")
        if approval["status"] != "pending":
            raise PolicyDenied(f"Approval {approval_id} has already been {approval['status']}.")
        self.auth_db.decide_approval(approval_id, "denied")
        self._log(user, "approve", "deny", {
            "reason": "explicit_denial",
            "approval_id": approval_id,
        })

    def expire_stale_approvals(self) -> int:
        """Expire any pending approvals past their TTL. Returns count expired."""
        expired = 0
        for approval in self.auth_db.list_pending_approvals():
            expires_at = datetime.fromisoformat(approval["expires_at"])
            if datetime.now(timezone.utc) > expires_at:
                self.auth_db.decide_approval(approval["id"], "expired")
                expired += 1
        return expired

    # ------------------------------------------------------------------
    # Permission matrix
    # ------------------------------------------------------------------
    def _check_permission(self, role: str, permission: str) -> bool:
        """Check if a role has a given permission type.

        Permission hierarchy:
            master   -> everything
            operator -> read, write, execute, backup
            user     -> read, write
            guest    -> read
        """
        perms = {
            "master": {"read", "write", "execute", "backup", "admin"},
            "operator": {"read", "write", "execute", "backup"},
            "user": {"read", "write"},
            "guest": {"read"},
        }
        return permission in perms.get(role, set())

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------
    def _log(self, user: User, action: str, decision: str, context: Dict) -> None:
        self.audit.log(
            user_id=user.id,
            role=user.role,
            action=action,
            decision=decision,
            context=context,
        )

    # ------------------------------------------------------------------
    # Convenience: full request flow
    # ------------------------------------------------------------------
    def execute(
        self,
        user: User,
        action: str,
        args: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Authorize and execute an action in one call.

        Returns the tool result. Raises PolicyDenied or HITLRequired.
        """
        decision = self.authorize(user, action, args, session)
        if decision.tool is None:
            raise PolicyDenied(f"Action '{action}' not found.")
        result = decision.tool.execute(args or {})
        self.audit.log(
            user_id=user.id,
            role=user.role,
            action=action,
            decision="executed",
            context={"result_summary": str(result)[:200]},
        )
        return result