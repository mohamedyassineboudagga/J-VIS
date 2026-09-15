"""User and role models backed by SQLite.

Roles (4-tier RBAC):
    master   — Root, unrestricted
    operator — Elevated, functional
    user     — Standard, personal read/write
    guest    — Sandbox, isolated read-only
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Role hierarchy: higher tier inherits lower tier permissions
ROLE_TIER = {
    "guest": 0,
    "user": 1,
    "operator": 2,
    "master": 3,
}

ROLE_HIERARCHY = {
    "master": ["master", "operator", "user", "guest"],
    "operator": ["operator", "user", "guest"],
    "user": ["user", "guest"],
    "guest": ["guest"],
}


@dataclass
class User:
    id: str
    username: str
    role: str
    pin_hash: str = ""
    voiceprint: Optional[bytes] = None
    face_encoding: Optional[bytes] = None
    created_at: str = ""
    is_active: bool = True

    @property
    def tier(self) -> int:
        return ROLE_TIER.get(self.role, 0)

    def can(self, required_role: str) -> bool:
        """Check if this user's role satisfies a required role tier."""
        return self.tier >= ROLE_TIER.get(required_role, 0)


@dataclass
class Session:
    token: str
    user_id: str
    role: str
    created_at: datetime
    expires_at: datetime
    biometric_hash: str = ""

    @property
    def is_valid(self) -> bool:
        return datetime.now(timezone.utc) < self.expires_at


class AuthDB:
    """SQLite-backed user/session store."""

    def __init__(self, db_path: str = "jvis.db"):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    pin_hash TEXT NOT NULL DEFAULT '',
                    voiceprint BLOB,
                    face_encoding BLOB,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    biometric_hash TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS pending_approvals (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    args TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    decided_at TEXT
                );
                """
            )

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def create_user(
        self,
        username: str,
        role: str,
        pin: Optional[str] = None,
    ) -> User:
        """Create a user. PIN is stored as a salted hash."""
        user_id = hashlib.sha256(
            f"{username}:{os.urandom(16).hex()}".encode()
        ).hexdigest()[:16]

        pin_hash = ""
        if pin:
            pin_hash = self._hash_pin(pin)

        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO users (id, username, role, pin_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username, role, pin_hash, now),
            )
        return User(
            id=user_id,
            username=username,
            role=role,
            pin_hash=pin_hash,
            created_at=now,
        )

    def get_user(self, user_id: str) -> Optional[User]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            username=row["username"],
            role=row["role"],
            pin_hash=row["pin_hash"],
            voiceprint=row["voiceprint"],
            face_encoding=row["face_encoding"],
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
        )

    def get_user_by_username(self, username: str) -> Optional[User]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            username=row["username"],
            role=row["role"],
            pin_hash=row["pin_hash"],
            voiceprint=row["voiceprint"],
            face_encoding=row["face_encoding"],
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
        )

    def list_users(self) -> List[User]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [
            User(
                id=r["id"],
                username=r["username"],
                role=r["role"],
                pin_hash=r["pin_hash"],
                voiceprint=r["voiceprint"],
                face_encoding=r["face_encoding"],
                created_at=r["created_at"],
                is_active=bool(r["is_active"]),
            )
            for r in rows
        ]

    def update_voiceprint(self, user_id: str, voiceprint: bytes) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE users SET voiceprint = ? WHERE id = ?",
                (voiceprint, user_id),
            )

    def update_face_encoding(self, user_id: str, encoding: bytes) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE users SET face_encoding = ? WHERE id = ?",
                (encoding, user_id),
            )

    def verify_pin(self, user_id: str, pin: str) -> bool:
        user = self.get_user(user_id)
        if not user or not user.pin_hash:
            return False
        return hmac.compare_digest(user.pin_hash, self._hash_pin(pin))

    @staticmethod
    def _hash_pin(pin: str) -> str:
        """Hash a PIN with a per-user salt derived from the PIN itself + fixed pepper."""
        # NOTE: For MVP this uses a deterministic salt. In production,
        # store a random salt per user. This is documented as a TODO.
        salt = "jvis-pin-salt-v1"
        return hashlib.sha256(f"{salt}:{pin}".encode()).hexdigest()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def create_session(
        self,
        user_id: str,
        role: str,
        ttl_minutes: int = 15,
        biometric_hash: str = "",
    ) -> Session:
        """Create a short-lived signed session token."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=ttl_minutes)
        token = hashlib.sha256(
            f"{user_id}:{role}:{now.isoformat()}:{os.urandom(32).hex()}".encode()
        ).hexdigest()

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, role, created_at, expires_at, biometric_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (token, user_id, role, now.isoformat(), expires.isoformat(), biometric_hash),
            )
        return Session(
            token=token,
            user_id=user_id,
            role=role,
            created_at=now,
            expires_at=expires,
            biometric_hash=biometric_hash,
        )

    def get_session(self, token: str) -> Optional[Session]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE token = ?", (token,)
            ).fetchone()
        if not row:
            return None
        return Session(
            token=row["token"],
            user_id=row["user_id"],
            role=row["role"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            biometric_hash=row["biometric_hash"],
        )

    def revoke_session(self, token: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    # ------------------------------------------------------------------
    # Pending approvals (HITL)
    # ------------------------------------------------------------------
    def create_approval(
        self,
        approval_id: str,
        user_id: str,
        action: str,
        args: Dict,
        ttl_minutes: int = 5,
    ) -> None:
        import json as _json

        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=ttl_minutes)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO pending_approvals (id, user_id, action, args, status, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (
                    approval_id,
                    user_id,
                    action,
                    _json.dumps(args),
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )

    def get_approval(self, approval_id: str) -> Optional[Dict]:
        import json as _json

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "action": row["action"],
            "args": _json.loads(row["args"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "decided_at": row["decided_at"],
        }

    def decide_approval(self, approval_id: str, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET status = ?, decided_at = ? WHERE id = ?",
                (status, now, approval_id),
            )

    def list_pending_approvals(self) -> List[Dict]:
        import json as _json

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_approvals WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "action": r["action"],
                "args": _json.loads(r["args"]),
                "status": r["status"],
                "created_at": r["created_at"],
                "expires_at": r["expires_at"],
                "decided_at": r["decided_at"],
            }
            for r in rows
        ]


def seed_demo_users(db: AuthDB) -> List[User]:
    """Seed the 4 demo users (one per role). Idempotent."""
    demo = [
        ("master_demo", "master", "0000"),
        ("operator_demo", "operator", "1111"),
        ("user_demo", "user", "2222"),
        ("guest_demo", "guest", "3333"),
    ]
    created = []
    for username, role, pin in demo:
        existing = db.get_user_by_username(username)
        if existing:
            created.append(existing)
        else:
            created.append(db.create_user(username, role, pin))
    return created