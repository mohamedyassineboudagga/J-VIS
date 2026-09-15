"""Hash-chained immutable audit logger.

Each audit entry is a JSONL line containing:
    {ts, user_id, role, action, decision, context_hash, prev_hash, entry_hash}

The chain is verified at startup; any mismatch means the log was tampered
with and the system refuses to boot (per security directive).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class AuditLogError(Exception):
    """Raised when the audit chain is invalid or cannot be written."""


class AuditLogger:
    """Append-only, hash-chained audit logger.

    Each entry's hash covers the previous entry's hash, forming a chain.
    verify() recomputes the chain and detects any tampering.
    """

    def __init__(self, log_path: str = "audit/jvis_audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._load_last_hash()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_last_hash(self) -> str:
        """Read the last entry's hash from the log file, if any."""
        if not self.log_path.exists():
            return hashlib.sha256(b"J-VIS-GENESIS").hexdigest()
        entries = self._read_entries()
        if not entries:
            return hashlib.sha256(b"J-VIS-GENESIS").hexdigest()
        return entries[-1]["entry_hash"]

    def _read_entries(self) -> list[Dict[str, Any]]:
        """Read all entries from the log file."""
        entries: list[Dict[str, Any]] = []
        if not self.log_path.exists():
            return entries
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    raise AuditLogError(
                        f"Corrupt audit log entry (not valid JSON): {line[:80]}"
                    )
        return entries

    def _compute_hash(self, payload: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a canonical JSON payload."""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log(
        self,
        user_id: str,
        role: str,
        action: str,
        decision: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a new audit entry. Returns the entry dict."""
        context = context or {}
        details = details or {}

        with self._lock:
            ts = datetime.now(timezone.utc).isoformat()
            context_hash = self._compute_hash(context)

            entry: Dict[str, Any] = {
                "ts": ts,
                "user_id": user_id,
                "role": role,
                "action": action,
                "decision": decision,
                "context_hash": context_hash,
                "prev_hash": self._last_hash,
                "details": details,
            }
            entry["entry_hash"] = self._compute_hash(
                {k: v for k, v in entry.items() if k != "entry_hash"}
            )

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            self._last_hash = entry["entry_hash"]
            return entry

    def verify(self) -> bool:
        """Verify the entire hash chain. Returns True if intact."""
        entries = self._read_entries()
        prev_hash = hashlib.sha256(b"J-VIS-GENESIS").hexdigest()

        for entry in entries:
            if entry.get("prev_hash") != prev_hash:
                return False
            computed = self._compute_hash(
                {k: v for k, v in entry.items() if k != "entry_hash"}
            )
            if computed != entry.get("entry_hash"):
                return False
            prev_hash = entry["entry_hash"]
        return True

    def verify_or_raise(self) -> None:
        """Verify the chain; raise AuditLogError if tampered."""
        if not self.verify():
            raise AuditLogError(
                "Audit log hash chain verification FAILED. "
                "The log may have been tampered with. Refusing to boot."
            )

    def get_entries(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """Return entries, newest first."""
        entries = self._read_entries()
        entries.reverse()
        if limit is not None:
            entries = entries[:limit]
        return entries

    def count(self) -> int:
        return len(self._read_entries())


# Module-level singleton for convenience
_default_logger: Optional[AuditLogger] = None


def get_audit_logger(log_path: Optional[str] = None) -> AuditLogger:
    """Get the shared audit logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger(log_path or os.getenv("AUDIT_LOG_PATH", "audit/jvis_audit.jsonl"))
    return _default_logger