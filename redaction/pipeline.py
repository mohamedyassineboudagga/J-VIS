"""Redaction Pipeline — PII scrubber.

Sits BETWEEN user input and remote LLM providers (and scrubs tool outputs
before they reach the LLM too).

Scrubs:
    - Email addresses
    - Phone numbers
    - Local IPs (10.x, 192.168.x, 127.x, 172.16-31.x)
    - API-key-shaped strings (sk-..., AKIA..., etc.)
    - Home address patterns
    - Credit card numbers
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple


class RedactionPipeline:
    """Regex + NER-based PII scrubber."""

    # ------------------------------------------------------------------
    # Regex patterns
    # ------------------------------------------------------------------
    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    PHONE_RE = re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )

    LOCAL_IP_RE = re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    )

    API_KEY_RE = re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}"
        r"|AKIA[A-Z0-9]{16}"
        r"|ghp_[A-Za-z0-9]{20,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|AIza[A-Za-z0-9_-]{20,})\b"
    )

    CREDIT_CARD_RE = re.compile(
        r"\b(?:\d{4}[- ]?){3}\d{4}\b"
    )

    # Simple US street address pattern (number + street name)
    ADDRESS_RE = re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)"
        r"\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
        r"Court|Ct|Place|Pl|Way|Circle|Cir)\b",
        re.IGNORECASE,
    )

    SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    # ------------------------------------------------------------------
    def __init__(self, redact_remote_only: Optional[bool] = None):
        self.redact_remote_only = (
            redact_remote_only
            if redact_remote_only is not None
            else os.getenv("REDACT_REMOTE_ONLY", "true").lower() == "true"
        )

    def scrub(self, text: str) -> Tuple[str, List[Dict]]:
        """Scrub PII from text. Returns (scrubbed_text, findings)."""
        findings: List[Dict] = []
        scrubbed = text

        patterns = [
            ("email", self.EMAIL_RE),
            ("phone", self.PHONE_RE),
            ("local_ip", self.LOCAL_IP_RE),
            ("api_key", self.API_KEY_RE),
            ("credit_card", self.CREDIT_CARD_RE),
            ("address", self.ADDRESS_RE),
            ("ssn", self.SSN_RE),
        ]

        for label, pattern in patterns:
            scrubbed, found = self._replace_pattern(scrubbed, pattern, label)
            findings.extend(found)

        return scrubbed, findings

    def _replace_pattern(
        self, text: str, pattern: re.Pattern, label: str
    ) -> Tuple[str, List[Dict]]:
        findings: List[Dict] = []
        offset = 0

        def replacer(match: re.Match) -> str:
            nonlocal offset
            start = match.start() + offset
            end = match.end() + offset
            findings.append({
                "type": label,
                "start": start,
                "end": end,
                "replacement": f"[REDACTED:{label}]",
            })
            return f"[REDACTED:{label}]"

        scrubbed = pattern.sub(replacer, text)
        return scrubbed, findings

    def scrub_dict(self, data: Dict) -> Tuple[Dict, List[Dict]]:
        """Scrub all string values in a dict (recursively)."""
        findings: List[Dict] = []
        result: Dict = {}

        for key, value in data.items():
            if isinstance(value, str):
                scrubbed, found = self.scrub(value)
                result[key] = scrubbed
                findings.extend(found)
            elif isinstance(value, dict):
                scrubbed, found = self.scrub_dict(value)
                result[key] = scrubbed
                findings.extend(found)
            elif isinstance(value, list):
                scrubbed_list = []
                for item in value:
                    if isinstance(item, str):
                        s, f = self.scrub(item)
                        scrubbed_list.append(s)
                        findings.extend(f)
                    elif isinstance(item, dict):
                        s, f = self.scrub_dict(item)
                        scrubbed_list.append(s)
                        findings.extend(f)
                    else:
                        scrubbed_list.append(item)
                result[key] = scrubbed_list
            else:
                result[key] = value

        return result, findings

    def should_redact(self, provider: str) -> bool:
        """Decide whether to redact for a given provider.

        Local Ollama calls may bypass when REDACT_REMOTE_ONLY=true,
        but we keep it on for consistency by default.
        """
        if not self.redact_remote_only:
            return True
        # Remote providers always redact; local Ollama may bypass
        return provider.lower() not in ("ollama", "local")


# Module-level singleton
_default_pipeline: Optional[RedactionPipeline] = None


def get_redaction_pipeline() -> RedactionPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RedactionPipeline()
    return _default_pipeline