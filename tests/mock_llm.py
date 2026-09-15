"""Mock LLM provider for testing without a real LLM backend.

This provider simulates intent classification so the full pipeline
(chat -> intent -> gateway -> tool) can be tested deterministically.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for tests.

    Maps known phrases to intents:
        - "weather" -> get_weather
        - "search"  -> web_search
        - "music"   -> play_music
        - "calendar" -> manage_calendar
        - "delete"  -> delete_file
        - "reboot"  -> reboot_server
        - "backup"  -> run_backup_script
        - "files"   -> list_files
        - "read"    -> read_file
        - "python"  -> run_python_snippet
        - anything else -> unknown
    """

    KEYWORD_MAP = [
        ("weather", "get_weather", {"location": "Unknown"}),
        ("search", "web_search", {"query": "search"}),
        ("music", "play_music", {"query": "music"}),
        ("calendar", "manage_calendar", {"action": "list"}),
        ("delete", "delete_file", {"path": "/tmp/mock.txt"}),
        ("reboot", "reboot_server", {}),
        ("backup", "run_backup_script", {"script_name": "daily"}),
        ("files", "list_files", {"path": "."}),
        ("read", "read_file", {"path": "/tmp/mock.txt"}),
        ("python", "run_python_snippet", {"code": "print('hi')"}),
    ]

    def __init__(self, responses: Dict[str, str] | None = None):
        self.responses = responses or {}

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Return a deterministic intent JSON based on the user message."""
        # Extract the last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # Check for explicit override
        if user_msg in self.responses:
            return self.responses[user_msg]

        # Keyword matching
        lowered = user_msg.lower()
        for keyword, action, args in self.KEYWORD_MAP:
            if keyword in lowered:
                return json.dumps({
                    "action": action,
                    "args": args,
                    "justification": f"Mock intent for keyword '{keyword}'",
                })

        return json.dumps({
            "action": "unknown",
            "args": {},
            "justification": "No matching tool found",
        })