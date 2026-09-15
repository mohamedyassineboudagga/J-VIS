"""Local intent classifier — rule-based, works WITHOUT any LLM.

Maps natural language to tool intents using keyword + pattern matching.
This is the fallback when Ollama/OpenAI is unavailable, and it makes
the assistant usable out of the box.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class LocalIntentClassifier:
    """Deterministic, offline intent classification."""

    # Pattern: (regex, action, arg_extractor)
    PATTERNS: List[tuple] = [
        # --- Weather ---
        (
            re.compile(r"(?:what'?s|what is|how'?s|how is|tell me).*weather.*(?:in|at|for)\s+([\w\s,]+?)(?:\?|$)", re.IGNORECASE),
            "get_weather",
            lambda m: {"location": m.group(1).strip()},
        ),
        (
            re.compile(r"weather\s+(?:in|at|for)\s+([\w\s,]+?)(?:\?|$)", re.IGNORECASE),
            "get_weather",
            lambda m: {"location": m.group(1).strip()},
        ),
        (
            re.compile(r"weather", re.IGNORECASE),
            "get_weather",
            lambda m: {"location": "Unknown"},
        ),

        # --- Web search ---
        (
            re.compile(r"(?:search|google|look up|find|look for)\s+(?:for\s+)?(?:the\s+)?(.+?)(?:\?|$)", re.IGNORECASE),
            "web_search",
            lambda m: {"query": m.group(1).strip()},
        ),
        (
            re.compile(r"search", re.IGNORECASE),
            "web_search",
            lambda m: {"query": "general search"},
        ),

        # --- Music ---
        (
            re.compile(r"(?:play|listen to)\s+(?:some\s+|the\s+)?(.+?)(?:\?|$)", re.IGNORECASE),
            "play_music",
            lambda m: {"query": m.group(1).strip()},
        ),
        (
            re.compile(r"music|song|playlist", re.IGNORECASE),
            "play_music",
            lambda m: {"query": "music"},
        ),

        # --- List files ---
        (
            re.compile(r"(?:list|show|what'?s in|what is in)\s+(?:the\s+)?(?:files|directory|folder|dir)\s*(?:in\s+)?([\w:\\/.\s-]*)$", re.IGNORECASE),
            "list_files",
            lambda m: {"path": m.group(1).strip() or "."},
        ),
        (
            re.compile(r"list files|show files|what files", re.IGNORECASE),
            "list_files",
            lambda m: {"path": "."},
        ),

        # --- Read file ---
        (
            re.compile(r"(?:read|open|show me|display)\s+(?:the\s+)?(?:file|content of|contents of)\s+([\w:\\/.\s-]+?)(?:\?|$)", re.IGNORECASE),
            "read_file",
            lambda m: {"path": m.group(1).strip()},
        ),
        (
            re.compile(r"read file", re.IGNORECASE),
            "read_file",
            lambda m: {"path": "."},
        ),

        # --- Calendar ---
        (
            re.compile(r"(?:add|create|schedule|set up)\s+(?:a\s+)?(?:calendar\s+)?(?:event|appointment|meeting)", re.IGNORECASE),
            "manage_calendar",
            lambda m: {"action": "create", "event_details": {"title": "New event"}},
        ),
        (
            re.compile(r"(?:show|list|what'?s on|what is on)\s+(?:my\s+)?calendar", re.IGNORECASE),
            "manage_calendar",
            lambda m: {"action": "list"},
        ),
        (
            re.compile(r"(?:delete|cancel|remove)\s+(?:a\s+)?(?:calendar\s+)?(?:event|appointment|meeting)", re.IGNORECASE),
            "manage_calendar",
            lambda m: {"action": "delete", "event_details": {}},
        ),
        (
            re.compile(r"calendar", re.IGNORECASE),
            "manage_calendar",
            lambda m: {"action": "list"},
        ),

        # --- Backup ---
        (
            re.compile(r"(?:run|execute|start|do)\s+(?:a\s+|the\s+)?backup", re.IGNORECASE),
            "run_backup_script",
            lambda m: {"script_name": "daily_backup"},
        ),
        (
            re.compile(r"backup", re.IGNORECASE),
            "run_backup_script",
            lambda m: {"script_name": "daily_backup"},
        ),

        # --- Delete file (DANGEROUS) ---
        (
            re.compile(r"(?:delete|remove|erase|destroy)\s+(?:all\s+)?(?:my\s+)?(?:files|file|data)", re.IGNORECASE),
            "delete_file",
            lambda m: {"path": "all"},
        ),
        (
            re.compile(r"(?:delete|remove|erase)\s+(?:the\s+)?(?:file\s+)?([\w:\\/.\s-]+?)(?:\?|$)", re.IGNORECASE),
            "delete_file",
            lambda m: {"path": m.group(1).strip()},
        ),

        # --- Reboot (DANGEROUS) ---
        (
            re.compile(r"(?:reboot|restart|shut ?down)\s+(?:the\s+)?(?:server|system|computer|machine)", re.IGNORECASE),
            "reboot_server",
            lambda m: {},
        ),
        (
            re.compile(r"reboot|restart server", re.IGNORECASE),
            "reboot_server",
            lambda m: {},
        ),

        # --- Python snippet ---
        (
            re.compile(r"(?:run|execute)\s+(?:a\s+|this\s+)?python", re.IGNORECASE),
            "run_python_snippet",
            lambda m: {"code": "print('hello from J-VIS')"},
        ),
        (
            re.compile(r"python code|python snippet", re.IGNORECASE),
            "run_python_snippet",
            lambda m: {"code": "print('hello from J-VIS')"},
        ),
    ]

    # Greetings / small talk — no tool needed
    GREETINGS = re.compile(
        r"^(hi|hello|hey|good morning|good afternoon|good evening|how are you|what'?s up|yo|sup)[\s!.,]*$",
        re.IGNORECASE,
    )

    THANKS = re.compile(r"^(thanks|thank you|thx|ty)[\s!.,]*$", re.IGNORECASE)

    BYE = re.compile(r"^(bye|goodbye|see you|good night|gtg)[\s!.,]*$", re.IGNORECASE)

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a message into an intent. Returns {action, args, justification}."""
        text = text.strip()

        # Small talk
        if self.GREETINGS.match(text):
            return {
                "action": "greeting",
                "args": {},
                "justification": "User greeted the assistant.",
            }
        if self.THANKS.match(text):
            return {
                "action": "thanks",
                "args": {},
                "justification": "User thanked the assistant.",
            }
        if self.BYE.match(text):
            return {
                "action": "bye",
                "args": {},
                "justification": "User said goodbye.",
            }

        # Tool patterns
        for pattern, action, extractor in self.PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    args = extractor(match)
                except Exception:
                    args = {}
                return {
                    "action": action,
                    "args": args,
                    "justification": f"Matched pattern for '{action}'.",
                }

        # Fallback: unknown
        return {
            "action": "unknown",
            "args": {},
            "justification": "No matching tool found for this message.",
        }

    def help_text(self) -> str:
        """Return a human-readable list of what the assistant can do."""
        return (
            "I can help you with:\n"
            "  - Weather: 'what's the weather in Tokyo?'\n"
            "  - Web search: 'search for latest AI news'\n"
            "  - Music: 'play some jazz'\n"
            "  - Files: 'list files', 'read file C:/path'\n"
            "  - Calendar: 'add a calendar event', 'show my calendar'\n"
            "  - Backup: 'run a backup'\n"
            "  - Python: 'run a python snippet'\n"
            "  - System: 'reboot the server' (requires master + confirmation)\n"
            "  - Delete: 'delete file C:/path' (requires operator + confirmation)"
        )


# Module-level singleton
_classifier: Optional[LocalIntentClassifier] = None


def get_local_classifier() -> LocalIntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LocalIntentClassifier()
    return _classifier