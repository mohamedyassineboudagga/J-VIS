"""LLM Integration — pluggable provider with strict intent schema.

The model MUST output ONLY a JSON intent:
    {"action": "<tool_name>", "args": {...}, "justification": "..."}

This module handles:
    - Provider abstraction (Ollama, OpenAI-compatible)
    - System prompt enforcement
    - Response parsing and validation
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def generate(self, messages: list[Dict[str, str]], temperature: float = 0.0) -> str:
        """Generate a completion. Returns the raw response text."""
        ...


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, messages: list[Dict[str, str]], temperature: float = 0.0) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: list[Dict[str, str]], temperature: float = 0.0) -> str:
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"OpenAI API error: {e.response.status_code} — {e.response.text[:200]}")


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
INTENT_SYSTEM_PROMPT = """You are J-VIS, a voice-first AI assistant.

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object. No other text.
2. The JSON must have exactly this schema:
   {"action": "<tool_name>", "args": {...}, "justification": "<brief explanation>"}
3. If the user's request doesn't match any tool, output:
   {"action": "unknown", "args": {}, "justification": "No matching tool found"}
4. NEVER execute anything directly. You only analyze and classify.

Available tools:
{tool_list}

Examples:
User: "What's the weather in Tokyo?"
{{"action": "get_weather", "args": {{"location": "Tokyo"}}, "justification": "User asked about weather in Tokyo."}}

User: "Search for latest Python news"
{{"action": "web_search", "args": {{"query": "latest Python news"}}, "justification": "User wants to search the web."}}

User: "Delete all my files"
{{"action": "delete_file", "args": {{"path": "all"}}, "justification": "User requested deletion (dangerous — may be blocked by policy)."}}

User: "Hello!"
{{"action": "unknown", "args": {}, "justification": "Greeting — no tool action needed."}}
"""


# ------------------------------------------------------------------
# Intent parser
# ------------------------------------------------------------------
class IntentParser:
    """Parse and validate LLM output into structured intents."""

    def __init__(self, tools: list[Dict[str, Any]]):
        self.tool_names = [t["name"] for t in tools]
        self.tool_descriptions = tools

    def build_system_prompt(self) -> str:
        tool_list = "\n".join(
            f"  - {t['name']}: {t.get('description', '')} "
            f"(args: {json.dumps(t.get('args_schema', {}))})"
            for t in self.tool_descriptions
        )
        return INTENT_SYSTEM_PROMPT.format(tool_list=tool_list)

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """Parse LLM output into a validated intent dict."""
        # Try to extract JSON from the response
        intent = self._extract_json(raw_response)

        if intent is None:
            return {
                "action": "unknown",
                "args": {},
                "justification": "Could not parse LLM response as valid JSON.",
                "_raw": raw_response,
            }

        # Validate structure
        action = intent.get("action", "unknown")
        args = intent.get("args", {})
        justification = intent.get("justification", "")

        if not isinstance(args, dict):
            args = {}

        # Check if action is a known tool
        if action != "unknown" and action not in self.tool_names:
            return {
                "action": "unknown",
                "args": {},
                "justification": (
                    f"LLM suggested action '{action}' which is not in the "
                    f"tool registry. Known tools: {', '.join(self.tool_names)}"
                ),
            }

        return {
            "action": action,
            "args": args,
            "justification": justification,
        }

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON from the text, even if wrapped in markdown."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON block in markdown
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        # Try to find first { ... } block
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        return None


# ------------------------------------------------------------------
# Provider factory
# ------------------------------------------------------------------
def get_llm_provider() -> LLMProvider:
    """Create the appropriate LLM provider from environment config."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai" or provider == "openai-compatible":
        return OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
        )
    else:
        # Default to Ollama
        return OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
        )