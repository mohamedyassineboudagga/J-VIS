"""Tool registry — every executable action is registered here.

Each tool has:
    name           — unique identifier
    description    — human-readable description
    required_role  — minimum role tier to invoke
    required_permission — permission type (read/write/execute/backup)
    hitl           — whether Human-in-the-Loop confirmation is required
    sandbox        — whether execution must happen inside a sandbox
    args_schema    — pydantic-compatible argument schema
    executor       — callable that performs the action
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ValidationError, create_model


class ToolError(Exception):
    """Raised when a tool fails to execute."""


@dataclass
class Tool:
    name: str
    description: str
    required_role: str
    required_permission: str
    hitl: bool
    sandbox: bool
    args_schema: Dict[str, Any]
    executor: Callable[..., Any]

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate args against the schema using pydantic."""
        fields = {}
        for arg_name, spec in self.args_schema.items():
            arg_type = _resolve_type(spec.get("type", "string"))
            required = spec.get("required", False)
            default = ... if required else spec.get("default", None)
            fields[arg_name] = (arg_type, default)

        model = create_model(f"Tool_{self.name}_Args", **fields)
        try:
            validated = model(**args)
            return validated.model_dump()
        except ValidationError as e:
            raise ToolError(f"Invalid arguments for {self.name}: {e}")

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with validated args."""
        validated = self.validate_args(args)
        result = self.executor(**validated)
        if isinstance(result, dict):
            return result
        return {"result": result}


def _resolve_type(type_name: str):
    """Map JSON schema type names to Python types."""
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(type_name, str)


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def load_from_manifest(self, manifest_path: str, executors: Dict[str, Callable]) -> None:
        """Load tools from a JSON manifest, wiring up executors."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for spec in manifest["tools"]:
            name = spec["name"]
            if name not in executors:
                raise ValueError(f"No executor provided for tool: {name}")
            self.register(
                Tool(
                    name=name,
                    description=spec.get("description", ""),
                    required_role=spec.get("required_role", "guest"),
                    required_permission=spec.get("required_permission", "read"),
                    hitl=spec.get("hitl", False),
                    sandbox=spec.get("sandbox", False),
                    args_schema=spec.get("args_schema", {}),
                    executor=executors[name],
                )
            )


# ----------------------------------------------------------------------
# Default tool executors (Phase 1 stubs — real implementations later)
# ----------------------------------------------------------------------

def _get_weather(location: str) -> Dict[str, Any]:
    """Stub: return mock weather data."""
    return {
        "location": location,
        "temperature_c": 21,
        "condition": "partly cloudy",
        "humidity": 55,
        "note": "Mock data — real weather API integration pending (TODO).",
    }


def _web_search(query: str) -> Dict[str, Any]:
    """Stub: return a mock search result."""
    return {
        "query": query,
        "results": [
            {"title": f"Mock result for '{query}'", "url": "https://example.com/1"},
            {"title": f"Second mock result for '{query}'", "url": "https://example.com/2"},
        ],
        "note": "Mock data — real search API integration pending (TODO).",
    }


def _play_music(query: str) -> Dict[str, Any]:
    """Stub: return a mock music playback response."""
    return {
        "query": query,
        "status": "queued",
        "note": "Mock data — real music provider integration pending (TODO).",
    }


def _list_files(path: str = ".") -> Dict[str, Any]:
    """List files in a directory (sandboxed in production)."""
    p = Path(path)
    if not p.exists():
        raise ToolError(f"Path does not exist: {path}")
    if not p.is_dir():
        raise ToolError(f"Not a directory: {path}")
    entries = []
    for child in sorted(p.iterdir()):
        entries.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "size": child.stat().st_size if child.is_file() else None,
        })
    return {"path": str(p.resolve()), "entries": entries}


def _read_file(path: str) -> Dict[str, Any]:
    """Read a file's contents (sandboxed in production)."""
    p = Path(path)
    if not p.exists():
        raise ToolError(f"File does not exist: {path}")
    if not p.is_file():
        raise ToolError(f"Not a file: {path}")
    # Limit read size to avoid memory exhaustion
    if p.stat().st_size > 1_000_000:
        raise ToolError("File too large to read (max 1MB)")
    content = p.read_text(encoding="utf-8", errors="replace")
    return {"path": str(p.resolve()), "content": content}


def _manage_calendar(action: str, event_details: Optional[Dict] = None) -> Dict[str, Any]:
    """Stub: manage calendar events."""
    return {
        "action": action,
        "event_details": event_details or {},
        "status": "ok",
        "note": "Mock data — real calendar integration pending (TODO).",
    }


def _run_backup_script(script_name: str, target_path: Optional[str] = None) -> Dict[str, Any]:
    """Stub: run a backup script."""
    return {
        "script_name": script_name,
        "target_path": target_path or "default",
        "status": "backup_completed",
        "note": "Mock data — real backup execution pending (TODO).",
    }


def _delete_file(path: str) -> Dict[str, Any]:
    """Delete a file. NOTE: HITL confirmation is enforced by the gateway."""
    p = Path(path)
    if not p.exists():
        raise ToolError(f"File does not exist: {path}")
    if not p.is_file():
        raise ToolError(f"Not a file: {path}")
    p.unlink()
    return {"path": str(p.resolve()), "status": "deleted"}


def _reboot_server() -> Dict[str, Any]:
    """Reboot the server. NOTE: HITL confirmation is enforced by the gateway."""
    return {"status": "reboot_scheduled", "note": "Mock — real reboot pending (TODO)."}


def _run_python_snippet(code: str) -> Dict[str, Any]:
    """Run a Python snippet inside the sandbox."""
    from sandbox.docker_executor import SandboxExecutor

    executor = SandboxExecutor()
    return executor.run_python(code)


DEFAULT_EXECUTORS: Dict[str, Callable] = {
    "get_weather": _get_weather,
    "web_search": _web_search,
    "play_music": _play_music,
    "list_files": _list_files,
    "read_file": _read_file,
    "manage_calendar": _manage_calendar,
    "run_backup_script": _run_backup_script,
    "delete_file": _delete_file,
    "reboot_server": _reboot_server,
    "run_python_snippet": _run_python_snippet,
}


def build_default_registry(manifest_path: Optional[str] = None) -> ToolRegistry:
    """Build the default registry from the manifest + default executors."""
    registry = ToolRegistry()
    manifest = manifest_path or os.path.join(
        os.path.dirname(__file__), "..", "configs", "tool_manifest.json"
    )
    registry.load_from_manifest(manifest, DEFAULT_EXECUTORS)
    return registry