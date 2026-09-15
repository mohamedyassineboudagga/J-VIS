"""J-VIS Persistent Power Mode.

"change my power" flips J-VIS into RED POWER MODE. The mode is persisted to
a small JSON file so it survives restarts and is enforced server-side, not
just as a transient in-browser toggle.

Modes:
    normal  — default operation (cyan theme, router decides engine)
    red     — RED POWER MODE (red theme, opencode-only routing)
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional

MODES = ("normal", "red")

_lock = threading.Lock()


def _default_path() -> str:
    return os.getenv("POWER_MODE_PATH", "jvis_power_mode.json")


def _read(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and data.get("mode") in MODES:
            return data
    except (OSError, ValueError):
        pass
    return {"mode": "normal"}


def _write(path: str, mode: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"mode": mode}, fh, indent=2)
    os.replace(tmp, path)


def get_power_mode(path: Optional[str] = None) -> str:
    """Return the persisted mode ('normal' or 'red')."""
    if path is None:
        path = _default_path()
    with _lock:
        return _read(path).get("mode", "normal")


def set_power_mode(mode: str, path: Optional[str] = None) -> str:
    """Persist a new power mode and return it. Raises ValueError on bad input."""
    if path is None:
        path = _default_path()
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise ValueError(f"Invalid power mode: {mode!r}. Must be one of {MODES}.")
    with _lock:
        _write(path, mode)
    return mode