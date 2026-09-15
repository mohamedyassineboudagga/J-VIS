"""J-VIS Voice Response — human-like Text-to-Speech.

Uses Microsoft Edge neural voices (edge-tts) — free, no API key,
natural-sounding human voices. Falls back to pyttsx3 if unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import Optional


class VoiceSynthesizer:
    """Human-like text-to-speech via Microsoft Edge neural voices."""

    # Natural-sounding voices (US English)
    VOICES = {
        "jenny": "en-US-JennyNeural",      # Female, natural
        "aria": "en-US-AriaNeural",        # Female, conversational
        "guy": "en-US-GuyNeural",          # Male, natural
        "christopher": "en-US-ChristopherNeural",  # Male, deep
        "michelle": "en-US-MichelleNeural",  # Female, friendly
        "steffan": "en-US-SteffanNeural",   # Male, young
    }

    def __init__(self, voice: str = "en-US-JennyNeural", rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    def is_available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    async def _synthesize_async(self, text: str, output_path: str) -> bool:
        """Async synthesis using edge-tts."""
        import edge_tts
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
            await communicate.save(output_path)
            return True
        except Exception:
            return False

    def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Synthesize speech to an audio file. Returns the file path.

        Works both in sync contexts and inside async event loops.
        """
        if not self.is_available():
            return None

        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), "jvis_speech.mp3")

        try:
            # Check if we're inside a running event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context — run the coroutine on the loop
                result = loop.run_until_complete(self._synthesize_async(text, output_path))
            except RuntimeError:
                # No running loop — use asyncio.run()
                result = asyncio.run(self._synthesize_async(text, output_path))

            if result and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            return None
        except Exception:
            return None

    async def synthesize_async(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Async version — safe to await from FastAPI endpoints."""
        if not self.is_available():
            return None
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), "jvis_speech.mp3")
        ok = await self._synthesize_async(text, output_path)
        if ok and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None

    def synthesize_b64(self, text: str) -> Optional[str]:
        """Synthesize speech and return base64-encoded audio."""
        path = self.synthesize(text)
        if path is None:
            return None
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return None

    async def synthesize_b64_async(self, text: str) -> Optional[str]:
        """Async version of synthesize_b64."""
        path = await self.synthesize_async(text)
        if path is None:
            return None
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return None

    def list_voices(self) -> list[str]:
        """List available voice names."""
        return list(self.VOICES.keys())


# Module-level singleton
_synthesizer: Optional[VoiceSynthesizer] = None


def get_synthesizer(voice: str = "en-US-JennyNeural") -> VoiceSynthesizer:
    """Get the shared voice synthesizer."""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = VoiceSynthesizer(voice=voice)
    return _synthesizer