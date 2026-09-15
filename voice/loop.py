"""Voice Loop — Wake word -> STT -> Intent -> Gateway -> Response.

Components:
    - Wake word detection (openwakeword or simple energy detector)
    - Speech-to-Text (faster-whisper, local)
    - Text-to-Speech (edge-tts or pyttsx3)

All optional dependencies — the voice loop degrades gracefully to
text-only mode when audio libraries are unavailable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class WakeWordDetector:
    """Detect a wake word in audio.

    Uses openwakeword if available; falls back to a simple energy
    detector (detects speech activity, not a specific word).
    """

    WAKE_WORD = "jarvis"

    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word.lower()
        self._model = None

    def is_available(self) -> bool:
        try:
            import openwakeword  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from openwakeword.model import Model
            self._model = Model()
        return self._model

    def detect(self, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """Detect wake word in raw audio bytes."""
        if self.is_available():
            try:
                import numpy as np
                model = self._load_model()
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                prediction = model.predict(audio)
                # Check if any wake word exceeded threshold
                for word, score in prediction.items():
                    if score > 0.5:
                        return True
                return False
            except Exception:
                return self._energy_detect(audio_data)
        return self._energy_detect(audio_data)

    def _energy_detect(self, audio_data: bytes) -> bool:
        """Simple energy-based speech detection fallback."""
        if len(audio_data) < 100:
            return False
        # Compute RMS energy
        samples = audio_data[: len(audio_data) - (len(audio_data) % 2)]
        import struct
        values = struct.unpack(f"<{len(samples)//2}h", samples)
        rms = (sum(v * v for v in values) / len(values)) ** 0.5
        return rms > 500  # heuristic threshold


class SpeechToText:
    """Speech-to-Text using faster-whisper (local)."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        if not self.is_available():
            return ""
        try:
            model = self._load_model()
            segments, _ = model.transcribe(audio_path)
            return " ".join(seg.text.strip() for seg in segments)
        except Exception:
            return ""


class TextToSpeech:
    """Text-to-Speech using edge-tts (preferred) or pyttsx3 (fallback)."""

    def __init__(self, voice: str = "en-US-JennyNeural"):
        self.voice = voice

    def is_available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            try:
                import pyttsx3  # noqa: F401
                return True
            except ImportError:
                return False

    def speak(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Convert text to speech. Returns output file path if saved."""
        if not self.is_available():
            return None

        try:
            import edge_tts
            import asyncio

            if output_path is None:
                output_path = os.path.join(tempfile.gettempdir(), "jvis_tts.mp3")

            async def _synthesize():
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(output_path)

            asyncio.run(_synthesize())
            return output_path
        except Exception:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                return None
            except Exception:
                return None


class VoiceLoop:
    """Full voice loop: wake word -> STT -> intent -> gateway -> TTS."""

    def __init__(
        self,
        intent_handler: Callable[[str], Dict[str, Any]],
        wake_word: str = "jarvis",
    ):
        self.wake_word_detector = WakeWordDetector(wake_word)
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.intent_handler = intent_handler

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process an audio file through the full voice loop."""
        # 1. Transcribe
        text = self.stt.transcribe(audio_path)
        if not text:
            return {"status": "error", "error": "STT unavailable or no speech detected"}

        # 2. Handle intent
        result = self.intent_handler(text)

        # 3. Synthesize response
        response_text = result.get("response", "I processed your request.")
        tts_path = self.tts.speak(response_text)

        return {
            "status": "ok",
            "transcribed_text": text,
            "intent_result": result,
            "tts_output": tts_path,
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text through the intent handler (no audio)."""
        result = self.intent_handler(text)
        response_text = result.get("response", "I processed your request.")
        tts_path = self.tts.speak(response_text)
        return {
            "status": "ok",
            "transcribed_text": text,
            "intent_result": result,
            "tts_output": tts_path,
        }