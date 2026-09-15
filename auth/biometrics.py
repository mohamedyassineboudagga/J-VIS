"""Biometric authentication — Voiceprint + Face enrollment/verification.

Uses SpeechBrain ECAPA-TDNN for speaker verification and
OpenCV + face_recognition for face identification.

Fallback: PIN entry when biometrics are unavailable. NEVER fall back to nothing.

HOW SPEECHBRAIN VOICEPRINTING WORKS:
    1. ENROLLMENT: Take 3 voice samples → ECAPA-TDNN encodes each into a
       192-dimensional speaker embedding (a vector that captures the
       unique characteristics of the voice) → average the 3 embeddings
       into a single "reference embedding" → store in SQLite.
    2. VERIFICATION: Take a new voice sample → encode it into an embedding
       → compute cosine similarity against the stored reference.
       If similarity >= threshold (0.85) → MATCH (same speaker).
    3. IDENTITY: Once verified, look up the user from the embedding
       → load their role → create a session.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class BiometricError(Exception):
    """Raised when biometric verification fails."""


class BiometricProvider(ABC):
    """Abstract base for biometric verification."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider's dependencies are installed."""
        ...

    @abstractmethod
    def verify(self, sample, reference) -> bool:
        """Verify a sample against a reference. Returns True if match."""
        ...

    @abstractmethod
    def encode(self, sample) -> Optional[bytes]:
        """Encode a biometric sample into a storable embedding."""
        ...


# ------------------------------------------------------------------
# Voiceprint (SpeechBrain ECAPA-TDNN)
# ------------------------------------------------------------------
class VoiceprintProvider(BiometricProvider):
    """Speaker verification using SpeechBrain ECAPA-TDNN.

    ECAPA-TDNN is a deep neural network trained on the VoxCeleb dataset
    to produce 192-dimensional speaker embeddings. These embeddings
    capture the unique characteristics of a person's voice — pitch,
    timbre, cadence, and vocal tract geometry.
    """

    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    EMBEDDING_DIM = 192  # ECAPA-TDNN outputs 192-dim vectors
    DEFAULT_THRESHOLD = 0.85  # Cosine similarity threshold for "same speaker"

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        try:
            import speechbrain  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Load the ECAPA-TDNN model (downloads on first run, ~8MB)."""
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source=self.MODEL_SOURCE,
                run_opts={"device": "cpu"},
            )
        return self._model

    def encode_audio(self, audio_path: str) -> Optional[bytes]:
        """Encode a voice sample into a speaker embedding.

        Takes an audio file path (WAV, MP3, OGG, FLAC, etc.)
        Returns a 192-dimensional float32 embedding as bytes.
        """
        if not self.is_available():
            return None
        try:
            model = self._load_model()
            from speechbrain.dataio.dataio import read_audio
            signal = read_audio(audio_path)
            embeddings = model.encode_batch(signal)
            embedding = embeddings.squeeze().cpu().numpy().astype(np.float32)
            return embedding.tobytes()
        except Exception as e:
            print(f"SpeechBrain encoding failed: {e}")
            return None

    def encode_wav_bytes(self, wav_bytes: bytes) -> Optional[bytes]:
        """Encode raw WAV bytes into a speaker embedding.

        Converts raw PCM bytes (16-bit, 16kHz mono) to a temp WAV file,
        then runs SpeechBrain encoding.
        """
        if not self.is_available():
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                # Write WAV header + PCM data
                sample_rate = 16000
                num_channels = 1
                bits_per_sample = 16
                data_size = len(wav_bytes)
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<I", 16))  # chunk size
                f.write(struct.pack("<H", 1))   # PCM format
                f.write(struct.pack("<H", num_channels))
                f.write(struct.pack("<I", sample_rate))
                f.write(struct.pack("<I", sample_rate * num_channels * bits_per_sample // 8))
                f.write(struct.pack("<H", num_channels * bits_per_sample // 8))
                f.write(struct.pack("<H", bits_per_sample))
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                f.write(wav_bytes)
                f.flush()
                temp_path = f.name

            result = self.encode_audio(temp_path)
            os.unlink(temp_path)
            return result
        except Exception:
            return None

    def encode_b64(self, b64_audio: str) -> Optional[bytes]:
        """Encode base64-encoded audio into a speaker embedding."""
        if not self.is_available():
            return None
        try:
            import base64
            audio_data = base64.b64decode(b64_audio)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                f.flush()
                temp_path = f.name
            result = self.encode_audio(temp_path)
            os.unlink(temp_path)
            return result
        except Exception:
            return None

    def compute_similarity(self, emb1: bytes, emb2: bytes) -> float:
        """Compute cosine similarity between two embeddings. Returns 0.0-1.0."""
        try:
            a = np.frombuffer(emb1, dtype=np.float32)
            b = np.frombuffer(emb2, dtype=np.float32)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0

    def verify(
        self,
        sample: bytes,
        reference: bytes,
        threshold: float = 0.85,
    ) -> Tuple[bool, float]:
        """Verify a voice embedding against a stored reference.

        Returns (is_match, similarity_score).
        """
        if not self.is_available():
            return False, 0.0
        similarity = self.compute_similarity(sample, reference)
        return similarity >= threshold, similarity

    def average_embeddings(self, embeddings: List[bytes]) -> bytes:
        """Average multiple embeddings into a single reference embedding.

        Called during enrollment with 3 voice samples to create a
        robust reference that's more stable than any single sample.
        """
        if not embeddings:
            return b""
        if len(embeddings) == 1:
            return embeddings[0]

        vectors = [np.frombuffer(e, dtype=np.float32) for e in embeddings]
        avg = np.mean(vectors, axis=0).astype(np.float32)
        return avg.tobytes()

    def encode(self, sample) -> Optional[bytes]:
        if isinstance(sample, str):
            return self.encode_audio(sample)
        return None


# ------------------------------------------------------------------
# Face ID (OpenCV + face_recognition)
# ------------------------------------------------------------------
class FaceProvider(BiometricProvider):
    """Face identification using OpenCV + face_recognition."""

    def is_available(self) -> bool:
        try:
            import face_recognition  # noqa: F401
            return True
        except ImportError:
            return False

    def encode_face(self, image_path: str) -> Optional[bytes]:
        """Encode a face from an image file."""
        if not self.is_available():
            return None
        try:
            import face_recognition
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                return None
            return encodings[0].tobytes()
        except Exception:
            return None

    def verify(self, sample: bytes, reference: bytes, tolerance: float = 0.6) -> bool:
        """Verify a face encoding against a stored reference."""
        if not self.is_available():
            return False
        try:
            import face_recognition
            sample_arr = np.frombuffer(sample, dtype=np.float64)
            ref_arr = np.frombuffer(reference, dtype=np.float64)
            distance = face_recognition.face_distance([ref_arr], sample_arr)[0]
            return distance <= tolerance
        except Exception:
            return False

    def encode(self, sample) -> Optional[bytes]:
        if isinstance(sample, str):
            return self.encode_face(sample)
        return None


# ------------------------------------------------------------------
# Biometric Manager (facade)
# ------------------------------------------------------------------
class BiometricManager:
    """Manages voice + face verification with PIN fallback."""

    def __init__(self):
        self.voice = VoiceprintProvider()
        self.face = FaceProvider()

    def verify_voice(
        self, audio_sample: bytes, stored_embedding: Optional[bytes],
        threshold: float = 0.85,
    ) -> Tuple[bool, float]:
        """Verify voice against stored embedding.

        Returns (is_match, similarity_score).
        """
        if not self.voice.is_available():
            return False, 0.0
        if stored_embedding is None:
            return False, 0.0
        return self.voice.verify(audio_sample, stored_embedding, threshold)

    def verify_face(
        self, face_sample: bytes, stored_encoding: Optional[bytes]
    ) -> bool:
        """Verify face against stored encoding."""
        if not self.face.is_available():
            return False
        if stored_encoding is None:
            return False
        return self.face.verify(face_sample, stored_encoding)

    def is_voice_available(self) -> bool:
        return self.voice.is_available()

    def is_face_available(self) -> bool:
        return self.face.is_available()

    @staticmethod
    def hash_embedding(data: bytes) -> str:
        """Create a stable hash of a biometric embedding for session tokens."""
        return hashlib.sha256(data).hexdigest()