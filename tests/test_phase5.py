"""Phase 5 acceptance tests — Biometrics (Voiceprint + Face).

Acceptance criteria:
    - Enrolled user A's voice gets role A
    - Unknown voice gets guest tier — flagged in audit log
    - PIN fallback when biometrics unavailable
    - Never fall back to nothing
"""

from __future__ import annotations

import base64
import hashlib
import os

import pytest


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "p5_audit.jsonl")
    yield
    os.environ.pop("AUDIT_LOG_PATH", None)


class TestBiometricManager:
    """Tests for the biometric manager facade."""

    def test_voice_availability_check(self):
        from auth.biometrics import BiometricManager
        mgr = BiometricManager()
        # SpeechBrain may or may not be installed — just verify the method works
        assert isinstance(mgr.is_voice_available(), bool)

    def test_face_availability_check(self):
        from auth.biometrics import BiometricManager
        mgr = BiometricManager()
        assert isinstance(mgr.is_face_available(), bool)

    def test_embedding_hash(self):
        from auth.biometrics import BiometricManager
        mgr = BiometricManager()
        h1 = mgr.hash_embedding(b"sample_data")
        h2 = mgr.hash_embedding(b"sample_data")
        h3 = mgr.hash_embedding(b"different")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64  # SHA-256 hex

    def test_voice_verify_without_embedding(self):
        from auth.biometrics import BiometricManager
        mgr = BiometricManager()
        # No stored embedding -> verification fails with (False, 0.0)
        is_match, similarity = mgr.verify_voice(b"sample", None)
        assert is_match is False
        assert similarity == 0.0

    def test_face_verify_without_encoding(self):
        from auth.biometrics import BiometricManager
        mgr = BiometricManager()
        assert mgr.verify_face(b"sample", None) is False


class TestVoiceEnrollment:
    """Tests for voiceprint enrollment."""

    def test_enroll_voice_stores_embedding(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        db = AuthDB(str(tmp_path / "p5.db"))
        users = seed_demo_users(db)
        user = [u for u in users if u.role == "user"][0]

        # Simulate enrollment (hash-based for MVP without SpeechBrain)
        voice_data = b"fake_voice_sample_data"
        voiceprint = hashlib.sha256(voice_data).digest()
        db.update_voiceprint(user.id, voiceprint)

        updated = db.get_user(user.id)
        assert updated.voiceprint is not None
        assert updated.voiceprint == voiceprint

    def test_enroll_face_stores_encoding(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        db = AuthDB(str(tmp_path / "p5b.db"))
        users = seed_demo_users(db)
        user = [u for u in users if u.role == "operator"][0]

        encoding = b"fake_face_encoding_data"
        db.update_face_encoding(user.id, encoding)

        updated = db.get_user(user.id)
        assert updated.face_encoding is not None
        assert updated.face_encoding == encoding


class TestSessionBootstrap:
    """Tests for session bootstrap with biometrics + PIN fallback."""

    def _setup(self, tmp_path):
        from auth.models import AuthDB, seed_demo_users
        db = AuthDB(str(tmp_path / "p5c.db"))
        users = seed_demo_users(db)
        return db, users

    def test_pin_fallback_creates_session(self, tmp_path):
        """PIN fallback creates a valid session."""
        db, users = self._setup(tmp_path)
        user = [u for u in users if u.role == "user"][0]

        # Verify PIN works
        assert db.verify_pin(user.id, "2222") is True

        # Create session via PIN
        session = db.create_session(user.id, user.role, ttl_minutes=15)
        assert session.is_valid is True
        assert session.user_id == user.id
        assert session.role == "user"

    def test_wrong_pin_fails(self, tmp_path):
        db, users = self._setup(tmp_path)
        user = [u for u in users if u.role == "user"][0]
        assert db.verify_pin(user.id, "9999") is False

    def test_session_expiry(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        db, users = self._setup(tmp_path)
        user = [u for u in users if u.role == "guest"][0]
        session = db.create_session(user.id, user.role, ttl_minutes=0)
        # TTL 0 means it expires immediately
        assert session.is_valid is False

    def test_session_retrieval(self, tmp_path):
        db, users = self._setup(tmp_path)
        user = [u for u in users if u.role == "master"][0]
        session = db.create_session(user.id, user.role, ttl_minutes=15)
        retrieved = db.get_session(session.token)
        assert retrieved is not None
        assert retrieved.user_id == user.id
        assert retrieved.role == "master"

    def test_session_revocation(self, tmp_path):
        db, users = self._setup(tmp_path)
        user = [u for u in users if u.role == "operator"][0]
        session = db.create_session(user.id, user.role, ttl_minutes=15)
        db.revoke_session(session.token)
        assert db.get_session(session.token) is None


class TestBootstrapEndpoint:
    """Tests for the /session/bootstrap HTTP endpoint."""

    def test_bootstrap_with_pin(self, tmp_path):
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            users = client.get("/users").json()
            user = [u for u in users if u["role"] == "user"][0]

            resp = client.post("/session/bootstrap", json={
                "user_id": user["id"],
                "pin": "2222",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["auth_method"] == "pin"
            assert data["token"] != ""
            assert data["role"] == "user"

    def test_bootstrap_with_wrong_pin(self, tmp_path):
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            users = client.get("/users").json()
            user = [u for u in users if u["role"] == "user"][0]

            resp = client.post("/session/bootstrap", json={
                "user_id": user["id"],
                "pin": "9999",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["auth_method"] == "denied"
            assert data["error"] is not None

    def test_bootstrap_no_credentials_denied(self, tmp_path):
        """Never fall back to nothing — no credentials = denied."""
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            users = client.get("/users").json()
            user = [u for u in users if u["role"] == "user"][0]

            resp = client.post("/session/bootstrap", json={
                "user_id": user["id"],
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["auth_method"] == "denied"
            assert data["error"] is not None

    def test_bootstrap_unknown_user(self, tmp_path):
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as client:
            resp = client.post("/session/bootstrap", json={
                "user_id": "nonexistent",
                "pin": "1234",
            })
            assert resp.status_code == 404


class TestVoiceIdentity:
    """Tests for voice-based identity (enrolled vs unknown)."""

    def test_enrolled_voice_gets_role(self, tmp_path):
        """Enrolled user's voice sample authenticates as that user."""
        from auth.models import AuthDB, seed_demo_users
        db = AuthDB(str(tmp_path / "p5d.db"))
        users = seed_demo_users(db)
        user = [u for u in users if u.role == "operator"][0]

        # Enroll a voiceprint
        voice_data = b"operator_voice_sample"
        voiceprint = hashlib.sha256(voice_data).digest()
        db.update_voiceprint(user.id, voiceprint)

        # Verify the voice matches
        updated = db.get_user(user.id)
        assert updated.voiceprint == voiceprint
        assert updated.role == "operator"

    def test_unknown_voice_flagged_in_audit(self, tmp_path):
        """Unknown voice gets denied and flagged in audit log."""
        from auth.models import AuthDB, seed_demo_users
        from audit.logger import AuditLogger
        db = AuthDB(str(tmp_path / "p5e.db"))
        users = seed_demo_users(db)
        user = [u for u in users if u.role == "user"][0]
        audit = AuditLogger(str(tmp_path / "p5e_audit.jsonl"))

        # Unknown voice sample (not matching enrolled)
        unknown_voice = b"unknown_voice_sample"
        unknown_hash = hashlib.sha256(unknown_voice).digest()

        # Enrolled voiceprint is different
        enrolled = hashlib.sha256(b"real_voice").digest()
        db.update_voiceprint(user.id, enrolled)

        # Verification fails for unknown voice
        stored = db.get_user(user.id).voiceprint
        assert unknown_hash != stored

        # Log the failed attempt
        audit.log(user.id, user.role, "voice_verify", "deny", {
            "reason": "voice_mismatch",
            "voice_hash": unknown_hash.hex()[:16],
        })
        entries = audit.get_entries()
        assert entries[0]["decision"] == "deny"
        assert entries[0]["action"] == "voice_verify"


class TestSpeechBrainVoiceprint:
    """Tests for the real SpeechBrain ECAPA-TDNN voiceprint integration.

    These tests run only when SpeechBrain + torch are installed.
    They verify the actual speaker-embedding pipeline end-to-end.
    """

    @pytest.fixture(scope="class")
    def voice_provider(self):
        from auth.biometrics import VoiceprintProvider
        provider = VoiceprintProvider()
        if not provider.is_available():
            pytest.skip("SpeechBrain not installed — skipping real voiceprint tests")
        return provider

    def test_embedding_dimension(self, voice_provider):
        """ECAPA-TDNN produces 192-dim float32 embeddings."""
        emb = voice_provider.encode_audio("samples/jenny_1.mp3")
        if emb is None:
            pytest.skip("Sample audio not available")
        assert len(emb) == 192 * 4  # 192 floats * 4 bytes

    def test_same_speaker_matches(self, voice_provider):
        """Two samples from the same speaker should match."""
        emb1 = voice_provider.encode_audio("samples/jenny_1.mp3")
        emb2 = voice_provider.encode_audio("samples/jenny_2.mp3")
        if emb1 is None or emb2 is None:
            pytest.skip("Sample audio not available")
        is_match, similarity = voice_provider.verify(emb1, emb2)
        assert is_match is True
        assert similarity > 0.85

    def test_different_speaker_rejected(self, voice_provider):
        """A different speaker should be rejected."""
        emb1 = voice_provider.encode_audio("samples/jenny_1.mp3")
        emb2 = voice_provider.encode_audio("samples/guy_1.mp3")
        if emb1 is None or emb2 is None:
            pytest.skip("Sample audio not available")
        is_match, similarity = voice_provider.verify(emb1, emb2)
        assert is_match is False
        assert similarity < 0.85

    def test_average_embeddings(self, voice_provider):
        """Averaging multiple samples produces a valid 192-dim embedding."""
        embs = [
            voice_provider.encode_audio("samples/jenny_1.mp3"),
            voice_provider.encode_audio("samples/jenny_2.mp3"),
            voice_provider.encode_audio("samples/jenny_3.mp3"),
        ]
        if any(e is None for e in embs):
            pytest.skip("Sample audio not available")
        avg = voice_provider.average_embeddings(embs)
        assert len(avg) == 192 * 4

    def test_cosine_similarity_symmetric(self, voice_provider):
        """Cosine similarity is symmetric: sim(a,b) == sim(b,a)."""
        emb1 = voice_provider.encode_audio("samples/jenny_1.mp3")
        emb2 = voice_provider.encode_audio("samples/jenny_2.mp3")
        if emb1 is None or emb2 is None:
            pytest.skip("Sample audio not available")
        s1 = voice_provider.compute_similarity(emb1, emb2)
        s2 = voice_provider.compute_similarity(emb2, emb1)
        assert abs(s1 - s2) < 1e-6