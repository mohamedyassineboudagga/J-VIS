"""Voice enrollment & verification CLI using SpeechBrain ECAPA-TDNN.

Enroll a user's voiceprint from audio files, then verify a live sample
against the stored embedding.

USAGE:
    # Enroll a user from 3 voice samples (recommended for robustness)
    python scripts/voice_enroll.py enroll --user master_demo sample1.wav sample2.wav sample3.wav

    # Enroll from a single sample
    python scripts/voice_enroll.py enroll --user master_demo sample.wav

    # Verify a live sample against a stored voiceprint
    python scripts/voice_enroll.py verify --user master_demo live_sample.wav

    # Check who a voice sample belongs to (1:N identification)
    python scripts/voice_enroll.py identify sample.wav

    # Show enrollment status for all users
    python scripts/voice_enroll.py status

HOW IT WORKS:
    1. SpeechBrain's ECAPA-TDNN (trained on VoxCeleb) encodes each audio
       sample into a 192-dimensional speaker embedding.
    2. Enrollment: multiple samples are averaged into one reference
       embedding (more robust than a single sample).
    3. Verification: cosine similarity between live embedding and stored
       reference. >= 0.85 threshold = same speaker.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth.biometrics import BiometricManager, VoiceprintProvider  # noqa: E402
from auth.models import AuthDB  # noqa: E402

DEFAULT_THRESHOLD = 0.85


def get_db() -> AuthDB:
    db_path = os.getenv("AUTH_DB_PATH", "jvis.db")
    return AuthDB(db_path)


def cmd_enroll(args) -> int:
    """Enroll a user's voiceprint from one or more audio files."""
    db = get_db()
    user = db.get_user_by_username(args.user)
    if user is None:
        print(f"ERROR: User '{args.user}' not found. Available users:")
        for u in db.list_users():
            print(f"  - {u.username} ({u.role})")
        return 1

    voice = VoiceprintProvider()
    if not voice.is_available():
        print("ERROR: SpeechBrain is not installed. Run: pip install speechbrain torch")
        return 1

    print(f"Enrolling voiceprint for '{args.user}' ({user.role})...")
    print(f"  Model: {voice.MODEL_SOURCE}")
    print(f"  Samples: {len(args.samples)}")

    embeddings = []
    for i, sample_path in enumerate(args.samples, 1):
        if not os.path.exists(sample_path):
            print(f"  [{i}/{len(args.samples)}] SKIP: {sample_path} not found")
            continue
        print(f"  [{i}/{len(args.samples)}] Encoding {sample_path}...")
        emb = voice.encode_audio(sample_path)
        if emb is None:
            print(f"  [{i}/{len(args.samples)}] FAILED to encode {sample_path}")
            continue
        embeddings.append(emb)
        print(f"  [{i}/{len(args.samples)}] OK ({len(emb) // 4}-dim embedding)")

    if not embeddings:
        print("ERROR: No samples could be encoded.")
        return 1

    reference = voice.average_embeddings(embeddings)
    db.update_voiceprint(user.id, reference)

    print(f"\nSUCCESS: Voiceprint enrolled for '{args.user}'")
    print(f"  Embedding dim: {len(reference) // 4}")
    print(f"  Samples averaged: {len(embeddings)}")
    print(f"  Threshold: {DEFAULT_THRESHOLD} (cosine similarity)")
    return 0


def cmd_verify(args) -> int:
    """Verify a live voice sample against a stored voiceprint."""
    db = get_db()
    user = db.get_user_by_username(args.user)
    if user is None:
        print(f"ERROR: User '{args.user}' not found.")
        return 1

    if not user.voiceprint:
        print(f"ERROR: '{args.user}' has no enrolled voiceprint. Enroll first:")
        print(f"  python scripts/voice_enroll.py enroll --user {args.user} sample.wav")
        return 1

    voice = VoiceprintProvider()
    if not voice.is_available():
        print("ERROR: SpeechBrain is not installed.")
        return 1

    print(f"Verifying '{args.user}' against {args.sample}...")
    live_emb = voice.encode_audio(args.sample)
    if live_emb is None:
        print("ERROR: Failed to encode the live sample.")
        return 1

    is_match, similarity = voice.verify(live_emb, user.voiceprint, threshold=args.threshold)
    print(f"  Cosine similarity: {similarity:.4f}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Result: {'MATCH - voice verified' if is_match else 'NO MATCH - voice rejected'}")
    return 0 if is_match else 1


def cmd_identify(args) -> int:
    """Identify which enrolled user a voice sample belongs to (1:N)."""
    db = get_db()
    voice = VoiceprintProvider()
    if not voice.is_available():
        print("ERROR: SpeechBrain is not installed.")
        return 1

    print(f"Encoding {args.sample}...")
    live_emb = voice.encode_audio(args.sample)
    if live_emb is None:
        print("ERROR: Failed to encode the sample.")
        return 1

    print(f"\nComparing against {len(db.list_users())} enrolled users:")
    best_user = None
    best_score = 0.0
    for u in db.list_users():
        if not u.voiceprint:
            continue
        score = voice.compute_similarity(live_emb, u.voiceprint)
        status = "MATCH" if score >= args.threshold else ""
        print(f"  {u.username:20s} {u.role:10s} similarity={score:.4f} {status}")
        if score > best_score:
            best_score = score
            best_user = u

    if best_user and best_score >= args.threshold:
        print(f"\nIDENTIFIED: {best_user.username} ({best_user.role}) with similarity {best_score:.4f}")
        return 0
    print(f"\nNO MATCH above threshold {args.threshold}. Best: {best_user.username if best_user else 'n/a'} ({best_score:.4f})")
    return 1


def cmd_status(args) -> int:
    """Show voiceprint enrollment status for all users."""
    db = get_db()
    print(f"{'Username':<20} {'Role':<12} {'Voiceprint':<12} {'Embedding dim'}")
    print("-" * 60)
    for u in db.list_users():
        if u.voiceprint:
            dim = len(u.voiceprint) // 4
            status = "ENROLLED"
        else:
            dim = "-"
            status = "not enrolled"
        print(f"{u.username:<20} {u.role:<12} {status:<12} {dim}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SpeechBrain voice enrollment & verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Enroll a user's voiceprint from audio files")
    p_enroll.add_argument("--user", required=True, help="Username (e.g. master_demo)")
    p_enroll.add_argument("samples", nargs="+", help="Audio files (WAV/MP3/OGG)")
    p_enroll.set_defaults(func=cmd_enroll)

    p_verify = sub.add_parser("verify", help="Verify a live sample against a stored voiceprint")
    p_verify.add_argument("--user", required=True, help="Username")
    p_verify.add_argument("sample", help="Live audio sample file")
    p_verify.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_verify.set_defaults(func=cmd_verify)

    p_id = sub.add_parser("identify", help="Identify which user a voice belongs to (1:N)")
    p_id.add_argument("sample", help="Audio sample file")
    p_id.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_id.set_defaults(func=cmd_identify)

    p_status = sub.add_parser("status", help="Show enrollment status for all users")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()