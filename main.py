"""J-VIS — Main FastAPI Application.

Endpoints:
    POST /chat            — Text chat (intent parsing -> gateway -> tool)
    POST /agent/task      — Delegate a mission to the opencode agent (J-VIS brain)
    POST /ollama/chat     — Ask the local Ollama model (fully local, no cloud)
    POST /router/decide   — J-VIS Router LLM: decides ollama vs opencode engine
    POST /enroll/voice    — Enroll voiceprint for a user
    POST /enroll/face     — Enroll face encoding for a user
    POST /approve         — Confirm a HITL approval
    GET  /audit/log       — View recent audit log entries
    GET  /audit/verify    — Verify audit log integrity
    GET  /health          — Health check
    GET  /users           — List users
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment
load_dotenv()

# ------------------------------------------------------------------
# Agent bridge configuration — J-VIS delegates missions to the
# opencode agent (the J-VIS brain) via the opencode CLI.
# ------------------------------------------------------------------
AGENT_CMD = os.getenv("AGENT_CMD", "opencode")
# Resolve the real executable (opencode is a .cmd shim on Windows).
import shutil as _shutil

_resolved_agent_cmd = _shutil.which(AGENT_CMD)
if _resolved_agent_cmd:
    AGENT_CMD = _resolved_agent_cmd
AGENT_WORKDIR = os.getenv("AGENT_WORKDIR", r"D:\J-VIS")
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))  # seconds
AGENT_AUTO_APPROVE = os.getenv("AGENT_AUTO_APPROVE", "true").lower() == "true"
AGENT_SYSTEM_PROMPT = os.getenv("AGENT_SYSTEM_PROMPT", "")  # optional extra context

# ------------------------------------------------------------------
# Power mode — "change my power" flips J-VIS into red power mode
# (red theme, opencode-only routing). State is persisted to a JSON file
# so the change survives restarts and is enforced server-side.
# ------------------------------------------------------------------
POWER_MODE_PATH = os.getenv("POWER_MODE_PATH", "jvis_power_mode.json")
POWER_TO_RED_COMMANDS = ("change my power", "red mode", "power mode", "disappear", "close yourself", "desaparece")
POWER_TO_NORMAL_COMMANDS = ("normal mode", "restore my power")

# ------------------------------------------------------------------
# Ollama bridge configuration — J-VIS consults the local Ollama model.
# Everything stays on the local host: J-VIS (localhost:8000) proxies to
# the Ollama API (localhost:11434). No cloud calls.
# ------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds
OLLAMA_SYSTEM = os.getenv(
    "OLLAMA_SYSTEM",
    "You are J-VIS, a helpful local AI assistant. Answer concisely and clearly.",
)

# ------------------------------------------------------------------
# J-VIS Router configuration — the custom LLM that decides whether a
# command goes to the local Ollama model or to the opencode agent.
# The model is created from D:\J-VIS\router\Modelfile (jvis-router).
# ------------------------------------------------------------------
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "jvis-router:latest")
ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "30"))  # seconds
ROUTER_DATASET = os.getenv("ROUTER_DATASET", r"D:\J-VIS\router\dataset.csv")
ROUTER_SYSTEM = os.getenv(
    "ROUTER_SYSTEM",
    "You are J-VIS Router. Read the user's command and decide which engine "
    "should handle it. Respond with ONLY a JSON object: "
    '{"engine": "ollama" or "opencode", "reason": "one short sentence"}. '
    "Questions, explanations, definitions, comparisons, advice and greetings "
    "go to ollama. Coding actions (create, write, build, fix, refactor, "
    "deploy, test, install, configure) and any mention of a programming "
    "language, framework, tool, database, DevOps, security, mobile, web or "
    "data/AI topic go to opencode.",
)

# Keyword fallback sets loaded from the dataset (used if the LLM is down).
_ROUTER_OLLAMA_KEYWORDS = set()
_ROUTER_OPENCODE_KEYWORDS = set()


def _load_router_keywords():
    """Load routing keywords from dataset.csv into in-memory sets."""
    global _ROUTER_OLLAMA_KEYWORDS, _ROUTER_OPENCODE_KEYWORDS
    try:
        import csv as _csv

        with open(ROUTER_DATASET, "r", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                kw = (row.get("keyword") or "").strip().lower()
                routing = (row.get("routing") or "").strip().lower()
                if not kw or kw == "question_mark":
                    continue
                if routing == "ollama":
                    _ROUTER_OLLAMA_KEYWORDS.add(kw)
                elif routing == "opencode":
                    _ROUTER_OPENCODE_KEYWORDS.add(kw)
        logger.info(
            f"Router dataset loaded: {len(_ROUTER_OLLAMA_KEYWORDS)} ollama, "
            f"{len(_ROUTER_OPENCODE_KEYWORDS)} opencode keywords"
        )
    except Exception as e:
        logger.warning(f"Router dataset load failed: {e}")


def _keyword_fallback(command: str) -> str:
    """Decide engine by keyword matching against the dataset (no LLM)."""
    t = command.lower()
    if t.endswith("?"):
        return "ollama"
    # Question words are strong signals — check them first.
    for kw in _ROUTER_OLLAMA_KEYWORDS:
        if kw in t:
            return "ollama"
    for kw in _ROUTER_OPENCODE_KEYWORDS:
        if kw in t:
            return "opencode"
    return "ollama"  # default: conversational


# Rate limiting middleware
from gateway.rate_limit import RateLimitMiddleware, get_rate_limit

# ------------------------------------------------------------------
# Module imports
# ------------------------------------------------------------------
from auth.models import AuthDB, ROLE_TIER, User, seed_demo_users
from audit.logger import AuditLogger, get_audit_logger
from gateway.policy_gateway import (
    HITLRequired,
    PolicyDenied,
    PolicyGateway,
)
from tools.registry import ToolRegistry, build_default_registry
from redaction.pipeline import RedactionPipeline, get_redaction_pipeline
from llm.provider import (
    IntentParser,
    get_llm_provider,
)
import power_mode

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jvis")

# ------------------------------------------------------------------
# Application state
# ------------------------------------------------------------------
auth_db: Optional[AuthDB] = None
audit: Optional[AuditLogger] = None
registry: Optional[ToolRegistry] = None
gateway: Optional[PolicyGateway] = None
intent_parser: Optional[IntentParser] = None
redaction: Optional[RedactionPipeline] = None


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    global auth_db, audit, registry, gateway, intent_parser, redaction

    logger.info("Starting J-VIS — Voice-first AI Assistant")

    # 1. Audit logger — verify chain integrity on boot
    audit = get_audit_logger()
    try:
        audit.verify_or_raise()
        logger.info("Audit log chain verified OK")
    except Exception as e:
        logger.error(f"CRITICAL: Audit log verification failed — {e}")
        # In production, we would refuse to boot here.
        # For development, log and continue with a warning.
        logger.warning("Continuing despite audit log failure (dev mode)")

    # 2. Auth database
    auth_db = AuthDB(db_path=os.getenv("AUTH_DB_PATH", "jvis.db"))
    demo_users = seed_demo_users(auth_db)
    logger.info(f"Seeded {len(demo_users)} demo users")

    # 3. Tool registry
    registry = build_default_registry()
    logger.info(f"Registered {len(registry.list())} tools")

    # 4. Policy gateway
    gateway = PolicyGateway(
        registry=registry,
        auth_db=auth_db,
        audit=audit,
        hitl_ttl_minutes=int(os.getenv("HITL_TTL_MINUTES", "5")),
    )

    # 5. Redaction pipeline
    redaction = get_redaction_pipeline()

    # 6. Intent parser
    tool_specs = [
        {
            "name": t.name,
            "description": t.description,
            "args_schema": t.args_schema,
        }
        for t in registry.list()
    ]
    intent_parser = IntentParser(tools=tool_specs)

    # 7. Router dataset — keyword fallback for the LLM router
    _load_router_keywords()

    logger.info("J-VIS startup complete")
    yield
    logger.info("J-VIS shutting down")


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="J-VIS",
    description="Voice-first AI Assistant with RBAC, Biometrics, Sandboxed Execution, and Audit Logging",
    version="0.1.0",
    lifespan=lifespan,
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, max_requests=get_rate_limit())

# Serve static assets (logo, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    use_llm: bool = True  # Set False to skip LLM and send raw intent


class ChatResponse(BaseModel):
    action: str
    args: Dict[str, Any]
    justification: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    hitl_required: bool = False
    approval_id: Optional[str] = None


class AgentTaskRequest(BaseModel):
    task: str
    user_id: Optional[str] = None


class AgentTaskResponse(BaseModel):
    result: str
    done: bool = True
    error: Optional[str] = None


class TranscribeRequest(BaseModel):
    audio_b64: str  # WAV audio, base64-encoded


class TranscribeResponse(BaseModel):
    text: str


class OllamaChatRequest(BaseModel):
    message: str
    system: Optional[str] = None


class OllamaChatResponse(BaseModel):
    reply: str = ""
    done: bool = False
    error: str = ""


class RouterDecideRequest(BaseModel):
    command: str


class RouterDecideResponse(BaseModel):
    engine: str = "ollama"
    reason: str = ""
    source: str = "llm"  # "llm" or "fallback"


class ApproveRequest(BaseModel):
    user_id: str
    approval_id: str


class ApproveResponse(BaseModel):
    approval_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PowerSetRequest(BaseModel):
    mode: str  # "normal" or "red"


class PowerStatusResponse(BaseModel):
    mode: str
    state: str


class PowerSetResponse(BaseModel):
    mode: str
    state: str
    message: str


class AuditEntry(BaseModel):
    ts: str
    user_id: str
    role: str
    action: str
    decision: str
    context_hash: str
    prev_hash: str
    entry_hash: str
    details: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    tools_registered: int
    audit_chain_valid: bool
    demo_users: int


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        tools_registered=len(registry.list()) if registry else 0,
        audit_chain_valid=audit.verify() if audit else False,
        demo_users=len(auth_db.list_users()) if auth_db else 0,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a chat message through the full pipeline:
    1. Load user + role
    2. Parse intent (LLM if available, else local classifier)
    3. Redact PII before LLM
    4. Authorize through gateway
    5. Execute tool (if allowed)
    6. Audit log everything
    """
    # 1. Load user
    user = auth_db.get_user(req.user_id) if auth_db else None
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {req.user_id}")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    # 2. Parse intent
    intent = None
    if req.use_llm:
        # Redact before sending to LLM
        scrubbed_msg, findings = redaction.scrub(req.message)

        llm = get_llm_provider()
        messages = [
            {"role": "system", "content": intent_parser.build_system_prompt()},
            {"role": "user", "content": scrubbed_msg},
        ]
        try:
            raw_response = llm.generate(messages)
            intent = intent_parser.parse(raw_response)
        except Exception as e:
            # LLM unavailable — fall back to local classifier
            audit.log(user.id, user.role, "chat", "llm_fallback", {"error": str(e)})
            from llm.local_intent import get_local_classifier
            intent = get_local_classifier().classify(req.message)

    if intent is None:
        # Local classifier (no LLM needed)
        from llm.local_intent import get_local_classifier
        intent = get_local_classifier().classify(req.message)

    action = intent.get("action", "unknown")
    args = intent.get("args", {})

    # 2b. Small talk — no tool needed
    if action in ("greeting", "thanks", "bye"):
        responses = {
            "greeting": "Hello! I'm J-VIS, your voice assistant. I can check the weather, search the web, play music, manage your calendar, and more. What would you like to do?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "bye": "Goodbye! I'll be here if you need me.",
        }
        audit.log(user.id, user.role, action, "allow", {"message": req.message[:100]})
        return ChatResponse(
            action=action,
            args={},
            justification=intent.get("justification", ""),
            result={"response": responses[action]},
        )

    # 2c. Help request
    if action == "unknown" and any(k in req.message.lower() for k in ["help", "what can you do", "what do you do", "capabilities"]):
        from llm.local_intent import get_local_classifier
        help_text = get_local_classifier().help_text()
        audit.log(user.id, user.role, "help", "allow", {})
        return ChatResponse(
            action="help",
            args={},
            justification="User asked what the assistant can do.",
            result={"response": help_text},
        )

    # 3. Gateway authorization
    try:
        decision = gateway.authorize(user, action, args)
        # Execute
        if decision.tool:
            result = decision.tool.execute(args)
        else:
            result = {}
        return ChatResponse(
            action=action,
            args=args,
            justification=intent.get("justification", ""),
            result=result,
        )
    except HITLRequired as e:
        return ChatResponse(
            action=action,
            args=args,
            justification=intent.get("justification", ""),
            hitl_required=True,
            approval_id=e.approval_id,
            error=e.message,
        )
    except PolicyDenied as e:
        return ChatResponse(
            action=action,
            args=args,
            justification=intent.get("justification", ""),
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}")
        audit.log(user.id, user.role, action, "error", {"error": str(e)})
        return ChatResponse(
            action=action,
            args=args,
            justification="",
            error=f"Unexpected error: {str(e)}",
        )


@app.post("/approve", response_model=ApproveResponse)
async def approve(req: ApproveRequest):
    """Confirm a HITL approval."""
    user = auth_db.get_user(req.user_id) if auth_db else None
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {req.user_id}")

    try:
        decision = gateway.confirm_approval(req.approval_id, user)
        if decision.tool:
            args = {}
            approval = auth_db.get_approval(req.approval_id)
            if approval:
                args = approval.get("args", {})
            result = decision.tool.execute(args)
            return ApproveResponse(
                approval_id=req.approval_id,
                status="executed",
                result=result,
            )
        return ApproveResponse(
            approval_id=req.approval_id,
            status="confirmed",
        )
    except PolicyDenied as e:
        return ApproveResponse(
            approval_id=req.approval_id,
            status="denied",
            error=str(e),
        )


@app.get("/audit/log", response_model=List[AuditEntry])
async def audit_log(limit: int = 50):
    """View recent audit log entries."""
    entries = audit.get_entries(limit=limit) if audit else []
    return entries


@app.get("/audit/verify")
async def audit_verify():
    """Verify the audit log hash chain integrity."""
    if audit is None:
        return {"valid": False, "error": "Audit logger not initialized"}
    try:
        valid = audit.verify()
        return {"valid": valid, "entry_count": audit.count()}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.get("/users")
async def list_users():
    """List all users (for admin/debug)."""
    users = auth_db.list_users() if auth_db else []
    return [
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "is_active": u.is_active,
            "has_voiceprint": u.voiceprint is not None,
            "has_face_encoding": u.face_encoding is not None,
        }
        for u in users
    ]


@app.get("/tools")
async def list_tools():
    """List all registered tools."""
    if registry is None:
        return []
    return [
        {
            "name": t.name,
            "description": t.description,
            "required_role": t.required_role,
            "required_permission": t.required_permission,
            "hitl": t.hitl,
            "sandbox": t.sandbox,
            "args_schema": t.args_schema,
        }
        for t in registry.list()
    ]


@app.get("/approvals")
async def list_pending_approvals():
    """List all pending HITL approvals."""
    if auth_db is None:
        return []
    return auth_db.list_pending_approvals()


# ------------------------------------------------------------------
# Power mode endpoints — "change my power" (persistent, server-side)
# ------------------------------------------------------------------
@app.get("/power/status", response_model=PowerStatusResponse)
async def power_status():
    """Return the current J-VIS power mode ('normal' or 'red')."""
    mode = power_mode.get_power_mode(POWER_MODE_PATH)
    return PowerStatusResponse(mode=mode, state=mode)


@app.post("/power/set", response_model=PowerSetResponse)
async def power_set(req: PowerSetRequest):
    """Persist a new power mode.

    'red'    -> RED POWER MODE (red theme, opencode-only routing)
    'normal' -> normal operation (cyan theme, router decides the engine)
    """
    mode = (req.mode or "").strip().lower()
    if mode not in ("normal", "red"):
        return PowerSetResponse(
            mode=mode,
            state="invalid",
            message=f"Invalid power mode {mode!r}. Use 'normal' or 'red'.",
        )
    power_mode.set_power_mode(mode, POWER_MODE_PATH)
    if auth_db:
        audit.log("system", "master", "set_power_mode", "allow", {"mode": mode})
    label = "RED POWER MODE activated — opencode-only routing" if mode == "red" else "Normal mode restored"
    return PowerSetResponse(mode=mode, state=mode, message=label)


# ------------------------------------------------------------------
# Biometric enrollment endpoints
# ------------------------------------------------------------------
from auth.biometrics import BiometricManager

biometric_mgr = BiometricManager()


class VoiceEnrollRequest(BaseModel):
    user_id: str
    voice_sample_b64: Optional[str] = None  # Single sample (backward compat)
    voice_samples_b64: Optional[List[str]] = None  # Multiple samples (averaged)


class FaceEnrollRequest(BaseModel):
    user_id: str
    face_image_path: str  # Path to face image file


class VoiceEnrollResponse(BaseModel):
    user_id: str
    enrolled: bool
    message: str


class FaceEnrollResponse(BaseModel):
    user_id: str
    enrolled: bool
    message: str


class SessionBootstrapRequest(BaseModel):
    user_id: str
    pin: Optional[str] = None
    voice_sample_b64: Optional[str] = None
    face_image_path: Optional[str] = None


class SessionBootstrapResponse(BaseModel):
    token: str
    user_id: str
    role: str
    expires_at: str
    auth_method: str
    error: Optional[str] = None
    voice_similarity: Optional[float] = None


@app.post("/enroll/voice", response_model=VoiceEnrollResponse)
async def enroll_voice(req: VoiceEnrollRequest):
    """Enroll a voiceprint for a user using SpeechBrain ECAPA-TDNN.

    How it works:
    1. Decode the base64 audio
    2. SpeechBrain ECAPA-TDNN encodes it into a 192-dim speaker embedding
    3. Store the embedding in the database
    4. If multiple samples provided, average them for a more robust reference
    """
    user = auth_db.get_user(req.user_id) if auth_db else None
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {req.user_id}")

    if not biometric_mgr.is_voice_available():
        audit.log(user.id, user.role, "enroll_voice", "error", {"error": "speechbrain_unavailable"})
        return VoiceEnrollResponse(
            user_id=user.id,
            enrolled=False,
            message="SpeechBrain is not installed. Install: pip install speechbrain torch",
        )

    try:
        # Collect all samples (single or multiple)
        samples = []
        if req.voice_sample_b64:
            samples.append(req.voice_sample_b64)
        if req.voice_samples_b64:
            samples.extend(req.voice_samples_b64)
        if not samples:
            return VoiceEnrollResponse(
                user_id=user.id,
                enrolled=False,
                message="No voice sample provided. Send voice_sample_b64 or voice_samples_b64.",
            )

        # SpeechBrain encode each sample into a 192-dim speaker embedding
        embeddings = []
        for i, sample_b64 in enumerate(samples, 1):
            emb = biometric_mgr.voice.encode_b64(sample_b64)
            if emb is None:
                return VoiceEnrollResponse(
                    user_id=user.id,
                    enrolled=False,
                    message=f"Failed to encode voice sample {i}. Ensure valid audio was provided (WAV, MP3, OGG).",
                )
            embeddings.append(emb)

        # Average multiple samples into one robust reference embedding
        if len(embeddings) > 1:
            embedding = biometric_mgr.voice.average_embeddings(embeddings)
        else:
            embedding = embeddings[0]

        # Store the real SpeechBrain embedding
        auth_db.update_voiceprint(user.id, embedding)

        audit.log(user.id, user.role, "enroll_voice", "allow",
            context={"samples": len(embeddings)},
            details={
                "status": "enrolled",
                "embedding_dim": len(embedding) // 4,  # float32 = 4 bytes each
                "samples_averaged": len(embeddings),
                "model": "ecapa-tdnn",
            })
        return VoiceEnrollResponse(
            user_id=user.id,
            enrolled=True,
            message=f"Voiceprint enrolled successfully using SpeechBrain ECAPA-TDNN ({len(embedding) // 4}-dim embedding, {len(embeddings)} sample(s)).",
        )
    except Exception as e:
        audit.log(user.id, user.role, "enroll_voice", "error", {"error": str(e)})
        return VoiceEnrollResponse(
            user_id=user.id,
            enrolled=False,
            message=f"Enrollment failed: {str(e)}",
        )


@app.post("/enroll/face", response_model=FaceEnrollResponse)
async def enroll_face(req: FaceEnrollRequest):
    """Enroll a face encoding for a user. Requires face_recognition + OpenCV."""
    user = auth_db.get_user(req.user_id) if auth_db else None
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {req.user_id}")

    if not biometric_mgr.is_face_available():
        audit.log(user.id, user.role, "enroll_face", "error", {"error": "face_recognition_unavailable"})
        return FaceEnrollResponse(
            user_id=user.id,
            enrolled=False,
            message="face_recognition is not installed. Install: pip install face-recognition opencv-python",
        )

    try:
        encoding = biometric_mgr.face.encode_face(req.face_image_path)
        if encoding is None:
            return FaceEnrollResponse(
                user_id=user.id,
                enrolled=False,
                message="No face detected in the image.",
            )
        auth_db.update_face_encoding(user.id, encoding)
        audit.log(user.id, user.role, "enroll_face", "allow", {"status": "enrolled"})
        return FaceEnrollResponse(
            user_id=user.id,
            enrolled=True,
            message="Face encoding enrolled successfully.",
        )
    except Exception as e:
        audit.log(user.id, user.role, "enroll_face", "error", {"error": str(e)})
        return FaceEnrollResponse(
            user_id=user.id,
            enrolled=False,
            message=f"Enrollment failed: {str(e)}",
        )


@app.post("/session/bootstrap", response_model=SessionBootstrapResponse)
async def bootstrap_session(req: SessionBootstrapRequest):
    """Bootstrap a session using biometrics or PIN fallback.

    Flow:
    1. Try voice verification (if sample provided + voiceprint enrolled)
    2. Try face verification (if image provided + face enrolled)
    3. Fall back to PIN (if provided)
    4. If biometrics unavailable and no PIN -> deny
    """
    user = auth_db.get_user(req.user_id) if auth_db else None
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {req.user_id}")

    auth_method = "unknown"
    voice_similarity = 0.0

    # 1. Voice verification (SpeechBrain ECAPA-TDNN cosine similarity)
    if req.voice_sample_b64 and user.voiceprint:
        if biometric_mgr.is_voice_available():
            # Encode the live sample into a speaker embedding
            live_embedding = biometric_mgr.voice.encode_b64(req.voice_sample_b64)
            if live_embedding is not None:
                is_match, similarity = biometric_mgr.verify_voice(
                    live_embedding, user.voiceprint
                )
                if is_match:
                    auth_method = "voiceprint"
                    voice_similarity = similarity

    # 2. Face verification
    if auth_method == "unknown" and req.face_image_path and user.face_encoding:
        if biometric_mgr.is_face_available():
            encoding = biometric_mgr.face.encode_face(req.face_image_path)
            if encoding and biometric_mgr.verify_face(encoding, user.face_encoding):
                auth_method = "face"

    # 3. PIN fallback
    if auth_method == "unknown" and req.pin:
        if auth_db.verify_pin(user.id, req.pin):
            auth_method = "pin"

    # 4. If biometrics unavailable, PIN is required
    if auth_method == "unknown":
        auth_method = "denied"
        audit.log(user.id, user.role, "session_bootstrap", "deny", {"reason": "auth_failed"})
        return SessionBootstrapResponse(
            token="",
            user_id=user.id,
            role=user.role,
            expires_at="",
            auth_method="denied",
            error="Authentication failed. Please provide valid biometrics or PIN.",
        )

    # Create session
    ttl = int(os.getenv("SESSION_TTL_MINUTES", "15"))
    session = auth_db.create_session(user.id, user.role, ttl_minutes=ttl)
    biometric_hash = hashlib.sha256(req.user_id.encode()).hexdigest() if auth_method in ("voiceprint", "face") else ""

    audit.log(user.id, user.role, "session_bootstrap", "allow",
        context={"auth_method": auth_method},
        details={
            "auth_method": auth_method,
            "voice_similarity": round(voice_similarity, 4) if auth_method == "voiceprint" else None,
        })
    return SessionBootstrapResponse(
        token=session.token,
        user_id=user.id,
        role=user.role,
        expires_at=session.expires_at.isoformat(),
        auth_method=auth_method,
        voice_similarity=round(voice_similarity, 4) if auth_method == "voiceprint" else None,
    )


# ------------------------------------------------------------------
# Voice response endpoints (human-like TTS)
# ------------------------------------------------------------------
from voice.tts import VoiceSynthesizer, get_synthesizer

synthesizer = get_synthesizer()


class SpeakRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"  # jenny, aria, guy, christopher, michelle, steffan
    rate: str = "+0%"
    pitch: str = "+0Hz"


class SpeakResponse(BaseModel):
    audio_b64: Optional[str] = None
    audio_path: Optional[str] = None
    voice: str
    error: Optional[str] = None


@app.post("/speak", response_model=SpeakResponse)
async def speak(req: SpeakRequest):
    """Convert text to human-like speech. Returns base64 MP3 audio."""
    if not synthesizer.is_available():
        return SpeakResponse(
            voice=req.voice,
            error="edge-tts is not installed. Run: pip install edge-tts",
        )

    # Map friendly voice names to edge-tts voice IDs
    voice_map = {
        "jenny": "en-US-JennyNeural",
        "aria": "en-US-AriaNeural",
        "guy": "en-US-GuyNeural",
        "christopher": "en-US-ChristopherNeural",
        "michelle": "en-US-MichelleNeural",
        "steffan": "en-US-SteffanNeural",
    }
    voice_id = voice_map.get(req.voice.lower(), req.voice)

    synth = VoiceSynthesizer(voice=voice_id, rate=req.rate, pitch=req.pitch)
    audio_b64 = await synth.synthesize_b64_async(req.text)
    if audio_b64 is None:
        return SpeakResponse(voice=req.voice, error="Speech synthesis failed")

    # Log the TTS request
    if auth_db:
        audit.log("tts", "system", "speak", "allow", {"text_length": len(req.text), "voice": voice_id})

    return SpeakResponse(audio_b64=audio_b64, voice=voice_id)


@app.get("/speak/audio")
async def speak_audio(text: str, voice: str = "en-US-JennyNeural"):
    """Get speech as a playable MP3 file (for <audio> tags)."""
    from fastapi.responses import FileResponse

    if not synthesizer.is_available():
        return {"error": "edge-tts not installed"}

    synth = VoiceSynthesizer(voice=voice)
    path = await synth.synthesize_async(text)
    if path is None:
        return {"error": "Speech synthesis failed"}
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/voices")
async def list_voices():
    """List available human voices."""
    return {
        "voices": [
            {"name": "jenny", "voice_id": "en-US-JennyNeural", "description": "Female, natural"},
            {"name": "aria", "voice_id": "en-US-AriaNeural", "description": "Female, conversational"},
            {"name": "guy", "voice_id": "en-US-GuyNeural", "description": "Male, natural"},
            {"name": "christopher", "voice_id": "en-US-ChristopherNeural", "description": "Male, deep"},
            {"name": "michelle", "voice_id": "en-US-MichelleNeural", "description": "Female, friendly"},
            {"name": "steffan", "voice_id": "en-US-SteffanNeural", "description": "Male, young"},
        ]
    }


# ------------------------------------------------------------------
# Voice chat page (simple HTML frontend)
# ------------------------------------------------------------------
from fastapi.responses import HTMLResponse

VOICE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>J-VIS — Voice Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body { height:100%; }

  /* ================================================================
     DARK THEME (always on)
  ================================================================ */
  :root {
    --bg: #02040a;
    --bg-grad-1: rgba(34,211,238,0.05);
    --bg-grad-2: rgba(59,130,246,0.04);
    --bg-grad-3: rgba(168,85,247,0.03);
    --text: #e6edf3;
    --text-muted: #8b949e;
    --text-faint: #484f58;
    --card-bg: rgba(6,12,24,0.88);
    --input-bg: rgba(4,8,16,0.85);
    --heard-bg: rgba(4,8,16,0.5);
    --status-bg: rgba(4,8,16,0.6);
    --status-border: rgba(34,211,238,0.15);
    --secondary-bg: rgba(48,54,61,0.8);
    --secondary-color: #e6edf3;
    --dot-bg: rgba(48,54,61,0.8);
    --accent: #22d3ee;
    --accent-strong: #22d3ee;
    --accent-2: #3b82f6;
    --accent-3: #a855f7;
    --ok: #3fb950;
    --fail: #f85149;
    --warn: #facc15;
    --logo-filter: brightness(0) invert(1);
    --shadow: 0 24px 70px rgba(0,0,0,0.7), 0 0 60px rgba(34,211,238,0.08);
    --shadow-soft: 0 4px 18px rgba(37,99,235,0.35);
    --shadow-hover: 0 8px 26px rgba(37,99,235,0.5);
    --scanline: rgba(255,255,255,0.025);
    --vignette: rgba(0,0,0,0.55);
  }

  body {
    font-family:'Segoe UI',system-ui,sans-serif;
    background:var(--bg);
    color:var(--text);
    overflow:hidden;
    background:
      radial-gradient(ellipse 90% 70% at 50% 35%, var(--bg-grad-1), transparent 65%),
      radial-gradient(ellipse 70% 60% at 15% 90%, var(--bg-grad-2), transparent 60%),
      radial-gradient(ellipse 60% 50% at 85% 80%, var(--bg-grad-3), transparent 60%),
      var(--bg);
  }
  #bgCanvas { position:fixed; inset:0; z-index:0; pointer-events:none; }
  .scanlines {
    position:fixed; inset:0; z-index:2; pointer-events:none;
    background:repeating-linear-gradient(0deg, var(--scanline) 0px, var(--scanline) 1px, transparent 1px, transparent 3px);
  }
  .vignette {
    position:fixed; inset:0; z-index:1; pointer-events:none;
    background:radial-gradient(ellipse at center, transparent 55%, var(--vignette) 100%);
  }

  /* ---- main ---- */
  main {
    position:relative; z-index:5;
    height:100vh;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:14px; padding:16px;
  }

  /* ---- circular HUD visualizer ---- */
  .viz-wrap { position:relative; width:min(760px, 96vw); height:min(760px, 96vw); }
  #viz { width:100%; height:100%; display:block; }
  .viz-center {
    position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    text-align:center; pointer-events:none; user-select:none; width:52%;
  }
  .viz-title { display:flex; justify-content:center; }
  .viz-title img {
    height:96px; width:auto;
    filter:var(--logo-filter) drop-shadow(0 4px 18px rgba(34,211,238,0.4));
  }
  .viz-status {
    font-family:'Orbitron','Consolas',monospace;
    font-size:14px; color:var(--text-muted); margin-top:12px; line-height:1.5; font-weight:500;
    letter-spacing:1px;
  }
  .viz-status .phrase { color:var(--accent); font-weight:700; text-shadow:0 0 12px rgba(34,211,238,0.3); }
  .viz-status.ok { color:var(--ok); }
  .viz-status.fail { color:var(--fail); }
  .viz-status.processing { color:var(--warn); }
  .viz-status.listening { color:var(--accent); animation:statusPulse 1.4s ease-in-out infinite; }
  @keyframes statusPulse {
    0%,100% { opacity:0.6; }
    50% { opacity:1; }
  }
  .viz-sub {
    font-family:'Orbitron','Consolas',monospace;
    font-size:10px; color:var(--text-faint); margin-top:10px; letter-spacing:2px;
    text-transform:uppercase;
  }

  /* ---- last heard line ---- */
  .heard {
    position:relative; z-index:5;
    font-size:14px; color:var(--text-muted); text-align:center; min-height:22px;
    padding:0 24px; max-width:85%;
    background:var(--heard-bg); border-radius:20px; padding:8px 24px;
    backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    border:1px solid var(--status-border);
  }
  .heard b { color:var(--accent); font-weight:600; }

  /* ================================================================
     WELCOME OVERLAY (double-clap awakening)
  ================================================================ */
  /* auth screen is a SEPARATE full-screen interface — the J-VIS
     circles are hidden while it is active */
  body.auth-mode main { display:none; }
  .welcome-overlay {
    position:fixed; inset:0; z-index:40;
    background:#000;
    display:flex; align-items:center; justify-content:center;
  }
  .welcome-hint {
    position:fixed; bottom:40px; left:0; right:0; text-align:center;
    font-family:'Orbitron',sans-serif; font-size:14px; color:#67e8f9;
    letter-spacing:3px; text-transform:uppercase;
    text-shadow:0 0 12px rgba(34,211,238,0.5);
    animation:hintPulse 1.6s ease-in-out infinite;
  }
  @keyframes hintPulse {
    0%,100% { opacity:0.55; }
    50% { opacity:1; }
  }
  .welcome-box {
    max-width:min(820px, 92vw);
    text-align:center; padding:40px 30px;
  }
  .welcome-logo { display:flex; justify-content:center; margin-bottom:26px; }
  .welcome-logo img {
    height:110px; width:auto;
    filter:var(--logo-filter) drop-shadow(0 4px 18px rgba(34,211,238,0.4));
  }
  .welcome-text {
    font-family:'Orbitron',sans-serif;
    font-size:22px; line-height:1.8; color:#22d3ee;
    text-shadow:0 0 18px rgba(34,211,238,0.5);
    min-height:160px; margin-top:30px; font-weight:500;
  }
  .welcome-cursor {
    display:inline-block; color:#22d3ee; font-weight:700;
    animation:blink 0.7s infinite;
  }
  @keyframes blink { 50% { opacity:0; } }
  .welcome-status {
    font-family:'Orbitron',sans-serif; font-size:13px; color:#8b949e;
    margin-top:26px; letter-spacing:3px; text-transform:uppercase; min-height:20px;
  }
  .welcome-status.ok { color:var(--ok); text-shadow:0 0 12px rgba(63,185,80,0.5); }
  .welcome-status.fail { color:var(--fail); text-shadow:0 0 12px rgba(248,81,73,0.5); }
  .welcome-reg { margin-top:22px; }
  .welcome-reg p { font-family:'Orbitron',sans-serif; font-size:12px; color:var(--text-muted); margin-bottom:14px; }
  .welcome-pin { margin-top:18px; }
  .welcome-pin .pin-input { margin-bottom:12px; }
  .row { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }
  button {
    background:linear-gradient(135deg,var(--accent-2),var(--accent-strong)); color:#fff; border:none;
    padding:13px 26px; border-radius:14px; font-size:14px; cursor:pointer; font-weight:600;
    font-family:'Orbitron',sans-serif; letter-spacing:1px;
    transition:transform .15s, box-shadow .15s;
    box-shadow:var(--shadow-soft);
  }
  button:hover { transform:translateY(-2px) scale(1.02); box-shadow:var(--shadow-hover); }
  button:disabled { opacity:0.4; cursor:not-allowed; transform:none; }
  button.record { background:linear-gradient(135deg,#dc2626,#f87171); box-shadow:0 4px 18px rgba(220,38,38,0.35); }
  button.record:hover { box-shadow:0 8px 26px rgba(220,38,38,0.5); }
  button.record.recording { animation:pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:0.5; } }
  .sample-dots { display:flex; gap:8px; justify-content:center; margin:16px 0; }
  .dot { width:14px; height:14px; border-radius:50%; background:var(--dot-bg); transition:all .3s; }
  .dot.filled { background:var(--accent-strong); box-shadow:0 0 14px rgba(34,211,238,0.7); }
  .link-btn { background:none; border:none; color:var(--accent); font-size:12px; cursor:pointer; padding:8px; box-shadow:none; font-family:'Orbitron',sans-serif; letter-spacing:1px; }
  .link-btn:hover { text-decoration:underline; transform:none; box-shadow:none; }
  .pin-input {
    background:var(--input-bg); border:1px solid var(--status-border); color:var(--text);
    padding:14px; border-radius:14px; font-size:24px; letter-spacing:14px; text-align:center;
    width:190px; outline:none; margin:0 auto; display:block;
    transition:border-color .3s, box-shadow .3s;
  }
  .pin-input:focus { border-color:var(--accent-strong); box-shadow:0 0 0 4px rgba(34,211,238,0.15); }
  .hidden { display:none !important; }

  /* typed mission bar (visible after unlock) */
  .cmd-bar {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 8px;
    z-index: 40;
    width: min(560px, 90vw);
  }
  .cmd-bar input {
    flex: 1;
    background: rgba(2,4,10,0.85);
    border: 1px solid rgba(34,211,238,0.35);
    border-radius: 999px;
    color: #e2f4ff;
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    letter-spacing: 0.08em;
    padding: 10px 18px;
    outline: none;
  }
  .cmd-bar input:focus { border-color:var(--accent-strong); box-shadow:0 0 0 4px rgba(34,211,238,0.15); }
  .cmd-bar button {
    background: rgba(34,211,238,0.12);
    border: 1px solid rgba(34,211,238,0.45);
    border-radius: 999px;
    color: #a5f3fc;
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    letter-spacing: 0.1em;
    padding: 10px 20px;
    cursor: pointer;
  }
  .cmd-bar button:hover { background: rgba(34,211,238,0.25); }

  /* ============================================================
     RED POWER MODE — activated by "disappear / close yourself /
     change my power". J-VIS turns red and uses opencode only.
     ============================================================ */
  body.red-mode {
    --bg: #0a0204;
    --bg-grad-1: rgba(248,113,113,0.08);
    --bg-grad-2: rgba(220,38,38,0.06);
    --bg-grad-3: rgba(127,29,29,0.05);
    --accent: #f87171;
    --accent-strong: #ef4444;
    --accent-2: #dc2626;
    --accent-3: #7f1d1d;
    --status-border: rgba(248,113,113,0.25);
    --shadow: 0 24px 70px rgba(0,0,0,0.7), 0 0 60px rgba(239,68,68,0.12);
    --shadow-soft: 0 4px 18px rgba(220,38,38,0.4);
    --shadow-hover: 0 8px 26px rgba(220,38,38,0.55);
  }
  body.red-mode .viz-status .phrase { text-shadow:0 0 12px rgba(248,113,113,0.4); }
  body.red-mode .dot.filled { box-shadow:0 0 14px rgba(239,68,68,0.8); }
  body.red-mode .cmd-bar button { background: rgba(239,68,68,0.15); }
  body.red-mode .cmd-bar button:hover { background: rgba(239,68,68,0.3); }
  body.red-mode .cmd-bar input:focus { border-color:var(--accent-strong); box-shadow:0 0 0 4px rgba(239,68,68,0.2); }
  body.red-mode .link-btn { color:var(--accent); }
</style>
</head>
<body class="auth-mode">
<canvas id="bgCanvas"></canvas>
<div class="vignette"></div>
<div class="scanlines"></div>

<main>
  <div class="viz-wrap">
    <canvas id="viz"></canvas>
    <div class="viz-center">
      <div class="viz-title"><img src="/static/jvis-logo.png" alt="J-VIS logo"></div>
      <div class="viz-status" id="vizStatus">locked</div>
      <div class="viz-sub" id="vizSub">👏 👏</div>
    </div>
  </div>
  <div class="heard" id="heardLine"></div>
</main>

<!-- typed mission bar — appears after unlock -->
<div class="cmd-bar hidden" id="cmdBar">
  <input type="text" id="cmdInput" placeholder="Type a mission for J-VIS..." autocomplete="off">
  <button id="cmdSend">Send</button>
</div>

<audio id="audioPlayer" style="display:none"></audio>

<!-- WELCOME OVERLAY — the FIRST interface: pure black screen -->
<div class="welcome-overlay" id="welcomeOverlay">
  <div class="welcome-box">
    <div class="welcome-logo"><img src="/static/jvis-logo.png" alt="J-VIS logo"></div>
    <div class="welcome-text hidden" id="welcomeText"></div>
    <div class="welcome-status hidden" id="welcomeStatus"></div>

    <div id="welcomeReg" class="welcome-reg hidden">
      <p>No voiceprint registered. Record 3 samples to bind your voice as the cipher.</p>
      <div class="row">
        <button id="wRecordBtn" class="record">Record</button>
        <button id="wEnrollBtn" disabled>Enroll Voice</button>
      </div>
      <div class="sample-dots" id="wSampleDots">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      </div>
    </div>

    <div id="welcomePin" class="welcome-pin hidden">
      <input type="password" id="wPinInput" class="pin-input" maxlength="4" inputmode="numeric" placeholder="••••">
      <div class="row">
        <button id="wPinBtn">Unlock with PIN</button>
      </div>
    </div>

    <button id="wPinLink" class="link-btn hidden">Use PIN instead</button>
    <div style="margin-top:16px;"><button id="wCancelBtn" class="link-btn hidden">Cancel</button></div>
  </div>
  <div class="welcome-hint hidden" id="welcomeHint">clap twice to awaken</div>
</div>

<script>
const vizStatus = document.getElementById('vizStatus');
const vizSub = document.getElementById('vizSub');
const heardLine = document.getElementById('heardLine');
const cmdBar = document.getElementById('cmdBar');
const cmdInput = document.getElementById('cmdInput');
const cmdSend = document.getElementById('cmdSend');
const audioPlayer = document.getElementById('audioPlayer');
const welcomeOverlay = document.getElementById('welcomeOverlay');
const welcomeText = document.getElementById('welcomeText');
const welcomeStatus = document.getElementById('welcomeStatus');
const welcomeReg = document.getElementById('welcomeReg');
const welcomePin = document.getElementById('welcomePin');
const wRecordBtn = document.getElementById('wRecordBtn');
const wEnrollBtn = document.getElementById('wEnrollBtn');
const wSampleDots = document.getElementById('wSampleDots');
const wPinInput = document.getElementById('wPinInput');
const wPinBtn = document.getElementById('wPinBtn');
const wPinLink = document.getElementById('wPinLink');
const wCancelBtn = document.getElementById('wCancelBtn');
const welcomeHint = document.getElementById('welcomeHint');

const OWNER = 'master_demo';
const VOICE = 'en-US-MichelleNeural';   // J-VIS always speaks with Michelle
const WELCOME_TEXT = 'Welcome to J-VIS—where the echoes of the past and the realities of the present forge your future. Please enter your cipher to awaken the system.';

let owner = null;
let unlocked = false;
let state = 'idle';
let welcomeActive = false;
let typeIndex = 0;
let recordedSamples = [];
let mediaRecorder = null;
let audioChunks = [];
let recognition = null;
let micStream = null;
let audioCtx = null;
let analyser = null;
let freqData = null;
let timeData = null;
let vizActive = false;
let burst = 0;
let idleT = 0;
let hintFlashT = null;

// whisper fallback STT state (used when browser SpeechRecognition is unavailable)
let whisperMode = false;
let passwordMode = false;   // whisper mode is listening for the cipher phrase
let inUtterance = false;
let lastSpeechAt = 0;
let ttsPlaying = false;
let whisperWatchdog = null;
let pwWatchdog = null;
let pwNoResultWatchdog = null;
let utteranceRecorder = null;   // dedicated recorder per utterance (finalized webm)
let utteranceChunks = [];

// clap detection state
let clapTimes = [];
let lastClapAt = 0;
let energyBaseline = 0.05;

// ================================================================
// BACKGROUND PARTICLES
// ================================================================
const bgCanvas = document.getElementById('bgCanvas');
const bgCtx = bgCanvas.getContext('2d');
let particles = [];
// Viz hue (190 = cyan). Red power mode switches it to 0 (red).
let hue = 190;
// Background particle palette — swapped in red power mode.
let bgPalette = ['rgba(34,211,238,', 'rgba(59,130,246,', 'rgba(168,85,247,'];
function initBg() {
  bgCanvas.width = window.innerWidth;
  bgCanvas.height = window.innerHeight;
  const count = Math.min(70, Math.floor(window.innerWidth / 20));
  particles = [];
  for (let i = 0; i < count; i++) {
    const colors = bgPalette;
    particles.push({
      x: Math.random() * bgCanvas.width,
      y: Math.random() * bgCanvas.height,
      r: Math.random() * 2 + 0.4,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      a: Math.random() * 0.35 + 0.08,
      c: colors[Math.floor(Math.random() * colors.length)],
    });
  }
}
window.addEventListener('resize', initBg);
initBg();
function drawBg() {
  requestAnimationFrame(drawBg);
  bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
  for (const p of particles) {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0) p.x = bgCanvas.width;
    if (p.x > bgCanvas.width) p.x = 0;
    if (p.y < 0) p.y = bgCanvas.height;
    if (p.y > bgCanvas.height) p.y = 0;
    bgCtx.beginPath();
    bgCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    bgCtx.fillStyle = p.c + p.a + ')';
    bgCtx.fill();
  }
}
drawBg();

// ================================================================
// HUD VISUALIZER — layered sci-fi interface
// ================================================================
const canvas = document.getElementById('viz');
const ctx = canvas.getContext('2d');
let W = 0, H = 0, DPR = 1;

function resizeViz() {
  DPR = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  W = rect.width; H = rect.height;
  canvas.width = W * DPR; canvas.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener('resize', resizeViz);
resizeViz();

// waveform trackers (radial bars)
const NBARS = 72;
const barLevels = new Array(NBARS).fill(0);
const barTargets = new Array(NBARS).fill(0);

// tachometer segments
const NSEG = 48;
const segLevels = new Array(NSEG).fill(0);
const segTargets = new Array(NSEG).fill(0);
const segTrails = new Array(NSEG).fill(0);

// data nodes
const NODES = [];
let nodesInit = false;

// tickers
const TICKERS = ['SYS:98%','UPLINK:ACTIVE','PWR:87%','NET:42%','CPU:63%','MEM:71%','LINK:OK','SCAN:ON','CORE:100%','AUDIO:LIVE'];

let sweepAngle = 0;

function initNodes(maxR) {
  NODES.length = 0;
  const radii = [0.3, 0.5, 0.7, 0.9];
  for (const fr of radii) {
    for (let i = 0; i < 12; i++) {
      NODES.push({r: maxR * fr, a: i * Math.PI / 6, phase: Math.random() * 10});
    }
  }
  nodesInit = true;
}

function drawViz(t) {
  requestAnimationFrame(drawViz);
  const cx = W / 2, cy = H / 2;
  ctx.clearRect(0, 0, W, H);

  // ---- double-clap detection (runs every frame) ----
  detectClaps();
  // ---- whisper-mode speech detection (runs every frame) ----
  detectSpeech();

  // ---- audio analysis ----
  let bass = 0, mid = 0, avg = 0;
  if (vizActive && analyser && freqData) {
    analyser.getByteFrequencyData(freqData);
    analyser.getByteTimeDomainData(timeData);
    const n = freqData.length;
    for (let i = 0; i < 10; i++) bass += freqData[i];
    bass /= 10 * 255;
    for (let i = 10; i < Math.min(60, n); i++) mid += freqData[i];
    mid /= Math.max(1, Math.min(60, n) - 10) * 255;
    for (let i = 0; i < n; i++) avg += freqData[i];
    avg /= n * 255;
  } else {
    idleT += 0.03;
    bass = 0.12 + 0.06 * Math.sin(idleT);
    mid = 0.08 + 0.05 * Math.sin(idleT * 1.7);
    avg = 0.1;
  }
  burst *= 0.90;

  const maxR = Math.min(W, H) * 0.46;
  const coreR = 96 + bass * 45 + burst * 35;

  // 1. POLAR COORDINATE GRID
  ctx.strokeStyle = 'hsla(' + hue + ', 100%, 60%, 0.07)';
  ctx.lineWidth = 1;
  for (let r = 55; r <= maxR; r += 55) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let a = 0; a < Math.PI * 2; a += Math.PI / 12) {
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * maxR, cy + Math.sin(a) * maxR);
    ctx.stroke();
  }

  // 2. SWEEPING SCANNER (radar wedge)
  sweepAngle += 0.015;
  const sa = sweepAngle;
  for (let i = 0; i < 24; i++) {
    const a0 = sa - 0.7 + i * 0.03;
    const alpha = (i / 24) * 0.10;
    ctx.fillStyle = 'hsla(' + hue + ', 100%, 60%, ' + alpha + ')';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, maxR, a0, a0 + 0.03);
    ctx.closePath();
    ctx.fill();
  }
  ctx.strokeStyle = 'hsla(' + hue + ', 100%, 70%, 0.55)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.cos(sa) * maxR, cy + Math.sin(sa) * maxR);
  ctx.stroke();

  // 3. DATA NODES (grid intersections, blinking + connecting)
  if (!nodesInit) initNodes(maxR);
  ctx.lineWidth = 1;
  for (let i = 0; i < NODES.length; i++) {
    const n1 = NODES[i];
    const a1 = n1.a + t / 4000;
    const x1 = cx + Math.cos(a1) * n1.r, y1 = cy + Math.sin(a1) * n1.r;
    const act1 = Math.sin(t / 500 + n1.phase) > 0.1;
    for (let j = i + 1; j < NODES.length; j++) {
      const n2 = NODES[j];
      const a2 = n2.a + t / 4000;
      const x2 = cx + Math.cos(a2) * n2.r, y2 = cy + Math.sin(a2) * n2.r;
      const d = Math.hypot(x2 - x1, y2 - y1);
      if (d < 110 && act1 && Math.sin(t / 500 + n2.phase) > 0.1) {
        ctx.strokeStyle = 'hsla(' + hue + ', 100%, 65%, 0.12)';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
  }
  for (const n of NODES) {
    const a = n.a + t / 4000;
    const x = cx + Math.cos(a) * n.r, y = cy + Math.sin(a) * n.r;
    const act = Math.sin(t / 500 + n.phase);
    if (act > 0.1) {
      ctx.fillStyle = 'hsla(' + hue + ', 100%, 70%, ' + (0.3 + act * 0.5) + ')';
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    } else if (act > -0.3) {
      ctx.fillStyle = 'hsla(' + hue + ', 100%, 70%, 0.15)';
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 4. SEGMENTED DASHBOARDS (outer tachometer with trails)
  const rOut = maxR - 6, rIn = maxR - 26;
  const segW = (Math.PI * 2 / NSEG) * 0.75;
  for (let i = 0; i < NSEG; i++) {
    const a = (i / NSEG) * Math.PI * 2 - Math.PI / 2;
    segTargets[i] = Math.max(0, Math.min(1, segTargets[i] + (Math.random() - 0.5) * 0.14 + avg * 0.1));
    segLevels[i] += (segTargets[i] - segLevels[i]) * 0.08;
    segTrails[i] = Math.max(segLevels[i], segTrails[i] * 0.96);
    const fill = segLevels[i], trail = segTrails[i];
    if (trail > 0.05) {
      ctx.strokeStyle = 'hsla(' + hue + ', 100%, 65%, ' + (trail * 0.15) + ')';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.arc(cx, cy, rOut - (rOut - rIn) * trail, a, a + segW);
      ctx.stroke();
    }
    if (fill > 0.03) {
      const col = fill > 0.75 ? 'hsla(40, 100%, 55%, 0.9)' : 'hsla(' + hue + ', 100%, 65%, 0.8)';
      ctx.strokeStyle = col;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(cx, cy, rOut - (rOut - rIn) * fill, a, a + segW);
      ctx.stroke();
    }
  }

  // 5. WARNING INDICATORS (hazard triangles, pulsing)
  const warnAngles = [0.4, 2.6, 4.9];
  for (let i = 0; i < warnAngles.length; i++) {
    const a = warnAngles[i];
    const pulse = (Math.sin(t / 300 + i * 2) + 1) / 2;
    const danger = avg > 0.55 || pulse > 0.85;
    const col = danger
      ? 'hsla(0, 90%, 55%, ' + (0.4 + pulse * 0.5) + ')'
      : 'hsla(40, 95%, 55%, ' + (0.2 + pulse * 0.4) + ')';
    const x = cx + Math.cos(a) * (maxR + 8), y = cy + Math.sin(a) * (maxR + 8);
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x - 5, y + 5);
    ctx.lineTo(x + 5, y + 5);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fillRect(x - 1, y - 3, 2, 4);
    ctx.fillRect(x - 1, y + 2, 2, 1.5);
  }

  // 6. CONTRAROTATING ARCS + PERCENTAGE TICKERS
  const rings = [
    {r: coreR + 34, speed: 1 / 6000, segs: 3, gap: 0.35, len: 1.7},
    {r: coreR + 48, speed: -1 / 4000, segs: 4, gap: 0.25, len: 1.3},
    {r: coreR + 62, speed: 1 / 9000, segs: 2, gap: 0.5, len: 2.6},
  ];
  for (const ring of rings) {
    const rot = t * ring.speed;
    ctx.strokeStyle = 'hsla(' + hue + ', 100%, 70%, 0.35)';
    ctx.lineWidth = 1.5;
    for (let s = 0; s < ring.segs; s++) {
      const a0 = rot + s * (ring.len + ring.gap);
      ctx.beginPath();
      ctx.arc(cx, cy, ring.r, a0, a0 + ring.len);
      ctx.stroke();
    }
    const ta = rot;
    const tx = cx + Math.cos(ta) * ring.r, ty = cy + Math.sin(ta) * ring.r;
    const label = TICKERS[Math.floor(t / 2000 + ring.r) % TICKERS.length];
    ctx.font = '9px Consolas, monospace';
    ctx.fillStyle = 'hsla(' + hue + ', 100%, 75%, 0.8)';
    ctx.fillText(label, tx + 6, ty - 4);
  }

  // 7. WAVEFORM TRACKERS (radial bars, cyan → amber when hot)
  for (let i = 0; i < NBARS; i++) {
    let target;
    if (vizActive && freqData) {
      target = freqData[Math.floor(i * freqData.length / NBARS)] / 255;
    } else {
      target = 0.1 + 0.08 * Math.sin(idleT * 2 + i * 0.4);
    }
    barTargets[i] = target;
    barLevels[i] += (barTargets[i] - barLevels[i]) * 0.35;
    const v = barLevels[i];
    const angle = (i / NBARS) * Math.PI * 2 - Math.PI / 2;
    const inner = coreR + 24;
    const outer = inner + 8 + v * 110;
    const hot = v > 0.7;
    ctx.strokeStyle = hot
      ? 'hsla(40, 100%, 55%, ' + (0.5 + v * 0.5) + ')'
      : 'hsla(' + hue + ', 100%, 65%, ' + (0.3 + v * 0.6) + ')';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner);
    ctx.lineTo(cx + Math.cos(angle) * outer, cy + Math.sin(angle) * outer);
    ctx.stroke();
    if (v > 0.2) {
      ctx.fillStyle = hot
        ? 'hsla(40, 100%, 70%, ' + Math.min(1, v) + ')'
        : 'hsla(' + hue + ', 100%, 80%, ' + Math.min(1, v) + ')';
      ctx.beginPath();
      ctx.arc(cx + Math.cos(angle) * outer, cy + Math.sin(angle) * outer, 1.5 + v * 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 8. MICRO-DATA STREAMS (hex ring + scrolling lines)
  const HEX = '0123456789ABCDEF';
  ctx.font = '9px Consolas, monospace';
  const NHEX = 40;
  for (let i = 0; i < NHEX; i++) {
    const a = (i / NHEX) * Math.PI * 2 + t / 2500;
    const x = cx + Math.cos(a) * (coreR + 10), y = cy + Math.sin(a) * (coreR + 10);
    const ch = HEX[Math.floor((t / 70 + i * 7) % 16)];
    ctx.fillStyle = 'hsla(' + hue + ', 100%, 70%, 0.3)';
    ctx.fillText(ch, x - 3, y + 3);
  }
  let line = '', bline = '';
  for (let i = 0; i < 26; i++) {
    line += HEX[Math.floor((t / 50 + i) % 16)];
    bline += Math.floor((t / 30 + i * 3) % 2);
  }
  ctx.fillStyle = 'hsla(' + hue + ', 100%, 70%, 0.35)';
  ctx.fillText(line, cx - 70, cy - coreR - 12);
  ctx.fillStyle = 'hsla(40, 100%, 60%, 0.3)';
  ctx.fillText(bline, cx - 70, cy + coreR + 14);

  // 9. AUDIO-REACTIVE ORB (core) with chromatic aberration + wireframe globe
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 1.6);
  g.addColorStop(0, 'hsla(' + hue + ', 100%, 65%, 0.18)');
  g.addColorStop(0.5, 'hsla(' + hue + ', 100%, 60%, 0.06)');
  g.addColorStop(1, 'hsla(' + hue + ', 100%, 60%, 0)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(cx, cy, coreR * 1.6, 0, Math.PI * 2);
  ctx.fill();

  // chromatic aberration (red/blue split)
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(255,60,60,0.22)';
  ctx.beginPath();
  ctx.arc(cx + 3, cy, coreR, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(60,120,255,0.22)';
  ctx.beginPath();
  ctx.arc(cx - 3, cy, coreR, 0, Math.PI * 2);
  ctx.stroke();

  // main ring
  ctx.strokeStyle = 'hsla(' + hue + ', 100%, 70%, ' + (0.5 + bass * 0.5) + ')';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
  ctx.stroke();

  // wireframe globe
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(t / 6000);
  ctx.strokeStyle = 'hsla(' + hue + ', 100%, 70%, 0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(0, 0, coreR * 0.92, coreR * 0.35, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.rotate(Math.PI / 2);
  ctx.beginPath();
  ctx.ellipse(0, 0, coreR * 0.92, coreR * 0.35, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.rotate(-Math.PI / 2);
  ctx.beginPath();
  ctx.ellipse(0, 0, coreR * 0.92, coreR * 0.92, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // 10. LENS FLARES
  const spots = [
    {x: cx - maxR * 0.8, y: cy - maxR * 0.7, s: 60},
    {x: cx + maxR * 0.75, y: cy + maxR * 0.6, s: 45},
  ];
  for (const sp of spots) {
    const fg = ctx.createRadialGradient(sp.x, sp.y, 0, sp.x, sp.y, sp.s);
    fg.addColorStop(0, 'hsla(' + hue + ', 100%, 70%, 0.10)');
    fg.addColorStop(1, 'hsla(' + hue + ', 100%, 70%, 0)');
    ctx.fillStyle = fg;
    ctx.beginPath();
    ctx.arc(sp.x, sp.y, sp.s, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = 'hsla(' + hue + ', 100%, 80%, 0.5)';
  ctx.beginPath();
  ctx.arc(cx - maxR * 0.8, cy - maxR * 0.7, 2, 0, Math.PI * 2);
  ctx.fill();
}
requestAnimationFrame(drawViz);

// ================================================================
// DOUBLE-CLAP DETECTION
// ================================================================
function detectClaps() {
  if (!analyser || !timeData) return;
  // time-domain RMS — unsmoothed, catches sharp transients like claps
  analyser.getByteTimeDomainData(timeData);
  let sumSq = 0;
  for (let i = 0; i < timeData.length; i++) {
    const s = (timeData[i] - 128) / 128;
    sumSq += s * s;
  }
  const rms = Math.sqrt(sumSq / timeData.length);
  energyBaseline = energyBaseline * 0.92 + rms * 0.08;
  const now = performance.now();
  // a clap = sharp transient spike well above the ambient level
  if (rms > Math.max(0.08, energyBaseline * 2.5) && now - lastClapAt > 180) {
    lastClapAt = now;
    burst = 1;                       // pulse the orb as feedback
    clapTimes.push(now);
    // visual feedback on the black screen so the user knows the mic hears them
    welcomeHint.textContent = '👏';
    clearTimeout(hintFlashT);
    hintFlashT = setTimeout(() => {
      if (!welcomeActive) welcomeHint.textContent = 'clap twice to awaken';
    }, 400);
    clapTimes = clapTimes.filter(t => now - t < 1000);
    // double clap = two claps within 700ms
    if (clapTimes.length >= 2 && clapTimes[clapTimes.length - 1] - clapTimes[clapTimes.length - 2] < 700) {
      clapTimes = [];
      onDoubleClap();
    }
  }
}

function onDoubleClap() {
  if (unlocked || welcomeActive) return;
  welcomeActive = true;
  welcomeHint.classList.add('hidden');
  startWelcome();
}

// ================================================================
// WELCOME SEQUENCE (typewriter + Michelle reads it)
// ================================================================
function startWelcome() {
  typeIndex = 0;
  welcomeText.innerHTML = '';
  welcomeText.classList.remove('hidden');
  welcomeStatus.classList.add('hidden');
  welcomeReg.classList.add('hidden');
  welcomePin.classList.add('hidden');
  wPinLink.classList.add('hidden');
  wCancelBtn.classList.remove('hidden');
  speak(WELCOME_TEXT);   // Michelle reads the welcome
  typeWelcome();
}

function typeWelcome() {
  if (typeIndex < WELCOME_TEXT.length) {
    welcomeText.innerHTML = WELCOME_TEXT.slice(0, typeIndex + 1) + '<span class="welcome-cursor">▌</span>';
    typeIndex++;
    setTimeout(typeWelcome, 22);
  } else {
    welcomeText.innerHTML = WELCOME_TEXT + '<span class="welcome-cursor">▌</span>';
    onWelcomeTyped();
  }
}

function onWelcomeTyped() {
  welcomeStatus.classList.remove('hidden');
  welcomeStatus.className = 'welcome-status';
  // Passphrase unlock — anyone who says the cipher awakens J-VIS.
  // No voiceprint is required or verified.
  welcomeStatus.textContent = 'Say the cipher to awaken';
  wPinLink.classList.remove('hidden');   // always keep the PIN escape hatch visible
  startPasswordListening();
}

// ================================================================
// PASSWORD LISTENING (voice-only unlock)
// ================================================================
function startPasswordListening() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { switchToWhisperPassword(); return; }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.onstart = () => {
    clearTimeout(pwWatchdog);
    welcomeStatus.textContent = 'Listening for cipher...';
    welcomeStatus.className = 'welcome-status';
    // If SR is active but hears nothing useful within 12s, fall back to Whisper.
    clearTimeout(pwNoResultWatchdog);
    pwNoResultWatchdog = setTimeout(() => {
      if (welcomeActive && !unlocked && !whisperMode) {
        switchToWhisperPassword();
      }
    }, 12000);
  };
  recognition.onresult = (e) => {
    let transcript = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      transcript += e.results[i][0].transcript;
    }
    const low = transcript.toLowerCase();
    if (low.includes('reveal my power') || low.includes('awaken') || low.includes('unlock')) {
      unlockPassphrase();
    } else {
      // heard something but not the cipher — keep listening, reset the timer
      clearTimeout(pwNoResultWatchdog);
      pwNoResultWatchdog = setTimeout(() => {
        if (welcomeActive && !unlocked && !whisperMode) {
          switchToWhisperPassword();
        }
      }, 12000);
    }
  };
  recognition.onerror = (e) => {
    if (e.error === 'not-allowed') {
      welcomeStatus.textContent = 'Microphone permission denied.';
      welcomeStatus.className = 'welcome-status fail';
    } else {
      switchToWhisperPassword();
    }
  };
  recognition.onend = () => {
    if (welcomeActive && !unlocked && !whisperMode) { try { recognition.start(); } catch (e) {} }
  };
  try { recognition.start(); } catch (e) { switchToWhisperPassword(); return; }
  // Watchdog: if the recognizer never actually starts, fall back to Whisper.
  clearTimeout(pwWatchdog);
  pwWatchdog = setTimeout(() => {
    if (welcomeActive && !unlocked && !whisperMode) {
      switchToWhisperPassword();
    }
  }, 3500);
}

// Whisper fallback for the cipher phrase — works in any browser.
function switchToWhisperPassword() {
  clearTimeout(pwWatchdog);
  clearTimeout(pwNoResultWatchdog);
  whisperMode = true;
  passwordMode = true;
  inUtterance = false;
  audioChunks = [];
  utteranceChunks = [];
  welcomeStatus.textContent = 'Whisper mode — say the cipher';
  welcomeStatus.className = 'welcome-status';
}

// Passphrase unlock — the cipher phrase itself is the key.
// No voiceprint is verified: anyone who says it awakens J-VIS.
function unlockPassphrase() {
  clearTimeout(pwWatchdog);
  clearTimeout(pwNoResultWatchdog);
  passwordMode = false;
  unlock({}, 'passphrase');
}

// ================================================================
// VOICE REGISTRATION (bind cipher to voice)
// ================================================================
async function startRecording() {
  if (wRecordBtn.classList.contains('recording')) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const rec = new MediaRecorder(stream);
    const chunks = [];
    rec.ondataavailable = e => chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, {type: rec.mimeType || 'audio/webm'});
      welcomeStatus.textContent = 'Processing audio...';
      welcomeStatus.className = 'welcome-status';
      try {
        const wavB64 = await blobToWavBase64(blob);
        recordedSamples.push(wavB64);
        updateDots();
        welcomeStatus.textContent = 'Sample ' + recordedSamples.length + '/3 captured. ' +
          (recordedSamples.length >= 3 ? 'Ready to enroll!' : 'Record again.');
        wEnrollBtn.disabled = recordedSamples.length < 3;
      } catch (err) {
        welcomeStatus.textContent = 'Audio conversion failed: ' + err.message;
        welcomeStatus.className = 'welcome-status fail';
      }
    };
    rec.start();
    wRecordBtn.textContent = 'Recording...';
    wRecordBtn.classList.add('recording');
    welcomeStatus.textContent = 'Recording... speak naturally for 3 seconds.';
    welcomeStatus.className = 'welcome-status';
    setTimeout(() => {
      if (wRecordBtn.classList.contains('recording')) {
        wRecordBtn.classList.remove('recording');
        wRecordBtn.textContent = 'Record';
        rec.stop();
      }
    }, 3500);
  } catch (e) {
    welcomeStatus.textContent = 'Microphone access denied: ' + e.message;
    welcomeStatus.className = 'welcome-status fail';
  }
}

function updateDots() {
  for (let i = 0; i < 3; i++) {
    wSampleDots.children[i].className = 'dot' + (i < recordedSamples.length ? ' filled' : '');
  }
}

async function enroll() {
  if (recordedSamples.length < 3) return;
  wEnrollBtn.disabled = true;
  welcomeStatus.textContent = 'Enrolling your voiceprint with SpeechBrain ECAPA-TDNN...';
  welcomeStatus.className = 'welcome-status';
  try {
    const resp = await fetch('/enroll/voice', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: owner.id, voice_samples_b64: recordedSamples})
    });
    const data = await resp.json();
    if (data.enrolled) {
      welcomeStatus.textContent = 'Voice bound. Awaiting your cipher...';
      welcomeStatus.className = 'welcome-status ok';
      welcomeReg.classList.add('hidden');
      wPinLink.classList.add('hidden');
      setTimeout(() => startPasswordListening(), 1200);
    } else {
      welcomeStatus.textContent = 'Enrollment failed: ' + data.message;
      welcomeStatus.className = 'welcome-status fail';
      wEnrollBtn.disabled = false;
    }
  } catch (e) {
    welcomeStatus.textContent = 'Enrollment error: ' + e.message;
    welcomeStatus.className = 'welcome-status fail';
    wEnrollBtn.disabled = false;
  }
}

// ================================================================
// PIN FALLBACK
// ================================================================
function showPinFallback() {
  passwordMode = false;
  if (recognition) { try { recognition.stop(); } catch (e) {} }
  welcomePin.classList.remove('hidden');
  wPinLink.classList.add('hidden');
  wPinInput.focus();
}

async function unlockWithPin() {
  const pin = wPinInput.value.trim();
  if (!pin) { welcomeStatus.textContent = 'Enter your PIN.'; welcomeStatus.className = 'welcome-status fail'; return; }
  welcomeStatus.textContent = 'Verifying PIN...';
  welcomeStatus.className = 'welcome-status';
  wPinBtn.disabled = true;
  try {
    const resp = await fetch('/session/bootstrap', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: owner.id, pin: pin})
    });
    const data = await resp.json();
    if (data.auth_method === 'pin') {
      unlock(data, 'pin');
    } else {
      welcomeStatus.textContent = 'Wrong PIN. Try again.';
      welcomeStatus.className = 'welcome-status fail';
      wPinInput.value = '';
      wPinInput.focus();
    }
  } catch (e) {
    welcomeStatus.textContent = 'Error: ' + e.message;
    welcomeStatus.className = 'welcome-status fail';
  }
  wPinBtn.disabled = false;
}

// ================================================================
// MIC + AUDIO
// ================================================================
async function startMic() {
  if (micStream) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({audio: true});
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.7;
    const source = audioCtx.createMediaStreamSource(micStream);
    source.connect(analyser);
    freqData = new Uint8Array(analyser.frequencyBinCount);
    timeData = new Uint8Array(analyser.fftSize);
    vizActive = true;
    // Browsers start AudioContext suspended until a user gesture.
    // Do NOT await resume() — it can hang forever without a gesture.
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }

    try {
      const ttsSource = audioCtx.createMediaElementSource(audioPlayer);
      ttsSource.connect(analyser);
      ttsSource.connect(audioCtx.destination);
    } catch (e) { /* element already connected */ }

    mediaRecorder = new MediaRecorder(micStream);
    mediaRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) {
        audioChunks.push(e.data);
        while (audioChunks.length > 10) audioChunks.shift();
      }
    };
    mediaRecorder.start(1000);
  } catch (e) {
    welcomeStatus.classList.remove('hidden');
    welcomeStatus.textContent = 'Microphone unavailable: ' + e.message;
    welcomeStatus.className = 'welcome-status fail';
  }
}

// Resume the AudioContext on the first user gesture (autoplay policy).
function resumeAudio() {
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume().then(() => {
      if (audioCtx.state === 'running') {
        welcomeHint.textContent = 'audio enabled — clap twice';
        setTimeout(() => {
          if (!welcomeActive) welcomeHint.textContent = 'clap twice to awaken';
        }, 3000);
      }
    }).catch(() => {});
  }
}
['click', 'keydown', 'touchstart'].forEach(ev => document.addEventListener(ev, resumeAudio));

function setVizStatus(text, cls) {
  vizStatus.textContent = text;
  vizStatus.className = 'viz-status' + (cls ? ' ' + cls : '');
}

function setState(s) {
  state = s;
}

// ================================================================
// INIT
// ================================================================
async function init() {
  try {
    const resp = await fetch('/users');
    const users = await resp.json();
    owner = users.find(u => u.username === OWNER);
    if (!owner) { setVizStatus('Owner account not found!', 'fail'); return; }
    // the FIRST interface is the pure black authentication screen
    document.body.classList.add('auth-mode');
    welcomeOverlay.classList.remove('hidden');
    await startMic();   // listen for the double clap
    if (audioCtx && audioCtx.state === 'suspended') {
      welcomeHint.textContent = 'tap to enable audio';
    } else {
      welcomeHint.textContent = 'clap twice to awaken';
    }
    welcomeHint.classList.remove('hidden');   // always guide the user on the black screen
  } catch (e) {
    setVizStatus('Cannot reach J-VIS: ' + e.message, 'fail');
  }
}

// ================================================================
// UNLOCK / LOCK
// ================================================================
function unlock(data, method) {
  unlocked = true;
  welcomeActive = false;
  passwordMode = false;
  burst = 1;
  setState('unlocked');
  welcomeOverlay.classList.add('hidden');
  document.body.classList.remove('auth-mode');   // bring the circles back
  resizeViz();
  if (method === 'pin') {
    setVizStatus('Unlocked with PIN', 'ok');
  } else if (method === 'passphrase') {
    setVizStatus('Cipher accepted — welcome back', 'ok');
  } else {
    setVizStatus('Voice verified — ' + (data.voice_similarity * 100).toFixed(1) + '% match', 'ok');
  }
  vizSub.textContent = 'speak your command';
  heardLine.innerHTML = '';
  if (recognition) { try { recognition.stop(); } catch (e) {} }
  setTimeout(() => {
    cmdBar.classList.remove('hidden');
    setVizStatus('J-VIS ready');
    speak("It's me J-VIS, how can I help you sir?").then(() => {
      setVizStatus('Listening... speak your command', 'listening');
      startCommandListening();
    });
  }, 2200);
}

// Dismiss the auth screen and return to the J-VIS circles (still locked).
function cancelAuth() {
  welcomeActive = false;
  passwordMode = false;
  welcomeText.classList.add('hidden');
  welcomeStatus.classList.add('hidden');
  welcomeReg.classList.add('hidden');
  welcomePin.classList.add('hidden');
  wPinLink.classList.add('hidden');
  wCancelBtn.classList.add('hidden');
  if (recognition) { try { recognition.stop(); } catch (e) {} }
  // back to the pure black screen, ready for another double clap
}

// ================================================================
// COMMAND LISTENING (unlocked) — fully vocal
// ================================================================
function startCommandListening() {
  // Whisper mode is sticky once active (browser STT failed earlier).
  if (whisperMode) { startWhisperListening(); return; }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setVizStatus('Speech recognition unavailable — switching to Whisper mode');
    whisperMode = true;
    startWhisperListening();
    return;
  }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.onstart = () => {
    clearTimeout(whisperWatchdog);
    setVizStatus('Listening... speak your command', 'listening');
  };
  recognition.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) {
        const text = e.results[i][0].transcript.trim();
        if (text) handleCommand(text);
      }
    }
  };
  recognition.onerror = (e) => {
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      setVizStatus('Mic blocked for speech recognition', 'fail');
      return;
    }
    // any other error (network, aborted, no-speech...) → Whisper fallback
    setVizStatus('Speech recognition error (' + e.error + ') — switching to Whisper mode');
    whisperMode = true;
    try { recognition.stop(); } catch (err) {}
    startWhisperListening();
  };
  recognition.onend = () => {
    if (unlocked && state !== 'processing' && state !== 'speaking' && !whisperMode) {
      try { recognition.start(); } catch (e) {}
    }
  };
  try { recognition.start(); } catch (e) {
    whisperMode = true;
    startWhisperListening();
    return;
  }
  // Watchdog: if the recognizer never actually starts, fall back to Whisper.
  clearTimeout(whisperWatchdog);
  whisperWatchdog = setTimeout(() => {
    if (unlocked && !whisperMode && state === 'unlocked') {
      whisperMode = true;
      try { recognition.stop(); } catch (e) {}
      setVizStatus('Switching to Whisper mode...');
      startWhisperListening();
    }
  }, 4000);
}

// ------------------------------------------------------------------
// WHISPER FALLBACK — silence-detection STT that works in any browser.
// detectSpeech() (called every frame from drawViz) watches the RMS and
// starts a dedicated per-utterance MediaRecorder when speech begins;
// stopping it finalizes the webm so it always decodes. The finalized
// blob is sent to /transcribe via transcribeUtterance().
// ------------------------------------------------------------------
function startWhisperListening() {
  inUtterance = false;
  audioChunks = [];
  utteranceChunks = [];
  setVizStatus('Whisper mode — speak your command', 'listening');
}

function detectSpeech() {
  if (!whisperMode || !analyser || !timeData || ttsPlaying) return;
  analyser.getByteTimeDomainData(timeData);
  let sumSq = 0;
  for (let i = 0; i < timeData.length; i++) {
    const s = (timeData[i] - 128) / 128;
    sumSq += s * s;
  }
  const rms = Math.sqrt(sumSq / timeData.length);
  const now = performance.now();
  const speechThreshold = Math.max(0.04, energyBaseline * 1.8);
  if (rms > speechThreshold) {
    if (!inUtterance) {
      inUtterance = true;
      startUtteranceRecording();
      if (passwordMode) {
        welcomeStatus.textContent = 'Heard you — verifying...';
        welcomeStatus.className = 'welcome-status';
      } else {
        setVizStatus('Heard you — transcribing...');
      }
    }
    lastSpeechAt = now;
  } else if (inUtterance && now - lastSpeechAt > 1400) {
    // 1.4s of silence → the utterance is over
    inUtterance = false;
    stopUtteranceRecording();
  }
}

// A dedicated MediaRecorder per utterance. Stopping it finalizes the webm
// container (writes the cues), so decodeAudioData can always decode it —
// the always-running recorder never finalizes, which is why its blobs
// failed with "Unable to decode audio data".
function startUtteranceRecording() {
  utteranceChunks = [];
  try {
    utteranceRecorder = new MediaRecorder(micStream);
    utteranceRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) utteranceChunks.push(e.data);
    };
    utteranceRecorder.onstop = () => transcribeUtterance();
    utteranceRecorder.start();
  } catch (e) {
    // Recorder unavailable — fall back to the continuous recorder's chunks.
    utteranceRecorder = null;
    audioChunks = [];
    setTimeout(transcribeUtterance, 1500);
  }
}

function stopUtteranceRecording() {
  if (utteranceRecorder && utteranceRecorder.state !== 'inactive') {
    try { utteranceRecorder.stop(); } catch (e) { transcribeUtterance(); }
  } else {
    transcribeUtterance();
  }
}

async function transcribeUtterance() {
  const chunks = utteranceRecorder ? utteranceChunks : audioChunks;
  if (chunks.length === 0) {
    if (passwordMode) {
      welcomeStatus.textContent = 'Whisper mode — say the cipher';
      welcomeStatus.className = 'welcome-status';
    } else {
      setVizStatus('Whisper mode — speak your command');
    }
    return;
  }
  if (passwordMode) {
    welcomeStatus.textContent = 'Transcribing...';
    welcomeStatus.className = 'welcome-status';
  } else {
    setVizStatus('Transcribing...');
  }
  try {
    const blob = new Blob(chunks, {type: utteranceRecorder ? utteranceRecorder.mimeType : (mediaRecorder ? mediaRecorder.mimeType : 'audio/webm')});
    const wavB64 = await blobToWavBase64(blob);
    const resp = await fetch('/transcribe', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({audio_b64: wavB64})
    });
    const data = await resp.json();
    const text = (data.text || '').trim();
    if (passwordMode) {
      const low = text.toLowerCase();
      if (low.includes('reveal my power') || low.includes('awaken') || low.includes('unlock')) {
        passwordMode = false;
        unlockPassphrase();
      } else {
        welcomeStatus.textContent = "Didn't catch the cipher — say 'J-VIS reveal my power'";
        welcomeStatus.className = 'welcome-status fail';
setTimeout(() => {
          if (welcomeActive && !unlocked) {
            welcomeStatus.textContent = 'Whisper mode — say the cipher';
            welcomeStatus.className = 'welcome-status';
            inUtterance = false;
            utteranceChunks = [];
          }
        }, 2200);
      }
      return;
    }
    if (text) {
      handleCommand(text);
    } else {
      setVizStatus('Didn\\'t catch that — speak again');
      setTimeout(() => {
        if (unlocked && state === 'unlocked') setVizStatus('Whisper mode — speak your command');
      }, 2000);
    }
  } catch (e) {
    if (passwordMode) {
      welcomeStatus.textContent = 'Transcription error: ' + e.message;
      welcomeStatus.className = 'welcome-status fail';
      setTimeout(() => {
        if (welcomeActive && !unlocked) {
          welcomeStatus.textContent = 'Whisper mode — say the cipher';
          welcomeStatus.className = 'welcome-status';
        }
      }, 2500);
    } else {
      setVizStatus('Transcription error: ' + e.message, 'fail');
      setTimeout(() => {
        if (unlocked && state === 'unlocked') setVizStatus('Whisper mode — speak your command');
      }, 2500);
    }
  }
}

// A name call = the user just says "J-VIS" (with optional greeting words),
// e.g. "J-VIS", "hey J-VIS", "J-VIS wake up", "are you there J-VIS".
function isNameCall(text) {
  const t = text.toLowerCase().replace(/[^a-z\\s]/g, ' ').replace(/\\s+/g, ' ').trim();
  if (!/j[\\s.\\-']?vis|javis|j[\\s.\\-']?v[\\s.\\-']?s/.test(t)) return false;
  const words = t.split(' ');
  const ok = new Set(['hey', 'hello', 'hi', 'yo', 'ok', 'okay', 'j', 'vis', 'jvis', 'j-vis', 'jvs',
    'v', 's', 'wake', 'up', 'are', 'you', 'there', 'listen', 'come', 'here', 'sir', 'boss', 'a', 'the']);
  return words.every(w => ok.has(w));
}

// A question goes to the local Ollama model; a mission goes to the
// opencode agent. Questions start with a question word or ask for
// information/explanation.
function isQuestion(text) {
  const t = text.trim().toLowerCase();
  if (t.endsWith('?')) return true;
  const first = t.split(/\\s+/).slice(0, 3).join(' ');
  return /^(what|who|when|where|why|how|which|is|are|do|does|did|can|could|should|would|tell me|explain|define|describe|summarize|compare|meaning|difference|ask|give me|list)\b/.test(first);
}

// Local Ollama brain — J-VIS answers questions with the local model.
// Returns true if the answer was spoken, false if Ollama is unavailable
// (the caller then falls back to the opencode agent).
async function ollamaChat(text) {
  setVizStatus('Consulting the local model...');
  try {
    const resp = await fetch('/ollama/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await resp.json();
    if (data.done && data.reply) {
      setState('speaking');
      setVizStatus('Answering...');
      heardLine.innerHTML = 'J-VIS: <b>' + escapeHtml(data.reply.slice(0, 400)) + '</b>';
      await speak(data.reply.slice(0, 600));
      setState('unlocked');
      setVizStatus('Listening... speak your command', 'listening');
      startCommandListening();
      return true;
    }
    return false;
  } catch (e) {
    return false;
  }
}

// ================================================================
// J-VIS Router — the custom LLM that reads the command and decides
// whether it goes to the local Ollama model or to the opencode agent.
// ================================================================
let opencodeOnly = false;  // red power mode: force opencode for everything

// Ask the jvis-router LLM which engine should handle the command.
// Falls back to isQuestion() if the router is unavailable.
async function routerDecide(text) {
  try {
    const resp = await fetch('/router/decide', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({command: text})
    });
    const data = await resp.json();
    if (data.engine === 'ollama' || data.engine === 'opencode') {
      return data.engine;
    }
  } catch (e) {}
  return isQuestion(text) ? 'ollama' : 'opencode';
}

// Special J-VIS commands handled locally (no engine needed).
function isSpecialCommand(text) {
  const t = text.trim().toLowerCase();
  return /^(describe yourself|disappear|desaparece|close yourself|change my power|red mode|power mode|normal mode|restore my power)\b/.test(t);
}

async function handleSpecialCommand(text) {
  const t = text.trim().toLowerCase();
  // "describe yourself" — J-VIS introduces itself
  if (/^describe yourself\b/.test(t)) {
    const reply = "I'm J-VIS, your virtual assistant that makes everything come true, welcome to the new world, advanced technology, how can I help you sir?";
    setState('speaking');
    setVizStatus('At your service');
    heardLine.innerHTML = 'J-VIS: <b>' + escapeHtml(reply) + '</b>';
    await speak(reply);
    setState('unlocked');
    setVizStatus('Listening... speak your command', 'listening');
    startCommandListening();
    return true;
  }
  // "disappear / close yourself / change my power" — red power mode,
  // J-VIS turns red and uses only opencode for coding.
  if (/^(disappear|desaparece|close yourself|change my power|red mode|power mode)\b/.test(t)) {
    opencodeOnly = true;
    document.body.classList.add('red-mode');
    hue = 0;  // red
    bgPalette = ['rgba(248,113,113,', 'rgba(239,68,68,', 'rgba(127,29,29,'];
    initBg();
    const reply = 'Power mode activated. I will only use opencode for coding.';
    setState('speaking');
    setVizStatus('Power mode');
    heardLine.innerHTML = 'J-VIS: <b>' + escapeHtml(reply) + '</b>';
    await speak(reply);
    setState('unlocked');
    setVizStatus('Listening... speak your command', 'listening');
    startCommandListening();
    return true;
  }
  // "normal mode / restore my power" — back to normal operation
  if (/^(normal mode|restore my power)\b/.test(t)) {
    opencodeOnly = false;
    document.body.classList.remove('red-mode');
    hue = 190;  // back to cyan
    bgPalette = ['rgba(34,211,238,', 'rgba(59,130,246,', 'rgba(168,85,247,'];
    initBg();
    const reply = 'Power mode deactivated. Returning to normal operation.';
    setState('speaking');
    setVizStatus('Normal mode');
    heardLine.innerHTML = 'J-VIS: <b>' + escapeHtml(reply) + '</b>';
    await speak(reply);
    setState('unlocked');
    setVizStatus('Listening... speak your command', 'listening');
    startCommandListening();
    return true;
  }
  return false;
}

async function handleCommand(text) {
  if (recognition) { try { recognition.stop(); } catch (e) {} }
  setState('processing');
  heardLine.innerHTML = 'You said: <b>' + escapeHtml(text) + '</b>';
  // calling J-VIS by name — acknowledge, no mission
  if (isNameCall(text)) {
    setState('speaking');
    setVizStatus('At your service');
    heardLine.innerHTML = 'J-VIS: <b>Yes sir, what can I do for you?</b>';
    await speak('Yes sir, what can I do for you?');
    setState('unlocked');
    setVizStatus('Listening... speak your command', 'listening');
    startCommandListening();
    return;
  }
  // special J-VIS commands — handled locally (describe yourself, power mode)
  if (isSpecialCommand(text)) {
    if (await handleSpecialCommand(text)) return;
  }
  setVizStatus('J-VIS is executing your order...');
  try {
    // The J-VIS Router LLM reads the command and decides the engine:
    // "ollama" (local chat) or "opencode" (coding agent). In red power
    // mode, opencode is forced for everything.
    let engine = await routerDecide(text);
    if (opencodeOnly) engine = 'opencode';
    if (engine === 'ollama') {
      if (await ollamaChat(text)) return;
    }
    const resp = await fetch('/agent/task', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: owner.id, task: text})
    });
    const data = await resp.json();
    if (data.done && data.result) {
      setState('speaking');
      setVizStatus('Mission complete');
      heardLine.innerHTML = 'J-VIS: <b>' + escapeHtml(data.result.slice(0, 400)) + '</b>';
      await speak('The job is done. ' + data.result.slice(0, 600));
      setState('unlocked');
      setVizStatus('Listening... speak your command', 'listening');
      startCommandListening();
    } else {
      await classicChat(text);
    }
  } catch (e) {
    await classicChat(text);
  }
}

// Fallback: the classic local intent pipeline (Michelle speaks the reply).
async function classicChat(text) {
  setVizStatus('Classic mode...');
  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: owner.id, message: text, use_llm: false})
    });
    const data = await resp.json();
    let reply;
    if (data.error) {
      reply = data.error;
    } else if (data.hitl_required) {
      reply = 'This action requires human confirmation. Approval ID: ' + data.approval_id;
    } else if (data.result && data.result.response) {
      reply = data.result.response;
    } else if (data.result) {
      reply = 'Done. ' + JSON.stringify(data.result).slice(0, 200);
    } else {
      reply = 'I processed your request.';
    }
    setState('speaking');
    setVizStatus('Speaking...');
    await speak(reply);
  } catch (e) {
    setVizStatus('Error: ' + e.message, 'fail');
  }
  setState('unlocked');
  setVizStatus('Listening... speak your command', 'listening');
  startCommandListening();
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ================================================================
// TTS — always Michelle
// ================================================================
function speak(text) {
  return new Promise((resolve) => {
    ttsPlaying = true;   // whisper mode must not hear Michelle as a command
    fetch('/speak/audio?text=' + encodeURIComponent(text) + '&voice=' + encodeURIComponent(VOICE))
      .then(r => { if (!r.ok) throw new Error('TTS failed'); return r.blob(); })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        audioPlayer.src = url;
        audioPlayer.onended = () => { ttsPlaying = false; resolve(); };
        audioPlayer.onerror = () => { ttsPlaying = false; resolve(); };
        audioPlayer.play().catch(() => { ttsPlaying = false; resolve(); });
      })
      .catch(() => { ttsPlaying = false; resolve(); });
  });
}

// ================================================================
// AUDIO HELPERS
// ================================================================
async function blobToWavBase64(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const audioBuffer = await tmpCtx.decodeAudioData(arrayBuffer);
    const targetRate = 16000;
    const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * targetRate), targetRate);
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start();
    const rendered = await offlineCtx.startRendering();
    const samples = rendered.getChannelData(0);
    const wav = encodeWav(samples, targetRate);
    return bytesToBase64(new Uint8Array(wav));
  } finally {
    try { tmpCtx.close(); } catch (e) {}
  }
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return buffer;
}

function bytesToBase64(bytes) {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// ================================================================
// EVENTS
// ================================================================
wRecordBtn.onclick = startRecording;
wEnrollBtn.onclick = enroll;
wPinBtn.onclick = unlockWithPin;
wPinInput.onkeydown = (e) => { if (e.key === 'Enter') unlockWithPin(); };
wPinLink.onclick = showPinFallback;
wCancelBtn.onclick = cancelAuth;

// typed missions — the writing option stays available after unlock
function sendTypedCommand() {
  if (state === 'processing') return;
  const text = cmdInput.value.trim();
  if (!text) return;
  cmdInput.value = '';
  handleCommand(text);
}
cmdSend.onclick = sendTypedCommand;
cmdInput.onkeydown = (e) => { if (e.key === 'Enter') sendTypedCommand(); };

init();
</script>
</body>
</html>
"""
@app.get("/", response_class=HTMLResponse)
async def voice_chat_page():
    """Voice chat frontend page."""
    from fastapi.responses import Response

    return Response(
        content=VOICE_PAGE,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ------------------------------------------------------------------
# Agent bridge — J-VIS delegates missions to the opencode agent
# ------------------------------------------------------------------
@app.post("/agent/task", response_model=AgentTaskResponse)
async def agent_task(req: AgentTaskRequest):
    """Delegate a mission to the opencode agent (the J-VIS brain).

    The user speaks an order to J-VIS; J-VIS transcribes it and sends it
    here. This endpoint runs the opencode CLI with the task, collects the
    agent's final text, and returns it so J-VIS can speak the results.
    """
    task = (req.task or "").strip()
    if not task:
        return AgentTaskResponse(result="", done=False, error="Empty task.")

    # Power mode mission — deterministic, no agent round-trip needed.
    task_lower = task.lower()
    for cmd in POWER_TO_RED_COMMANDS:
        if task_lower.startswith(cmd):
            power_mode.set_power_mode("red", POWER_MODE_PATH)
            if auth_db:
                audit.log("system", "master", "set_power_mode", "allow", {"mode": "red", "source": "agent_task"})
            return AgentTaskResponse(
                result="Power changed. RED POWER MODE activated — J-VIS is now in red mode using opencode only. Use 'normal mode' or 'restore my power' to switch back.",
                done=True,
            )
    for cmd in POWER_TO_NORMAL_COMMANDS:
        if task_lower.startswith(cmd):
            power_mode.set_power_mode("normal", POWER_MODE_PATH)
            if auth_db:
                audit.log("system", "master", "set_power_mode", "allow", {"mode": "normal", "source": "agent_task"})
            return AgentTaskResponse(
                result="Power restored. Normal mode activated — J-VIS routing is back to normal.",
                done=True,
            )

    prompt = (
        f"Mission from the user: {task}. Execute it and reply with a concise "
        "summary of what you did and the final results."
    )
    if AGENT_SYSTEM_PROMPT.strip():
        prompt = f"{AGENT_SYSTEM_PROMPT.strip()}\n\n{prompt}"
    cmd = [AGENT_CMD, "run", prompt, "--format", "json", "--dir", AGENT_WORKDIR]
    if AGENT_AUTO_APPROVE:
        cmd.append("--auto")

    logger.info(f"Agent task delegated: {task[:120]}")
    # Strip OPENCODE* env vars so the CLI starts a fresh session instead of
    # attaching to any running opencode server (e.g. this one).
    clean_env = {k: v for k, v in os.environ.items() if "OPENCODE" not in k.upper()}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=AGENT_WORKDIR,
            env=clean_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=AGENT_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentTaskResponse(
                result="",
                done=False,
                error=f"Agent timed out after {AGENT_TIMEOUT}s.",
            )
    except FileNotFoundError:
        return AgentTaskResponse(
            result="",
            done=False,
            error=f"Agent command '{AGENT_CMD}' not found. Install opencode or set AGENT_CMD.",
        )
    except Exception as e:
        logger.error(f"Agent task error: {e}")
        return AgentTaskResponse(result="", done=False, error=str(e))

    # Parse newline-delimited JSON events; collect the agent's text parts.
    texts = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        if evt.get("type") == "text":
            part = evt.get("part") or {}
            txt = part.get("text", "")
            if txt:
                texts.append(txt)

    result = "\n".join(texts).strip()
    if not result:
        err = stderr.decode("utf-8", errors="replace").strip()
        return AgentTaskResponse(
            result="",
            done=False,
            error=err[-500:] if err else "Agent produced no output.",
        )
    logger.info(f"Agent task completed ({len(result)} chars)")
    return AgentTaskResponse(result=result, done=True)


# ------------------------------------------------------------------
# Ollama bridge — J-VIS consults the local Ollama model.
# Fully local: this endpoint proxies to the Ollama API on the same
# machine (default http://localhost:11434). No cloud involved.
# ------------------------------------------------------------------
@app.post("/ollama/chat", response_model=OllamaChatResponse)
async def ollama_chat(req: OllamaChatRequest):
    """Ask the local Ollama model a question and return its reply."""
    message = (req.message or "").strip()
    if not message:
        return OllamaChatResponse(reply="", done=False, error="Empty message.")
    system = (req.system or OLLAMA_SYSTEM).strip()
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        "stream": False,
    }
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    http_req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info(f"Ollama chat asked ({OLLAMA_MODEL}): {message[:120]}")
    try:
        with urllib.request.urlopen(http_req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Ollama chat error: {e}")
        return OllamaChatResponse(reply="", done=False, error=str(e))
    reply = (data.get("message") or {}).get("content", "").strip()
    if not reply:
        return OllamaChatResponse(reply="", done=False, error="Ollama returned no reply.")
    logger.info(f"Ollama chat answered ({len(reply)} chars)")
    return OllamaChatResponse(reply=reply, done=True)


# ------------------------------------------------------------------
# J-VIS Router — the custom LLM that reads the command and decides
# whether it goes to the local Ollama model or to the opencode agent.
# The decision comes from the jvis-router model (an LLM); if it is
# unavailable, a keyword fallback built from dataset.csv is used.
# ------------------------------------------------------------------
@app.post("/router/decide", response_model=RouterDecideResponse)
async def router_decide(req: RouterDecideRequest):
    """Read the command and decide: ollama (local chat) or opencode (agent)."""
    command = (req.command or "").strip()
    if not command:
        return RouterDecideResponse(engine="ollama", reason="Empty command.", source="fallback")
    payload = {
        "model": ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": command},
        ],
        "stream": False,
        "format": "json",
    }
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    http_req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info(f"Router deciding ({ROUTER_MODEL}): {command[:120]}")
    try:
        with urllib.request.urlopen(http_req, timeout=ROUTER_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data.get("message") or {}).get("content", "").strip()
        decision = json.loads(content)
        engine = str(decision.get("engine", "")).strip().lower()
        reason = str(decision.get("reason", "")).strip()
        if engine not in ("ollama", "opencode"):
            raise ValueError(f"Bad engine: {engine}")
        logger.info(f"Router decided: {engine} ({reason})")
        return RouterDecideResponse(engine=engine, reason=reason, source="llm")
    except Exception as e:
        logger.warning(f"Router LLM failed ({e}) — using keyword fallback")
        engine = _keyword_fallback(command)
        return RouterDecideResponse(engine=engine, reason="Keyword fallback.", source="fallback")


# ------------------------------------------------------------------
# Speech-to-text — server-side Whisper fallback for vocal orders.
# The browser tries SpeechRecognition first; if it is unavailable or
# fails, J-VIS records the utterance and transcribes it here.
# ------------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium
_whisper_model = None
_whisper_lock = asyncio.Lock()


def get_whisper_model():
    """Lazily load the faster-whisper model (first call downloads it)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        logger.info(f"Whisper model '{WHISPER_MODEL}' loaded")
    return _whisper_model


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    """Transcribe a WAV audio sample (base64) using faster-whisper."""
    import base64
    import io

    try:
        wav_bytes = base64.b64decode(req.audio_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload.")

    async with _whisper_lock:  # one model, one transcription at a time
        try:
            model = get_whisper_model()
            segments, _info = model.transcribe(
                io.BytesIO(wav_bytes), language="en", vad_filter=True
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    logger.info(f"Transcribed ({len(text)} chars): {text[:120]}")
    return TranscribeResponse(text=text)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )