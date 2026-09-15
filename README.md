# J-VIS — Voice-first AI Assistant

**J-VIS** is a multi-role, voice-first AI assistant with Role-Based Access Control (RBAC), biometric identity, sandboxed execution, immutable audit logging, and a **custom LLM router** that balances between a local Ollama model and the opencode coding agent.

> **SECURITY IS THE PRODUCT.** The LLM NEVER executes anything directly. Every action passes through the deterministic Policy Gateway. No exceptions.

---

## Current Status

| Component | Status |
|-----------|--------|
| API Server | ✅ Running on `http://localhost:8000` |
| Registered Tools | ✅ 10 tools |
| RBAC Enforcement | ✅ 4-tier model verified |
| Audit Chain | ✅ Valid (hash-verified at boot) |
| Test Suite | ✅ **153 tests passing** |
| Demo Users | ✅ 4 seeded (master/operator/user/guest) |
| Voice-first UI | ✅ Clap to awaken, cipher unlock, PIN fallback |
| Local Ollama Brain | ✅ `qwen2.5-coder` answers questions locally |
| opencode Agent Bridge | ✅ Coding missions delegated to the opencode agent |
| **J-VIS Router LLM** | ✅ Custom `jvis-router` model decides Ollama vs opencode |
| Power Mode | ✅ "change my power" → red theme + opencode-only |

---

## Architecture

```
[Voice/Vision Layer] --> [Intent Schema] --> [POLICY GATEWAY] --> [Tool Executor]
        |                        |                    |
        |                        |                    +--> (RBAC + Biometrics + HITL)
        |                        |                    |
        |                        |                    +--> [Sandbox / System APIs]
        |                        |                    |
        |                        +--> [Immutable Audit Log]
```

### Command Routing (the J-VIS Router)

```
[User command (voice or typed)]
        |
        v
[Special commands?]  describe yourself / change my power / disappear ...
        | (no)
        v
[J-VIS Router LLM — jvis-router]  reads the command, outputs JSON decision
        |
        +--> "ollama"   --> local Ollama model (qwen2.5-coder) answers
        |
        +--> "opencode" --> opencode agent executes the coding mission
        |
        +--> fallback   --> keyword matching from router/dataset.csv
```

- **Questions** (what/why/how/explain/define/compare/teach me ...) → **Ollama** (fully local)
- **Coding missions** (create/fix/build/deploy/test + any language/framework/tool) → **opencode agent**
- **Red power mode** forces everything to opencode
- If the router LLM is down, a keyword fallback built from `router/dataset.csv` (529 keywords) takes over

### Dual-Layer Design

| Layer | Responsibility |
|-------|---------------|
| **Voice/Vision Layer** | Wake word detection, STT (faster-whisper), TTS (edge-tts), voiceprint (SpeechBrain), face ID (OpenCV) |
| **Intent Schema** | LLM outputs ONLY structured JSON intents — never commands |
| **Policy Gateway** | Deterministic RBAC enforcement, HITL triggers, default-deny |
| **Tool Executor** | Whitelisted tools with validated args (pydantic) |
| **Sandbox** | Docker containers with no network, memory limits, read-only root FS |
| **Audit Log** | Append-only JSONL with SHA-256 hash chaining |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| API Backend | FastAPI + Uvicorn |
| STT | faster-whisper (local, "base" or "small") + browser SpeechRecognition |
| Voiceprint | SpeechBrain (ECAPA-TDNN) |
| Face ID | OpenCV + face_recognition |
| Local LLM | Ollama (`qwen2.5-coder:latest`) — fully local, no cloud |
| **Router LLM** | Custom `jvis-router` model (Ollama Modelfile) |
| Coding Agent | opencode CLI (agent bridge) |
| Agent Framework | LangChain (thin wrapper) |
| Policy Engine | Custom Policy Gateway (Casbin-compatible CSV) |
| Sandboxing | Docker SDK (primary), subprocess fallback |
| Audit Log | Append-only JSONL + SHA-256 hash chaining |
| Storage | SQLite (users, roles, sessions) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally (`http://localhost:11434`)
- [opencode](https://opencode.ai) CLI available on PATH

### Setup (Windows)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env from .env.example
copy .env.example .env

# 3. Pull the local model
ollama pull qwen2.5-coder:latest

# 4. Create the J-VIS Router LLM (your custom decision model)
ollama create jvis-router -f router/Modelfile

# 5. Run the server
python main.py
```

### Setup (Linux/macOS)

```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5-coder:latest
ollama create jvis-router -f router/Modelfile
python main.py
```

### Verify it's running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "tools_registered": 10,
  "audit_chain_valid": true,
  "demo_users": 4
}
```

---

## The Voice-First Interface

Open `http://localhost:8000` in a browser:

1. **Black screen** — clap twice (or tap) to awaken J-VIS
2. **Say the cipher** — "J-VIS reveal my power" (or "awaken" / "unlock") to unlock
3. **PIN fallback** — the PIN link is always available (demo PINs below)
4. **Give orders** — by voice or by typing in the command bar

### Special Commands

| Command | Effect |
|---------|--------|
| `describe yourself` | J-VIS introduces itself: *"I'm J-VIS, your virtual assistant that makes everything come true, welcome to the new world, advanced technology, how can I help you sir?"* |
| `change my power` / `disappear` / `close yourself` / `red mode` | **RED POWER MODE** — interface turns red, opencode-only routing |
| `normal mode` / `restore my power` | Back to normal (cyan) operation |

Power mode is persisted server-side (`power_mode.py` → `jvis_power_mode.json`) and enforced on every command.

---

## Demo Users

| Username | Role | PIN | Permissions |
|----------|------|-----|-------------|
| `master_demo` | master | `0000` | Unrestricted (firewall, reboot, policy changes) |
| `operator_demo` | operator | `1111` | Backup scripts, cameras, DB queries |
| `user_demo` | user | `2222` | Calendar, emails, private docs |
| `guest_demo` | guest | `3333` | Weather, web search, music |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Process a chat message (intent -> gateway -> tool) |
| `POST` | `/agent/task` | Delegate a mission to the opencode agent (J-VIS brain) |
| `POST` | `/ollama/chat` | Ask the local Ollama model (fully local, no cloud) |
| `POST` | `/router/decide` | **J-VIS Router LLM** — decides `ollama` vs `opencode` |
| `POST` | `/transcribe` | Server-side Whisper STT fallback |
| `POST` | `/enroll/voice` | Enroll a voiceprint for a user |
| `POST` | `/enroll/face` | Enroll a face encoding for a user |
| `POST` | `/session/bootstrap` | Create a session (biometrics or PIN) |
| `POST` | `/approve` | Confirm a HITL approval |
| `GET` | `/audit/log` | View recent audit log entries |
| `GET` | `/audit/verify` | Verify audit log hash chain integrity |
| `GET` | `/health` | Health check |
| `GET` | `/users` | List users |
| `GET` | `/tools` | List registered tools |
| `GET` | `/approvals` | List pending HITL approvals |

### Example: Router decision

```bash
curl -X POST http://localhost:8000/router/decide \
  -H "Content-Type: application/json" \
  -d '{"command": "What is a REST API?"}'
# {"engine": "ollama", "reason": "Definition of a REST API.", "source": "llm"}

curl -X POST http://localhost:8000/router/decide \
  -H "Content-Type: application/json" \
  -d '{"command": "Create a React component for the navbar"}'
# {"engine": "opencode", "reason": "Action verb 'create' with framework 'React'", "source": "llm"}
```

### Example: Chat

```bash
# Get a user ID
curl http://localhost:8000/users

# Send a chat message (direct mode, no LLM)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<guest_user_id>", "message": "weather in Tokyo", "use_llm": false}'

# With LLM (requires Ollama running)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<guest_user_id>", "message": "What is the weather in Tokyo?"}'
```

### Example: HITL Approval Flow

```bash
# 1. Operator requests a dangerous action
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<operator_user_id>", "message": "delete file /tmp/test.txt"}'

# Response includes approval_id
# {"action": "delete_file", "hitl_required": true, "approval_id": "ABC123DEF456", ...}

# 2. Confirm within 5 minutes
curl -X POST http://localhost:8000/approve \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<operator_user_id>", "approval_id": "ABC123DEF456"}'
```

---

## The J-VIS Router (your custom LLM)

The router is a **real LLM** created from `router/Modelfile` (based on `qwen2.5-coder`), registered in Ollama as `jvis-router:latest`.

- It reads the command coming from J-VIS and outputs a JSON decision:
  `{"engine": "ollama"|"opencode", "reason": "..."}`
- Its system prompt encodes 18 routing rules (question words → ollama, action verbs/languages/frameworks/tools/databases/DevOps → opencode)
- **Training/guidance data:** `router/dataset.csv` — **529 coding keywords** across 29 categories (question words, code actions, languages, frameworks, tools, databases, security, DevOps, mobile, data/AI, special commands)
- **Fallback:** if the router LLM is unavailable, `_keyword_fallback()` in `main.py` matches the command against the dataset keywords (36 ollama + 485 opencode)

### Rebuilding the router model

```bash
ollama create jvis-router -f router/Modelfile
```

---

## RBAC Model

| Role | Tier | Scope | Example Actions |
|------|------|-------|-----------------|
| `master` | 3 (Root) | Unrestricted | firewall rules, server reboot, policy changes |
| `operator` | 2 (Elevated) | Functional | run backup scripts, toggle cameras, DB queries |
| `user` | 1 (Standard) | Personal read/write | calendar, draft emails, summarize private docs |
| `guest` | 0 (Sandbox) | Isolated read-only | weather, web search, play music |

### Rules

- Every tool/action is tagged with a required role.
- Gateway check: `session.role >= action.required_role`, else **DENY**.
- Denials are logged and the assistant replies with a natural-language refusal.
- **Default deny**: unknown actions are denied and logged.
- **All destructive ops require HITL** regardless of role (even master).

### Registered Tools

| Tool | Required Role | HITL | Sandbox |
|------|--------------|------|---------|
| `get_weather` | guest | No | No |
| `web_search` | guest | No | No |
| `play_music` | guest | No | No |
| `list_files` | user | No | No |
| `read_file` | user | No | No |
| `manage_calendar` | user | No | No |
| `run_backup_script` | operator | No | Yes |
| `delete_file` | operator | **Yes** | Yes |
| `reboot_server` | master | **Yes** | No |
| `run_python_snippet` | operator | No | **Always** |

---

## Biometric Enrollment Walkthrough

### Voice Enrollment (SpeechBrain ECAPA-TDNN)

J-VIS uses **SpeechBrain's ECAPA-TDNN** (trained on VoxCeleb) for speaker
verification. Each voice sample is encoded into a **192-dimensional speaker
embedding** — a vector capturing the unique characteristics of a voice
(pitch, timbre, cadence, vocal tract geometry). Verification uses **cosine
similarity** between embeddings: ≥ 0.85 = same speaker.

**Install:**
```bash
pip install speechbrain torch
```

**Option A — CLI (recommended for enrollment):**
```bash
# Enroll from 3 samples (averaged for robustness)
python scripts/voice_enroll.py enroll --user master_demo sample1.wav sample2.wav sample3.wav

# Verify a live sample
python scripts/voice_enroll.py verify --user master_demo live_sample.wav

# Identify which user a voice belongs to (1:N)
python scripts/voice_enroll.py identify sample.wav

# Show enrollment status
python scripts/voice_enroll.py status
```

**Option B — API:**
```bash
# 1. Capture 3 voice samples (e.g., via microphone)
# 2. Encode each sample and send to the enrollment endpoint
curl -X POST http://localhost:8000/enroll/voice \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<user_id>", "voice_sample_b64": "<base64_audio>"}'
```

**How it works:**
1. **Enrollment:** ECAPA-TDNN encodes each sample into a 192-dim embedding → 3 samples are averaged into one reference embedding → stored in SQLite.
2. **Verification:** A live sample is encoded → cosine similarity computed against the stored reference → ≥ 0.85 = MATCH.
3. **Identity:** Once verified, the user's role is loaded and a session token is issued (bound to the biometric hash).

**Measured results (edge-tts voices):**
- Same speaker (Jenny vs Jenny): similarity **0.97** → MATCH
- Different speaker (Jenny vs Guy): similarity **0.46** → REJECTED

### Face Enrollment

```bash
# 1. Capture a face image via webcam
# 2. Send the image path to the enrollment endpoint
curl -X POST http://localhost:8000/enroll/face \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<user_id>", "face_image_path": "/path/to/face.jpg"}'
```

### Session Bootstrap

```bash
# With PIN (fallback)
curl -X POST http://localhost:8000/session/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<user_id>", "pin": "2222"}'

# With voice sample
curl -X POST http://localhost:8000/session/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<user_id>", "voice_sample_b64": "<base64_audio>"}'
```

**Fallback chain:** Voiceprint → Face → PIN. **Never falls back to nothing.**

---

## Security Model

### Non-Negotiable Rules

1. **LLM output is DATA, never commands.** Only the registry whitelist executes.
2. **Default deny**: unknown action = deny + log.
3. **Guest tier**: no filesystem, no network beyond whitelisted APIs, no shell.
4. **All destructive ops require HITL** regardless of role (even master).
5. **Audit log**: hash chain verified at startup; mismatch = refuse to boot.
6. **No secrets in code, prompts, logs, or LLM context.**
7. **Redaction runs both ways** (scrub tool outputs containing keys/PII before they reach the LLM too).

### Redaction Pipeline

The redaction pipeline sits between user input and remote LLM providers:

- Emails, phone numbers, local IPs (10.x, 192.168.x, 127.x), API-key-shaped strings, home addresses, credit cards, SSNs
- `REDACT_REMOTE_ONLY=true` allows local Ollama calls to bypass (kept on for consistency by default)
- Original text is preserved locally for gateway use

### Sandboxing

- Tools with `sandbox: true` run inside a disposable Docker container
- `ubuntu:22.04`, **no network by default**, memory limit 512MB, 30s timeout
- Read-only root FS except `/tmp`
- Only the specific path the tool needs is mounted
- `run_python_snippet` is **ALWAYS sandboxed**

### Audit Log

- Append-only JSONL with SHA-256 hash chaining
- Each entry includes: `{ts, user_id, role, action, decision, context_hash, prev_hash, entry_hash}`
- `verify()` detects any tampering
- Chain verified at startup; mismatch = refuse to boot

---

## Threat Model

| Threat | Mitigation |
|--------|-----------|
| **Prompt injection** (LLM told to execute arbitrary code) | LLM output is parsed as JSON intent only; only whitelisted tools execute |
| **Privilege escalation** (guest tries to access admin tools) | RBAC tier check in gateway; default deny |
| **Data exfiltration** (LLM leaks PII to remote provider) | Redaction pipeline scrubs PII before remote calls |
| **Destructive actions** (delete/reboot) | HITL confirmation with 5-min TTL |
| **Audit tampering** (attacker modifies logs) | SHA-256 hash chaining; verify() at startup |
| **Sandbox escape** (malicious code) | Docker isolation, no network, memory limits, read-only root FS |
| **Session hijacking** | HMAC-signed tokens, short TTL (15 min), bound to biometric hash |
| **DoS / abuse** | Rate limiting (30 req/min per session) |
| **Secret leakage** | No hardcoded secrets; .env + python-dotenv; .gitignore |
| **Biometric spoofing** | Voiceprint + face verification; PIN fallback (never nothing) |

---

## Testing

```bash
# Run all tests (153 tests)
python -m pytest tests/ -v

# Run a specific phase
python -m pytest tests/test_comprehensive.py -v   # Phase 1 (46 tests)
python -m pytest tests/test_phase2.py -v          # Phase 2 (14 tests)
python -m pytest tests/test_phase3.py -v          # Phase 3 (13 tests)
python -m pytest tests/test_phase4.py -v          # Phase 4 (10 tests)
python -m pytest tests/test_phase5.py -v          # Phase 5 (18 tests)
python -m pytest tests/test_phase6.py -v          # Phase 6 (17 tests)
python -m pytest tests/test_phase7.py -v          # Phase 7 (23 tests)
```

**153 tests** covering all security invariants across all 7 build phases plus the agent bridge, Ollama bridge, and router.

---

## Project Structure

```
J-VIS/
├── main.py                  # FastAPI entrypoint (agent bridge, Ollama, router)
├── power_mode.py            # Persistent power mode (normal / red)
├── router/
│   ├── Modelfile            # jvis-router LLM definition (Ollama)
│   └── dataset.csv          # 529 coding keywords for routing (training data)
├── gateway/
│   ├── policy_gateway.py    # Policy enforcement (RBAC + HITL)
│   └── rate_limit.py        # Rate limiting middleware (30 req/min)
├── auth/
│   ├── models.py            # Users, roles, sessions (SQLite)
│   ├── biometrics.py        # Voiceprint + face verification
│   └── tokens.py            # HMAC-signed session tokens
├── tools/
│   └── registry.py          # Tool registry + executors
├── sandbox/
│   └── docker_executor.py   # Docker sandbox (subprocess fallback)
├── redaction/
│   └── pipeline.py          # PII scrubber
├── audit/
│   └── logger.py            # Hash-chained audit logger
├── llm/
│   └── provider.py          # Pluggable LLM (Ollama/OpenAI)
├── voice/
│   └── loop.py              # Wake word, STT, TTS
├── configs/
│   ├── rbac_policy.csv      # RBAC policy
│   └── tool_manifest.json   # Tool definitions
├── tests/                   # 153 tests across 7 phase files
│   ├── test_comprehensive.py
│   ├── test_phase2.py ... test_phase7.py
│   └── mock_llm.py          # Deterministic mock LLM for tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL (bridge) |
| `OLLAMA_MODEL` | `qwen2.5-coder:latest` | Local chat model |
| `OLLAMA_TIMEOUT` | `120` | Ollama chat timeout (seconds) |
| `ROUTER_MODEL` | `jvis-router:latest` | J-VIS Router LLM model |
| `ROUTER_TIMEOUT` | `30` | Router decision timeout (seconds) |
| `ROUTER_DATASET` | `D:\J-VIS\router\dataset.csv` | Keyword fallback dataset |
| `AGENT_CMD` | `opencode` | Coding agent command |
| `AGENT_WORKDIR` | `D:\J-VIS` | Agent working directory |
| `AGENT_TIMEOUT` | `300` | Agent timeout (seconds) |
| `AGENT_AUTO_APPROVE` | `true` | Auto-approve agent actions |
| `WHISPER_MODEL` | `base` | faster-whisper model size |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4` | OpenAI model |
| `REDACT_REMOTE_ONLY` | `true` | Only redact for remote providers |
| `SANDBOX_ENABLED` | `true` | Enable sandboxed execution |
| `SANDBOX_MEMORY_LIMIT` | `512m` | Sandbox memory limit |
| `SANDBOX_TIMEOUT` | `30` | Sandbox timeout (seconds) |
| `SANDBOX_IMAGE` | `ubuntu:22.04` | Sandbox container image |
| `SESSION_TTL_MINUTES` | `15` | Session token TTL |
| `SESSION_SECRET` | *(dev fallback)* | HMAC signing secret |
| `AUDIT_LOG_PATH` | `audit/jvis_audit.jsonl` | Audit log location |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per minute per session |
| `HITL_TTL_MINUTES` | `5` | HITL approval TTL |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable auto-reload |

---

## Roadmap / TODOs

- [ ] Real weather API integration (currently mock)
- [ ] Real web search API integration (currently mock)
- [ ] Real music provider integration (currently mock)
- [ ] Real calendar integration (currently mock)
- [ ] Real backup script execution (currently mock)
- [x] SpeechBrain voiceprint encoding (ECAPA-TDNN, 192-dim embeddings, cosine similarity)
- [ ] face_recognition face encoding (requires optional deps)
- [ ] faster-whisper STT integration (requires optional deps)
- [x] edge-tts TTS integration (human-like neural voices)
- [x] opencode agent bridge (coding missions)
- [x] Local Ollama brain (fully local questions)
- [x] Custom J-VIS Router LLM (ollama vs opencode)
- [x] Power mode (red theme + opencode-only)
- [ ] openwakeword wake word detection (requires optional deps)
- [ ] Casbin policy engine integration (currently custom gateway)
- [ ] OPA policy engine integration (alternative to Casbin)
- [ ] Per-user random salt for PIN hashing
- [ ] POSIX resource limits for subprocess sandbox fallback
- [ ] WebSocket streaming for real-time voice chat
- [ ] Multi-user concurrent session support
- [ ] Session token rotation on privilege change

---

## Build Phases (Completed)

| Phase | Deliverable | Tests |
|-------|-------------|-------|
| **1** | Foundation & Policy Gateway — RBAC, tool registry, audit logger | 46 |
| **2** | Conversational Core — LLM intent schema, /chat endpoint | 14 |
| **3** | Privacy / Redaction Pipeline — PII scrubber | 13 |
| **4** | HITL Triggers — dangerous tool confirmation with TTL | 10 |
| **5** | Biometrics — voiceprint + face + PIN fallback | 18 |
| **6** | Voice Loop & Sandboxed Execution — Docker isolation | 17 |
| **7** | Hardening & Docs — rate limiting, HMAC tokens, README | 23 |
| **8** | Agent Bridge & Local AI — opencode agent, Ollama brain, J-VIS Router LLM, power mode | 12 |

---

## License

Proprietary. For internal use only.

**J-VIS** — Voice-first AI Assistant. Security is the product.