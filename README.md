# JARVIS

> **Current development line: NEXUS V2 / TABY.** The repository now contains the integrated owner-authority kernel, adaptive Gemini/Ollama routing, TABY/FRIDAY/JARVIS personas, durable semantic memory/sync, signed peer authentication, explicit phone/device pairing, and the typed Windows broker contract. Physical Windows microphone/camera/iGPU/admin execution remains deliberately uncertified until tested on the target machine.

## NEXUS V2

The cloud-tested NEXUS V2 stack is developed in this repository and is no longer a separate experimental project. Its main architecture is:

- **TABY** — the single user-visible identity and runtime presence.
- **JARVIS** — strategic/engineering cognition profile.
- **FRIDAY** — low-latency realtime cognition profile.
- **OwnerKernel / ActionGateway** — exact-consent, capability-based authority. Model text cannot grant privileges.
- **Adaptive model stack** — Gemini + local Ollama, current hardware snapshots, warm/cold runtime lifecycle and deterministic fallbacks.
- **Semantic local-first sync** — vector clocks, deterministic merge policies, transactional materialization and per-peer ACK/retry.
- **Device trust** — Ed25519-signed sync, explicit trusted-peer registry, revocation and one-time pairing.
- **Windows broker contract** — short-lived signed permits for typed capabilities only; there is no generic shell/PowerShell execution surface.

Cloud CI and the full legacy regression gate exercise these layers together. The manual Windows hardware-certification workflow remains a release gate for functionality that cannot be honestly proven in GitHub-hosted runners.


Local Gemini Live desktop assistant with a PyQt6 interface, voice interaction, detachable panels, and optional browser, file, screen, and messaging tools.

JARVIS also includes a dedicated presentation studio that creates, edits,
redesigns, and extends editable widescreen `.pptx` decks from documents, data,
images, audio, and video, with optional PDF export. See the
[usage guide](docs/USAGE.md#6-powerpoint-presentations) for examples.

## Requirements

You need **Python 3.11 or newer** installed to set up and run JARVIS. Confirm
your Python version before continuing:

```bash
python --version
```

## Quick start (Windows, macOS, Linux)

In Terminal, run:

```bash
git clone https://github.com/MAL19INDUSTRIES/JARVIS-OS-V.2.git
cd JARVIS-OS-V.2
python scripts/setup_jarvis.py
```

On Windows, you can double-click `scripts/setup_jarvis.bat` instead.

Open `.env`, add your `GEMINI_API_KEY`, then launch JARVIS:

```bash
jarvis
```

You only need to run setup once. Activate `.venv` when opening a new terminal,
then type `jarvis`.

JARVIS's core UI, Gemini connection, presentations, research, files, and CLI are
cross-platform. Some computer-control, email, media, and browser integrations
depend on permissions and available applications on each operating system.

## Hosted web application

The repository also contains a multi-user FastAPI service and a Next.js web
client. Hosted sessions use Postgres for user-scoped memory and configuration,
Redis for request quotas, encrypted per-user Gemini keys, and Gemini Live over
an authenticated WebSocket. The desktop launcher continues to use its local
stores and full local action inventory.

Start the complete local web stack with Docker:

```bash
docker compose up --build
```

Then open `http://localhost:3000`. To run each service directly:

```bash
# API
cp .env.example .env
alembic upgrade head
uvicorn api.server:app --reload

# Web client
cd web
cp .env.example .env.local
npm install
npm run dev
```

Production templates are included for Fly.io (`fly.toml`), Render
(`render.yaml`), and Vercel (`web/vercel.json`). Configure `DATABASE_URL`,
`REDIS_URL`, `JWT_SECRET`, `JARVIS_ENCRYPTION_KEY`, and `CORS_ORIGINS` on the
API host. Configure `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL` on Vercel.
The deployment workflow runs manually after the Fly and Vercel repository
secrets have been added.

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
cp .env.example .env
./scripts/install_jarvis_cli.sh
jarvis
```

Set `GEMINI_API_KEY` in `.env` before launch. Optional settings such as voice and local API keys are documented in `.env.example`.

### Launch with `jarvis`

The CLI launcher is included in this repository. After cloning and completing
the one-time setup, install it for your user with:

```bash
./scripts/install_jarvis_cli.sh
```

Open a new terminal (or reload your shell profile), then start JARVIS with:

```bash
jarvis
```

Before packaging or releasing the desktop app, run the side-effect-safe
capability audit:

```bash
jarvis --self-test
```

The audit exercises voice/tool contracts, messaging routing and approval
boundaries, a local browser interaction, isolated file operations, vision,
agent recovery, and memory. It never sends a real message or performs a live
desktop mutation. Results that still require a person, account, or physical
device are labeled `LIVE CHECK REQUIRED`, and a JSON report is written under
`.qa-artifacts/`.

Alternatively, from an activated virtual environment, `python3 -m pip install -e .`
installs the same `jarvis` command through the standard Python package entry point.

## Documentation

- [Usage guide](docs/USAGE.md)
- [Tutorial](docs/TUTORIAL.md)
- [Complete QA and bug-audit guide](docs/QA.md)
- [Contribution notes](CONTRIBUTING.md)

## Configuration files

Template files are included for local setup:

- `.env.example`
- `config/api_keys.example.json`
- `config/layout_settings.example.json`
- `config/ui_settings.example.json`
- `memory/long_term.example.json`
- `memory/task_history.example.json`

## Publishing checklist

- Keep `.env` and local secret files out of git.
- Do not commit `memory/long_term.json` or `config/api_keys.json`.
- Run `python3 -m py_compile main.py ui.py` before tagging a release.

## License

MIT License, see [LICENSE](LICENSE).
