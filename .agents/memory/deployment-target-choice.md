---
name: Deployment target — single persistent instance (Docker on Render), not autoscale
description: Why this app must run as one stateful instance, and how it deploys via Docker on Render (current) — and previously as a Replit Reserved VM (legacy)
---

# Deployment target: single stateful instance, never autoscale

## Current: Docker on Render (the app left Replit)
The app deploys via the repo's `Dockerfile` + `render.yaml` Blueprint as a single
Render **Web Service** (the equivalent of the old Reserved VM — one persistent
instance, not autoscaled to many).
- One image bundles all four runtimes: Python 3.12, ffmpeg, deno, Node. The
  bgutil PO-token provider is cloned + built into `vendor/` at image build time.
- `scripts/start.sh` is the entrypoint: backgrounds `run_pot_provider.sh` then
  `exec gunicorn --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:$PORT app:app`.
- **Workers MUST stay at 1** (in-memory `download_jobs` dict). Render injects
  `$PORT`; bind gunicorn to it. `healthCheckPath: /`.
- Free plan cold-starts after idle; bump to `starter` for always-on. 512 MB RAM
  is fine for typical songs; very long tracks can OOM the STFT (see app.py).
- Render is also a datacenter IP, so YouTube anti-bot behaves like it did on
  Replit — deno + android_vr + PO tokens is still the working path.

## Legacy: Replit Reserved VM (no longer used)
Originally published on Replit as a **Reserved VM**, not autoscale.

## Why VM (do not switch back to autoscale)
The YouTube download flow is **stateful and asynchronous**:
- `POST /youtube-download` starts a **background thread** and returns a job id immediately.
- Progress + results live in an **in-memory dict** (`download_jobs`); the frontend polls
  `/youtube-status` for it.
- A **companion localhost service** (bgutil PO-token provider on 127.0.0.1:4416) must run
  alongside the web server.

Autoscale (Cloud Run) breaks all three: it is stateless (polls can hit a different instance →
"job not found"), it freezes/kills CPU + background threads once the HTTP response is sent (so
downloads stall), and it serves a **single port** with a **single run command** (so the
companion service never runs). A publish on autoscale fails the promote/health-check step.

## How production runs both processes
Deployment run command (set via deployConfig, lives in `.replit` `[deployment]`):
`bash -c "bash scripts/run_pot_provider.sh & exec gunicorn --bind=0.0.0.0:5000 --reuse-port app:app"`
- PO provider backgrounded with `&`; `exec gunicorn` is the foreground process so it answers the
  startup probe on `GET /` immediately (verified: 200 in ~11ms).
- The provider script self-heals (builds vendor/ if missing) but the deployment image already
  contains the prebuilt `vendor/` (gitignored files ARE included in Replit's filesystem snapshot).
- If the provider is down, downloads still work via yt-dlp's android_vr fallback — graceful.

## Operational notes
- The deployment **type** (autoscale/vm) can't be changed purely in code — the user must select
  **Reserved VM** in the Publishing/Deployments pane before publishing.
- `.replit` can't be edited directly (port mappings/run command are tool-owned). Port entries are
  auto-managed; VM tolerates multiple ports, so leftover port mappings don't block a VM publish.
- Workflows (dev) ≠ deployment (prod): the dev "POT Provider" workflow does not run in prod; only
  the single `[deployment]` run command does — hence bundling both processes into it.
