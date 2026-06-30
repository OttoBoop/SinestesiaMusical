---
name: Deployment target — why this app needs Reserved VM, not autoscale
description: When an app must be a Reserved VM instead of autoscale on Replit, and how to run a companion service in production
---

# Deployment target: Reserved VM vs autoscale

This Musical Frequency Visualizer must be published as a **Reserved VM**, not autoscale.

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
