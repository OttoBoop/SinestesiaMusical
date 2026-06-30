#!/usr/bin/env bash
# Production entrypoint: run the PO-token provider alongside the web server in a
# single container. This is the Docker equivalent of the old Replit deployment
# run command ("bash run_pot_provider.sh & exec gunicorn ...").
set -uo pipefail

# Start the bgutil PO-token provider in the background. It self-heals (rebuilds
# vendor/ if missing), but the image already ships it prebuilt, so this is
# instant. If it ever fails, YouTube downloads still work via yt-dlp's
# android_vr path — so we never let it take the container down.
bash scripts/run_pot_provider.sh &

# Foreground web server. ONE worker on purpose: download-job state lives in an
# in-memory dict, so a second worker would 404 the status polls (the same reason
# the app needed a Reserved VM and not autoscale on Replit). Threads handle the
# concurrent status polling while a download runs in its own background thread.
# --timeout 180 gives the synchronous /upload analysis room for large files.
exec gunicorn \
    --bind "0.0.0.0:${PORT:-5000}" \
    --workers 1 \
    --threads 8 \
    --timeout 180 \
    app:app
