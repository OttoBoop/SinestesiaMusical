#!/usr/bin/env bash
# Production entrypoint: bring up the WARP egress proxy + the PO-token provider,
# then run the web server. The Docker equivalent of the old Replit run command.
set -uo pipefail

# ── Egress proxy selection (to dodge YouTube's datacenter-IP block) ─────────────
# Priority: an externally provided YTDLP_PROXY (e.g. a paid residential proxy) wins;
# otherwise bring up the free, built-in Cloudflare WARP proxy if its profile exists.
if [ -n "${YTDLP_PROXY:-}" ]; then
    echo "[proxy] Using external proxy from YTDLP_PROXY=${YTDLP_PROXY}"
elif [ -f /app/wireproxy.conf ]; then
    echo "[proxy] Starting Cloudflare WARP (wireproxy)…"
    wireproxy -c /app/wireproxy.conf >/tmp/wireproxy.log 2>&1 &
    export YTDLP_PROXY="http://127.0.0.1:25345"
    sleep 2
    TRACE="$(curl -fsS --max-time 12 -x "${YTDLP_PROXY}" https://www.cloudflare.com/cdn-cgi/trace 2>/dev/null || true)"
    if echo "${TRACE}" | grep -q 'warp=on'; then
        echo "[proxy] WARP is up — egress identity:"
        echo "${TRACE}" | grep -E '^(warp|ip)='
    else
        echo "[proxy] WARNING: WARP did not come up cleanly; downloads will try it then fall back to direct. See /tmp/wireproxy.log"
    fi
else
    echo "[proxy] No WARP profile and no YTDLP_PROXY set — yt-dlp will run direct (likely blocked from a datacenter IP)."
fi

# ── PO-token provider (helps clients that need a Proof-of-Origin token) ──────────
bash scripts/run_pot_provider.sh &

# ── Web server. ONE worker on purpose: download-job state lives in an in-memory
# dict, so a second worker would 404 status polls. Threads handle concurrent status
# polling while a download runs in its own background thread. --timeout 180 gives
# the synchronous /upload analysis room for large files. Inherits YTDLP_PROXY.
exec gunicorn \
    --bind "0.0.0.0:${PORT:-5000}" \
    --workers 1 \
    --threads 8 \
    --timeout 180 \
    app:app
