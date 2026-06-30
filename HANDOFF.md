# Sinestesia Musical — Project Handoff & YouTube-Download Problem Brief

> **Purpose of this document.** It is a complete, self-contained briefing for an
> AI/engineer **with real web access and the ability to deploy + test**. It
> explains the project, where it lives, how it's deployed, and — the main event —
> the YouTube-download reliability problem: what's been tried, what failed, what's
> deployed right now, how to verify it, and what to do if it doesn't work.
>
> **The single most important task:** make YouTube → audio download work
> **reliably and seamlessly for end users** (paste a link → it just works), with
> **no paid third-party download API** and **no per-user friction** (no "press F12
> and copy cookies"). A one-time setup by the owner is acceptable.
>
> **Current status (2026-06-30):** a free **Cloudflare WARP** egress proxy was
> just implemented and deployed to `main`. It is **NOT yet verified end-to-end** —
> the author had no way to run a real download from their build sandbox. Your first
> job is to deploy/observe and confirm whether it works (see §7–§8).

---

## 1. Repository

- **GitHub:** `https://github.com/OttoBoop/SinestesiaMusical` (owner `OttoBoop`).
  If it's private, the user must grant you access.
- **Branches:**
  - `main` — production / what Render deploys from. Contains everything below.
  - `claude/code-review-explanation-so2zac` — working branch (currently identical
    to `main`).
- **Hosting:** Render (Docker web service). Migrated off Replit.

---

## 2. What the app is

A **musical frequency visualizer**. The user provides audio (file upload **or** a
YouTube search/link); the server detects the dominant pitch frame-by-frame; the
browser draws a logarithmic spiral whose **radius and color follow the melody in
real time**, synced to audio playback.

- **File upload path is rock-solid** and unaffected by any YouTube issues.
- **The YouTube path is the only fragile part** and the entire subject of this doc.

---

## 3. Tech stack & file map

| File | Role |
|------|------|
| `app.py` | Flask backend: upload, audio analysis, YouTube search/download, proxy wiring |
| `templates/index.html` | Single-page frontend (tabs: upload / YouTube search / YouTube link), canvas spiral renderer, polling logic |
| `Dockerfile` | Builds the one image: Python 3.12 + ffmpeg + deno + Node + **wgcf + wireproxy (WARP)** + bgutil PO-token provider |
| `scripts/start.sh` | Container entrypoint: starts WARP tunnel + PO provider, then gunicorn |
| `scripts/run_pot_provider.sh` | Runs the bgutil PO-token provider (Node) on `127.0.0.1:4416` |
| `scripts/setup_pot_provider.sh` | Builds the PO provider into `vendor/` (gitignored), pinned to v1.3.1 |
| `requirements.txt` | Pinned Python deps (exported from `uv.lock`) |
| `render.yaml` | Render Blueprint: Docker web service, `plan: free`, healthcheck `/` |
| `.agents/memory/*.md` | Prior decisions/history (deployment, YouTube saga) |
| `README.md` | User-facing project + deploy docs |
| `*.py` at root (PyGame/Matplotlib) | Original standalone reference scripts; NOT used in production |

**Backend dependencies of note:** `yt-dlp==2026.6.9`, `librosa`, `numpy`, `scipy`,
`flask`, `gunicorn`, `bgutil-ytdlp-pot-provider==1.3.1`. `deno` is installed as the
JS runtime yt-dlp needs.

### Key HTTP endpoints (in `app.py`)
- `GET /` → the page.
- `POST /upload` → analyze an uploaded file; returns `{times[], frequencies[]}`.
- `GET /search?q=` → `ytsearch8:` results (title/thumb/duration/url).
- `POST /youtube-download` `{url}` → starts a **background thread**, returns `{jobId}`.
- `GET /youtube-status?jobId=` → `{status, progress, times, frequencies, error}`.
- `GET /audio/<filename>` → serves the produced mp3.

### Key functions (in `app.py`)
- `analyze_frequencies()` — STFT + Harmonic Product Spectrum pitch tracking.
- `ytdl_base_opts(proxy=None)` — base yt-dlp opts; adds `proxy` when given.
- `current_proxy()` — reads `YTDLP_PROXY` env (the proxy selector).
- `download_via_ytdlp()` — audio-only download into an isolated temp dir; **tries the
  proxy first, then falls back to a direct connection**; transcodes to mp3.
- `yt_extractor_args()` — wires the bgutil PO-token provider + the player-client list.
- `friendly_youtube_error()` — maps yt-dlp errors to user messages (bot-block /
  unavailable / rate-limit) and we log the raw cause.
- `YTDLP_PLAYER_CLIENTS = ['android_vr', 'tv', 'default', 'web_safari', 'mweb']` —
  no-PO-token clients first.

---

## 4. Deployment (Docker on Render)

- One Docker image runs everything. `render.yaml` defines a single **web service**
  (Docker runtime, free plan, healthcheck `/`). Render injects `$PORT`.
- **Must stay a single instance / single gunicorn worker.** Download-job state lives
  in an **in-memory dict** (`download_jobs`) and downloads run in **background
  threads**; multiple workers/instances would 404 the status polls. `start.sh` runs
  `gunicorn --workers 1 --threads 8 --timeout 180`. **Do not enable autoscaling.**
- First build is a few minutes (installs ffmpeg/deno/Node, pip deps, builds the PO
  provider, fetches wgcf/wireproxy, registers a WARP identity).
- To run locally: `docker build -t sinestesia . && docker run --rm -p 5000:5000 sinestesia`
  then open `http://localhost:5000`. **Note:** building/testing from a datacenter IP
  reproduces the YouTube block; test WARP from the actual deploy or a residential
  connection.

---

## 5. THE CORE PROBLEM

YouTube blocks the **majority of download requests coming from datacenter IPs**
(Render is a datacenter), regardless of cookies or PO-tokens. In early 2026 YouTube
also rolled out a new streaming protocol (**SABR**) that further degrades/kills
requests it flags as automated. Reported success rates: datacenter IP ~20–40%;
residential IP ~85–95%.

**The public downloader sites (y2mate-style) succeed only because their servers
egress through pools of residential/mobile IPs** — not because of any client-side
trick. So the fix is fundamentally about **IP reputation**, i.e. making our server's
traffic look like an ordinary home connection.

---

## 6. Dead ends — DO NOT re-attempt (confirmed via research, mid-2026)

These were each investigated and ruled out. Re-litigating them wastes time.

- **Client-side / in-browser download (use the visitor's own residential IP).**
  *Impossible for a normal web page.* `googlevideo.com` never sends
  `Access-Control-Allow-Origin`, so browser JS cannot read the audio bytes —
  confirmed across `fetch({mode:'no-cors'})` (opaque/null body), YouTube.js
  (`youtubei.js`, needs a server proxy), `@distube/ytdl-core` (archived Aug 2025),
  and Web Audio `createMediaElementSource`/`captureStream` (outputs silent zeros for
  cross-origin media). The only people who cross this wall ship a **browser
  extension** (e.g. LuanRT/kira's "ytc-bridge") — which violates "seamless, no
  install."
- **Harvesting the visitor's YouTube cookies/login.** A site can only read its own
  cookies; ours can't read `youtube.com`'s. Browser security forbids it, and it
  wouldn't help anyway (the block is IP-based, not login-based).
- **Paid third-party download APIs** (Apify, RapidAPI YouTube-to-mp3, etc.). They
  work but the **user explicitly rejected renting a downloader** — we must own it.
- **Self-hosted Cobalt** (`imputnet/cobalt`). Inherits the exact same datacenter-IP
  problem; adds its own brittle cookie handling and extra service. No better than our
  yt-dlp.
- **Public Invidious / Piped instances** — 403/401 / empty `audioStreams`.
- **yt-dlp OAuth login plugin** — killed by Google in 2024.
- **Google "Sign in with Google" / Data API** — metadata only, never stream/download
  access.
- **Auto-refreshing one account's cookies via headless login.** Fragile: 2FA/CAPTCHA
  on automated Google login; cookie lifetime collapsed to ~3–7 days in 2026
  (rotation on any active session); risk of the throwaway account getting flagged.
  High-maintenance last resort only. (One existing tool: `devkulemannege/yt-dlp-Cookie-Sync`,
  Playwright-based, barely maintained, explicitly fails on CAPTCHA.)

---

## 7. CURRENT SOLUTION (deployed, needs verification): free Cloudflare WARP egress

**Idea:** route our server's yt-dlp traffic through **Cloudflare WARP**, whose IP
ranges YouTube does **not** treat as blacklisted datacenter IPs — for free. This is
the "be a downloader site" move (residential-looking egress) without paying for a
proxy pool.

**Implementation (already in `main`):**
- `Dockerfile`:
  - Installs **`wgcf` v2.2.31** (ViRb3) — registers a free WARP identity
    (`wgcf register --accept-tos && wgcf generate`).
  - Installs **`wireproxy` v1.1.2** (`windtf/wireproxy`, the transferred home of
    `pufferffish/wireproxy`) — runs the WireGuard tunnel in **userspace** (no root,
    no kernel module, no `cap_net_admin` — important because Render doesn't grant
    those).
  - Both binaries are **pinned by SHA-256** (supply-chain safety):
    - `wgcf` = `69147e1a517c66129edd8ac8cb60484d6c9515178d7b4a2f95e3c925f225572a`
    - `wireproxy.tar.gz` = `b7dcff8f6e9d3410364e432aff24154eaa8db8206e0c6faac35d6c6ab06dac51`
  - At build time it writes `/app/wireproxy.conf` = the generated WireGuard profile +
    `[Socks5] 127.0.0.1:25344` and `[http] 127.0.0.1:25345`. **Non-fatal** if
    Cloudflare's API hiccups (app still builds, runs direct).
- `scripts/start.sh`:
  - If `YTDLP_PROXY` is already set (e.g. a paid residential proxy) → use it, skip WARP.
  - Else if `/app/wireproxy.conf` exists → run `wireproxy`, set
    `YTDLP_PROXY=http://127.0.0.1:25345`, then **probe and log**:
    `curl -x $YTDLP_PROXY https://www.cloudflare.com/cdn-cgi/trace` → expects `warp=on`.
  - Then starts the PO provider and `exec gunicorn`.
- `app.py`: `download_via_ytdlp()` tries `current_proxy()` first, then **falls back to
  direct**. `/search` also uses the proxy.

**Known caveat:** WARP is a **shared free egress pool** ("free tier of a free tier").
It may be rate-limited or partially flagged by YouTube and could degrade with little
warning. It is the best *free* shot, **not** a guarantee.

---

## 8. HOW TO VERIFY (your first task)

1. **Confirm deploy + WARP came up.** In Render → service → **Logs**, look for:
   - ✅ `[proxy] WARP is up — egress identity:` then `warp=on` and `ip=...`
   - ⚠️ `[proxy] WARNING: WARP did not come up cleanly…` (check `/tmp/wireproxy.log`)
2. **Manual proxy check inside the container** (Render shell, if available):
   `curl --max-time 12 -x http://127.0.0.1:25345 https://www.cloudflare.com/cdn-cgi/trace`
   → should print `warp=on` and a non-datacenter `ip=`.
3. **End-to-end download test.** In the live app, paste a YouTube URL (or use the
   search tab). Watch:
   - Frontend: progress bar → spiral renders = success.
   - On failure, the on-screen message states the category; Render logs show
     `[youtube-download] proxy ... attempt failed: <real yt-dlp error>` and
     `[youtube-download] direct attempt failed: <...>`.
4. **Direct API test** (bypass the UI):
   ```
   curl -s -X POST https://<service>.onrender.com/youtube-download \
     -H 'Content-Type: application/json' \
     -d '{"url":"https://www.youtube.com/watch?v=<id>"}'
   # then poll:
   curl -s "https://<service>.onrender.com/youtube-status?jobId=yt_job"
   ```
   Success → `status:"done"` with `times`/`frequencies`. Failure → `status:"error"`.

**Define success as:** several different YouTube videos download + analyze reliably
(not just one lucky hit). Try ~10 varied videos and note the success rate.

---

## 9. IF WARP IS INSUFFICIENT — ranked next steps

In order of recommendation for a hobby app that must stay seamless and own the
downloader:

1. **Residential proxy via the existing `YTDLP_PROXY` env var (no code change).**
   Set `YTDLP_PROXY=http://user:pass@host:port` in Render → Environment. It takes
   priority over WARP automatically. Best-evidenced cheap option: **Webshare
   residential** (~$6/mo; a free tier of 10 proxies/1 GB exists; the
   `youtube-transcript-api` maintainer independently recommends Webshare residential
   over Bright Data/ScraperAPI for YouTube). DataImpulse (~$1/GB) and Decodo are
   alternatives. This is "renting an IP," not a download API — we still own the code.
   *This is the most reliable known fix; expect ~85–95% success.*
2. **Pair cookies + PO-token WITH the residential proxy**, only if #1 alone still
   misses. Use cookies from a **throwaway** Google account, exported from an
   incognito session and never reused for browsing (YouTube rotates cookies on active
   sessions). Accept a 3–7 day refresh burden. The bgutil PO provider is already
   wired (`127.0.0.1:4416`); for IP consistency its egress should also go through the
   proxy. High maintenance — last resort.
3. **Verify yt-dlp is current** (it ships near-daily fixes for YouTube breakage).
   `requirements.txt` pins `yt-dlp==2026.6.9`; bumping it may resolve transient
   breakage.

**Do NOT** pivot to client-side, browser extensions, or paid download APIs (see §6).

---

## 10. Known flaws / tech debt (not yet fixed — safe to improve)

- **Single global job id `'yt_job'`** and fixed output filename `audio.mp3`
  (`/youtube-download`). Two concurrent users clobber each other. Make job ids
  unique (e.g. per video id); the frontend already polls whatever `jobId` the
  backend returns.
- **No caching.** The same URL re-downloads every time — wasteful and increases
  bot-detection exposure. Cache audio + analysis by video id.
- **In-memory job state** is lost on restart and forces single-worker/single-instance.
- **Memory:** `analyze_frequencies()` builds a large STFT (`n_fft=4096`) and a float64
  HPS copy; very long songs can OOM the 512 MB free/starter tiers. Consider chunked/
  streaming analysis or a bigger plan for long tracks.
- **No rate limiting** on `/youtube-download` and `/search`.

---

## 11. Open questions for the user (decide if relevant)

- If WARP is unreliable: OK to spend ~$6/mo on a residential proxy (Webshare)? (This
  is just an IP, not a download API — consistent with "own the downloader.")
- Acceptable for very long songs to require a bigger Render plan, or should analysis
  be optimized to stream in chunks?

---

## 12. Primary research sources (mid-2026)

- yt-dlp PO Token Guide & FAQ: `https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide`,
  `https://github.com/yt-dlp/yt-dlp/wiki/FAQ`
- Cookie lifetime collapse: `https://github.com/yt-dlp/yt-dlp/issues/13964`,
  `https://github.com/yt-dlp/yt-dlp/issues/16229`
- SABR / datacenter failures with valid cookies+PO token:
  `https://github.com/yt-dlp/yt-dlp/issues/16082`, `.../issues/15793`, `.../issues/12482`
- bgutil PO provider: `https://github.com/Brainicism/bgutil-ytdlp-pot-provider`
- googlevideo has no CORS (client-side dead end):
  `https://issuetracker.google.com/issues/229013699`,
  `https://github.com/WebAudio/web-audio-api/issues/2547`,
  `https://github.com/fent/node-ytdl-core/issues/75`
- YouTube.js browser usage needs a proxy: `https://ytjs.dev/guide/browser-usage`,
  `https://github.com/LuanRT/kira`
- Residential proxy success/cost: `https://github.com/jdepoix/youtube-transcript-api/discussions/335`,
  `https://dev.to/osovsky/i-was-building-a-cloud-video-service-youtube-turned-me-into-an-ip-trafficker-1l9o`
- WARP-as-proxy tooling: `https://github.com/ViRb3/wgcf`, `https://github.com/windtf/wireproxy`
  (formerly `pufferffish/wireproxy`)

---

## 13. One-paragraph summary for the next AI

Sinestesia Musical is a Flask audio-visualizer on Render (Docker) that downloads
YouTube audio server-side with yt-dlp; YouTube blocks our datacenter IP. We cannot
move the download into the user's browser (googlevideo has no CORS) and the user
won't pay for a download API, so the fix is residential-looking egress. We just
deployed a **free Cloudflare WARP** proxy (wgcf + wireproxy, userspace, in the
Dockerfile; yt-dlp routed via `YTDLP_PROXY`, with direct fallback). **Verify whether
it actually works** (logs show `warp=on`; test ~10 videos). **If it's not reliable,
set `YTDLP_PROXY` to a ~$6/mo Webshare residential proxy — no code change.** Keep it
a single gunicorn worker. Don't revisit the §6 dead ends.
