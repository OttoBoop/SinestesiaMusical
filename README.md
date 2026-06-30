# Musical Frequency Visualizer

A web app that analyzes audio and renders an animated spiral whose size and color
track the detected pitch in real time. Feed it an uploaded file or a YouTube
track; the Python backend detects the dominant frequency frame-by-frame and the
browser draws a logarithmic spiral synced to playback.

## How It Works

1. Upload an audio file (MP3, WAV, OGG, FLAC, M4A — up to 50 MB) **or** search /
   paste a YouTube link.
2. The server extracts the dominant frequency over time using an FFT + Harmonic
   Product Spectrum (HPS) pitch tracker (`librosa` + `numpy` + `scipy`).
3. As you play the audio, a colored spiral animates in sync — radius and hue map
   to the current frequency.

## Architecture

- **Backend:** Flask (`app.py`) — audio upload, frequency analysis, YouTube
  download. Served by gunicorn in production.
- **Frontend:** single-page HTML/CSS/JS (`templates/index.html`) — drag-and-drop
  upload, YouTube search/link, audio playback, and the canvas spiral renderer.
- **YouTube downloads:** authenticated `yt-dlp` (audio-only) with the `deno` JS
  runtime + automatic Proof-of-Origin (PO) tokens. No login/account required.
- **PO token provider:** a small Node companion service
  (`bgutil-ytdlp-pot-provider`) on `127.0.0.1:4416` that generates PO tokens
  automatically. Built into the image and started alongside the web server.

The whole stack (Python, ffmpeg, deno, Node) is packaged in a single
**Dockerfile**, so it runs identically on any Docker host — Render, Fly.io,
Railway, or your laptop. No platform lock-in.

## Deploy to Render

This repo ships a Render Blueprint (`render.yaml`), so deployment is mostly
clicks:

1. Push this repo to GitHub (already done if you're reading this there).
2. In the [Render dashboard](https://dashboard.render.com): **New + → Blueprint**.
3. Pick this repository. Render reads `render.yaml` and proposes a Docker web
   service named `sinestesia-musical`.
4. Click **Apply**. The first build takes a few minutes (it installs ffmpeg,
   deno, Node, the Python deps, and compiles the PO-token provider).
5. When it goes live, open the `*.onrender.com` URL.

That's it — no run command or system packages to configure by hand; the
Dockerfile and `scripts/start.sh` handle everything.

### Notes & trade-offs

- **Free plan** (the blueprint default) sleeps after ~15 min of inactivity and
  cold-starts in ~1 min on the next visit. For always-on, change `plan: free` to
  `plan: starter` in `render.yaml` (or switch it in the dashboard).
- **One worker on purpose.** Download-job state lives in memory, so the service
  runs a single gunicorn worker with threads (see `scripts/start.sh`). Don't
  scale to multiple instances/workers or status polls will 404.
- **Memory.** Analyzing a long song builds a large STFT matrix. The 512 MB free/
  starter tiers are fine for typical 3–5 min tracks; very long tracks may need a
  bigger plan (Standard, 2 GB) — or open an issue to make the analysis stream in
  chunks.
- **YouTube from a datacenter IP.** Render is a datacenter, like Replit was, so
  YouTube's anti-bot checks apply the same way. The `deno` + `android_vr` path
  plus PO tokens is what keeps it working; **file upload is rock-solid** and not
  subject to this.

## Run Locally with Docker

```bash
docker build -t sinestesia .
docker run --rm -p 5000:5000 sinestesia
# open http://localhost:5000
```

## Run Locally without Docker

You'll need Python 3.12, `ffmpeg`, `deno`, and Node on your PATH.

```bash
uv sync                       # or: pip install -r requirements.txt
bash scripts/run_pot_provider.sh &   # optional: PO-token provider on :4416
python app.py                 # http://localhost:5000
```

The app still works if the PO provider is down — downloads just lose the
automatic anti-bot token and rely on the `android_vr` path.

## Original Scripts

The original standalone Python scripts are preserved as reference (they require a
local MP3 and, for the pygame ones, a display):

- `Musical frequency reader.py` — reads an MP3 and writes `frequency_table.csv`
- `Musical SPiral create images.py` — pre-generates colored spiral images
- `Spiral Create Images Single Color.py` — single-hue spiral generator
- `Pygame Animation+analysis.py` — combined analysis + pygame animation
- `PyGame Spiral ANimation.py` — pygame animation driven by a CSV

## Legacy: Replit

This project used to run on Replit. The old config (`.replit`, `replit.nix`,
`scripts/post-merge.sh`) is left in the repo for history but is no longer used —
the Dockerfile + `render.yaml` are the source of truth now.
