# Musical Frequency Visualizer

A web-based musical frequency visualizer that analyzes audio files and renders an animated spiral whose shape and color correspond to the detected pitch in real time.

## How It Works

1. Upload an audio file (MP3, WAV, OGG, FLAC, M4A — up to 50MB)
2. The server analyzes the audio using autocorrelation to extract dominant frequencies chunk-by-chunk
3. As you play the audio back, a colored spiral animates in sync — the radius and hue map to the current frequency

## Architecture

- **Backend:** Flask (Python) — handles audio upload, frequency analysis with `librosa`, and on-demand spiral image generation with `matplotlib`
- **Frontend:** Single-page HTML/CSS/JS app served by Flask — handles drag-and-drop upload, audio playback, and live spiral updates
- **YouTube downloads:** Primary path is authenticated `yt-dlp` (audio-only) with automatic Proof-of-Origin (PO) tokens to pass YouTube's anti-bot check; public Invidious mirrors are a secondary fallback. No login/account is ever required.
- **PO token provider:** A small companion service (`bgutil-ytdlp-pot-provider`, Node) runs on `127.0.0.1:4416` and generates PO tokens automatically. It is started by the "POT Provider" workflow via `scripts/run_pot_provider.sh`, which self-heals by building the server (`scripts/setup_pot_provider.sh`) if missing.

## Running the App

Two workflows run together:

- **POT Provider** — `bash scripts/run_pot_provider.sh` (background PO-token service on port 4416)
- **Start application** — `python app.py` (web app on `0.0.0.0:5000`)

The app still works if the PO provider is down — downloads simply lose the automatic anti-bot token and fall back to mirrors. The vendored provider lives under `vendor/` (gitignored) and is rebuilt automatically after merges by `scripts/post-merge.sh`.

## Original Scripts

The original Python scripts are preserved as reference:

- `Musical frequency reader.py` — reads an MP3 and writes `frequency_table.csv`
- `Musical SPiral create images.py` — pre-generates colored spiral images for each frequency radius
- `Spiral Create Images Single Color.py` — single-hue variant of the spiral generator
- `Pygame Animation+analysis.py` — combined analysis + pygame animation (requires local MP3 and display)
- `PyGame Spiral ANimation.py` — pygame animation driven by a pre-generated CSV

## User Preferences

- No specific preferences recorded yet
