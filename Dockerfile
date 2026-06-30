# syntax=docker/dockerfile:1
#
# Sinestesia Musical — production image (Render, Fly.io, Railway, any Docker host).
#
# The app needs FOUR runtimes in one place, which is exactly why a container is
# the right call instead of a platform-specific buildpack:
#   • Python 3.12 — Flask web app + audio analysis (librosa/numpy/scipy)
#   • ffmpeg      — decode/transcode audio for librosa and the yt-dlp postprocessor
#   • deno        — JavaScript runtime yt-dlp uses to solve YouTube's challenges
#   • Node        — runs the bgutil Proof-of-Origin (PO) token provider

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────────────
#   ffmpeg : audio decode/transcode (librosa + yt-dlp postprocessor)
#   nodejs : runs the bgutil PO-token provider (installed from NodeSource 20.x)
#   git    : clones the PO-token provider at build time
#   curl/unzip/gnupg/ca-certificates : fetch deno + the NodeSource key
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg git curl unzip ca-certificates gnupg \
 && mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# ── deno (yt-dlp's JavaScript runtime for YouTube) ─────────────────────────────
RUN curl -fsSL \
        https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
        -o /tmp/deno.zip \
 && unzip /tmp/deno.zip -d /usr/local/bin \
 && rm /tmp/deno.zip \
 && chmod +x /usr/local/bin/deno \
 && deno --version

WORKDIR /app

# ── Python dependencies (fully pinned, exported from uv.lock) ──────────────────
# Copied first so this layer caches unless the dependency set changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# ── Pre-build the bgutil PO-token provider into vendor/ ────────────────────────
# Baked into the image so the container boots instantly with no git clone at
# startup. (vendor/ is gitignored and excluded by .dockerignore, so this build
# artifact is not clobbered by the "COPY . ." below.)
COPY scripts/setup_pot_provider.sh scripts/setup_pot_provider.sh
RUN bash scripts/setup_pot_provider.sh

# ── Application source ─────────────────────────────────────────────────────────
COPY . .

# Render injects $PORT at runtime; this default is only for local `docker run`.
ENV PORT=5000
EXPOSE 5000

CMD ["bash", "scripts/start.sh"]
