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

# ── Cloudflare WARP egress (wgcf + wireproxy, userspace — no root/kernel needed) ─
# This is the whole fix for YouTube's datacenter-IP block: route yt-dlp through a
# free Cloudflare WARP tunnel so our traffic looks like an ordinary home connection
# instead of a flagged datacenter. wgcf (ViRb3) registers a free WARP identity;
# wireproxy runs the WireGuard tunnel in userspace and exposes it as a local proxy.
#
# Both binaries are downloaded from their GitHub releases and PINNED by sha256, so
# the build runs only the exact bytes vetted on 2026-06-30 — a tampered URL or repo
# transfer would fail the checksum instead of silently running different code.
# (wireproxy lives at windtf/wireproxy, the transferred home of pufferffish/wireproxy.)
ARG WGCF_SHA256=69147e1a517c66129edd8ac8cb60484d6c9515178d7b4a2f95e3c925f225572a
ARG WIREPROXY_SHA256=b7dcff8f6e9d3410364e432aff24154eaa8db8206e0c6faac35d6c6ab06dac51
RUN curl -fsSL -o /tmp/wgcf \
        https://github.com/ViRb3/wgcf/releases/download/v2.2.31/wgcf_2.2.31_linux_amd64 \
 && echo "${WGCF_SHA256}  /tmp/wgcf" | sha256sum -c - \
 && install -m 0755 /tmp/wgcf /usr/local/bin/wgcf && rm /tmp/wgcf \
 && curl -fsSL -o /tmp/wireproxy.tar.gz \
        https://github.com/windtf/wireproxy/releases/download/v1.1.2/wireproxy_linux_amd64.tar.gz \
 && echo "${WIREPROXY_SHA256}  /tmp/wireproxy.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/wireproxy.tar.gz -C /usr/local/bin wireproxy \
 && rm /tmp/wireproxy.tar.gz \
 && chmod +x /usr/local/bin/wireproxy

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

# ── Register a free Cloudflare WARP identity and build the wireproxy config ──────
# Non-fatal on purpose: if Cloudflare's API hiccups during the build, the image
# still builds and the app runs with yt-dlp going direct (just without the WARP IP
# advantage). A rebuild gets a fresh identity. The [Socks5]/[http] sections turn the
# generated WireGuard profile into a wireproxy config exposing local proxy ports.
RUN set +e; cd /tmp; \
    if wgcf register --accept-tos && wgcf generate; then \
        cp /tmp/wgcf-profile.conf /app/wireproxy.conf; \
        printf '\n[Socks5]\nBindAddress = 127.0.0.1:25344\n\n[http]\nBindAddress = 127.0.0.1:25345\n' >> /app/wireproxy.conf; \
        rm -f /tmp/wgcf-account.toml /tmp/wgcf-profile.conf; \
        echo "WARP profile generated at /app/wireproxy.conf"; \
    else \
        echo "WARNING: WARP registration failed at build; app will run without the WARP proxy."; \
    fi; \
    true

# Render injects $PORT at runtime; this default is only for local `docker run`.
ENV PORT=5000
EXPOSE 5000

CMD ["bash", "scripts/start.sh"]
