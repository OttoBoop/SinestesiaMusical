---
name: yt-dlp YouTube downloads from datacenter IPs
description: Why yt-dlp "bot detection"/download failures usually aren't an IP block, and what actually fixes them
---

# yt-dlp YouTube audio downloads (Replit / datacenter IP)

When yt-dlp fails to download YouTube audio here, the instinct is "the datacenter IP is
bot-blocked." That was **wrong** in practice — metadata extraction and audio downloads both
work fine from this server's IP.

## The real causes (2026-era yt-dlp)
1. **Missing JavaScript runtime.** Modern yt-dlp needs a JS runtime (deno) to solve YouTube's
   signature / n-challenge. Without it you get warnings and many formats get skipped.
2. **`bestaudio` selecting a format that needs signature solving.** yt-dlp's default client
   list includes `android_vr`, which serves direct audio URLs needing **no** signature solving —
   that's what makes downloads work with zero user setup.

## The fix that works
- Install `deno` as a system dependency (Nix). It's yt-dlp's default/recommended JS runtime.
- In the **Python API**, pass `js_runtimes` as a **dict**: `{'deno': {}}`. A list like
  `['deno']` raises `Invalid js_runtimes format, expected a dict of {runtime: {config}}`.
  (The CLI flag form `--js-runtimes deno` is different — don't copy it into the Python opts.)
- Use `format: 'bestaudio/best'` + `FFmpegExtractAudio` postprocessor. Downloads succeed via
  android_vr even when a cosmetic "No supported JavaScript runtime" warning still prints.

## Dead ends — do not waste time re-attempting
- **Invidious** public instances: all return 403/401/connection-refused.
- **Piped** public instances: respond with metadata but `audioStreams: []` (YouTube blocks the
  stream extraction). Useless for audio.
- **Cobalt** public API: now requires auth.
- **cookies.txt**: works but requires per-end-user browser-extension export — unacceptable for a
  public app.
- **"Automatic YouTube login"**: not possible. OAuth/Google Sign-In only grants metadata (Data
  API), never stream-download access. Browser security prevents an app from reading a login
  popup's cookies. yt-dlp's old OAuth login was killed by Google in 2024.

**Why this matters:** the user needs a zero-setup public app. The only thing that delivers that
is yt-dlp + deno running on the server, not any proxy/login scheme.
