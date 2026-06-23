---
name: YouTube download reliability
description: How automatic YouTube downloads are made robust (deno runtime, PO tokens, audio-only)
---

# YouTube download reliability

This app downloads YouTube audio with **zero user steps** (no login/account). The
reliable path layers two independently-validated mechanisms on top of yt-dlp:

- **deno JS runtime + default clients (incl. android_vr)** — the primary fix. See
  `youtube-ytdlp-downloads.md` for why this is what actually works from a datacenter
  IP. `ytdl_base_opts()` sets `js_runtimes={'deno': {}}`.
- **Automatic PO tokens via the bgutil HTTP provider** (`bgutil-ytdlp-pot-provider`,
  Node server on 127.0.0.1:4416) — belt-and-suspenders for clients that need a
  Proof-of-Origin token. Python plugin installed via pip; Node server vendored under
  `vendor/` (gitignored) and built by `scripts/setup_pot_provider.sh`. Wired via
  extractor args `youtubepot-bgutilhttp:base_url` + a `player_client` list that keeps
  `android_vr`.
  **How to apply:** keep the Node server VERSION in `setup_pot_provider.sh` in lockstep
  with the pip plugin version. `scripts/post-merge.sh` rebuilds it after merges; the run
  script self-heals on fresh envs.

- **Never override `player_client` without including `android_vr`** — that client serves
  signature-free audio and is the most reliable path. Dropping it breaks downloads.

- **Audio-only format string must never contain `/best`** — use
  `bestaudio[ext=m4a]/bestaudio/bestaudio*`.
  **Why:** the old `bestaudio/best` pulled multi-hundred-MB video files.

- **Download into an isolated `tempfile.mkdtemp` dir, then move the produced mp3** to the
  final path; always `cleanup_downloads()` (clears stale `audio.*`/`ytdl_*`) first and on
  failure.
  **Why:** stale destination files caused the intermittent
  `Unable to rename file: audio.mp4.part -> audio.mp4` crash. Isolated temp + pre-clean
  eliminates clashes and leaves no partial files on failure.

- **No Invidious / mirror fallback.** It was removed during a rebase: the main app proved
  public Invidious/Piped/cobalt instances are dead ends (403/empty/auth). A dead fallback
  only makes the failure path hang for minutes, hurting the friendly-error UX. On total
  failure, surface the friendly retryable error immediately; log technical detail
  server-side only.
