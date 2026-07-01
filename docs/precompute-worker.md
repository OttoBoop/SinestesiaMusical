# Precompute worker — enabling the HD ML separation on the free tier

The ML engine (MDX-Net, "Vocal HD") needs ~1.8 GB RAM and ~1× realtime (measured), so it
**never runs on the 512 MB free web tier**. Instead:

- The **web app** serves the ML engine from a **shared cache** only. On a cache miss it tells
  the user "HD separation is being prepared" — it never downloads+separates inline (no OOM).
- A **precompute worker** (`precompute.py`) runs the heavy separation off-tier (your machine,
  or a Render one-off Job on a ≥2 GB plan) and writes the ~1 MB analysis into the shared cache.

The other engines (melody/bands/hpss/repet/stereo) run inline on the free tier as usual and
also benefit from the cache (repeat requests are instant).

## Shared cache: local vs R2/S3
`cache.py` picks its backend from env:
- **local** (default): `cache/` dir on disk — per-instance. Fine only when the worker and the
  web app share a filesystem (same box).
- **S3-compatible (Cloudflare R2 / S3 / Backblaze / MinIO)** — SHARED across machines. Set:
  ```
  CACHE_S3_BUCKET=sinestesia-cache
  CACHE_S3_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com   # R2
  CACHE_S3_KEY=<access key id>
  CACHE_S3_SECRET=<secret access key>
  # optional: CACHE_S3_REGION=auto  CACHE_S3_PREFIX=analysis/
  ```
  Set the SAME four vars on both the Render web service and wherever the worker runs. `boto3`
  is imported lazily — only when `CACHE_S3_BUCKET` is set.

R2 free tier: 10 GB storage + zero egress → serving cached analyses costs nothing.

## Running the worker
```
# one-time: pip install -r requirements.txt ; have ffmpeg on PATH ; download the MDX model:
#   curl -L -o models/UVR-MDX-NET-Inst_HQ_3.onnx \
#     "https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx?download=true"
export MDX_MODEL_PATH=models/UVR-MDX-NET-Inst_HQ_3.onnx
export CACHE_S3_BUCKET=... CACHE_S3_ENDPOINT=... CACHE_S3_KEY=... CACHE_S3_SECRET=...

# precompute HD vocals for a track (any YouTube URL or bare video id):
python precompute.py https://youtu.be/dQw4w9WgXcQ ml
# you can pass several engines to warm at once:
python precompute.py dQw4w9WgXcQ ml hpss repet
```
After that, opening the same track on the site with **Vocal HD** serves the cached result
instantly. Re-runs skip already-cached (video, engine) pairs.

## Verified (2026-07-01)
Local e2e: worker `precompute_file(...,'ml')` → cache; `app.run_analysis(...,'ml')` serves it
from cache with **no inline compute**; an uncached track raises `AnalysisQueued` (friendly
"being prepared" message). pytest `tests/test_phase5.py` green.
