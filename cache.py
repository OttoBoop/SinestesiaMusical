"""Analysis cache — stores the derived per-component analysis (~1 MB), NOT audio.

Keyed by (source id, engine, params version). Two interchangeable backends behind the same
get/put:
  • local filesystem (default) — per-instance, fine for same-box worker+web.
  • S3-compatible (Cloudflare R2 / S3 / Backblaze / MinIO) — SHARED, so a precompute worker
    on one machine and the web app on Render see the same cache. Enabled by setting
    CACHE_S3_BUCKET (+ CACHE_S3_ENDPOINT / CACHE_S3_KEY / CACHE_S3_SECRET). boto3 is imported
    lazily so it's only needed when S3 is configured.

This is what lets the heavy ML engine work on the free tier: the worker precomputes and
puts the result here; the web app only ever get()s it.
"""
import os
import json
import hashlib

# Bump when the analysis output format / DSP params change → old entries become clean misses.
PARAMS_VERSION = 'v1'

CACHE_DIR = os.environ.get('CACHE_DIR') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cache')

S3_BUCKET   = os.environ.get('CACHE_S3_BUCKET')
S3_ENDPOINT = os.environ.get('CACHE_S3_ENDPOINT')          # e.g. https://<acct>.r2.cloudflarestorage.com
S3_PREFIX   = os.environ.get('CACHE_S3_PREFIX', 'analysis/')


def backend_kind() -> str:
    return 's3' if S3_BUCKET else 'local'


def analysis_key(source_id: str, engine: str, params_version: str = PARAMS_VERSION) -> str:
    """Stable cache key for a (source, engine, params) triple."""
    raw = f'{source_id}|{engine}|{params_version}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


# ── local filesystem backend ────────────────────────────────────────────────────
def _local_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + '.json')


def _local_get(key: str):
    try:
        with open(_local_path(key)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _local_put(key: str, result: dict) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = _local_path(key) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(result, f)
        os.replace(tmp, _local_path(key))
    except OSError:
        pass


# ── S3 / R2 backend (lazy boto3) ─────────────────────────────────────────────────
_s3 = None


def _s3_client():
    global _s3
    if _s3 is None:
        import boto3  # lazy: only required when S3 is configured
        _s3 = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=os.environ.get('CACHE_S3_KEY'),
            aws_secret_access_key=os.environ.get('CACHE_S3_SECRET'),
            region_name=os.environ.get('CACHE_S3_REGION', 'auto'),
        )
    return _s3


def _s3_obj(key: str) -> str:
    return f'{S3_PREFIX}{key}.json'


def _s3_get(key: str):
    try:
        obj = _s3_client().get_object(Bucket=S3_BUCKET, Key=_s3_obj(key))
        return json.loads(obj['Body'].read())
    except Exception:
        return None


def _s3_put(key: str, result: dict) -> None:
    try:
        _s3_client().put_object(Bucket=S3_BUCKET, Key=_s3_obj(key),
                                Body=json.dumps(result).encode('utf-8'),
                                ContentType='application/json')
    except Exception:
        pass  # cache is best-effort; never fail the request on a cache write error


# ── public API (backend-agnostic) ────────────────────────────────────────────────
def get(key: str):
    """Return the cached result dict, or None on miss / unreadable entry."""
    return _s3_get(key) if backend_kind() == 's3' else _local_get(key)


def put(key: str, result: dict) -> None:
    """Store a result dict (best-effort)."""
    (_s3_put if backend_kind() == 's3' else _local_put)(key, result)
