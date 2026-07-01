"""Analysis cache — store the derived per-component analysis (~1 MB), NOT audio.

Keyed by (source id, engine, params version) so a repeat (video, engine) request is served
instantly, and — critically for the heavy ML engine — a precomputed result can be served
without re-running separation on the web tier. Local-filesystem backend for now (a future
Phase can swap in Cloudflare R2 behind the same get/put; see docs/visualizer-v2-research.md).
"""
import os
import json
import hashlib

# Bump when the analysis output format / DSP params change → old entries become clean misses.
PARAMS_VERSION = 'v1'

CACHE_DIR = os.environ.get('CACHE_DIR') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cache')


def analysis_key(source_id: str, engine: str, params_version: str = PARAMS_VERSION) -> str:
    """Stable cache key for a (source, engine, params) triple."""
    raw = f'{source_id}|{engine}|{params_version}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + '.json')


def get(key: str):
    """Return the cached result dict, or None on miss / unreadable entry."""
    try:
        with open(_path(key)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def put(key: str, result: dict) -> None:
    """Store a result dict atomically (tmp + os.replace)."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = _path(key) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(result, f)
        os.replace(tmp, _path(key))
    except OSError:
        pass  # cache is best-effort; never fail the request on a cache write error
