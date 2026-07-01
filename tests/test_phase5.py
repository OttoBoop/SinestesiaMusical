"""Phase 5 tests (TDD): precompute worker + shared-cache backend + ml-cache-only web flow.

RED until precompute.py exists, cache.py grows an S3/R2 backend, and app.py serves the ml
engine from cache only (never running the ~1.8 GB separation inline on the free web tier)."""
import sys, os, importlib
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ── shared-cache backend selection ──────────────────────────────────────────────
def test_cache_backend_defaults_local(tmp_path, monkeypatch):
    monkeypatch.delenv('CACHE_S3_BUCKET', raising=False)
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    import cache; importlib.reload(cache)
    assert cache.backend_kind() == 'local'


def test_cache_backend_is_s3_when_configured(monkeypatch):
    monkeypatch.setenv('CACHE_S3_BUCKET', 'sinestesia-cache')
    monkeypatch.setenv('CACHE_S3_ENDPOINT', 'https://example.r2.cloudflarestorage.com')
    import cache; importlib.reload(cache)
    assert cache.backend_kind() == 's3'
    monkeypatch.delenv('CACHE_S3_BUCKET', raising=False)
    importlib.reload(cache)


# ── precompute worker: run an engine and populate the shared cache ──────────────
def test_precompute_populates_cache(mono_song, tmp_path, monkeypatch):
    monkeypatch.delenv('CACHE_S3_BUCKET', raising=False)
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    import cache; importlib.reload(cache)
    import precompute; importlib.reload(precompute)

    result = precompute.precompute_file(mono_song, source_id='vidZZZ', engine='hpss')
    assert result['engine'] == 'hpss'
    # the web app must find it under the SAME key it would compute
    got = cache.get(cache.analysis_key('vidZZZ', 'hpss'))
    assert got is not None
    assert {c['name'] for c in got['components']} == {'harmonic', 'percussive'}


# ── web app: ml is served from cache only, never computed inline ────────────────
def test_ml_is_cache_only_on_web(mono_song, tmp_path, monkeypatch):
    monkeypatch.delenv('CACHE_S3_BUCKET', raising=False)
    monkeypatch.delenv('ENABLE_ML', raising=False)
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    import cache; importlib.reload(cache)
    import app; importlib.reload(app)

    # miss → queued signal, and NO inline computation (must not raise a plain AnalysisError)
    with pytest.raises(app.AnalysisQueued):
        app.run_analysis(mono_song, 'ml', source_id='vidML')

    # after a worker precomputes it, the web serves it from cache
    payload = {'engine': 'ml', 'sr': 22050,
               'components': [{'name': 'vocals', 'times': [0.0], 'freqs': [220.0], 'energy': [1.0]},
                              {'name': 'instrumental', 'times': [0.0], 'freqs': [110.0], 'energy': [1.0]}]}
    cache.put(cache.analysis_key('vidML', 'ml'), payload)
    served = app.run_analysis(mono_song, 'ml', source_id='vidML')
    assert served == payload
