"""Phase 4 analysis-cache tests (TDD). RED until cache.py exists.

Cache stores the derived analysis (~1 MB), keyed by (source id, engine, params version),
so a repeated (video, engine) request is served instantly instead of re-analyzed."""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cache_module_importable():
    import cache  # noqa: F401


def test_key_is_deterministic_and_engine_specific():
    import cache
    k1 = cache.analysis_key('vid123', 'hpss')
    k2 = cache.analysis_key('vid123', 'hpss')
    k3 = cache.analysis_key('vid123', 'repet')
    assert k1 == k2
    assert k1 != k3
    assert isinstance(k1, str) and len(k1) >= 8


def test_put_then_get_roundtrips(tmp_path, monkeypatch):
    import importlib, cache
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    importlib.reload(cache)
    key = cache.analysis_key('vidABC', 'bands')
    assert cache.get(key) is None
    payload = {'engine': 'bands', 'sr': 22050,
               'components': [{'name': 'bass', 'times': [0.0, 0.1], 'freqs': [100, 110], 'energy': [1.0, 0.9]}]}
    cache.put(key, payload)
    got = cache.get(key)
    assert got == payload


def test_missing_key_returns_none(tmp_path, monkeypatch):
    import importlib, cache
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    importlib.reload(cache)
    assert cache.get(cache.analysis_key('never', 'hpss')) is None
