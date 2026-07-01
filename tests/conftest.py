"""Shared pytest fixtures: synthetic audio written to temp WAVs (no network)."""
import os
import numpy as np
import soundfile as sf
import pytest

SR = 22050


def _clicks(t, sr, period_s=0.5, width=60):
    """Percussive click train (broadband transients) — the 'drums' of a synthetic mix."""
    x = np.zeros_like(t)
    rng = np.random.default_rng(0)
    for k in range(int(t[-1] / period_s) + 1):
        i = int(k * period_s * sr)
        x[i:i + width] += (np.hanning(width) * rng.standard_normal(width))[: len(x[i:i + width])]
    return 0.4 * x


@pytest.fixture(scope='session')
def mono_song(tmp_path_factory):
    """8 s mono: sustained harmonic tone (220 Hz + harmonic) + periodic percussive clicks."""
    dur = 8.0
    t = np.arange(int(dur * SR)) / SR
    harmonic = 0.30 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)
    y = (harmonic + _clicks(t, SR)).astype(np.float32)
    y /= np.max(np.abs(y)) + 1e-9
    p = tmp_path_factory.mktemp('audio') / 'mono.wav'
    sf.write(str(p), y, SR)
    return str(p)


@pytest.fixture(scope='session')
def repeating_song(tmp_path_factory):
    """10 s mono: a 2 s REPEATING chord loop (background) + a non-repeating vocal-ish sweep.

    This matches REPET's model (repeating bed + sparse non-repeating foreground)."""
    dur = 10.0
    t = np.arange(int(dur * SR)) / SR
    loop_len = int(2.0 * SR)
    one = (0.25 * np.sin(2 * np.pi * 130.81 * t[:loop_len])      # C3
           + 0.20 * np.sin(2 * np.pi * 196.00 * t[:loop_len]))   # G3
    bg = np.tile(one, int(np.ceil(len(t) / loop_len)))[:len(t)]
    sweep = 0.30 * np.sin(2 * np.pi * (300 + 200 * t / dur) * t)  # rising, non-repeating
    y = (bg + sweep).astype(np.float32)
    y /= np.max(np.abs(y)) + 1e-9
    p = tmp_path_factory.mktemp('audio') / 'repeating.wav'
    sf.write(str(p), y, SR)
    return str(p)


def _harm(t, f0, n=5, amp=0.3):
    """A harmonic-rich tone (fundamental + n-1 harmonics) — HPS needs harmonics to find f0."""
    return sum((amp / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, n + 1))


@pytest.fixture(scope='session')
def stereo_song(tmp_path_factory):
    """8 s stereo: a harmonic CENTER voice (~300 Hz, equal L/R) + hard-panned L and R tones."""
    dur = 8.0
    t = np.arange(int(dur * SR)) / SR
    center = _harm(t, 300, n=5, amp=0.30)            # equal in both channels (the 'vocal')
    left   = 0.25 * np.sin(2 * np.pi * 130 * t)      # L only
    right  = 0.25 * np.sin(2 * np.pi * 700 * t)      # R only (not a harmonic of 300)
    L = (center + left).astype(np.float32)
    R = (center + right).astype(np.float32)
    st = np.stack([L, R], axis=1)
    st /= np.max(np.abs(st)) + 1e-9
    p = tmp_path_factory.mktemp('audio') / 'stereo.wav'
    sf.write(str(p), st, SR)
    return str(p)


def _valid_component(c, sr):
    """Assert a component dict is well-formed (shared contract check)."""
    assert set(('name', 'times', 'freqs', 'energy')).issubset(c), c.keys()
    n = len(c['times'])
    assert n > 0
    assert len(c['freqs']) == n and len(c['energy']) == n
    tt = np.asarray(c['times'])
    assert tt[0] >= 0 and np.all(np.diff(tt) >= 0)             # monotonic, starts at 0
    fr = np.asarray(c['freqs'])
    assert np.all(np.isfinite(fr)) and np.all(fr >= 0) and np.all(fr <= sr / 2 + 1)
    en = np.asarray(c['energy'])
    assert np.all(np.isfinite(en)) and np.all(en >= -1e-6) and np.all(en <= 1 + 1e-6)


@pytest.fixture
def valid_component():
    return _valid_component
