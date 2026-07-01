"""Phase 3 engine tests (TDD). RED until repet/stereo are implemented + HPSS is block-bounded.

Structure/contract level (classical separation quality on synthetic audio is not asserted)."""
import sys, os, subprocess, resource, textwrap
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines import run_engine, ENGINES, ANALYSIS_SR


# ── existing engines stay green (regression) ────────────────────────────────────
@pytest.mark.parametrize('engine,expected', [
    ('melody', {'melody'}),
    ('bands',  {'bass', 'mid', 'high'}),
    ('hpss',   {'harmonic', 'percussive'}),
])
def test_existing_engines(mono_song, valid_component, engine, expected):
    r = run_engine(mono_song, engine)
    assert r['engine'] == engine and r['sr'] == ANALYSIS_SR
    names = {c['name'] for c in r['components']}
    assert names == expected
    for c in r['components']:
        valid_component(c, r['sr'])


# ── NEW: REPET-SIM vocal engine ─────────────────────────────────────────────────
def test_repet_registered():
    assert 'repet' in ENGINES

def test_repet_components(repeating_song, valid_component):
    r = run_engine(repeating_song, 'repet')
    names = {c['name'] for c in r['components']}
    assert names == {'vocal', 'background'}
    for c in r['components']:
        valid_component(c, r['sr'])
    # both components share the same time base
    lens = {len(c['times']) for c in r['components']}
    assert len(lens) == 1


# ── NEW: stereo (ADRess / mid-side) engine ──────────────────────────────────────
def test_stereo_registered():
    assert 'stereo' in ENGINES

def test_stereo_components(stereo_song, valid_component):
    r = run_engine(stereo_song, 'stereo')
    names = {c['name'] for c in r['components']}
    assert {'center', 'sides'}.issubset(names)
    for c in r['components']:
        valid_component(c, r['sr'])

def test_stereo_center_is_vocal_frequency(stereo_song):
    """The synthetic center tone is 330 Hz; the 'center' component's dominant pitch should
    sit near it (well below the 660 Hz right-panned tone and above the 110 Hz left)."""
    r = run_engine(stereo_song, 'stereo')
    center = next(c for c in r['components'] if c['name'] == 'center')
    med = float(np.median(center['freqs']))
    assert 240 < med < 430


# ── NEW: HPSS must be memory-bounded (block-processed) on a long track ──────────
def test_hpss_memory_bounded_on_long_track(tmp_path):
    """A 4-min track must not blow the analysis subprocess past a tight cap. The old
    full-spectrogram HPSS grows with length; block processing keeps it ~flat."""
    import soundfile as sf
    sr = ANALYSIS_SR
    dur = 240
    t = np.arange(int(dur * sr)) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav = tmp_path / 'long.wav'
    sf.write(str(wav), y, sr)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = textwrap.dedent(f'''
        import resource, sys
        sys.path.insert(0, {repo!r})
        from engines import run_engine
        r = run_engine({str(wav)!r}, 'hpss')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(int(peak), len(r['components'][0]['times']))
    ''')
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    peak_mb, n = (int(x) for x in out.stdout.split())
    assert n > 0
    assert peak_mb < 400, f'HPSS peak RSS {peak_mb}MB on a 4-min track exceeds the 400MB cap'
