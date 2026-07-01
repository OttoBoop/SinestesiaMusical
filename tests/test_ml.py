"""Phase 4 ML (self-hosted ONNX MDX) engine tests (TDD).

`test_ml_registered` is RED until the engine exists. The full separation test needs the
MDX ONNX model; it's skipped when the model isn't present (set MDX_MODEL_PATH), so the
suite stays runnable without the 66 MB weight in CI."""
import sys, os
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MODEL = os.environ.get('MDX_MODEL_PATH') or os.path.join(REPO, 'models', 'UVR-MDX-NET-Inst_HQ_3.onnx')
HAS_MODEL = os.path.exists(MODEL)


def test_ml_registered():
    from engines import ENGINES
    assert 'ml' in ENGINES


@pytest.mark.skipif(not HAS_MODEL, reason='MDX ONNX model not present (set MDX_MODEL_PATH)')
def test_ml_separation(mono_song, valid_component):
    from engines import run_engine
    r = run_engine(mono_song, 'ml')
    names = {c['name'] for c in r['components']}
    assert names == {'vocals', 'instrumental'}
    for c in r['components']:
        valid_component(c, r['sr'])
    lens = {len(c['times']) for c in r['components']}
    assert len(lens) == 1
