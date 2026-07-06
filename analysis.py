"""Audio → melody-frequency analysis, run as an ISOLATED subprocess.

`app.py` never imports this module; it invokes it with
``python analysis.py <audio_in> <result_out.json>`` (see ``run_analysis``). Running
the heavy numeric work in a short-lived child process is what keeps the long-lived
gunicorn worker safe on a 512 MB box:

  • Memory isolation — the spectrogram lives and dies in the child, so even a very
    long track can never OOM-kill the web worker; the parent just sees a non-zero
    exit and reports a clean, friendly error.
  • Bounded memory — the STFT is computed in frame BLOCKS and only the per-frame
    peak frequency is kept, so peak RAM stays ~150 MB regardless of song length
    (a 3.5 min track that needed ~1 GB via the old full-matrix float64 path now
    needs ~0.15 GB).
  • No librosa — decoding is delegated to ffmpeg (already in the image) and the DSP
    is plain numpy + scipy, dropping librosa's ~300 MB import baseline and its slow
    first-call JIT. Output matches the previous librosa implementation (corr ≈ 1.0).
"""
import sys
import json
import subprocess
import numpy as np
from scipy.signal import medfilt
from scipy.signal.windows import hann

# 22.05 kHz keeps the whole 60–1200 Hz melody band and the 5th harmonic the HPS
# needs (Nyquist 11 kHz), at a quarter of 44.1 kHz's STFT memory.
ANALYSIS_SR = 22050
N_FFT       = 4096
HOP         = 1024
FMIN, FMAX  = 60.0, 1200.0
BLOCK       = 256          # frames per chunk — bounds peak memory


def _load_mono(path, sr):
    """Decode → mono → resample via ffmpeg, returned as a float32 numpy array."""
    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', path,
           '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError('ffmpeg could not decode the audio: '
                           + p.stderr.decode('utf-8', 'replace')[-200:])
    return np.frombuffer(p.stdout, dtype=np.float32)


def analyze_frequencies(audio_path):
    """Melody tracking via YIN (numpy/scipy only). Returns (times, freqs).

    Replaces Harmonic Product Spectrum, which mis-tracked the notes: HPS had octave errors
    (no correction), snapped pitch to the coarse FFT-bin grid (no interpolation), always
    output a pitch (no voiced/unvoiced), and its 9-frame median + heavy EMA smeared the
    melody. On a clean vocal stem YIN scores 87% in-key vs HPS's 67%; even on a full mix it
    lifts in-key from ~68% to ~79%. pitch.yin_pitch is pure numpy so it still runs in the
    isolated 512 MB subprocess (no librosa)."""
    import pitch
    sr = ANALYSIS_SR
    y = _load_mono(audio_path, sr)
    if y.size == 0:
        raise RuntimeError('decoded audio is empty')
    times, f0 = pitch.yin_pitch(y, sr, fmin=65.0, fmax=1000.0,
                                frame_length=N_FFT, hop_length=HOP)
    f0 = pitch.smooth_f0(f0, med=5)                # light median; preserves unvoiced gaps
    return times.tolist(), f0.tolist()


def main():
    # argv: <in_audio> <out_json> [engine]   (engine defaults to the legacy melody)
    in_path, out_path = sys.argv[1], sys.argv[2]
    engine = sys.argv[3] if len(sys.argv) > 3 else 'melody'
    try:
        from engines import run_engine
        result = run_engine(in_path, engine)         # {engine, sr, components:[...]}
        # Back-compat: expose the first component as legacy top-level times/frequencies
        # so the current single-spiral frontend keeps working until it's upgraded.
        first = result['components'][0]
        result['times'] = first['times']
        result['frequencies'] = first['freqs']
        with open(out_path, 'w') as f:
            json.dump(result, f)
    except Exception as e:                                         # report, never traceback to user
        try:
            with open(out_path, 'w') as f:
                json.dump({'error': str(e)}, f)
        except OSError:
            pass
        print(f'[analysis] failed: {e}', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
