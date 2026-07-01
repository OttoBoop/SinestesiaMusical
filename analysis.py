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
    """FFT + Harmonic Product Spectrum melody tracking. Returns (times, freqs)."""
    sr = ANALYSIS_SR
    y = _load_mono(audio_path, sr)
    if y.size == 0:
        raise RuntimeError('decoded audio is empty')
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))

    y = np.pad(y, N_FFT // 2, mode='reflect')                 # center frames (librosa-style)
    n_frames = 1 + (len(y) - N_FFT) // HOP
    win   = hann(N_FFT, sym=False).astype(np.float32)         # periodic Hann, matches librosa
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    lo    = int(np.searchsorted(freqs, FMIN))
    hi    = int(np.searchsorted(freqs, FMAX))
    band  = freqs[lo:hi]

    windows = np.lib.stride_tricks.sliding_window_view(y, N_FFT)   # view, no copy
    peak = np.empty(n_frames, dtype=np.float64)
    for s in range(0, n_frames, BLOCK):
        e = min(s + BLOCK, n_frames)
        frames = windows[s * HOP:(e - 1) * HOP + 1:HOP]           # strided view (chunk, N_FFT)
        spec   = np.abs(np.fft.rfft(frames * win, axis=1)).astype(np.float32)
        hps = spec.copy()
        for h in range(2, 6):                                      # harmonic product spectrum
            down = spec[:, ::h]
            n = min(hps.shape[1], down.shape[1])
            hps[:, :n] *= down[:, :n]
        peak[s:e] = band[np.argmax(hps[:, lo:hi], axis=1)]

    peak = medfilt(peak, kernel_size=9)
    alpha = 0.2
    smoothed = peak.copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * peak[i] + (1.0 - alpha) * smoothed[i - 1]

    times = (np.arange(len(smoothed)) * HOP / sr)
    return times.tolist(), smoothed.tolist()


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
