"""Separation ENGINES for the multi-spiral visualizer.

Imported only by the analysis subprocess (`analysis.py`), never by the web worker, so
the heavy numpy/scipy stays out of the long-lived gunicorn process.

Each engine turns one audio file into a list of COMPONENTS; each component is a
per-frame time-series the frontend draws as its own spiral:
    {"name": str, "times": [s...], "freqs": [Hz...], "energy": [0..1...]}

Engines here are the free/instant CLASSICAL tier (numpy+scipy, run inline). The ML
engine (ONNX MDX/SCNet) is added later via precompute+cache (it needs ~1.8 GB, measured
in Phase 0 — see docs/visualizer-v2-research.md — so it can't run inline on 512 MB).

Design: build ONE full magnitude spectrogram at the analysis params (sr 22.05 kHz,
n_fft 4096, hop 1024 → ~2049 bins). At those params the full float32 spectrogram is
tens of MB even for long songs, cheap enough to hold — unlike the old 44.1 kHz/float64
path that OOM'd. HPSS/bands read from it; the melody engine reuses the shipped
memory-bounded `analyze_frequencies` verbatim (keeps the verified single-spiral output).
"""
import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.signal.windows import hann

from analysis import ANALYSIS_SR, HOP, FMIN, FMAX, analyze_frequencies, _load_mono

# Engines build a full spectrogram, so they use a smaller FFT than the melody engine
# (which stays at analysis.N_FFT=4096 for its verified output). 2048 halves every
# spectrogram array — keeping the analysis subprocess well under the 512 MB container
# budget it shares with the web worker + WARP + PO-token processes. Same HOP as melody
# so component time-bases line up.
ENG_N_FFT = 2048


# ── shared spectral helpers ─────────────────────────────────────────────────────

def _full_mag(y):
    """Full magnitude spectrogram (n_bins, n_frames) float32 + bin frequencies.

    Same framing as analysis.py (center reflect pad, periodic Hann) so pitch results
    line up with the melody engine.
    """
    nfft = ENG_N_FFT
    if y.size < nfft:
        y = np.pad(y, (0, nfft - y.size))
    y = np.pad(y, nfft // 2, mode='reflect')
    n_frames = 1 + (len(y) - nfft) // HOP
    win = hann(nfft, sym=False).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(y, nfft)           # view, no copy
    n_bins = nfft // 2 + 1
    mag = np.empty((n_bins, n_frames), dtype=np.float32)
    # Fill the spectrogram in frame BLOCKS so the transient complex128 rfft buffer stays
    # small (a whole-song rfft was the ~0.5 GB spike). Peak transient ≈ BLOCK*n_bins*16B.
    BLOCK = 512
    for s in range(0, n_frames, BLOCK):
        e = min(s + BLOCK, n_frames)
        frames = windows[s * HOP:(e - 1) * HOP + 1:HOP]                   # (blk, nfft) strided view
        mag[:, s:e] = np.abs(np.fft.rfft(frames * win, axis=1)).astype(np.float32).T
    freqs = np.fft.rfftfreq(nfft, d=1.0 / ANALYSIS_SR)
    return mag, freqs


def _times(n_frames):
    return (np.arange(n_frames) * HOP / ANALYSIS_SR).tolist()


def _energy(mag):
    """Per-frame RMS energy, normalised to [0,1] (drives spiral intensity/opacity)."""
    e = np.sqrt((mag ** 2).sum(axis=0, dtype=np.float64))   # float64 accumulator, no full copy
    m = float(e.max())
    return (e / m if m > 0 else e).tolist()


def _hps_pitch(mag, freqs, fmin=FMIN, fmax=FMAX):
    """Harmonic Product Spectrum dominant pitch per frame (same recipe as melody)."""
    hps = mag.copy()
    for h in range(2, 6):
        down = mag[::h, :]
        n = min(hps.shape[0], down.shape[0])
        hps[:n, :] *= down[:n, :]
    lo = int(np.searchsorted(freqs, fmin))
    hi = int(np.searchsorted(freqs, fmax))
    band = hps[lo:hi, :]
    fb = freqs[lo:hi]
    peak = fb[np.argmax(band, axis=0)].astype(np.float64)
    peak = scipy.signal.medfilt(peak, kernel_size=9)
    alpha = 0.2
    sm = peak.copy()
    for i in range(1, len(sm)):
        sm[i] = alpha * peak[i] + (1.0 - alpha) * sm[i - 1]
    return sm.tolist()


def _centroid_pitch(mag, freqs, lo_hz, hi_hz):
    """Spectral centroid within a band per frame (better than HPS for wide bands)."""
    lo = int(np.searchsorted(freqs, lo_hz))
    hi = int(np.searchsorted(freqs, hi_hz))
    sub = mag[lo:hi, :].astype(np.float64)
    fb = freqs[lo:hi][:, None]
    denom = sub.sum(axis=0)
    denom[denom < 1e-9] = 1e-9
    cen = (fb * sub).sum(axis=0) / denom
    cen = scipy.signal.medfilt(cen, kernel_size=9)
    return cen.tolist()


def _component(name, mag, freqs, pitch):
    return {'name': name, 'times': _times(mag.shape[1]), 'freqs': pitch,
            'energy': _energy(mag)}


# ── engines ─────────────────────────────────────────────────────────────────────

def engine_melody(path):
    """Legacy single spiral: the shipped, verified HPS melody track (memory-bounded)."""
    times, freqs = analyze_frequencies(path)
    # energy isn't produced by the legacy path; use flat 1.0 (spiral radius = pitch as before)
    energy = [1.0] * len(times)
    return [{'name': 'melody', 'times': times, 'freqs': freqs, 'energy': energy}]


BANDS = [('bass', 60.0, 250.0), ('mid', 250.0, 2000.0), ('high', 2000.0, 8000.0)]

def engine_bands(path):
    """Frequency bands ≈ (kick/bass) · (vocals/keys body) · (hats/air). Always works."""
    y = _load_mono(path, ANALYSIS_SR)
    mag, freqs = _full_mag(y)
    comps = []
    for name, lo, hi in BANDS:
        loi = int(np.searchsorted(freqs, lo)); hii = int(np.searchsorted(freqs, hi))
        band_mag = mag[loi:hii, :]
        comps.append({'name': name, 'times': _times(mag.shape[1]),
                      'freqs': _centroid_pitch(mag, freqs, lo, hi),
                      'energy': _energy(band_mag)})
    return comps


def engine_hpss(path, beta=2.0):
    """Harmonic (melody/voice/keys) vs Percussive (drums) via median-filter HPSS.

    Fitzgerald/Driedger: median along time → harmonic-enhanced; along freq → percussive-
    enhanced; Driedger β-masks split the spectrogram. Each masked spectrogram then gets an
    HPS pitch track + energy curve.
    """
    y = _load_mono(path, ANALYSIS_SR)
    mag, freqs = _full_mag(y)
    Lt, Lf = 17, 17  # Fitzgerald's kernel; ~time 0.8s / ~freq 90Hz at our params
    henh = scipy.ndimage.median_filter(mag, size=(1, Lt), mode='reflect')  # harmonic
    penh = scipy.ndimage.median_filter(mag, size=(Lf, 1), mode='reflect')  # percussive
    eps = 1e-9
    mh = (henh / (penh + eps)) > beta
    mp = (penh / (henh + eps)) >= beta
    del henh, penh                          # free before allocating the masked spectrograms
    harm = mag * mh; del mh
    perc = mag * mp; del mp
    return [
        _component('harmonic', harm, freqs, _hps_pitch(harm, freqs)),
        _component('percussive', perc, freqs, _centroid_pitch(perc, freqs, FMIN, 8000.0)),
    ]


ENGINES = {
    'melody': engine_melody,   # default, back-compat single spiral
    'bands':  engine_bands,
    'hpss':   engine_hpss,
}


def run_engine(path, engine):
    fn = ENGINES.get(engine)
    if fn is None:
        raise ValueError(f'unknown engine {engine!r}; choices: {sorted(ENGINES)}')
    components = fn(path)
    return {'engine': engine, 'sr': ANALYSIS_SR, 'components': components}
