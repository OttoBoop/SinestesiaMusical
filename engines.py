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
import os
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
        # Search the harmonic pitch from 120 Hz up: the harmonic mask keeps sustained BASS
        # too, and unrestricted HPS locks onto the ~65 Hz bass fundamental (a near-static
        # tiny spiral). Starting at 120 Hz surfaces the actual melody/voice/keys instead.
        _component('harmonic', harm, freqs, _hps_pitch(harm, freqs, fmin=120.0)),
        _component('percussive', perc, freqs, _centroid_pitch(perc, freqs, FMIN, 8000.0)),
    ]


def _load_stereo(path, sr):
    """Decode → stereo → resample via ffmpeg; returns (L, R) float32 arrays."""
    import subprocess
    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', path,
           '-ac', '2', '-ar', str(sr), '-f', 'f32le', '-']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError('ffmpeg could not decode the audio: '
                           + p.stderr.decode('utf-8', 'replace')[-200:])
    a = np.frombuffer(p.stdout, dtype=np.float32)
    a = a[:(a.size // 2) * 2].reshape(-1, 2)      # trim any half-frame so reshape never fails
    if a.shape[0] == 0:
        raise RuntimeError('decoded audio is empty')
    return a[:, 0].copy(), a[:, 1].copy()


def engine_repet(path, k=100, min_dist_s=1.0):
    """REPET-SIM: separate a repeating musical background from a non-repeating vocal.

    The repeating model W is, per frame, the median of that frame's k most-similar frames
    (cosine similarity). The self-similarity is computed in row BLOCKS so the full T×T matrix
    is never materialised. Background mask M = min(W,V)/V; vocal = (1−M)·V. Magnitude only
    (no resynthesis — we just read pitch+energy per component).
    """
    y = _load_mono(path, ANALYSIS_SR)
    V, freqs = _full_mag(y)                       # (n_bins, T)
    T = V.shape[1]
    Vn = (V / (np.linalg.norm(V, axis=0) + 1e-9)).astype(np.float32)   # unit-norm columns
    W = np.empty_like(V)
    kk = min(k, T)
    for s in range(0, T, 256):
        e = min(s + 256, T)
        sims = Vn[:, s:e].T @ Vn                  # (blk, T) cosine similarities
        for jj in range(e - s):
            idx = np.argpartition(sims[jj], -kk)[-kk:]     # k most-similar frames, O(T)
            W[:, s + jj] = np.median(V[:, idx], axis=1)
    del Vn
    M = np.minimum(W, V) / (V + 1e-9)             # background mask in [0,1]
    voice = (1.0 - M) * V
    bg = M * V
    del W, M
    voice[:int(np.searchsorted(freqs, 100.0)), :] = 0.0   # vocals rarely below 100 Hz
    return [
        _component('vocal', voice, freqs, _hps_pitch(voice, freqs)),
        _component('background', bg, freqs, _hps_pitch(bg, freqs)),
    ]


def engine_stereo(path):
    """Stereo center/side split. center = max(0, |mid| − |side|) isolates content panned to
    the middle (usually vocals/bass/kick); sides = |L−R| is the hard-panned rest."""
    L, R = _load_stereo(path, ANALYSIS_SR)
    n = min(len(L), len(R))
    mid_mag, freqs = _full_mag((L[:n] + R[:n]) * 0.5)
    side_mag, _ = _full_mag((L[:n] - R[:n]) * 0.5)
    center = np.maximum(0.0, mid_mag - side_mag).astype(np.float32)
    return [
        _component('center', center, freqs, _hps_pitch(center, freqs)),
        _component('sides', side_mag, freqs, _centroid_pitch(side_mag, freqs, FMIN, 8000.0)),
    ]


_HERE = os.path.dirname(os.path.abspath(__file__))
MDX_MODEL_PATH = os.environ.get('MDX_MODEL_PATH') or os.path.join(
    _HERE, 'models', 'UVR-MDX-NET-Inst_HQ_3.onnx')


def _pyin_pitch(y, fmin, fmax, energy_gate=0.06, conf_gate=0.0):
    """Correct monophonic F0 via probabilistic-YIN (librosa) on a SEPARATED stem, returning
    ({name-less} times, freqs, energy) aligned to the app's HOP time-base.

    Replaces HPS for the HD path: pyin does voiced/unvoiced detection (so the spiral goes
    quiet when nobody is singing) + Viterbi continuity + no octave lock + no FFT-bin snapping.
    Measured on Starlight/Zitti: in-scale 84–86% vs HPS 67–74%. Unvoiced frames → 0 Hz so the
    frontend draws no wedge (radius clamps to the centre)."""
    import librosa
    f0, _voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=ANALYSIS_SR,
                                     frame_length=2048, hop_length=HOP, center=True)
    freqs = np.where(np.isfinite(f0), f0, 0.0).astype(np.float64)
    times = librosa.times_like(f0, sr=ANALYSIS_SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP, center=True)[0]
    n = min(len(freqs), len(times), len(rms), len(prob))
    freqs, times, rms, prob = freqs[:n], times[:n], rms[:n].astype(np.float64), prob[:n]
    mx = float(rms.max())
    energy = (rms / mx if mx > 0 else rms)
    # Silence the pitch where the stem is quiet (a separated stem still bleeds during
    # non-vocal sections → pyin would track that bleed as out-of-key noise). conf_gate adds a
    # pyin-confidence floor (good for the vocal; left at 0 for the polyphonic bass where pyin is
    # inherently less certain, so its spiral doesn't go dead).
    mask = energy < energy_gate
    if conf_gate > 0:
        mask = mask | (np.asarray(prob, dtype=np.float64) < conf_gate)
    freqs[mask] = 0.0
    return times.tolist(), freqs.tolist(), energy.tolist()


def _pitched_component(name, times, freqs, energy):
    return {'name': name, 'times': times, 'freqs': freqs, 'energy': energy}


def _drum_series(y):
    """Drums have no pitch — drive the spiral by percussive BRIGHTNESS (spectral centroid:
    kick→small, snare/hats→big) with loudness as intensity, so the spiral pulses on the beat."""
    import librosa
    cen = librosa.feature.spectral_centroid(y=y, sr=ANALYSIS_SR, n_fft=2048, hop_length=HOP)[0]
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP, center=True)[0]
    times = librosa.times_like(cen, sr=ANALYSIS_SR, hop_length=HOP)
    n = min(len(cen), len(rms), len(times))
    cen, rms, times = cen[:n], rms[:n].astype(np.float64), times[:n]
    frac = np.clip((cen[:n] - 300.0) / (6000.0 - 300.0), 0.0, 1.0)   # map brightness → radius band
    freqs = 70.0 + frac * (500.0 - 70.0)
    mx = float(rms.max())
    energy = rms / mx if mx > 0 else rms
    freqs = np.where(energy > 0.05, freqs, 0.0)                       # only on a hit
    return times.tolist(), freqs.tolist(), energy.tolist()


def _stem_component(name, y, fmin, fmax):
    t, f, e = _pyin_pitch(np.asarray(y, np.float32), fmin, fmax)
    return _pitched_component(name, t, f, e)


# per-stem pitch bands (Hz) for the melodic stems
STEM_BANDS = {'vocals': (65.0, 1000.0), 'bass': (30.0, 350.0),
              'guitar': (80.0, 1200.0), 'piano': (50.0, 1200.0), 'other': (60.0, 1200.0)}
STEM_ORDER = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']


def engine_ml(path):
    """HD multi-instrument separation (Demucs htdemucs_6s) → SIX stems, each with the right
    signal: melodic stems (vocals/bass/guitar/piano/other) get a correct pyin F0 track; drums
    get a percussive brightness+loudness track. One spiral per instrument.

    Heavy (torch/demucs) so it runs ONLY in the off-tier precompute worker; the web tier serves
    the cached result. Lazy imports keep torch off the 512 MB tier."""
    import stems
    st = stems.separate_6stem(path)                        # {drums,bass,other,vocals,guitar,piano}
    comps = []
    for name in STEM_ORDER:
        y = st.get(name)
        if y is None:
            continue
        if name == 'drums':
            dt, df, de = _drum_series(np.asarray(y, np.float32))
            comps.append(_pitched_component('drums', dt, df, de))
        else:
            lo, hi = STEM_BANDS[name]
            comps.append(_stem_component(name, y, lo, hi))
    return comps


ENGINES = {
    'melody': engine_melody,   # default, back-compat single spiral
    'bands':  engine_bands,
    'hpss':   engine_hpss,
    'repet':  engine_repet,    # REPET-SIM: vocal vs repeating background
    'stereo': engine_stereo,   # stereo center (vocal) vs sides
    'ml':     engine_ml,       # HD ONNX MDX: vocals vs instrumental (precompute+cache path)
}


def run_engine(path, engine):
    fn = ENGINES.get(engine)
    if fn is None:
        raise ValueError(f'unknown engine {engine!r}; choices: {sorted(ENGINES)}')
    components = fn(path)
    return {'engine': engine, 'sr': ANALYSIS_SR, 'components': components}
