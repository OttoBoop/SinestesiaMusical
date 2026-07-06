"""Lightweight monophonic pitch tracking — numpy/scipy only (NO librosa/torch), so it runs in
the 512 MB free-tier analysis subprocess. Replaces Harmonic Product Spectrum, which had octave
errors, no interpolation (FFT-bin quantization) and tracked the bass on a mix.

YIN (de Cheveigné & Kawahara 2002): cumulative-mean-normalized difference function + absolute
threshold (defeats octave errors) + parabolic interpolation (beats the FFT-bin grid). Returns a
per-frame F0 (0 Hz where unvoiced), aligned to the given hop.
"""
import numpy as np


def yin_pitch(y, sr, fmin=65.0, fmax=1000.0, frame_length=2048, hop_length=1024,
              threshold=0.15, voiced_max=0.60):
    """Per-frame F0 via YIN. 0 Hz = unvoiced. Aligned to hop_length (like the STFT frames)."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    tau_min = max(1, int(sr / fmax))
    tau_max = min(frame_length - 1, int(sr / fmin) + 1)
    if tau_max <= tau_min:
        return np.zeros(0), np.zeros(0)

    # pad so frames are centred (same convention as the STFT path)
    pad = frame_length // 2
    yp = np.pad(y, pad, mode='reflect')
    n_frames = 1 + (len(yp) - frame_length) // hop_length
    taus = np.arange(tau_max + 1)
    nfft = 1
    while nfft < 2 * frame_length:
        nfft *= 2

    f0 = np.zeros(n_frames)
    for i in range(n_frames):
        x = yp[i * hop_length: i * hop_length + frame_length]
        if x.shape[0] < frame_length:
            break
        x = x - x.mean()
        # difference function d(tau) via FFT autocorrelation + running power sums
        X = np.fft.rfft(x, nfft)
        acf = np.fft.irfft(X * np.conj(X), nfft)[:tau_max + 1]
        cumsq = np.concatenate([[0.0], np.cumsum(x * x)])
        W = frame_length
        term1 = cumsq[W - taus]                    # sum_{0}^{W-tau-1} x^2
        term2 = cumsq[W] - cumsq[taus]             # sum_{tau}^{W-1} x^2
        d = term1 + term2 - 2.0 * acf
        d[d < 0] = 0.0
        # cumulative mean normalized difference
        dp = np.ones_like(d)
        cs = np.cumsum(d[1:])
        nz = cs > 1e-12
        idx = np.arange(1, len(d))
        dp[1:] = np.where(nz, d[1:] * idx / np.where(nz, cs, 1.0), 1.0)

        # absolute threshold: first local min below `threshold` past tau_min
        band = dp[tau_min:tau_max + 1]
        below = np.where(band < threshold)[0]
        if below.size:
            t = tau_min + int(below[0])
            while t + 1 <= tau_max and dp[t + 1] < dp[t]:
                t += 1
        else:
            t = tau_min + int(np.argmin(band))
            if dp[t] > voiced_max:
                continue                            # unvoiced → leave 0
        # parabolic interpolation of the trough
        if tau_min < t < tau_max:
            a, b, c = dp[t - 1], dp[t], dp[t + 1]
            denom = a - 2 * b + c
            t_int = t + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
        else:
            t_int = float(t)
        f0[i] = sr / t_int if t_int > 0 else 0.0

    times = np.arange(n_frames) * hop_length / sr
    return times, f0


def smooth_f0(f0, med=5):
    """Light median smoothing that PRESERVES unvoiced (0) gaps — only smooths voiced runs."""
    from scipy.signal import medfilt
    out = f0.copy()
    voiced = f0 > 0
    if voiced.sum() > med:
        sm = medfilt(np.where(voiced, f0, np.nan_to_num(f0)), kernel_size=med)
        out[voiced] = sm[voiced]
    return out
