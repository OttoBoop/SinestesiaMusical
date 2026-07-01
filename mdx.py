"""Torch-free MDX-Net separation (numpy + onnxruntime + ffmpeg).

Reimplements the MDX STFT/iSTFT in numpy (no torch/librosa) and runs the ONNX model
CHUNK-BY-CHUNK to bound RAM. For an "Inst" model the network outputs the INSTRUMENTAL;
vocals = mix − instrumental. Returns both as mono at target_sr.

NOTE (measured, Phase 0): this needs ~1.8 GB peak RAM and ~1× realtime even on 4 fast
CPU threads — it CANNOT run inline on the 512 MB free web tier. It's meant for a
precompute/one-off-Job path with the result cached (see docs/visualizer-v2-research.md).
"""
import math
import subprocess
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
from scipy.signal.windows import hann

# UVR-MDX-NET-Inst_HQ_3 params
N_FFT = 6144
HOP = 1024
DIM_T = 2 ** 8            # 256 frames / model chunk
DIM_F = 3072             # bins the model keeps
N_BINS = N_FFT // 2 + 1  # 3073
CHUNK = HOP * (DIM_T - 1)
TRIM = N_FFT // 2
GEN = CHUNK - 2 * TRIM
MDX_SR = 44100
_WIN = hann(N_FFT, sym=False).astype(np.float32)


def _load_stereo(path):
    p = subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', path,
                        '-ac', '2', '-ar', str(MDX_SR), '-f', 'f32le', '-'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError('ffmpeg could not decode the audio: '
                           + p.stderr.decode('utf-8', 'replace')[-200:])
    a = np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2).T.copy()  # (2, n)
    if a.shape[1] == 0:
        raise RuntimeError('decoded audio is empty')
    return a


def _stft(x):                       # x: (C, CHUNK) -> (C, DIM_F, DIM_T) complex64
    pad = N_FFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode='reflect')
    nf = 1 + (xp.shape[1] - N_FFT) // HOP
    sw = np.lib.stride_tricks.sliding_window_view(xp, N_FFT, axis=1)[:, ::HOP][:, :nf]
    spec = np.fft.rfft(sw * _WIN, axis=2).transpose(0, 2, 1)               # (C, N_BINS, nf)
    return spec[:, :DIM_F, :].astype(np.complex64)


def _istft(spec):                   # spec: (C, DIM_F, DIM_T) complex -> (C, CHUNK)
    C = spec.shape[0]
    full = np.zeros((C, N_BINS, DIM_T), dtype=np.complex64)
    full[:, :DIM_F, :] = spec
    frames = np.fft.irfft(full, n=N_FFT, axis=1).astype(np.float32) * _WIN[None, :, None]
    out_len = N_FFT + HOP * (DIM_T - 1)
    y = np.zeros((C, out_len), dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    for t in range(DIM_T):
        s = t * HOP
        y[:, s:s + N_FFT] += frames[:, :, t]
        wsum[s:s + N_FFT] += _WIN ** 2
    wsum[wsum < 1e-8] = 1e-8
    y /= wsum[None, :]
    return y[:, N_FFT // 2:N_FFT // 2 + CHUNK]


def _to_input(spec2):               # (2, DIM_F, DIM_T) complex -> (1, 4, DIM_F, DIM_T)
    ri = np.stack([spec2.real, spec2.imag], axis=1).astype(np.float32)
    return ri.reshape(1, 4, DIM_F, DIM_T)


def _from_output(out):              # (1,4,DIM_F,DIM_T) -> (2, DIM_F, DIM_T) complex
    o = out.reshape(2, 2, DIM_F, DIM_T)
    return (o[:, 0] + 1j * o[:, 1]).astype(np.complex64)


def _to_mono(x, target_sr):
    mono = x.mean(axis=0)
    if target_sr != MDX_SR:
        g = math.gcd(target_sr, MDX_SR)
        mono = resample_poly(mono, target_sr // g, MDX_SR // g).astype(np.float32)
    return mono


def separate(path, model_path, target_sr=MDX_SR, threads=1):
    """Return (vocals_mono, instrumental_mono) at target_sr for an Inst MDX model."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False                 # ~halves peak RSS (Phase 0)
    sess = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    mix = _load_stereo(path)
    n = mix.shape[1]
    pad = GEN - (n % GEN)
    mixp = np.concatenate([np.zeros((2, TRIM), np.float32), mix,
                           np.zeros((2, pad), np.float32), np.zeros((2, TRIM), np.float32)], axis=1)
    stem = []
    i = 0
    while i < n + pad:
        wav = mixp[:, i:i + CHUNK]
        if wav.shape[1] < CHUNK:
            wav = np.pad(wav, ((0, 0), (0, CHUNK - wav.shape[1])))
        out = sess.run(None, {in_name: _to_input(_stft(wav))})[0]
        stem.append(_istft(_from_output(out))[:, TRIM:-TRIM])
        i += GEN
    instrumental = np.concatenate(stem, axis=1)[:, :n]      # Inst model → instrumental
    vocals = mix - instrumental
    return _to_mono(vocals, target_sr), _to_mono(instrumental, target_sr)
