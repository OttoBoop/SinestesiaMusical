"""6-stem source separation (Demucs htdemucs_6s) for the multi-instrument visualizer.

Runs ONLY in the off-tier precompute worker (torch/demucs are heavy — never on the 512 MB web
tier). Returns each stem as a mono float32 array at ANALYSIS_SR so engines.py can pitch-track it.

htdemucs_6s sources: drums, bass, other, vocals, guitar, piano.
"""
import numpy as np

ANALYSIS_SR = 22050
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from demucs.pretrained import get_model
        m = get_model('htdemucs_6s')
        m.cpu().eval()
        _MODEL = m
    return _MODEL


def separate_6stem(path):
    """Separate `path` into {name: mono float32 @ ANALYSIS_SR} for the 6 htdemucs_6s stems."""
    import torch
    import librosa
    from demucs.apply import apply_model
    model = _model()
    # Load via librosa (newer torchaudio.load needs torchcodec); stereo at the model's rate.
    y, _ = librosa.load(path, sr=model.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]
    wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = float(ref.mean()), float(ref.std()) + 1e-8
    wav = (wav - mean) / std
    with torch.no_grad():
        sources = apply_model(model, wav[None], device='cpu', split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * std + mean                       # (S, C, T) @ model.samplerate
    out = {}
    for name, src in zip(model.sources, sources):
        mono = src.mean(0).numpy().astype(np.float32)
        mono = librosa.resample(mono, orig_sr=model.samplerate, target_sr=ANALYSIS_SR)
        out[name] = mono.astype(np.float32)
    return out
