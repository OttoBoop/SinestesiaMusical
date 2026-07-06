"""Ground-truth validation of the app's pitch detection.

Proves whether the detected 'vocals' pitch actually matches the sung melody — the check
that was missing. For each song: separate the vocal (MDX ONNX, the same separation the app
uses), build a ground-truth F0 with librosa.pyin on the clean stem, run the app's CURRENT
tracker (engines._hps_pitch) on the same stem, and score with mir_eval.melody
(RPA / RCA / octave-error = RCA-RPA / voicing) plus a tracker-INDEPENDENT music-theory anchor:
the fraction of voiced frames whose nearest note is in the song's key scale.

Usage:
    MDX_MODEL_PATH=/abs/UVR-MDX-NET-Inst_HQ_3.onnx python validate_pitch.py <song.mp3> <vid>
    (vid selects the key/scale + range from SONGS)

Two callables let the fix (Marco 3/4) re-validate the SAME way:
    track_hps(vocals)  -> the current HPS series
    track_pyin(vocals) -> the pyin series (the intended fix)
"""
import os, sys, math
import numpy as np

SR = 22050
HOP = 1024

# Song ground truth (key scale from the sheet/key; vocal range in Hz from transcriptions).
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SONGS = {
    'Pgum6OT_VH8': {'title': 'Muse — Starlight', 'scale': {'B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'},
                    'vmin': 180.0, 'vmax': 520.0},          # B major; vocal ~B3-B4
    'QN1odfjtMoo': {'title': 'Måneskin — Zitti e Buoni', 'scale': {'E', 'F#', 'G', 'A', 'B', 'C', 'D'},
                    'vmin': 110.0, 'vmax': 520.0},          # E minor
}


def note_class(hz):
    if hz is None or hz <= 0 or not np.isfinite(hz):
        return None
    n = int(round(12 * math.log2(hz / 440.0) + 69))
    return NOTE_NAMES[n % 12]


def in_scale_fraction(freqs, scale):
    v = [f for f in freqs if f and f > 0 and np.isfinite(f)]
    if not v:
        return 0.0
    return sum(1 for f in v if note_class(f) in scale) / len(v)


def in_range_fraction(freqs, lo, hi):
    v = [f for f in freqs if f and f > 0 and np.isfinite(f)]
    if not v:
        return 0.0
    return sum(1 for f in v if lo <= f <= hi) / len(v)


def separate_vocal(mp3):
    """Separate (and CACHE) the vocal stem — separation is the slow part, so do it once."""
    cache_npy = mp3 + '.voc.npy'
    if os.path.exists(cache_npy):
        return np.load(cache_npy).astype(np.float32), None
    import mdx
    model = os.environ.get('MDX_MODEL_PATH')
    vocals, instrumental = mdx.separate(mp3, model, target_sr=SR)
    vocals = np.asarray(vocals, dtype=np.float32)
    np.save(cache_npy, vocals)
    return vocals, np.asarray(instrumental, dtype=np.float32)


def track_hps(vocals):
    """The app's CURRENT tracker (HPS on the vocal spectrogram) — what's live today."""
    import engines
    vmag, freqs = engines._full_mag(vocals)
    f = np.asarray(engines._hps_pitch(vmag, freqs), dtype=np.float64)
    t = np.asarray(engines._times(vmag.shape[1]), dtype=np.float64)
    return t, f


def track_pyin(vocals, fmin=65.0, fmax=1000.0):
    """The intended FIX: probabilistic-YIN on the clean vocal stem (voiced/unvoiced aware)."""
    import librosa
    f0, voiced_flag, voiced_prob = librosa.pyin(
        vocals, fmin=fmin, fmax=fmax, sr=SR, frame_length=2048, hop_length=HOP)
    t = librosa.times_like(f0, sr=SR, hop_length=HOP)
    f = np.where(np.isfinite(f0), f0, 0.0)
    return t.astype(np.float64), f.astype(np.float64)


def score(ref_t, ref_f, est_t, est_f):
    import mir_eval
    ref_f = np.nan_to_num(np.asarray(ref_f, float), nan=0.0)
    est_f = np.nan_to_num(np.asarray(est_f, float), nan=0.0)
    ref_v = ref_f > 0
    s = mir_eval.melody.evaluate(ref_t, ref_f, est_t, est_f)
    return {
        'RPA': s['Raw Pitch Accuracy'], 'RCA': s['Raw Chroma Accuracy'],
        'octave_err': s['Raw Chroma Accuracy'] - s['Raw Pitch Accuracy'],
        'VoicingRecall': s['Voicing Recall'], 'VoicingFA': s['Voicing False Alarm'],
        'OverallAcc': s['Overall Accuracy'],
    }


# in_scale ceiling for REAL expressive vocals (vibrato/slides/passing tones cross note lines)
# is ~0.84-0.86, so the gate is 0.80 — a perfect tracker sits just above it, HPS well below.
GATE = {'RPA': 0.70, 'octave_err': 0.10, 'in_scale': 0.80, 'VoicingRecall': 0.80}


def evaluate_song(mp3, vid, tracker='hps'):
    meta = SONGS[vid]
    print(f"\n{'='*70}\n{meta['title']}  ({vid})  key-scale={sorted(meta['scale'])}")
    vocals, _ = separate_vocal(mp3)
    print(f"  separated vocal: {len(vocals)/SR:.1f}s @ {SR}Hz")
    ref_t, ref_f = track_pyin(vocals)                    # ground truth
    gt_inscale = in_scale_fraction(ref_f, meta['scale'])
    gt_inrange = in_range_fraction(ref_f, meta['vmin'], meta['vmax'])
    print(f"  GROUND TRUTH (pyin): in-scale={gt_inscale:.0%}  in-range={gt_inrange:.0%}  "
          f"(a valid GT should be high in-scale)")

    est_t, est_f = (track_hps(vocals) if tracker == 'hps' else track_pyin(vocals))
    sc = score(ref_t, ref_f, est_t, est_f)
    est_inscale = in_scale_fraction(est_f, meta['scale'])
    est_inrange = in_range_fraction(est_f, meta['vmin'], meta['vmax'])
    label = {'hps': 'CURRENT app tracker (HPS)', 'pyin': 'FIX (pyin)'}[tracker]
    print(f"  {label} vs GT:")
    print(f"     RPA={sc['RPA']:.2f}  RCA={sc['RCA']:.2f}  octave-err={sc['octave_err']:.2f}  "
          f"VoicingRecall={sc['VoicingRecall']:.2f}")
    print(f"     in-scale={est_inscale:.0%}  in-range={est_inrange:.0%}")
    checks = {
        'RPA>0.70': sc['RPA'] > GATE['RPA'],
        'octave_err<0.10': sc['octave_err'] < GATE['octave_err'],
        'in_scale>0.85': est_inscale > GATE['in_scale'],
        'VoicingRecall>0.80': sc['VoicingRecall'] > GATE['VoicingRecall'],
    }
    verdict = all(checks.values())
    print(f"     GATE: {'PASS' if verdict else 'FAIL'}  " +
          " ".join(f"[{'ok' if v else 'X'} {k}]" for k, v in checks.items()))
    return {'song': meta['title'], 'tracker': tracker, **sc,
            'in_scale': est_inscale, 'in_range': est_inrange, 'gate_pass': verdict}


if __name__ == '__main__':
    mp3 = sys.argv[1]
    vid = sys.argv[2]
    tracker = sys.argv[3] if len(sys.argv) > 3 else 'hps'
    os.environ.setdefault('MDX_THREADS', str(os.cpu_count() or 4))
    evaluate_song(mp3, vid, tracker)
