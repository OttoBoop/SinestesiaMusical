"""Ground-truth validation for the 6-stem (engine_ml) pitch tracks — piano/guitar edition.

Replicates the recipe that made the VOCAL analyzer good (validate_pitch.py): separate the
stems the same way the app does (Demucs htdemucs_6s), run the app's tracker on each stem,
and score against tracker-independent gates. A pyin ground truth is INVALID for polyphonic
stems (piano/guitar play chords — several simultaneous F0s), so the gates here are anchors
that hold regardless of polyphony:

  • in-scale %  — voiced frames whose nearest note is in the song's key scale. Piano/guitar
                  play chord tones with no vibrato/slides, so a good line sits ABOVE the
                  vocal ceiling (0.84–0.86): gate 0.85.
  • big-jump %  — consecutive voiced frames leaping > 7 semitones. Chord-tone/octave flicker
                  is exactly how a monophonic tracker fails on chords; melodies rarely leap
                  a fifth+ between adjacent 46 ms frames: gate 0.10.
  • coverage %  — voiced fraction of the frames where the stem is actually sounding
                  (energy > gate). A dead spiral fails this: gate 0.50.
  • median F0 + in-range % vs the instrument's typical register (informational).

Usage:
    python validate_stems.py <song.mp3> <vid> [tracker] [stem ...]
    tracker: pyin (current app path) | salience (the fix)   default: pyin
    stems default: vocals bass guitar piano

Demucs is the slow part → each stem is cached as <mp3>.<stem>.npy next to the file.
"""
import os
import sys
import math
import numpy as np

from validate_pitch import SONGS, in_scale_fraction, note_class  # same GT table as the vocal fix

SR = 22050
HOP = 1024

# Typical sounding register per stem (Hz) — informational sanity, NOT the tracker's search band.
TYPICAL_RANGE = {'vocals': (110.0, 660.0), 'bass': (30.0, 200.0),
                 'guitar': (82.0, 700.0), 'piano': (55.0, 1050.0), 'other': (60.0, 1200.0)}

POLYPHONIC = {'guitar', 'piano', 'other'}   # the fix targets; vocals/bass are the reference
GATES = {'in_scale': 0.85, 'big_jump': 0.10, 'coverage': 0.50}


def separate_stems(mp3):
    """Demucs htdemucs_6s once per song; each stem cached as <mp3>.<stem>.npy."""
    names = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
    if all(os.path.exists(f'{mp3}.{n}.npy') for n in names):
        return {n: np.load(f'{mp3}.{n}.npy').astype(np.float32) for n in names}
    import stems
    st = stems.separate_6stem(mp3)
    for n, y in st.items():
        np.save(f'{mp3}.{n}.npy', y.astype(np.float32))
    return st


def track(name, y, tracker):
    """Run the chosen tracker with the app's own per-stem band (same call path as engine_ml)."""
    import engines
    lo, hi = engines.STEM_BANDS[name]
    fn = {'pyin': engines._pyin_pitch,
          'salience': lambda *a, **k: engines._salience_pitch(*a, **k)}[tracker]
    t, f, e = fn(np.asarray(y, np.float32), lo, hi)
    return np.asarray(t, float), np.asarray(f, float), np.asarray(e, float)


def stem_metrics(freqs, energy, meta, name, energy_gate=0.06):
    f = np.nan_to_num(freqs, nan=0.0)
    voiced = f > 0
    sounding = energy > energy_gate
    v = f[voiced]

    in_scale = in_scale_fraction(v.tolist(), meta['scale'])
    lo, hi = TYPICAL_RANGE[name]
    in_range = float(((v >= lo) & (v <= hi)).mean()) if v.size else 0.0
    coverage = float(voiced[sounding].mean()) if sounding.any() else 0.0

    # jump stats over consecutive voiced frame pairs
    both = voiced[:-1] & voiced[1:]
    if both.any():
        jumps = np.abs(12.0 * np.log2(f[1:][both] / f[:-1][both]))
        big_jump = float((jumps > 7.0).mean())
        mean_jump = float(jumps.mean())
    else:
        big_jump, mean_jump = 1.0, 0.0
    return {'in_scale': in_scale, 'in_range': in_range, 'coverage': coverage,
            'big_jump': big_jump, 'mean_jump_st': mean_jump,
            'median_f0': float(np.median(v)) if v.size else 0.0,
            'voiced_frames': int(v.size)}


def evaluate_song(mp3, vid, tracker='pyin', names=('vocals', 'bass', 'guitar', 'piano')):
    meta = SONGS[vid]
    print(f"\n{'=' * 70}\n{meta['title']}  ({vid})  tracker={tracker}  "
          f"key-scale={sorted(meta['scale'])}")
    st = separate_stems(mp3)
    results = {}
    for name in names:
        t, f, e = track(name, st[name], tracker)
        m = stem_metrics(f, e, meta, name)
        gated = name in POLYPHONIC
        checks = {
            f"in_scale>{GATES['in_scale']}": m['in_scale'] > GATES['in_scale'],
            f"big_jump<{GATES['big_jump']}": m['big_jump'] < GATES['big_jump'],
            f"coverage>{GATES['coverage']}": m['coverage'] > GATES['coverage'],
        }
        verdict = all(checks.values())
        tag = ('PASS' if verdict else 'FAIL') if gated else 'ref '
        print(f"  {name:<7} in-scale={m['in_scale']:5.0%}  big-jump={m['big_jump']:5.0%}  "
              f"mean-jump={m['mean_jump_st']:4.1f}st  coverage={m['coverage']:5.0%}  "
              f"in-range={m['in_range']:5.0%}  medF0={m['median_f0']:6.1f}Hz  "
              f"voiced={m['voiced_frames']:5d}  [{tag}]"
              + ('' if verdict or not gated else '  ' +
                 ' '.join(f"[X {k}]" for k, ok in checks.items() if not ok)))
        results[name] = {**m, 'gate_pass': verdict if gated else None}
    return results


if __name__ == '__main__':
    mp3, vid = sys.argv[1], sys.argv[2]
    tracker = sys.argv[3] if len(sys.argv) > 3 else 'pyin'
    names = tuple(sys.argv[4:]) or ('vocals', 'bass', 'guitar', 'piano')
    evaluate_song(mp3, vid, tracker, names)
