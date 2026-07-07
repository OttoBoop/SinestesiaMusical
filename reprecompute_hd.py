"""Re-run the HD (ml) analysis for tracks already in the shared cache — used after an
analyzer change (e.g. the polyphonic salience fix for guitar/piano) — WITHOUT re-downloading
from YouTube: each track's mp3 already lives in the bucket (audio/<vid>.mp3).

For each track: back up the current analysis JSON locally, recompute engine_ml, overwrite
the cache entry, and VERIFY the write by reading it back (cache.put is best-effort and
swallows S3 errors — a silent failure here would look like success). The web tier serves
the new JSON on its next get() — no deploy needed.

Usage (CACHE_S3_* env must be set, same vars as the web service):
    python reprecompute_hd.py              # every track in the HD Library manifest
    python reprecompute_hd.py <vid> [...]  # specific tracks
Backups land in cache-backups/<vid>.ml.json; restore one with:
    python -c "import json,cache; cache.put(cache.analysis_key('<vid>','ml'), json.load(open('cache-backups/<vid>.ml.json')))"
"""
import os
import sys
import json
import time
import tempfile

import cache
import precompute

BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache-backups')


def reprocess(vid):
    key = cache.analysis_key(vid, 'ml')
    old = cache.get(key)
    if old is not None:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        with open(os.path.join(BACKUP_DIR, f'{vid}.ml.json'), 'w') as f:
            json.dump(old, f)
    audio = cache.get_bytes(cache.audio_name(vid))
    if audio is None:
        print(f'[reprecompute] {vid}: no cached audio — skipped', flush=True)
        return False
    fd, path = tempfile.mkstemp(suffix='.mp3')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(audio)
        t0 = time.time()
        result = precompute.precompute_file(path, vid, 'ml')
        back = cache.get(key)
        if back != result:
            print(f'[reprecompute] {vid}: WRITE VERIFY FAILED — cache still stale', flush=True)
            return False
        voiced = {c['name']: sum(1 for x in c['freqs'] if x > 0) for c in result['components']}
        print(f'[reprecompute] {vid}: ok in {time.time() - t0:.0f}s — voiced frames {voiced}',
              flush=True)
        return True
    finally:
        os.unlink(path)


def main():
    vids = sys.argv[1:] or [e['id'] for e in cache.get_library()]
    print(f'[reprecompute] backend={cache.backend_kind()} tracks={len(vids)}', flush=True)
    ok = 0
    for i, vid in enumerate(vids, 1):
        print(f'[reprecompute] ({i}/{len(vids)}) {vid} …', flush=True)
        try:
            ok += bool(reprocess(vid))
        except Exception as e:
            print(f'[reprecompute] {vid}: FAILED — {e}', flush=True)
    print(f'[reprecompute] done: {ok}/{len(vids)} updated', flush=True)


if __name__ == '__main__':
    main()
