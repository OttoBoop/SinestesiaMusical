"""Precompute worker — run separation ENGINES (including the heavy ML one) off the web tier
and write the derived analysis into the SHARED cache, so the free-tier web app can serve it
with a cheap get() (no OOM, no inline compute).

Run it wherever there's RAM (your machine, or a Render one-off Job with a bigger plan). Point
it at the same cache the web app reads — set CACHE_S3_* for a shared R2/S3 bucket (see
cache.py / docs/visualizer-v2-research.md). For the ML engine also set MDX_MODEL_PATH.

Usage:
    python precompute.py <youtube_url_or_video_id> <engine> [<engine> ...]
e.g. python precompute.py https://youtu.be/dQw4w9WgXcQ ml
"""
import os
import sys
import glob
import shutil
import tempfile
from urllib.parse import urlparse, parse_qs

import cache
import engines


def video_id(url_or_id: str):
    """Accept a full YouTube URL or a bare 11-char id; return the video id."""
    if '/' not in url_or_id and '?' not in url_or_id:
        return url_or_id
    p = urlparse(url_or_id)
    host = (p.hostname or '').lower()
    if 'youtu.be' in host:
        return p.path.lstrip('/').split('/')[0] or None
    return parse_qs(p.query).get('v', [None])[0]


def precompute_file(audio_path: str, source_id: str, engine: str) -> dict:
    """Run one engine on a local audio file and store the result in the shared cache."""
    result = engines.run_engine(audio_path, engine)        # {engine, sr, components}
    first = result['components'][0]
    result['times'] = first['times']                       # legacy fields for the frontend
    result['frequencies'] = first['freqs']
    cache.put(cache.analysis_key(source_id, engine), result)
    return result


def _download(url: str) -> str:
    """Download audio → mp3 into a temp dir; return the mp3 path (caller cleans the dir)."""
    import yt_dlp
    tmp = tempfile.mkdtemp(prefix='precompute_')
    opts = {
        'quiet': True, 'no_warnings': True,
        'format': 'bestaudio[ext=m4a]/bestaudio/bestaudio*',
        'outtmpl': os.path.join(tmp, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3', 'preferredquality': '192'}],
    }
    if os.environ.get('YTDLP_PROXY'):
        opts['proxy'] = os.environ['YTDLP_PROXY']
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)   # downloads AND returns metadata
    mp3s = glob.glob(os.path.join(tmp, '*.mp3'))
    if not mp3s:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError('download produced no mp3')
    return mp3s[0], (info or {})


def precompute_url(url: str, engine: str) -> dict:
    """Download a YouTube track and precompute one engine into the shared cache."""
    vid = video_id(url) or url
    have_analysis = cache.get(cache.analysis_key(vid, engine)) is not None
    have_audio = cache.get_bytes(cache.audio_name(vid)) is not None
    if have_analysis and have_audio:
        print(f'[precompute] {vid} {engine}: analysis + audio already cached, skipping')
        return cache.get(cache.analysis_key(vid, engine))
    mp3, info = _download(url)            # needed for analysis and/or the audio upload
    try:
        result = cache.get(cache.analysis_key(vid, engine))
        if not have_analysis:
            result = precompute_file(mp3, vid, engine)
            print(f'[precompute] {vid} {engine}: analysis cached '
                  f'({[c["name"] for c in result["components"]]})')
        # Stash the audio so the site plays HD tracks from the shared cache (no live download).
        if not have_audio:
            with open(mp3, 'rb') as f:
                cache.put_bytes(cache.audio_name(vid), f.read(), 'audio/mpeg')
            print(f'[precompute] {vid}: audio stored')
        # Make the HD track discoverable in the site's library.
        if engine == 'ml':
            cache.add_to_library({'id': vid, 'title': info.get('title') or vid,
                                  'duration': info.get('duration'),
                                  'thumb': f'https://i.ytimg.com/vi/{vid}/mqdefault.jpg'})
            print(f'[precompute] {vid}: added to HD library')
        return result
    finally:
        shutil.rmtree(os.path.dirname(mp3), ignore_errors=True)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src = sys.argv[1]
    print(f'[precompute] cache backend: {cache.backend_kind()}')
    for engine in sys.argv[2:]:
        try:
            precompute_url(src, engine)
        except Exception as e:
            print(f'[precompute] {engine}: FAILED — {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
