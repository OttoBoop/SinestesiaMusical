import os
import glob
import hashlib
import json
import shutil
import sys
import tempfile
import threading
import subprocess
import time
import uuid
from urllib.parse import urlparse, parse_qs
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, abort
import yt_dlp


def youtube_video_id(url: str):
    """Extract the YouTube video id from a URL, for use as a cache source id."""
    try:
        p = urlparse(url)
    except ValueError:
        return None
    host = (p.hostname or '').lower()
    if 'youtu.be' in host:
        return (p.path.lstrip('/').split('/')[0] or None)
    vid = parse_qs(p.query).get('v', [None])[0]
    return vid or None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# The frequency analysis runs as an isolated subprocess (analysis.py) so a heavy
# or very long track can never OOM-kill or hang this web worker. ANALYSIS_TIMEOUT
# bounds the wait; on the free tier the numpy/scipy analysis is a few seconds.
ANALYSIS_SCRIPT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis.py')
ANALYSIS_TIMEOUT = 120

download_jobs = {}
download_lock = threading.Lock()

# ── Proof-of-Origin (PO) token provider ─────────────────────────────────────────

# A small companion service (bgutil-ytdlp-pot-provider, started by
# scripts/run_pot_provider.sh) generates the PO tokens YouTube requires to pass
# its "confirm you're not a bot" check — automatically, with no account/login.
# yt-dlp talks to it over HTTP via the bgutil plugin.
POTOKEN_PORT    = 4416
POTOKEN_BASEURL = f'http://127.0.0.1:{POTOKEN_PORT}'

# Player clients to try with yt-dlp, in order. 'android_vr' and 'tv' serve audio
# URLs that need NO PO token (the most reliable path), so they go first; the
# remaining clients work hand-in-hand with the PO token provider as a fallback.
YTDLP_PLAYER_CLIENTS = ['android_vr', 'tv', 'default', 'web_safari', 'mweb']


# ── Outbound proxy (to dodge YouTube's datacenter-IP block) ─────────────────────
#
# YouTube blocks most download requests from datacenter IPs (Render is one). The fix
# is to egress through a residential-looking IP. By default scripts/start.sh brings up
# a free Cloudflare WARP proxy and points YTDLP_PROXY at it. Set YTDLP_PROXY to a
# residential-proxy URL (e.g. http://user:pass@host:port) to use that instead.

def current_proxy():
    """The proxy URL yt-dlp should egress through, or None for a direct connection."""
    return (os.environ.get('YTDLP_PROXY') or '').strip() or None


# ── yt-dlp options ────────────────────────────────────────────────────────────

def ytdl_base_opts(proxy=None):
    """
    Base yt-dlp options shared by search and download.

    'deno' is installed as the JS runtime so signature / n-challenge solving works.
    The client list (YTDLP_PLAYER_CLIENTS) puts android_vr/tv first — they serve
    audio without a PO token. ``proxy`` (when given) routes the request through a
    residential-looking IP so YouTube doesn't reject it as datacenter traffic.
    """
    opts = {
        'quiet':        True,
        'no_warnings':  True,
        'js_runtimes':  {'deno': {}},
    }
    if proxy:
        opts['proxy'] = proxy
    return opts


def yt_extractor_args():
    """Extractor args wiring the PO-token provider and player clients.

    Layered on top of ``ytdl_base_opts()`` so downloads benefit from both the
    deno/android_vr path and automatic PO tokens.
    """
    return {
        'youtube': {'player_client': YTDLP_PLAYER_CLIENTS},
        # Tell the bgutil plugin where the PO-token provider is listening.
        'youtubepot-bgutilhttp': {'base_url': [POTOKEN_BASEURL]},
    }


def cleanup_old_downloads(max_age_s: int = 600) -> None:
    """Drop audio/ytdl leftovers from FINISHED jobs, but never touch fresh files — so two
    concurrent downloads don't delete each other's in-progress audio (the old blanket wipe
    of every ``audio.*`` did, which is why simultaneous users clobbered one another)."""
    now = time.time()
    for fname in os.listdir(UPLOAD_FOLDER):
        if not (fname.startswith('audio') or fname.startswith('ytdl_')):
            continue
        path = os.path.join(UPLOAD_FOLDER, fname)
        try:
            if now - os.path.getmtime(path) < max_age_s:
                continue                      # still in use by an active job — leave it
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass


def cleanup_job(job_id: str) -> None:
    """Remove just THIS job's audio file(s) — used on error so we don't touch other jobs."""
    for path in glob.glob(os.path.join(UPLOAD_FOLDER, f'audio_{job_id}*')):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass


def download_via_ytdlp(url: str, base_path: str, progress_cb=None) -> str:
    """
    Authenticated yt-dlp download with deno JS runtime + automatic PO-token support.

    Downloads **audio only** into an isolated temp directory (so a stale file can
    never cause the 'Unable to rename file' crash), transcodes to mp3, then moves
    the result to ``base_path + '.mp3'``. The temp dir is always cleaned up.
    """
    tmp_dir = tempfile.mkdtemp(prefix='ytdl_', dir=UPLOAD_FOLDER)
    out_tpl = os.path.join(tmp_dir, '%(id)s.%(ext)s')

    def hook(d):
        if d['status'] == 'downloading' and progress_cb:
            total      = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
            downloaded = d.get('downloaded_bytes', 0)
            progress_cb(int(downloaded / total * 65))
        elif d['status'] == 'finished' and progress_cb:
            progress_cb(80)

    def build_opts(proxy):
        opts = ytdl_base_opts(proxy)
        opts.update({
            # Audio-only selection. Never falls back to '/best', which could pull a
            # full multi-hundred-MB video file.
            'format':       'bestaudio[ext=m4a]/bestaudio/bestaudio*',
            'outtmpl':      out_tpl,
            'paths':        {'home': tmp_dir, 'temp': tmp_dir},
            'noplaylist':   True,
            'overwrites':   True,
            'postprocessors': [{
                'key':             'FFmpegExtractAudio',
                'preferredcodec':  'mp3',
                'preferredquality': '192',
            }],
            'progress_hooks': [hook],
            'extractor_args': yt_extractor_args(),
        })
        return opts

    # Try the proxy first (residential-looking IP), then fall back to a direct
    # connection — some videos work direct and it costs nothing to try.
    proxy = current_proxy()
    attempts = ([(f'proxy ({proxy})', proxy)] if proxy else []) + [('direct', None)]

    try:
        last_err = None
        for label, p in attempts:
            try:
                with yt_dlp.YoutubeDL(build_opts(p)) as ydl:
                    ydl.download([url])
                if glob.glob(os.path.join(tmp_dir, '*.mp3')):
                    break
                last_err = RuntimeError('yt-dlp finished but no mp3 was produced')
                print(f'[youtube-download] {label}: no mp3 produced', flush=True)
            except Exception as e:
                last_err = e
                print(f'[youtube-download] {label} attempt failed: {e}', flush=True)

        produced = glob.glob(os.path.join(tmp_dir, '*.mp3'))
        if not produced:
            raise last_err or RuntimeError('download failed')

        mp3_path = base_path + '.mp3'
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        shutil.move(produced[0], mp3_path)
        return mp3_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Audio analysis ────────────────────────────────────────────────────────────

class AnalysisError(Exception):
    """Raised when the isolated analysis subprocess fails (timeout/OOM/decode)."""


class AnalysisQueued(Exception):
    """Raised when an ML result isn't cached yet and must be precomputed off-tier."""


import cache

# The ML engine is a VALID choice, but on the free web tier it is served from the shared
# cache ONLY — never computed inline (it needs ~1.8 GB / ~1x realtime, measured Phase 0, and
# would OOM the 512 MB instance). A precompute worker (precompute.py) populates the cache.
# Set ENABLE_ML=1 on a bigger instance to also allow inline ML computation there.
VALID_ENGINES = {'melody', 'bands', 'hpss', 'repet', 'stereo', 'ml'}
INLINE_ML = os.environ.get('ENABLE_ML') == '1'


def run_analysis(audio_path: str, engine: str = 'melody', source_id: str = None) -> dict:
    """Analyze ``audio_path`` with the chosen ENGINE in an isolated subprocess.

    Returns the v2 result dict ``{engine, sr, components:[{name,times,freqs,energy}],
    times, frequencies}`` (the last two are the first component, kept for the legacy
    single-spiral frontend). The real DSP lives in analysis.py + engines.py and runs
    as a separate, short-lived process: a track too long/heavy fails the child alone —
    this worker stays up and we raise an ``AnalysisError`` mapped to a friendly message,
    instead of hanging at 85% or crashing the instance.

    When ``source_id`` is given (YouTube video id, or an uploaded file's content hash) the
    derived analysis is cached by (source_id, engine) — a repeat request is served instantly
    and, for the heavy ML engine, a precomputed result can be served without recomputing.
    """
    if engine not in VALID_ENGINES:
        engine = 'melody'

    key = cache.analysis_key(source_id, engine) if source_id else None
    if key:
        cached = cache.get(key)
        if cached is not None:
            return cached

    # ML is precomputed off-tier and served from cache only — never run inline on the free
    # web worker (it would OOM). A cache miss means "not prepared yet".
    if engine == 'ml' and not INLINE_ML:
        raise AnalysisQueued(
            "HD vocal separation isn't ready for this track yet — it's prepared in the "
            "background. Try again in a bit, or use the Vocal/Harmonic engines meanwhile.")

    out_path = audio_path + '.analysis.json'
    try:
        os.remove(out_path)
    except OSError:
        pass

    try:
        proc = subprocess.run(
            [sys.executable, ANALYSIS_SCRIPT, audio_path, out_path, engine],
            capture_output=True, timeout=ANALYSIS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise AnalysisError(
            f'analysis timed out after {ANALYSIS_TIMEOUT}s (track too long for this server)')

    try:
        if proc.returncode != 0:
            detail = (proc.stderr or b'').decode('utf-8', 'replace').strip()[-300:]
            # A negative return code means a signal — typically -9 (OOM-killed child).
            if proc.returncode < 0:
                detail = f'analysis process was killed (signal {-proc.returncode}); {detail}'
            raise AnalysisError(detail or f'analysis exited with code {proc.returncode}')

        try:
            with open(out_path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise AnalysisError(f'analysis produced no usable result: {e}')

        if 'error' in data:
            raise AnalysisError(data['error'])
        if key:
            cache.put(key, data)
        return data
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


ALLOWED_YT_HOSTS = {
    'youtube.com', 'www.youtube.com', 'm.youtube.com',
    'music.youtube.com', 'youtu.be',
}


def is_valid_youtube_url(url: str) -> bool:
    """Only allow real YouTube URLs — yt-dlp will otherwise fetch arbitrary
    (incl. internal) URLs supplied by the client."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ('http', 'https'):
        return False
    host = (parsed.hostname or '').lower()
    return host in ALLOWED_YT_HOSTS


def friendly_youtube_error(exc) -> str:
    """Map a yt-dlp exception to a useful user-facing message.

    The raw error is always logged server-side; this just decides what the user
    sees, and whether retrying is worth it.
    """
    msg = str(exc).lower()
    if 'sign in to confirm' in msg or 'not a bot' in msg or 'bot' in msg:
        return ("YouTube is blocking this download from our server right now "
                "(anti-bot check). Try again shortly, or try another song.")
    if any(s in msg for s in ('private', 'unavailable', 'removed', 'age-restricted', 'age restricted')):
        return "This video can't be downloaded (private, removed, or age-restricted). Try another."
    if '429' in msg or 'too many requests' in msg:
        return "We're being rate-limited by YouTube. Please wait a minute and try again."
    return ("Couldn't fetch this track from YouTube right now. It may be unavailable "
            "or temporarily blocked — please try again or pick another song.")


def friendly_analysis_error(exc) -> str:
    """Map an analysis (subprocess) failure to a user-facing message.

    The technical cause is logged server-side; the user just learns it's about the
    audio being too long/heavy for the server, not a YouTube problem.
    """
    msg = str(exc).lower()
    if 'timed out' in msg or 'killed' in msg or 'signal' in msg or 'memory' in msg:
        return ("This track is too long or heavy to analyze on our small server. "
                "Try a shorter song or clip.")
    if 'decode' in msg or 'empty' in msg:
        return "We couldn't read this audio. Try another file or song."
    return ("Something went wrong analyzing this track. Please try again or pick "
            "another song.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    f = request.files['audio']
    if not f.filename:
        return jsonify({'error': 'No file selected'}), 400

    # Per-request filename: two simultaneous uploads must never overwrite each other's
    # file mid-analysis (the fixed 'audio.mp3' did exactly that). The browser plays the
    # user's own local copy (objectURL), so the server file is analysis-only → delete after.
    save_path = os.path.join(UPLOAD_FOLDER, f'upload_{uuid.uuid4().hex[:12]}.mp3')
    f.save(save_path)

    engine = (request.form.get('engine') or 'melody').strip()
    try:
        with open(save_path, 'rb') as fh:
            sid = 'file:' + hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        sid = None
    try:
        try:
            result = run_analysis(save_path, engine, source_id=sid)
        except AnalysisQueued:
            # ML isn't precomputed for uploads — fall back to the standard engine instead
            # of dead-ending the upload (same trap as the YouTube path: the HD Library
            # leaves the engine set to ml).
            result = run_analysis(save_path, 'melody', source_id=sid)
            result['notice'] = ("Vocal HD isn't available for uploaded files — using the "
                                "standard Melody engine instead.")
    except AnalysisError as e:
        print(f'[upload] analysis failed for {f.filename!r} (engine={engine}) — {e}', flush=True)
        return jsonify({'error': friendly_analysis_error(e)}), 500
    except Exception as e:
        print(f'[upload] unexpected error for {f.filename!r} — {e}', flush=True)
        return jsonify({'error': 'Something went wrong analyzing this file.'}), 500
    finally:
        try:
            os.remove(save_path)
        except OSError:
            pass

    return jsonify(result)


# ── YouTube search ────────────────────────────────────────────────────────────

@app.route('/search')
def search_youtube():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    opts = ytdl_base_opts(current_proxy())
    opts['extract_flat']  = True
    opts['extractor_args'] = yt_extractor_args()

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info    = ydl.extract_info(f'ytsearch8:{query}', download=False)
            results = []
            for entry in info.get('entries', []):
                if not entry:
                    continue
                duration = entry.get('duration')
                dur_str  = ''
                if duration:
                    m, s = divmod(int(duration), 60)
                    dur_str = f'{m}:{s:02d}'
                vid_id = entry.get('id', '')
                results.append({
                    'id':        vid_id,
                    'title':     entry.get('title', 'Unknown'),
                    'channel':   entry.get('uploader') or entry.get('channel', ''),
                    'duration':  dur_str,
                    'thumbnail': entry.get('thumbnail') or
                                 f'https://i.ytimg.com/vi/{vid_id}/mqdefault.jpg',
                    'url':       f'https://www.youtube.com/watch?v={vid_id}',
                })
            return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── YouTube download ──────────────────────────────────────────────────────────

@app.route('/youtube-download', methods=['POST'])
def youtube_download():
    data = request.get_json()
    url  = (data or {}).get('url', '').strip()
    engine = ((data or {}).get('engine') or 'melody').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if not is_valid_youtube_url(url):
        return jsonify({'error': 'Please provide a valid YouTube URL'}), 400

    # Unique per request so concurrent users never clobber each other's job or audio file.
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with download_lock:
        # Prune stale entries so the dict can't grow forever on a long-lived worker.
        # 1h is far beyond any real job (download minutes + 120s analysis timeout).
        for jid in [j for j, job in download_jobs.items() if now - job.get('ts', now) > 3600]:
            del download_jobs[jid]
        download_jobs[job_id] = {'status': 'downloading', 'progress': 0, 'error': None, 'ts': now}

    base_path = os.path.join(UPLOAD_FOLDER, f'audio_{job_id}')

    def do_download():
        def set_progress(pct):
            with download_lock:
                download_jobs[job_id]['progress'] = pct

        cleanup_old_downloads()

        eng    = engine
        notice = None
        sid    = youtube_video_id(url)
        # ML is cache-only on the web tier: serve a precomputed result if there is one —
        # WITHOUT downloading + separating inline (that would OOM the free tier).
        if eng == 'ml' and not INLINE_ML:
            cached = cache.get(cache.analysis_key(sid, 'ml')) if sid else None
            if cached is not None:
                # audio was stashed in the shared cache by the worker → play it from there
                cached = {**cached, 'audioUrl': f'/cached-audio/{sid}'}
                with download_lock:
                    download_jobs[job_id].update({'status': 'done', 'progress': 100, **cached})
                return
            # Not cached, and nothing on this tier can prepare it — fall back to the standard
            # engine so the download still works. Dead-ending here is what made every search
            # download "fail" once the HD Library had set the engine to ml (2026-07-07).
            eng    = 'melody'
            notice = ("Vocal HD isn't available for this track — using the standard Melody "
                      "engine instead. Pre-separated songs are in the HD Library.")

        try:
            download_via_ytdlp(url, base_path, set_progress)
            mp3_path = base_path + '.mp3'

            set_progress(85)
            result = run_analysis(mp3_path, eng, source_id=sid)
            if notice:
                result['notice'] = notice
            # Play this job's own downloaded file (unless the engine already provided an
            # audioUrl, e.g. a cached HD track). Per-job filename → no cross-user clobber.
            result.setdefault('audioUrl', f'/audio/audio_{job_id}.mp3')

            with download_lock:
                download_jobs[job_id].update({
                    'status':   'done',
                    'progress': 100,
                    **result,          # engine, sr, components[], audioUrl, + legacy times/frequencies
                })

        except AnalysisQueued as e:
            cleanup_job(job_id)
            with download_lock:
                download_jobs[job_id].update({'status': 'error', 'error': str(e)})

        except AnalysisError as e:
            # Download succeeded but analysis failed — not a YouTube problem.
            cleanup_job(job_id)
            print(f'[youtube-download] analysis failed for {url!r} — {e}', flush=True)
            with download_lock:
                download_jobs[job_id].update({
                    'status': 'error',
                    'error':  friendly_analysis_error(e),
                })

        except Exception as e:
            # Log the technical detail server-side, show a friendly message to the user.
            cleanup_job(job_id)
            print(f'[youtube-download] failed for {url!r} — {e}', flush=True)
            with download_lock:
                download_jobs[job_id].update({
                    'status': 'error',
                    'error':  friendly_youtube_error(e),
                })

    threading.Thread(target=do_download, daemon=True).start()
    return jsonify({'jobId': job_id})


@app.route('/youtube-status')
def youtube_status():
    job_id = request.args.get('jobId', 'yt_job')
    with download_lock:
        job = download_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def _ranged_audio_response(data, mimetype='audio/mpeg'):
    """Serve in-memory audio bytes WITH HTTP Range support (206 Partial Content).

    Without this the browser can't SEEK a cached HD track — a plain 200 with no
    Accept-Ranges makes the <audio> element non-seekable, so clicking the progress bar
    does nothing and the timer feels dead. Honour `Range: bytes=start-end` (and suffix
    ranges) → 206 + Content-Range; a bad range → 416; no range → full 200."""
    total = len(data)
    common = {'Accept-Ranges': 'bytes', 'Cache-Control': 'public, max-age=86400'}
    rng = request.headers.get('Range')
    if rng:
        try:
            units, spec = rng.split('=', 1)
            if units.strip().lower() != 'bytes':
                raise ValueError
            start_s, _, end_s = spec.partition('-')
            if start_s == '':                       # suffix: last N bytes
                n = int(end_s); start = max(0, total - n); end = total - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else total - 1
            end = min(end, total - 1)
            if start > end or start >= total or start < 0:
                raise ValueError
        except (ValueError, TypeError):
            resp = Response(status=416)
            resp.headers['Content-Range'] = f'bytes */{total}'
            resp.headers.update(common)
            return resp
        chunk = data[start:end + 1]
        resp = Response(chunk, status=206, mimetype=mimetype)
        resp.headers['Content-Range'] = f'bytes {start}-{end}/{total}'
        resp.headers['Content-Length'] = str(len(chunk))
        resp.headers.update(common)
        return resp
    resp = Response(data, mimetype=mimetype)
    resp.headers['Content-Length'] = str(total)
    resp.headers.update(common)
    return resp


@app.route('/cached-audio/<vid>')
def cached_audio(vid):
    """Stream a precomputed HD track's audio from the shared cache, WITH range/seek support."""
    if not vid or '/' in vid or '\\' in vid:
        abort(404)
    data = cache.get_bytes(cache.audio_name(vid))
    if data is None:
        abort(404)
    return _ranged_audio_response(data)


@app.route('/hd-library')
def hd_library():
    """The browsable list of pre-saved HD tracks (populated by the precompute worker)."""
    return jsonify({'tracks': cache.get_library()})


@app.route('/hd-track/<vid>')
def hd_track(vid):
    """Serve a pre-saved HD track STRAIGHT from the shared cache — analysis + audioUrl —
    with NO YouTube download and NO job machinery. This is the path the HD Library uses so
    clicking a saved song never triggers a download (yt-dlp is never even reached here)."""
    if not vid or '/' in vid or '\\' in vid:
        abort(404)
    cached = cache.get(cache.analysis_key(vid, 'ml'))
    if cached is None:
        return jsonify({'error': 'This track is not ready yet — it is still being prepared.'}), 404
    result = {**cached, 'audioUrl': f'/cached-audio/{vid}'}
    return jsonify(result)


@app.route('/hd-analyze/<vid>', methods=['POST'])
def hd_analyze(vid):
    """Run a CLASSICAL engine (melody/bands/hpss/repet/stereo) on a pre-saved HD track's cached
    audio, so the engine buttons work on HD-Library tracks too — not just 'Vocal HD'. Async job
    (reuses the youtube-status poller) since inline analysis of a full song takes ~30–60s on the
    free tier. The mp3 is already in the shared cache; result is cached by (vid, engine) so a
    repeat click is instant. ML stays cache-only via /hd-track (never runs inline)."""
    if not vid or '/' in vid or '\\' in vid:
        abort(404)
    engine = ((request.get_json(silent=True) or {}).get('engine') or 'melody').strip()
    if engine == 'ml' or engine not in VALID_ENGINES:
        return jsonify({'error': 'Vocal HD is served from the cache — pick a classical engine here.'}), 400

    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with download_lock:
        for jid in [j for j, job in download_jobs.items() if now - job.get('ts', now) > 3600]:
            del download_jobs[jid]
        # progress starts at 85 → the poller shows "Analyzing frequencies…" (there is no download)
        download_jobs[job_id] = {'status': 'downloading', 'progress': 85, 'error': None, 'ts': now}

    def do_analyze():
        tmp = None
        try:
            result = cache.get(cache.analysis_key(vid, engine))       # instant on a repeat click
            if result is None:
                data = cache.get_bytes(cache.audio_name(vid))
                if data is None:
                    raise AnalysisError('This track has no saved audio to analyze.')
                fd, tmp = tempfile.mkstemp(prefix=f'hd_{vid}_', suffix='.mp3', dir=UPLOAD_FOLDER)
                with os.fdopen(fd, 'wb') as f:
                    f.write(data)
                result = run_analysis(tmp, engine, source_id=vid)    # isolated subprocess, caches
            payload = {**result, 'audioUrl': f'/cached-audio/{vid}'}
            with download_lock:
                download_jobs[job_id].update({'status': 'done', 'progress': 100, **payload})
        except AnalysisError as e:
            with download_lock:
                download_jobs[job_id].update({'status': 'error', 'error': friendly_analysis_error(e)})
        except Exception as e:
            print(f'[hd-analyze] {vid} {engine} failed — {e}', flush=True)
            with download_lock:
                download_jobs[job_id].update({'status': 'error',
                                              'error': 'Could not analyze this track — try another engine.'})
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    threading.Thread(target=do_analyze, daemon=True).start()
    return jsonify({'jobId': job_id})


if __name__ == '__main__':
    # Production runs under gunicorn (see scripts/start.sh); this branch is for
    # local `python app.py`. Honour $PORT so it matches the container default.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
