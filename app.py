import os
import re
import threading
import subprocess
import numpy as np
import requests as req_lib
from flask import Flask, render_template, request, jsonify, send_from_directory
import librosa
from scipy.signal import medfilt
import yt_dlp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RATE = 44100

download_jobs = {}
download_lock = threading.Lock()

# ── Invidious config ──────────────────────────────────────────────────────────

# Multiple public Invidious instances — tried in order, first success wins.
# Invidious is an open-source YouTube frontend whose API returns working audio
# stream URLs without triggering YouTube's cloud-IP bot detection.
INVIDIOUS_INSTANCES = [
    'https://invidious.fdn.fr',
    'https://yewtu.be',
    'https://invidious.privacydev.net',
    'https://iv.ggtyler.dev',
    'https://invidious.nerdvpn.de',
    'https://invidious.projectsegfau.lt',
]

# itag 140 = m4a/AAC 128 kbps (best quality, easiest to transcode)
# itag 251 = webm/Opus ~160 kbps
# itag 250 = webm/Opus ~64 kbps
ITAG_EXTS = {140: 'm4a', 251: 'webm', 250: 'webm', 249: 'webm'}
PREFERRED_ITAGS = [140, 251, 250, 249]


def extract_video_id(url: str) -> str | None:
    """Extract the 11-char YouTube video ID from any YouTube URL format."""
    m = re.search(r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None


def download_via_invidious(video_id: str, base_path: str, progress_cb=None) -> str:
    """
    Attempt to fetch audio via Invidious proxy URLs, convert to MP3.

    Uses /latest_version?local=true so Invidious proxies the bytes through its
    own server — this avoids YouTube's IP-binding on signed CDN URLs.

    Returns the final mp3 path on success, raises RuntimeError on total failure.
    """
    last_err = RuntimeError('No instances tried')

    for instance in INVIDIOUS_INSTANCES:
        for itag, ext in ((i, ITAG_EXTS[i]) for i in PREFERRED_ITAGS):
            raw_path = base_path + '.' + ext
            try:
                stream_url = (
                    f'{instance}/latest_version'
                    f'?id={video_id}&itag={itag}&local=true'
                )
                r = req_lib.get(
                    stream_url, stream=True, timeout=20,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible)'},
                )
                r.raise_for_status()

                total      = int(r.headers.get('content-length', 0))
                downloaded = 0
                with open(raw_path, 'wb') as fh:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
                            downloaded += len(chunk)
                            if progress_cb and total:
                                progress_cb(int(downloaded / total * 65))

                # Transcode to mp3
                mp3_path = base_path + '.mp3'
                subprocess.run(
                    ['ffmpeg', '-y', '-i', raw_path,
                     '-acodec', 'libmp3lame', '-q:a', '2', mp3_path],
                    check=True, capture_output=True,
                )
                os.remove(raw_path)
                if progress_cb:
                    progress_cb(80)
                return mp3_path

            except Exception as exc:
                last_err = exc
                if os.path.exists(raw_path):
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass
                # Try next itag / next instance

    raise RuntimeError(f'All Invidious instances failed — {last_err}')


def download_via_ytdlp(url: str, base_path: str, progress_cb=None) -> str:
    """Fallback: direct yt-dlp download (may be blocked by YouTube on cloud IPs)."""
    out_tpl = base_path + '.%(ext)s'

    def hook(d):
        if d['status'] == 'downloading' and progress_cb:
            total      = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
            downloaded = d.get('downloaded_bytes', 0)
            progress_cb(int(downloaded / total * 65))
        elif d['status'] == 'finished' and progress_cb:
            progress_cb(80)

    opts = {
        'format':       'bestaudio/best',
        'outtmpl':      out_tpl,
        'postprocessors': [{
            'key':             'FFmpegExtractAudio',
            'preferredcodec':  'mp3',
            'preferredquality': '192',
        }],
        'quiet':        True,
        'no_warnings':  True,
        'progress_hooks': [hook],
        'extractor_args': {
            'youtube': {'player_client': ['tv_embedded', 'ios', 'android', 'mweb']},
        },
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    mp3_path = base_path + '.mp3'
    if not os.path.isfile(mp3_path):
        raise RuntimeError('yt-dlp finished but mp3 not found')
    return mp3_path


# ── Audio analysis ────────────────────────────────────────────────────────────

def analyze_frequencies(audio_path: str):
    """FFT + Harmonic Product Spectrum (HPS) frequency analysis."""
    y, sr = librosa.load(audio_path, sr=RATE, mono=True)

    n_fft      = 4096
    hop_length = 512

    D     = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=RATE, n_fft=n_fft)

    D_hps = D.astype(np.float64).copy()
    for h in range(2, 6):
        D_down = D[::h, :]
        n = min(D_hps.shape[0], D_down.shape[0])
        D_hps[:n, :] *= D_down[:n, :]

    fmin_idx = np.searchsorted(freqs, 60.0)
    fmax_idx = np.searchsorted(freqs, 1200.0)
    D_band   = D_hps[fmin_idx:fmax_idx, :]
    f_band   = freqs[fmin_idx:fmax_idx]

    peak_indices = np.argmax(D_band, axis=0)
    peak_freqs   = f_band[peak_indices].astype(float)
    peak_freqs   = medfilt(peak_freqs, kernel_size=9).astype(float)

    alpha    = 0.2
    smoothed = peak_freqs.copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * peak_freqs[i] + (1.0 - alpha) * smoothed[i - 1]

    time_axis = librosa.frames_to_time(
        np.arange(len(smoothed)), sr=RATE, hop_length=hop_length
    ).tolist()

    return time_axis, smoothed.tolist()


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

    save_path = os.path.join(UPLOAD_FOLDER, 'audio.mp3')
    f.save(save_path)

    try:
        times, freqs = analyze_frequencies(save_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'times': times, 'frequencies': freqs})


# ── YouTube search ────────────────────────────────────────────────────────────

@app.route('/search')
def search_youtube():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Try Invidious search first
    for instance in INVIDIOUS_INSTANCES:
        try:
            r = req_lib.get(
                f'{instance}/api/v1/search',
                params={'q': query, 'type': 'video', 'fields':
                        'videoId,title,author,lengthSeconds,videoThumbnails'},
                timeout=8,
                headers={'User-Agent': 'Mozilla/5.0 (compatible)'},
            )
            r.raise_for_status()
            items   = r.json()
            results = []
            for item in items[:8]:
                vid_id   = item.get('videoId', '')
                duration = item.get('lengthSeconds', 0)
                m, s     = divmod(int(duration), 60)
                thumbs   = item.get('videoThumbnails') or []
                thumb    = next(
                    (t['url'] for t in thumbs if t.get('quality') == 'medium'),
                    f'https://i.ytimg.com/vi/{vid_id}/mqdefault.jpg',
                )
                results.append({
                    'id':        vid_id,
                    'title':     item.get('title', 'Unknown'),
                    'channel':   item.get('author', ''),
                    'duration':  f'{m}:{s:02d}',
                    'thumbnail': thumb,
                    'url':       f'https://www.youtube.com/watch?v={vid_id}',
                })
            if results:
                return jsonify({'results': results})
        except Exception:
            pass  # Try next instance

    # Fallback: yt-dlp flat search (metadata only, less likely to trigger bot check)
    try:
        opts = {
            'quiet':        True,
            'no_warnings':  True,
            'extract_flat': True,
            'extractor_args': {'youtube': {'player_client': ['tv_embedded', 'ios']}},
        }
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
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    job_id = 'yt_job'
    with download_lock:
        download_jobs[job_id] = {'status': 'downloading', 'progress': 0, 'error': None}

    base_path = os.path.join(UPLOAD_FOLDER, 'audio')

    def do_download():
        def set_progress(pct):
            with download_lock:
                download_jobs[job_id]['progress'] = pct

        # Clean up previous audio files
        for fname in os.listdir(UPLOAD_FOLDER):
            if fname.startswith('audio.'):
                try:
                    os.remove(os.path.join(UPLOAD_FOLDER, fname))
                except OSError:
                    pass

        try:
            vid_id = extract_video_id(url)

            # Primary: Invidious (no bot detection)
            if vid_id:
                try:
                    download_via_invidious(vid_id, base_path, set_progress)
                    mp3_path = base_path + '.mp3'
                except Exception as inv_err:
                    # Secondary: yt-dlp direct (may fail on cloud IPs)
                    set_progress(0)
                    try:
                        download_via_ytdlp(url, base_path, set_progress)
                        mp3_path = base_path + '.mp3'
                    except Exception as ydl_err:
                        raise RuntimeError(
                            f'Invidious: {inv_err} | yt-dlp: {ydl_err}'
                        )
            else:
                # Non-standard URL — go straight to yt-dlp
                download_via_ytdlp(url, base_path, set_progress)
                mp3_path = base_path + '.mp3'

            set_progress(85)
            times, freqs = analyze_frequencies(mp3_path)

            with download_lock:
                download_jobs[job_id].update({
                    'status':      'done',
                    'progress':    100,
                    'times':       times,
                    'frequencies': freqs,
                })

        except Exception as e:
            with download_lock:
                download_jobs[job_id].update({'status': 'error', 'error': str(e)})

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
