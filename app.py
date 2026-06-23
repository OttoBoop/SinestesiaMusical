import os
import threading
from urllib.parse import urlparse
import numpy as np
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


# ── yt-dlp options ────────────────────────────────────────────────────────────

def ytdl_base_opts():
    """
    Base yt-dlp options.

    'deno' is installed as the JS runtime so signature / n-challenge solving
    works. yt-dlp's default client list includes android_vr, which serves
    direct audio URLs without signature solving — this is what makes downloads
    work reliably from a datacenter IP with zero user setup.
    """
    return {
        'quiet':        True,
        'no_warnings':  True,
        'js_runtimes':  {'deno': {}},
    }


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

    opts = ytdl_base_opts()
    opts['extract_flat'] = True

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
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if not is_valid_youtube_url(url):
        return jsonify({'error': 'Please provide a valid YouTube URL'}), 400

    job_id = 'yt_job'
    with download_lock:
        download_jobs[job_id] = {'status': 'downloading', 'progress': 0, 'error': None}

    out_tpl = os.path.join(UPLOAD_FOLDER, 'audio.%(ext)s')

    def do_download():
        def progress_hook(d):
            if d['status'] == 'downloading':
                total      = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
                downloaded = d.get('downloaded_bytes', 0)
                pct = int(downloaded / total * 75)
                with download_lock:
                    download_jobs[job_id]['progress'] = pct
            elif d['status'] == 'finished':
                with download_lock:
                    download_jobs[job_id]['progress'] = 80

        opts = ytdl_base_opts()
        opts.update({
            'format':  'bestaudio/best',
            'outtmpl': out_tpl,
            'postprocessors': [{
                'key':             'FFmpegExtractAudio',
                'preferredcodec':  'mp3',
                'preferredquality': '192',
            }],
            'progress_hooks': [progress_hook],
        })

        try:
            # Clear any previous audio file
            for fname in os.listdir(UPLOAD_FOLDER):
                if fname.startswith('audio.'):
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, fname))
                    except OSError:
                        pass

            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            with download_lock:
                download_jobs[job_id]['progress'] = 85

            mp3_path = os.path.join(UPLOAD_FOLDER, 'audio.mp3')
            if not os.path.exists(mp3_path):
                raise RuntimeError('Download finished but no audio file was produced')
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
