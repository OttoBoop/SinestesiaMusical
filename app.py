import os
import io
import re
import base64
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from flask import Flask, render_template, request, jsonify, send_from_directory
import librosa
import yt_dlp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CHUNK_SIZE = 1024
RATE = 44100

# Track download progress per job
download_jobs = {}
download_lock = threading.Lock()


def autocorrelate(signal):
    correlation = np.correlate(signal, signal, mode="full")
    return correlation[len(correlation) // 2:]


def analyze_frequencies(audio_path):
    y, sr = librosa.load(audio_path, sr=RATE, mono=True)
    audio_data = y.astype(np.float32)

    peak_freqs = []
    for i in range(0, len(audio_data), CHUNK_SIZE):
        data = audio_data[i:i + CHUNK_SIZE]
        if len(data) < CHUNK_SIZE:
            continue
        window = np.hamming(len(data))
        windowed_data = data * window
        autocorr_data = autocorrelate(windowed_data)
        peak_idx = np.argmax(autocorr_data[100:CHUNK_SIZE // 2]) + 100
        peak_freq = RATE / peak_idx
        peak_freqs.append(float(peak_freq))

    time_axis = (np.arange(len(peak_freqs)) * (CHUNK_SIZE / RATE)).tolist()
    return time_axis, peak_freqs


def spiral_r(phi, a=16.5, rotations=5, r_max=1046.5 / 2):
    b = np.log(r_max / a) / (2 * np.pi * rotations)
    return a * np.exp(b * phi)


def generate_spiral_image(radius):
    phi = np.linspace(0, 10 * np.pi, 1000)
    r = spiral_r(phi)
    colored_r = r[r <= radius]
    colored_phi = phi[:len(colored_r)]

    if len(colored_phi) == 0:
        colored_phi = phi[:1]
        colored_r = r[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.plot(phi, r, 'k', linewidth=0.5, alpha=0.3)

    final_angle = colored_phi[-1] % (2 * np.pi)
    hue = final_angle / (2 * np.pi)
    color = mcolors.hsv_to_rgb((hue, 1, 1))
    ax.fill_between(colored_phi, 0, colored_r, color=color, alpha=0.8)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.grid(False)
    ax.spines['polar'].set_visible(False)
    fig.patch.set_facecolor('white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def is_youtube_url(text):
    return bool(re.search(r'(youtube\.com/watch|youtu\.be/)', text))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    f = request.files['audio']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, 'audio.mp3')
    f.save(save_path)

    try:
        times, freqs = analyze_frequencies(save_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'times': times, 'frequencies': freqs, 'audioFile': 'audio.mp3'})


@app.route('/search')
def search_youtube():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'default_search': 'ytsearch8',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'ytsearch8:{query}', download=False)
            results = []
            for entry in info.get('entries', []):
                if entry:
                    duration = entry.get('duration')
                    dur_str = ''
                    if duration:
                        m, s = divmod(int(duration), 60)
                        dur_str = f'{m}:{s:02d}'
                    results.append({
                        'id': entry.get('id', ''),
                        'title': entry.get('title', 'Unknown'),
                        'channel': entry.get('uploader') or entry.get('channel', ''),
                        'duration': dur_str,
                        'thumbnail': entry.get('thumbnail') or f"https://i.ytimg.com/vi/{entry.get('id','')}/mqdefault.jpg",
                        'url': f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                    })
            return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/youtube-download', methods=['POST'])
def youtube_download():
    data = request.get_json()
    url = (data or {}).get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    job_id = 'yt_job'
    with download_lock:
        download_jobs[job_id] = {'status': 'downloading', 'progress': 0, 'error': None}

    out_path = os.path.join(UPLOAD_FOLDER, 'audio.%(ext)s')

    def do_download():
        def progress_hook(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
                downloaded = d.get('downloaded_bytes', 0)
                pct = int(downloaded / total * 80)
                with download_lock:
                    download_jobs[job_id]['progress'] = pct
            elif d['status'] == 'finished':
                with download_lock:
                    download_jobs[job_id]['progress'] = 85

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [progress_hook],
        }

        try:
            # Remove old audio file
            for f in os.listdir(UPLOAD_FOLDER):
                if f.startswith('audio.'):
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, f))
                    except Exception:
                        pass

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            mp3_path = os.path.join(UPLOAD_FOLDER, 'audio.mp3')
            with download_lock:
                download_jobs[job_id]['progress'] = 90

            times, freqs = analyze_frequencies(mp3_path)

            with download_lock:
                download_jobs[job_id].update({
                    'status': 'done',
                    'progress': 100,
                    'times': times,
                    'frequencies': freqs,
                    'audioFile': 'audio.mp3',
                })
        except Exception as e:
            with download_lock:
                download_jobs[job_id].update({'status': 'error', 'error': str(e)})

    t = threading.Thread(target=do_download, daemon=True)
    t.start()
    return jsonify({'jobId': job_id})


@app.route('/youtube-status')
def youtube_status():
    job_id = request.args.get('jobId', 'yt_job')
    with download_lock:
        job = download_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/spiral')
def spiral():
    try:
        freq = float(request.args.get('freq', 200))
        r_max = 1046.5 / 2
        a = 16.5
        rotations = 5
        b = np.log(r_max / a) / (2 * np.pi * rotations)
        radius = a * np.exp(b * (freq / 100.0 * 10 * np.pi / 600))
        radius = max(16, min(radius, r_max))
        img_b64 = generate_spiral_image(radius)
        return jsonify({'image': img_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
