import os
import io
import json
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from flask import Flask, render_template, request, jsonify, send_from_directory
import librosa

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CHUNK_SIZE = 1024
RATE = 44100


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
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


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

    return jsonify({'times': times, 'frequencies': freqs})


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
