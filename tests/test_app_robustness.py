"""Robustness fixes: per-request upload files, ml→melody fallback on /upload,
and pruning of the in-memory download-jobs dict (long-lived worker, no leaks)."""
import io
import os
import sys
import glob
import importlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _fresh_app(tmp_path, monkeypatch):
    monkeypatch.delenv('CACHE_S3_BUCKET', raising=False)
    monkeypatch.delenv('ENABLE_ML', raising=False)
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    import cache; importlib.reload(cache)
    import app; importlib.reload(app)
    return app


def _fake_analysis(audio_path, engine='melody', source_id=None):
    return {'engine': engine, 'sr': 22050,
            'components': [{'name': 'melody', 'times': [0.0], 'freqs': [220.0], 'energy': [1.0]}],
            'times': [0.0], 'frequencies': [220.0]}


def test_upload_uses_per_request_file_and_cleans_up(tmp_path, monkeypatch):
    app = _fresh_app(tmp_path, monkeypatch)
    seen = {}

    def spy_analysis(audio_path, engine='melody', source_id=None):
        seen['path'] = audio_path
        return _fake_analysis(audio_path, engine, source_id)

    monkeypatch.setattr(app, 'run_analysis', spy_analysis)
    client = app.app.test_client()
    r = client.post('/upload', data={'audio': (io.BytesIO(b'ID3fake'), 'song.mp3')},
                    content_type='multipart/form-data')
    assert r.status_code == 200
    # not the old shared filename, and the temp file is gone afterwards
    assert os.path.basename(seen['path']) != 'audio.mp3'
    assert not os.path.exists(seen['path'])
    assert not glob.glob(os.path.join(app.UPLOAD_FOLDER, 'upload_*'))


def test_upload_ml_falls_back_to_melody_with_notice(tmp_path, monkeypatch):
    app = _fresh_app(tmp_path, monkeypatch)
    calls = []

    def queued_then_ok(audio_path, engine='melody', source_id=None):
        calls.append(engine)
        if engine == 'ml':
            raise app.AnalysisQueued('not cached')
        return _fake_analysis(audio_path, engine, source_id)

    monkeypatch.setattr(app, 'run_analysis', queued_then_ok)
    client = app.app.test_client()
    r = client.post('/upload', data={'audio': (io.BytesIO(b'ID3fake'), 'song.mp3'),
                                     'engine': 'ml'},
                    content_type='multipart/form-data')
    data = r.get_json()
    assert r.status_code == 200
    assert calls == ['ml', 'melody']         # fell back instead of dead-ending
    assert data['engine'] == 'melody'
    assert 'notice' in data


def test_download_jobs_dict_is_pruned(tmp_path, monkeypatch):
    app = _fresh_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app, 'download_via_ytdlp',
                        lambda url, base, progress_cb=None: base + '.mp3')
    monkeypatch.setattr(app, 'run_analysis', _fake_analysis)

    with app.download_lock:
        app.download_jobs.clear()
        app.download_jobs['ancient'] = {'status': 'done', 'progress': 100, 'ts': 0}
        app.download_jobs['recent']  = {'status': 'downloading', 'progress': 10,
                                        'ts': __import__('time').time()}

    client = app.app.test_client()
    client.post('/youtube-download',
                json={'url': 'https://www.youtube.com/watch?v=prunetest01', 'engine': 'melody'})

    with app.download_lock:
        assert 'ancient' not in app.download_jobs     # stale entry dropped
        assert 'recent' in app.download_jobs          # live one kept
