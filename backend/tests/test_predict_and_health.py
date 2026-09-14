import io

from PIL import Image


def _tiny_jpeg_bytes():
    img = Image.new("RGB", (64, 64), color=(120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_reports_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classifier_loaded"] is True
    assert body["database_ok"] is True
    # ENABLE_GENERATION=false in the test env, per conftest.
    assert body["generation_enabled"] is False


def test_predict_rejects_bad_content_type(client):
    resp = client.post(
        "/predict",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 415


def test_predict_rejects_empty_file(client):
    """Regression test: empty bytes used to make cv2.imdecode raise a raw
    C++ assertion instead of returning None, crashing the request instead of
    producing a clean 422."""
    resp = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_predict_rejects_corrupt_image_bytes(client):
    resp = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_predict_accepts_valid_image_anonymously(client):
    resp = client.post(
        "/predict",
        files={"file": ("face.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "identity" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["saved_to_history"] is False  # no auth token supplied
    assert "reconstruction_trust_score" in body
    assert body["reconstruction_trust_score"] is None  # ENABLE_GENERATION=false in test env


def test_predict_saves_history_when_authenticated(client, auth_headers):
    resp = client.post(
        "/predict",
        headers=auth_headers,
        files={"file": ("face.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["saved_to_history"] is True

    history = client.get("/history", headers=auth_headers)
    assert history.status_code == 200
    assert len(history.json()) == 1


def test_history_requires_auth(client):
    resp = client.get("/history")
    assert resp.status_code == 401