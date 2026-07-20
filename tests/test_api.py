"""
Tests for api/main.py's endpoints. Uses FastAPI's TestClient and monkeypatches
`retrieval_core` (imported by api.main as `utils`) rather than the real CLIP/DINOv2
models — conftest.py's session fixture already made `import retrieval_core` safe to
do at all, but individual endpoint behavior (galleries loaded vs. not, search results)
is controlled per-test here.
"""
import io
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from api.main import app
import retrieval_core as utils

client = TestClient(app)


def _make_jpeg_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_expected_shape(monkeypatch):
    monkeypatch.setattr(utils, "CLIP_GALLERY", np.zeros((3, 512)))
    monkeypatch.setattr(utils, "DINOV2_GALLERY", None)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["clip_gallery"]["ready"] is True
    assert body["clip_gallery"]["shape"] == [3, 512]
    assert body["dinov2_gallery"]["ready"] is False
    # Regression guard for reviewLog.md v3: ResNet was removed entirely, /health must
    # not reference it (would otherwise raise AttributeError on utils.RESNET_GALLERY).
    assert "resnet_gallery" not in body


# ── /search — input validation ────────────────────────────────────────────────

def test_search_no_input_returns_400():
    response = client.post("/search", data={})
    assert response.status_code == 400


def test_search_wrong_extension_returns_400():
    response = client.post(
        "/search",
        files={"image": ("photo.txt", b"whatever", "text/plain")},
    )
    assert response.status_code == 400


def test_search_corrupt_image_returns_400():
    response = client.post(
        "/search",
        files={"image": ("photo.jpg", b"not actually a jpeg", "image/jpeg")},
    )
    assert response.status_code == 400


def test_search_oversized_upload_returns_413():
    oversized = b"0" * (config.MAX_CONTENT_LENGTH + 1)
    response = client.post(
        "/search",
        files={"image": ("photo.jpg", oversized, "image/jpeg")},
    )
    assert response.status_code == 413


@pytest.mark.parametrize("top_k", [0, -1, 101])
def test_search_invalid_top_k_returns_422(top_k):
    # Regression test for reviewLog.md v1 finding #5/#7: top_k previously had no bounds.
    response = client.post(
        "/search",
        data={"user_query": "red shirt", "top_k": top_k},
    )
    assert response.status_code == 422


# ── /search — service-unavailable paths (and the 503-vs-500 regression) ──────

def test_search_text_only_503_when_clip_gallery_missing(monkeypatch):
    # Regression test for reviewLog.md v1 finding #1: this used to get rewritten to a
    # 500 by a broad `except Exception` that caught its own HTTPException.
    monkeypatch.setattr(utils, "CLIP_GALLERY", None)
    response = client.post("/search", data={"user_query": "red shirt"})
    assert response.status_code == 503


def test_search_image_503_when_dinov2_missing(monkeypatch):
    monkeypatch.setattr(utils, "DINOV2_GALLERY", None)
    monkeypatch.setattr(utils, "DINOV2_MODEL", None)
    response = client.post(
        "/search",
        files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503


# ── /search — happy path (mocked model calls) ─────────────────────────────────

def test_search_image_happy_path(monkeypatch):
    monkeypatch.setattr(utils, "DINOV2_GALLERY", np.zeros((3, 768)))
    monkeypatch.setattr(utils, "DINOV2_MODEL", object())
    monkeypatch.setattr(utils, "embed_image_dinov2", lambda img: np.zeros((1, 768)))
    monkeypatch.setattr(
        utils, "nearest_search",
        lambda query_emb, gallery_emb, top_k: [
            {"rank": 1, "similarity": 91.0, "image_path": "/x/a.jpg", "image_data": ""}
        ][:top_k],
    )
    monkeypatch.setattr(utils, "_tag_attributes", lambda img: ["striped"])

    response = client.post(
        "/search",
        files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
        data={"top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "image_search"
    assert body["confidence"] == "high"
    assert body["result_count"] == 1


def test_search_text_only_happy_path(monkeypatch):
    monkeypatch.setattr(utils, "CLIP_GALLERY", np.zeros((3, 512)))
    monkeypatch.setattr(
        utils, "text_search",
        lambda query, top_k: [{"rank": 1, "similarity": 80.0, "image_path": "/x/a.jpg", "image_data": ""}],
    )
    response = client.post("/search", data={"user_query": "red shirt"})
    assert response.status_code == 200
    assert response.json()["mode"] == "text_search"


# ── /text-search ──────────────────────────────────────────────────────────────

def test_text_search_503_when_clip_gallery_missing(monkeypatch):
    monkeypatch.setattr(utils, "CLIP_GALLERY", None)
    response = client.post("/text-search", json={"query": "red shirt"})
    assert response.status_code == 503


@pytest.mark.parametrize("top_k", [0, -1, 101, None])
def test_text_search_invalid_top_k_returns_422(top_k):
    response = client.post("/text-search", json={"query": "red shirt", "top_k": top_k})
    assert response.status_code == 422


def test_text_search_happy_path(monkeypatch):
    monkeypatch.setattr(utils, "CLIP_GALLERY", np.zeros((3, 512)))
    monkeypatch.setattr(
        utils, "text_search",
        lambda query, top_k: [{"rank": 1, "similarity": 77.0, "image_path": "/x/a.jpg", "image_data": ""}],
    )
    response = client.post("/text-search", json={"query": "red shirt", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 1
    assert body["query"] == "red shirt"


# ── Frontend & Middleware Tests ────────────────────────────────────────────────

def test_serve_frontend():
    response = client.get("/")
    # Could be 200 if templates/index.html exists, or 404 with specific content if not.
    assert response.status_code in (200, 404)
    if response.status_code == 404:
        assert b"StyleLens Demo Interface is not found" in response.content


def test_serve_frontend_missing_template(monkeypatch):
    # Force mock os.path.exists to return False to test the 404 fallback path
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    response = client.get("/")
    assert response.status_code == 404
    assert b"StyleLens Demo Interface is not found" in response.content


def test_request_id_middleware_generated():
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    # Should be a valid UUID or string
    assert len(response.headers["X-Request-ID"]) > 0


def test_request_id_middleware_propagated():
    custom_id = "test-req-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers.get("X-Request-ID") == custom_id


def test_search_non_ascii_query(monkeypatch):
    monkeypatch.setattr(utils, "CLIP_GALLERY", np.zeros((3, 512)))
    monkeypatch.setattr(
        utils, "text_search",
        lambda query, top_k: [{"rank": 1, "similarity": 95.0, "image_path": "/x/a.jpg", "image_data": ""}],
    )
    # Unicode emoji and script query
    unicode_query = "👗 red dress 👘"
    response = client.post("/search", data={"user_query": unicode_query})
    assert response.status_code == 200
    assert response.json()["mode"] == "text_search"
    assert response.json()["query"] == unicode_query


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
