"""
Tests for retrieval_core.py's pure helper functions — l2_normalize, nearest_search,
_validate_gallery, _clean_tag, _image_to_base64. Deliberately does NOT test
embed_text/embed_image_clip/embed_image_dinov2 directly: those require the real
CLIP/DINOv2 tensor pipelines end-to-end, which would need much deeper mocking for
low return — the model objects are mocked in conftest.py only so the module imports
cleanly, not to fake real inference.
"""
import os

import numpy as np
import pytest
from PIL import Image

import retrieval_core as rc


# ── l2_normalize ──────────────────────────────────────────────────────────────

def test_l2_normalize_unit_length():
    v = np.array([[3.0, 4.0]])  # length 5.0
    normed = rc.l2_normalize(v)
    assert np.allclose(normed, [[0.6, 0.8]])
    assert np.allclose(np.linalg.norm(normed, axis=-1), 1.0)


def test_l2_normalize_zero_vector_no_division_by_zero():
    v = np.array([[0.0, 0.0]])
    normed = rc.l2_normalize(v)
    assert np.all(np.isfinite(normed))
    assert np.allclose(normed, [[0.0, 0.0]])


# ── nearest_search ────────────────────────────────────────────────────────────

def test_nearest_search_exact_match_ranks_first():
    query = np.array([[1.0, 0.0]])
    gallery = np.array([[0.0, 1.0], [1.0, 0.0]])
    results = rc.nearest_search(query, gallery, top_k=2)
    assert results[0]["rank"] == 1
    assert results[0]["similarity"] == 100.0


def test_nearest_search_respects_top_k():
    query = np.array([[1.0, 0.0]])
    gallery = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    results = rc.nearest_search(query, gallery, top_k=2)
    assert len(results) == 2


def test_nearest_search_empty_gallery_returns_empty_list():
    query = np.array([[1.0, 0.0]])
    gallery = np.zeros((0, 2))
    results = rc.nearest_search(query, gallery, top_k=5)
    assert results == []


# ── _validate_gallery (reviewLog.md v2, finding #10) ─────────────────────────

def test_validate_gallery_none_stays_none():
    assert rc._validate_gallery(None, "test") is None


def test_validate_gallery_matching_row_count_passes_through():
    # conftest.py's fake read_csv gives GALLERY_PATHS length 3
    gallery = np.zeros((len(rc.GALLERY_PATHS), 4))
    assert rc._validate_gallery(gallery, "test") is gallery


def test_validate_gallery_mismatched_row_count_returns_none():
    # Regression test for reviewLog.md v1 finding #10 / v2 fix: a gallery whose row count
    # doesn't match GALLERY_PATHS must be disabled, not silently trusted.
    mismatched = np.zeros((len(rc.GALLERY_PATHS) + 1, 4))
    assert rc._validate_gallery(mismatched, "test") is None


# ── _clean_tag ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("cotton fabric", "cotton"),
    ("A-line silhouette", "A-line"),
    ("black clothing", "black"),
])
def test_clean_tag_strips_stopwords(raw, expected):
    assert rc._clean_tag(raw) == expected


def test_clean_tag_returns_original_if_all_words_are_stopwords():
    assert rc._clean_tag("fabric") == "fabric"


# ── _image_to_base64 ──────────────────────────────────────────────────────────

def test_image_to_base64_valid_image_returns_nonempty_string(tmp_path):
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 400), color=(10, 20, 30)).save(img_path)
    encoded = rc._image_to_base64(str(img_path))
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_image_to_base64_missing_file_returns_empty_string():
    encoded = rc._image_to_base64("/nonexistent/path/does-not-exist.jpg")
    assert encoded == ""


# ── Model & Search Helper Tests with Mocks ────────────────────────────────────
from unittest.mock import MagicMock

def test_embed_text(monkeypatch):
    mock_input_ids = MagicMock()
    mock_input_ids.to.return_value = mock_input_ids
    mock_attention_mask = MagicMock()
    mock_attention_mask.to.return_value = mock_attention_mask
    
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "input_ids": mock_input_ids,
        "attention_mask": mock_attention_mask
    }
    monkeypatch.setattr(rc, "CLIP_PROCESSOR", mock_processor)
    
    mock_pooler_output = MagicMock()
    mock_text_model_out = MagicMock()
    mock_text_model_out.pooler_output = mock_pooler_output
    
    mock_text_model = MagicMock()
    mock_text_model.return_value = mock_text_model_out
    
    mock_numpy_arr = np.array([[1.0] * 512], dtype=np.float32)
    mock_proj_out = MagicMock()
    mock_proj_out.cpu.return_value.numpy.return_value = mock_numpy_arr
    
    mock_model = MagicMock()
    mock_model.text_model = mock_text_model
    mock_model.text_projection = MagicMock(return_value=mock_proj_out)
    monkeypatch.setattr(rc, "CLIP_MODEL", mock_model)
    
    emb = rc.embed_text("test query")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 512)
    assert np.allclose(np.linalg.norm(emb, axis=-1), 1.0)


def test_embed_image_clip(monkeypatch):
    mock_processor = MagicMock()
    mock_processor.return_value = MagicMock()
    monkeypatch.setattr(rc, "CLIP_PROCESSOR", mock_processor)
    
    mock_numpy_arr = np.array([[2.0] * 512], dtype=np.float32)
    mock_proj_out = MagicMock()
    mock_proj_out.cpu.return_value.numpy.return_value = mock_numpy_arr
    
    mock_model = MagicMock()
    mock_model.vision_model.return_value.pooler_output = MagicMock()
    mock_model.visual_projection.return_value = mock_proj_out
    monkeypatch.setattr(rc, "CLIP_MODEL", mock_model)
    
    img = Image.new("RGB", (224, 224))
    emb = rc.embed_image_clip(img)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 512)
    assert np.allclose(np.linalg.norm(emb, axis=-1), 1.0)


def test_embed_image_dinov2(monkeypatch):
    mock_model = MagicMock()
    mock_numpy_arr = np.array([[3.0] * 768], dtype=np.float32)
    mock_model.return_value.cpu.return_value.numpy.return_value = mock_numpy_arr
    monkeypatch.setattr(rc, "DINOV2_MODEL", mock_model)
    
    img = Image.new("RGB", (224, 224))
    emb = rc.embed_image_dinov2(img)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 768)
    assert np.allclose(np.linalg.norm(emb, axis=-1), 1.0)


def test_text_search(monkeypatch):
    monkeypatch.setattr(rc, "embed_text", lambda query: np.array([[1.0, 0.0]]))
    monkeypatch.setattr(rc, "CLIP_GALLERY", np.array([[1.0, 0.0], [0.0, 1.0]]))
    monkeypatch.setattr(rc, "CLIP_MODEL", object())
    
    results = rc.text_search("query", top_k=2)
    assert len(results) == 2
    assert results[0]["similarity"] == 100.0


def test_text_refine(monkeypatch):
    monkeypatch.setattr(rc, "embed_text", lambda query: np.array([[1.0, 0.0]]))
    monkeypatch.setattr(rc, "CLIP_GALLERY", np.array([[1.0, 0.0], [0.0, 1.0]]))
    monkeypatch.setattr(rc, "PATH_TO_GALLERY_IDX", {"/x/a.jpg": 0, "/x/b.jpg": 1})
    
    candidates = [
        {"similarity": 80.0, "image_path": "/x/a.jpg"},
        {"similarity": 90.0, "image_path": "/x/b.jpg"},
    ]
    refined = rc._text_refine(candidates, "query", visual_weight=0.5, text_weight=0.5)
    assert refined[0]["image_path"] == "/x/a.jpg"
    assert refined[0]["rank"] == 1


def test_tag_attributes(monkeypatch):
    monkeypatch.setattr(rc, "embed_image_clip", lambda img: np.array([[1.0, 0.0]]))
    def mock_embed_text(attr):
        if attr == rc.ATTRIBUTE_VOCABULARY[0]:
            return np.array([[1.0, 0.0]])
        return np.array([[0.0, 1.0]])
    monkeypatch.setattr(rc, "embed_text", mock_embed_text)
    
    img = Image.new("RGB", (224, 224))
    tags = rc._tag_attributes(img)
    expected_tag = rc._clean_tag(rc.ATTRIBUTE_VOCABULARY[0])
    assert expected_tag in tags


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
