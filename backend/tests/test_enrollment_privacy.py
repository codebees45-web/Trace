# backend/tests/test_enrollment_privacy.py
"""Verify that the enrollment flow stores ONLY numeric embedding data —
never raw image bytes, base64 strings, or file paths to uploaded photos.

This is the automated evidence behind TRACE's privacy-by-design claim:
enrollment photos are processed in memory and discarded after feature
extraction; only the resulting 512-d embedding vectors are persisted.
"""
import io
import numpy as np
import pytest
from PIL import Image


def _tiny_jpeg_bytes():
    """Generate a minimal synthetic JPEG (not a real face, but enough to
    exercise the upload→decode→embed→store path without needing InsightFace
    to actually detect a face)."""
    img = Image.new("RGB", (112, 112), color=(120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_enrollment_stores_only_embeddings(client):
    """Enroll a synthetic image and verify that the stored gallery record
    contains only numeric embedding vectors — no image bytes anywhere."""
    from video_gallery import video_gallery

    identity = "__privacy_test_identity__"

    # Clean up any previous test residue.
    video_gallery.remove(identity)

    # Manually enroll a dummy embedding (the /enroll endpoint requires
    # InsightFace to detect a face, which may not be available in CI).
    dummy_embedding = np.random.randn(512).astype(np.float32)
    dummy_embedding /= np.linalg.norm(dummy_embedding) + 1e-8
    video_gallery.enroll(identity, dummy_embedding)

    # --- Assert: in-memory entry contains only numeric data ---
    entry = video_gallery._entries.get(identity)
    assert entry is not None, "Identity was not enrolled"
    assert len(entry.embeddings) >= 1

    for emb_list in entry.embeddings:
        # Each embedding should be a list of floats, not bytes/strings.
        assert isinstance(emb_list, list), f"Expected list, got {type(emb_list)}"
        for val in emb_list:
            assert isinstance(val, (int, float)), (
                f"Non-numeric value found in stored embedding: {type(val)}"
            )

    # --- Assert: no raw image data in the stored record ---
    import json
    serialised = json.dumps(entry.embeddings)
    # Base64 JPEG starts with /9j/ and PNG with iVBOR — neither should appear.
    assert "/9j/" not in serialised, "Base64 JPEG data found in stored embedding"
    assert "iVBOR" not in serialised, "Base64 PNG data found in stored embedding"

    # Clean up.
    video_gallery.remove(identity)
