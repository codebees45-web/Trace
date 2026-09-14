"""Tests for compute_occlusion_heatmap — occlusion sensitivity explainability."""
import os
import sys

# Ensure backend/ is importable when running from backend/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from cnn_backbone import compute_occlusion_heatmap, extract_cnn_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_face(h: int = 112, w: int = 112) -> np.ndarray:
    """Create a synthetic BGR face-like image with some spatial variance."""
    rng = np.random.RandomState(42)
    img = rng.randint(60, 200, (h, w, 3), dtype=np.uint8)
    # Draw a brighter "eye region" in the upper third so there's a
    # distinguishing area for the heatmap to potentially highlight.
    img[20:40, 25:45] = [255, 220, 200]
    img[20:40, 65:85] = [255, 220, 200]
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_heatmap_shape_matches_grid():
    """Output shape must equal the expected (grid_h, grid_w) for the given
    patch_size and stride on a 112x112 input."""
    face = _make_synthetic_face()
    ref_emb = extract_cnn_features(face)

    patch_size, stride = 16, 8
    heatmap = compute_occlusion_heatmap(face, ref_emb,
                                         patch_size=patch_size, stride=stride)

    expected_h = (112 - patch_size) // stride + 1  # 13
    expected_w = (112 - patch_size) // stride + 1  # 13
    assert heatmap.shape == (expected_h, expected_w), (
        f"Expected ({expected_h}, {expected_w}), got {heatmap.shape}"
    )


def test_heatmap_values_in_unit_range():
    """All heatmap values must lie in [0, 1]."""
    face = _make_synthetic_face()
    ref_emb = extract_cnn_features(face)
    heatmap = compute_occlusion_heatmap(face, ref_emb)

    assert heatmap.min() >= 0.0, f"min value {heatmap.min()} < 0"
    assert heatmap.max() <= 1.0, f"max value {heatmap.max()} > 1"


def test_heatmap_near_uniform_face():
    """A perfectly uniform face (all same colour) should still produce a
    valid heatmap without crashing — edge case where every patch has near-
    identical importance and max might be near zero."""
    uniform = np.full((112, 112, 3), 128, dtype=np.uint8)
    ref_emb = extract_cnn_features(uniform)
    heatmap = compute_occlusion_heatmap(uniform, ref_emb)

    # Shape check
    assert heatmap.ndim == 2
    # Value check — should be all zeros or all normalized to [0,1]
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_heatmap_with_different_patch_sizes():
    """Verify the heatmap adapts grid size correctly for different configs."""
    face = _make_synthetic_face()
    ref_emb = extract_cnn_features(face)

    # Larger patch, same stride → smaller grid
    heatmap_big = compute_occlusion_heatmap(face, ref_emb,
                                             patch_size=28, stride=14)
    expected_h = (112 - 28) // 14 + 1  # 7
    expected_w = (112 - 28) // 14 + 1  # 7
    assert heatmap_big.shape == (expected_h, expected_w)
    assert heatmap_big.min() >= 0.0
    assert heatmap_big.max() <= 1.0
