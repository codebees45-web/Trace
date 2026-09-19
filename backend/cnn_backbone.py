# backend/cnn_backbone.py
"""
Deep Convolutional Neural Network (CNN) Backbone for Masked Face Recognition

This module provides the core deep learning architecture and feature extraction backbone
for the TRACE facial recognition ecosystem, replacing legacy HOG/LBP engineering across:
1. Live REST API inference endpoints (/predict, /predict/batch, /predict/multi-face)
2. Standalone & batch training scripts (train_model_v4.py and train_model_cnn.py)
3. Jupyter Data Science Notebook experiments
4. Real-time surveillance video feature identification

Architecture:
    MaskedFaceCNN: A 5-block Convolutional Neural Network with Residual short-circuit paths,
    Batch Normalization, LeakyReLU activations, Adaptive Avg Pooling, and a Dense embedding
    projection layer that compresses occluded surveillance face images into L2-normalized
    512-dimensional feature vectors.
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("trace.cnn_backbone")

# ---------------------------------------------------------------------------
# State-of-the-Art Deep Feature Extraction via ArcFace ResNet-50 (512-d)
# ---------------------------------------------------------------------------
_rec_engine = None


def get_recognition_engine():
    """Lazy-loads the InsightFace ArcFace ResNet-50 (w600k_r50) recognition engine."""
    global _rec_engine
    if _rec_engine is None:
        logger.info("Loading InsightFace ArcFace ResNet-50 512-d recognition engine...")
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _rec_engine = app.models["recognition"]
            logger.info("ArcFace ResNet-50 512-d engine ready.")
        except Exception as e:
            logger.error("Could not initialize InsightFace recognition module: %s", e)
            raise
    return _rec_engine


def _preprocess_image_bgr112(img: np.ndarray) -> np.ndarray:
    """Ensure image is 112x112 BGR uint8 array as required by ArcFace w600k_r50."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def extract_cnn_features(face_input: np.ndarray) -> np.ndarray:
    """
    Primary interface for external modules: transforms a BGR or Grayscale facial matrix
    into a robust 512-dimensional NumPy embedding vector via the Deep CNN / ArcFace Backbone.
    """
    engine = get_recognition_engine()
    proc = _preprocess_image_bgr112(face_input)
    emb = engine.get_feat(proc).flatten()
    return emb


def make_occlusion_weight_mask(face_bgr: np.ndarray,
                               alpha_occluded: float = 0.15) -> np.ndarray:
    """Build a (H, 1) float32 soft weighting mask for an already-cropped face.

    Upper region (forehead → nose bridge, top ~42% of crop height): weight 1.0.
    Lower region (surgical-mask area, bottom ~58%): weight ``alpha_occluded``.
    A 10%-height linear taper connects the two zones to avoid a hard edge.

    The 0.42 split mirrors the ``y + h*0.42`` boundary used by
    ``build_lower_face_mask()`` in main.py for the SD inpainting mask, so the
    "visible" / "occluded" split is consistent across the whole pipeline.

    Returned shape is (H, 1) so it broadcasts directly against (H, W, C)
    arrays: ``face_bgr * mask`` dims occluded pixels toward zero before
    ArcFace sees them, biasing spatial attention toward the visible upper-face
    region without touching the pretrained network internals (item 3 of the
    occlusion-aware embedding plan).

    Args:
        face_bgr: BGR face crop — any size; shape used only to read H.
        alpha_occluded: pixel multiplier for the occluded lower region.
            0.0 blacks out the mask area completely; 0.15 (default) preserves
            enough texture to avoid hard-edge artefacts at the boundary while
            still strongly down-weighting mask pixels.
    Returns:
        np.ndarray of shape (H, 1) and dtype float32.
    """
    h = face_bgr.shape[0]
    mask = np.ones(h, dtype=np.float32)

    split_y  = int(h * 0.42)            # nose-bridge boundary — same as build_lower_face_mask
    taper_px = max(1, int(h * 0.10))    # 10% soft transition band
    taper_end = min(h, split_y + taper_px)

    for row in range(split_y, taper_end):
        t = (row - split_y) / taper_px  # 0.0 at split → 1.0 at taper_end
        mask[row] = 1.0 - t * (1.0 - alpha_occluded)

    mask[taper_end:] = alpha_occluded

    return mask[:, np.newaxis]  # (H, 1) — broadcast-ready


def extract_batch_cnn_features(face_list: list[np.ndarray]) -> np.ndarray:
    """
    Batch extraction optimization: processes multiple face images simultaneously.
    Returns array of shape (N, 512).
    """
    if not face_list:
        return np.empty((0, 512), dtype=np.float32)
    engine = get_recognition_engine()
    proc_list = [_preprocess_image_bgr112(img) for img in face_list]
    embs = engine.get_feat(proc_list)
    return embs


def extract_cnn_features_occlusion_aware(face_bgr: np.ndarray,
                                          alpha_occluded: float = 0.15) -> np.ndarray:
    """Occlusion-aware variant of ``extract_cnn_features()``.

    Applies ``make_occlusion_weight_mask()`` to the BGR input crop before
    passing it to ArcFace.  The occluded lower-face region (surgical-mask
    area, below ~42% of face height) is dimmed to ``alpha_occluded`` of its
    original pixel values, biasing ArcFace's spatial attention toward the
    visible identity cues in the upper face.

    The original ``extract_cnn_features()`` is **not modified** and remains
    independently callable — this function is an additive option for A/B
    accuracy comparison (item 4 of the occlusion-aware embedding plan).

    Args:
        face_bgr: BGR face crop (any size; resized to 112×112 internally by
            ArcFace's preprocessing, same as the standard path).
        alpha_occluded: pixel multiplier for the occluded lower-face region.
            Forwarded directly to ``make_occlusion_weight_mask()``.
            Default 0.15 matches the config default ``occlusion_mask_alpha``.
    Returns:
        512-d L2-normalized embedding, identical shape to ``extract_cnn_features()``.
    """
    weight_mask = make_occlusion_weight_mask(face_bgr, alpha_occluded=alpha_occluded)
    masked = np.clip(face_bgr.astype(np.float32) * weight_mask, 0, 255).astype(np.uint8)
    return extract_cnn_features(masked)


def compute_occlusion_heatmap(face_bgr: np.ndarray, reference_embedding: np.ndarray,
                              patch_size: int = 16, stride: int = 8) -> np.ndarray:
    """Computes an occlusion sensitivity heatmap to explain match decisions.

    Slides a gray patch across the face, re-embeds the occluded face, and measures
    the drop in cosine similarity against the reference_embedding. A large drop
    means that region was highly important for the match.

    Args:
        face_bgr: BGR face crop.
        reference_embedding: 512-d ArcFace embedding to compare against.
        patch_size: Size of the occluding square patch in pixels (relative to 112x112).
        stride: Step size for the sliding patch.

    Returns:
        A 2D float32 numpy array of importance scores, normalized to [0, 1].
    """
    # 1. Resize to 112x112 (ArcFace native size) to cap latency and standardize grid
    face_112 = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    
    # 2. Compute baseline embedding and similarity
    baseline_emb = extract_cnn_features(face_112)
    
    def cosine_sim(e1: np.ndarray, e2: np.ndarray) -> float:
        n1 = e1 / (np.linalg.norm(e1) + 1e-8)
        n2 = e2 / (np.linalg.norm(e2) + 1e-8)
        return float(np.dot(n1, n2))
        
    baseline_sim = cosine_sim(baseline_emb, reference_embedding)
    
    h, w = face_112.shape[:2]
    grid_h = (h - patch_size) // stride + 1
    grid_w = (w - patch_size) // stride + 1
    
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
    gray_patch = np.full((patch_size, patch_size, 3), 127, dtype=np.uint8)
    
    # Prepare batch of occluded images to optimize inference time
    patches_list = []
    coords = []
    
    for i, y in enumerate(range(0, h - patch_size + 1, stride)):
        for j, x in enumerate(range(0, w - patch_size + 1, stride)):
            occluded = face_112.copy()
            occluded[y:y+patch_size, x:x+patch_size] = gray_patch
            patches_list.append(occluded)
            coords.append((i, j))
            
    if patches_list:
        # 3. Batch extract embeddings for all occluded versions
        embs = extract_batch_cnn_features(patches_list)
        
        # 4. Compute importance (similarity drop)
        for idx, emb in enumerate(embs):
            sim = cosine_sim(emb, reference_embedding)
            importance = max(0.0, baseline_sim - sim)
            i, j = coords[idx]
            heatmap[i, j] = importance
            
    # 5. Normalize 0-1
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap

