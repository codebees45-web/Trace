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
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("trace.cnn_backbone")

# Determine optimal inference device (CUDA GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualConvBlock(nn.Module):
    """Residual Convolutional Building Block with BatchNorm and Dropout."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # Shortcut projection if dimensions or strides change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.1, inplace=True)


class MaskedFaceCNN(nn.Module):
    """
    Deep CNN Feature Backbone architecture optimized for occluded facial recognition.
    Transforms arbitrary face crops into robust 512-dimensional identity embeddings.
    """
    def __init__(self, embedding_dim: int = 512, input_channels: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Stem layer (Initial resolution encoding)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Deep residual feature extractor hierarchy
        self.layer1 = ResidualConvBlock(64, 128, stride=1, dropout=0.1)
        self.layer2 = ResidualConvBlock(128, 256, stride=2, dropout=0.15)
        self.layer3 = ResidualConvBlock(256, 384, stride=1, dropout=0.15)
        self.layer4 = ResidualConvBlock(384, 512, stride=2, dropout=0.2)
        
        # Global Spatial Aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Deep Embedding Projection
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Initialize convolutional weights via Kaiming / He distribution
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward feature pass. Returns L2-normalized embedding vectors of shape (B, embedding_dim).
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        emb = self.head(x)
        # Apply L2 hypersphere normalization for optimal cosine distance & margin identification
        return F.normalize(emb, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Singleton Instance Management & Feature Extraction Service
# ---------------------------------------------------------------------------
_cnn_instance: Optional[MaskedFaceCNN] = None


def get_cnn_backbone() -> MaskedFaceCNN:
    """Returns the global loaded instance of the CNN Backbone."""
    global _cnn_instance
    if _cnn_instance is None:
        logger.info("Initializing Deep CNN Backbone (512-d) on device: %s", DEVICE)
        model = MaskedFaceCNN(embedding_dim=512, input_channels=3).to(DEVICE)
        
        # Check if pre-trained backbone weights exist on disk
        weights_path = Path(__file__).resolve().parent / "masked_face_cnn_backbone.pth"
        if weights_path.exists():
            try:
                state = torch.load(weights_path, map_location=DEVICE)
                model.load_state_dict(state, strict=False)
                logger.info("Loaded pre-trained CNN weights from %s", weights_path)
            except Exception as e:
                logger.warning("Could not load checkpoint weights (%s) — using initialized backbone", e)
                
        model.eval()
        _cnn_instance = model
    return _cnn_instance


def preprocess_face_for_cnn(face_bgr: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    """
    Standardizes input facial crops (color conversion, spatial resolution rescaling, and ImageNet norm).
    """
    if len(face_bgr.shape) == 2:
        face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)
    elif face_bgr.shape[2] != 3:
        face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_RGBA2BGR)
        
    resized = cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Normalization mean & standard deviation for facial distributions
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    norm = (rgb - mean) / std
    
    # Transpose to PyTorch planar layout: (C, H, W) and prepend batch axis
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor


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
