# backend/enrollment.py
"""
Gallery-based enrollment and matching.

The RWMFD-trained classifier only knows its 23 training identities. For an
evaluation where organizers supply photos of *new* people + a masked video
of one of them, we can't retrain in time and shouldn't try. Instead we
reuse the same feature extractor as an embedding function and do
nearest-neighbor matching against a small enrolled gallery. No retraining
needed to add a person — just enroll their photos.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from database import init_db

logger = logging.getLogger("trace.enrollment")


@dataclass
class GalleryEntry:
    identity: str
    embeddings: list  # list of list[float] — one per enrolled photo/variant, PCA-space
    created_at: float = field(default_factory=time.time)


class Gallery:
    """In-memory + MongoDB-persisted store of enrolled identity embeddings."""

    def __init__(self, path: Optional[str] = None):
        self._entries: dict[str, GalleryEntry] = {}
        self._load()

    def _load(self):
        try:
            db = init_db()
            for doc in db.gallery_embeddings.find():
                identity = doc["identity"]
                self._entries[identity] = GalleryEntry(
                    identity=identity,
                    embeddings=doc.get("embeddings", []),
                    created_at=doc.get("created_at", time.time()),
                )
            logger.info("Loaded gallery from MongoDB: %d identities", len(self._entries))
        except Exception as e:
            logger.warning("Could not load gallery embeddings from MongoDB: %s", e)

    def _save_entry(self, entry: GalleryEntry):
        try:
            db = init_db()
            db.gallery_embeddings.update_one(
                {"identity": entry.identity},
                {"$set": {"identity": entry.identity, "embeddings": entry.embeddings, "created_at": entry.created_at}},
                upsert=True,
            )
        except Exception as e:
            logger.warning("Failed to save gallery embedding to MongoDB: %s", e)

    def enroll(self, identity: str, embedding: np.ndarray):
        entry = self._entries.get(identity)
        if entry is None:
            entry = GalleryEntry(identity=identity, embeddings=[])
            self._entries[identity] = entry
        entry.embeddings.append(embedding.tolist())
        self._save_entry(entry)

    def remove(self, identity: str) -> bool:
        if identity in self._entries:
            del self._entries[identity]
            try:
                db = init_db()
                db.gallery_embeddings.delete_one({"identity": identity})
            except Exception as e:
                logger.warning("Failed to remove gallery embedding from MongoDB: %s", e)
            return True
        return False

    def identities(self) -> list[str]:
        return list(self._entries.keys())

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def match(self, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Cosine similarity against every enrolled embedding; a person's
        score is their single best-matching embedding (max, not average —
        averaging would penalize someone enrolled with dissimilar angles)."""
        if self.is_empty():
            return []
        scores = []
        q = embedding / (np.linalg.norm(embedding) + 1e-8)
        for identity, entry in self._entries.items():
            best = -1.0
            for emb in entry.embeddings:
                e = np.asarray(emb)
                e = e / (np.linalg.norm(e) + 1e-8)
                sim = float(np.dot(q, e))
                best = max(best, sim)
            scores.append({"identity": identity, "similarity": best})
        scores.sort(key=lambda s: s["similarity"], reverse=True)
        return scores[:top_k]


def augment_face(face_bgr) -> list:
    """Generates a handful of perturbed copies of a single enrollment photo
    so 2 photos become ~14-16 embeddings instead of 2 — more robust than
    trusting 2 raw points, without needing more real photos."""
    variants = [face_bgr]

    flipped = cv2.flip(face_bgr, 1)
    variants.append(flipped)

    h, w = face_bgr.shape[:2]
    center = (w // 2, h // 2)
    for base in (face_bgr, flipped):
        for angle in (-8, 8):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            variants.append(cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT))
        for alpha, beta in ((1.15, 10), (0.85, -10)):  # brightness/contrast jitter
            variants.append(cv2.convertScaleAbs(base, alpha=alpha, beta=beta))

    return variants


gallery = Gallery()