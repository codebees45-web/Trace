# backend/video_gallery.py
"""
Gallery store for deep-embedding (ArcFace, 512-d) identity vectors.

This is deliberately separate from enrollment.py's Gallery, which stores
HOG/LBP+PCA embeddings for the legacy single-image /predict pipeline.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from database import init_db

logger = logging.getLogger("trace.video_gallery")

# Local JSON fallback path (sits next to this file in the backend directory).
_GALLERY_JSON = os.path.join(os.path.dirname(__file__), "video_gallery_local.json")


@dataclass
class VideoGalleryEntry:
    identity: str
    embeddings: list  # list of list[float], each L2-normalized 512-d ArcFace vector
    created_at: float = field(default_factory=time.time)


class VideoGallery:
    """In-memory + persisted store of enrolled ArcFace identity embeddings.

    Privacy by design: only L2-normalized 512-d numeric embedding vectors are
    persisted.  Raw enrollment photos uploaded via ``/enroll`` are decoded and
    embedded entirely in memory by the endpoint handler and are never written to
    disk, logged, or stored in any database collection.  The embeddings are not
    reversible — there is no way to reconstruct the original face image from the
    stored vector.

    Persistence priority:
      1. MongoDB (``video_gallery_embeddings`` collection, when available) --
         shared across processes, survives restarts.
      2. Local JSON file (``video_gallery_local.json``) -- fallback when MongoDB
         is not running. Survives server restarts; lost only if the file is
         deleted.
    """

    def __init__(self, path: Optional[str] = None):
        self._entries: dict[str, VideoGalleryEntry] = {}
        self._mongo_ok = False
        self._load()

    def _load(self):
        try:
            db = init_db()
            docs = list(db.video_gallery_embeddings.find())
            if docs:
                for doc in docs:
                    identity = doc["identity"]
                    self._entries[identity] = VideoGalleryEntry(
                        identity=identity,
                        embeddings=doc.get("embeddings", []),
                        created_at=doc.get("created_at", time.time()),
                    )
                self._mongo_ok = True
                logger.info("Loaded video gallery from MongoDB: %d identities", len(self._entries))
                return
        except Exception as e:
            logger.warning("Could not load video gallery from MongoDB: %s", e)
        self._load_json()

    def _load_json(self):
        if not os.path.exists(_GALLERY_JSON):
            return
        try:
            with open(_GALLERY_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            for identity, item in data.items():
                self._entries[identity] = VideoGalleryEntry(
                    identity=identity,
                    embeddings=item.get("embeddings", []),
                    created_at=item.get("created_at", time.time()),
                )
            logger.info("Loaded video gallery from local JSON: %d identities", len(self._entries))
        except Exception as e:
            logger.warning("Failed to load local JSON gallery: %s", e)

    def _save_json(self):
        try:
            data = {
                identity: {"embeddings": entry.embeddings, "created_at": entry.created_at}
                for identity, entry in self._entries.items()
            }
            with open(_GALLERY_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Failed to save local JSON gallery: %s", e)

    def _save_entry(self, entry: VideoGalleryEntry):
        try:
            db = init_db()
            db.video_gallery_embeddings.update_one(
                {"identity": entry.identity},
                {"$set": {"identity": entry.identity, "embeddings": entry.embeddings,
                          "created_at": entry.created_at}},
                upsert=True,
            )
            self._mongo_ok = True
        except Exception as e:
            logger.warning("Failed to save video gallery embedding to MongoDB: %s", e)
            self._mongo_ok = False
        self._save_json()

    def enroll(self, identity: str, embedding: np.ndarray):
        norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        entry = self._entries.get(identity)
        if entry is None:
            entry = VideoGalleryEntry(identity=identity, embeddings=[])
            self._entries[identity] = entry
        entry.embeddings.append(norm.tolist())
        self._save_entry(entry)

    def remove(self, identity: str) -> bool:
        if identity in self._entries:
            del self._entries[identity]
            try:
                db = init_db()
                db.video_gallery_embeddings.delete_one({"identity": identity})
            except Exception as e:
                logger.warning("Failed to remove video gallery embedding from MongoDB: %s", e)
            self._save_json()
            return True
        return False

    def identities(self) -> list[str]:
        return list(self._entries.keys())

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def match(self, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        if self.is_empty():
            return []
        q = embedding / (np.linalg.norm(embedding) + 1e-8)
        scores = []
        for identity, entry in self._entries.items():
            best = -1.0
            for emb in entry.embeddings:
                e = np.asarray(emb, dtype=np.float32)
                sim = float(np.dot(q, e))
                best = max(best, sim)
            scores.append({"identity": identity, "similarity": best})
        scores.sort(key=lambda s: s["similarity"], reverse=True)
        return scores[:top_k]


video_gallery = VideoGallery()
