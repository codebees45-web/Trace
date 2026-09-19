# backend/reid_store.py
"""Cross-camera re-identification sighting store.

Links the same person's appearances across different camera feeds into a
single ``global_person_id``, even before they are enrolled by name.

Storage pattern mirrors ``video_gallery.py``: MongoDB-first with a local
JSON file fallback.  Each sighting contains a 512-d ArcFace embedding
tagged with its source ``camera_id`` and ``track_id``.  When a new sighting
is ingested, it is compared (cosine similarity) against all existing
sightings from *different* cameras; above a configurable threshold they
are linked under the same ``global_person_id``.

Privacy note: only numeric embeddings are stored — no raw images.
"""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger("trace.reid_store")

_REID_JSON = os.path.join(os.path.dirname(__file__), "reid_sightings.json")


@dataclass
class Sighting:
    """A single appearance of a person at a specific camera."""
    sighting_id: str
    camera_id: str
    track_id: int
    embedding: list          # list[float], 512-d L2-normalised ArcFace vector
    first_seen: float        # epoch timestamp
    last_seen: float         # epoch timestamp
    global_person_id: Optional[str] = None
    identity_name: Optional[str] = None   # propagated from video_gallery match, if any


class ReIDStore:
    """In-memory + persisted store of cross-camera sightings.

    Persistence priority (same as VideoGallery):
      1. MongoDB ``reid_sightings`` collection (when available).
      2. Local JSON file ``reid_sightings.json`` (fallback).
    """

    def __init__(self):
        self._sightings: dict[str, Sighting] = {}  # keyed by sighting_id
        self._mongo_ok = False
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        try:
            from database import init_db
            db = init_db()
            docs = list(db.reid_sightings.find())
            if docs:
                for doc in docs:
                    sid = doc["sighting_id"]
                    self._sightings[sid] = Sighting(
                        sighting_id=sid,
                        camera_id=doc["camera_id"],
                        track_id=doc.get("track_id", 0),
                        embedding=doc["embedding"],
                        first_seen=doc.get("first_seen", 0),
                        last_seen=doc.get("last_seen", 0),
                        global_person_id=doc.get("global_person_id"),
                        identity_name=doc.get("identity_name"),
                    )
                self._mongo_ok = True
                logger.info("Loaded re-ID store from MongoDB: %d sightings", len(self._sightings))
                return
        except Exception as e:
            logger.warning("Could not load re-ID store from MongoDB: %s", e)
        self._load_json()

    def _load_json(self):
        if not os.path.exists(_REID_JSON):
            return
        try:
            with open(_REID_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, item in data.items():
                self._sightings[sid] = Sighting(
                    sighting_id=sid,
                    camera_id=item["camera_id"],
                    track_id=item.get("track_id", 0),
                    embedding=item["embedding"],
                    first_seen=item.get("first_seen", 0),
                    last_seen=item.get("last_seen", 0),
                    global_person_id=item.get("global_person_id"),
                    identity_name=item.get("identity_name"),
                )
            logger.info("Loaded re-ID store from local JSON: %d sightings", len(self._sightings))
        except Exception as e:
            logger.warning("Failed to load re-ID JSON: %s", e)

    def _save_json(self):
        try:
            data = {}
            for sid, s in self._sightings.items():
                data[sid] = {
                    "camera_id": s.camera_id,
                    "track_id": s.track_id,
                    "embedding": s.embedding,
                    "first_seen": s.first_seen,
                    "last_seen": s.last_seen,
                    "global_person_id": s.global_person_id,
                    "identity_name": s.identity_name,
                }
            with open(_REID_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Failed to save re-ID JSON: %s", e)

    def _save_sighting(self, sighting: Sighting):
        if self._mongo_ok:
            try:
                from database import init_db
                db = init_db()
                db.reid_sightings.update_one(
                    {"sighting_id": sighting.sighting_id},
                    {"$set": {
                        "sighting_id": sighting.sighting_id,
                        "camera_id": sighting.camera_id,
                        "track_id": sighting.track_id,
                        "embedding": sighting.embedding,
                        "first_seen": sighting.first_seen,
                        "last_seen": sighting.last_seen,
                        "global_person_id": sighting.global_person_id,
                        "identity_name": sighting.identity_name,
                    }},
                    upsert=True,
                )
            except Exception as e:
                logger.warning("Failed to save re-ID sighting to MongoDB: %s", e)
                self._mongo_ok = False
        self._save_json()

    def _save_sighting_batch(self, sightings: list[Sighting]):
        """Persist multiple sightings (e.g. after a global_person_id update)."""
        for s in sightings:
            self._save_sighting(s)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_sighting(
        self,
        camera_id: str,
        track_id: int,
        embedding: np.ndarray,
        similarity_threshold: float = 0.5,
        identity_name: Optional[str] = None,
    ) -> Sighting:
        """Persist a new sighting and attempt cross-camera matching.

        Args:
            camera_id: which camera/session produced this detection.
            track_id: within-video track id.
            embedding: 512-d ArcFace embedding (L2-normalised).
            similarity_threshold: cosine similarity above which two sightings
                from different cameras are considered the same person.
            identity_name: named identity if already matched via video_gallery.
        Returns:
            The created (and possibly linked) Sighting.
        """
        norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        now = time.time()

        sighting = Sighting(
            sighting_id=uuid.uuid4().hex,
            camera_id=camera_id,
            track_id=track_id,
            embedding=norm.tolist(),
            first_seen=now,
            last_seen=now,
            global_person_id=None,
            identity_name=identity_name,
        )

        # --- Cross-camera matching ---
        self._match_cross_camera(sighting, similarity_threshold)

        self._sightings[sighting.sighting_id] = sighting
        self._save_sighting(sighting)

        return sighting

    def _match_cross_camera(self, new_sighting: Sighting, threshold: float):
        """Search existing sightings from DIFFERENT cameras and link if similar."""
        if not self._sightings:
            return

        candidates = [s for s in self._sightings.values() if s.camera_id != new_sighting.camera_id]
        if not candidates:
            return

        q = np.asarray(new_sighting.embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        # Vectorized batch computation of cosine similarities
        candidate_matrix = np.array([s.embedding for s in candidates], dtype=np.float32)
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        candidate_matrix = candidate_matrix / (candidate_norms + 1e-8)

        similarities = np.dot(candidate_matrix, q)
        best_idx = np.argmax(similarities)
        best_sim = float(similarities[best_idx])
        best_match = candidates[best_idx]

        if best_sim < threshold:
            return

        logger.info(
            "Cross-camera match: cam=%s track=%d ↔ cam=%s track=%d (sim=%.4f)",
            new_sighting.camera_id, new_sighting.track_id,
            best_match.camera_id, best_match.track_id, best_sim,
        )

        # Link under the same global_person_id.
        if best_match.global_person_id:
            gpid = best_match.global_person_id
        else:
            gpid = uuid.uuid4().hex[:12]
            best_match.global_person_id = gpid
            self._save_sighting(best_match)

        new_sighting.global_person_id = gpid

        # Propagate identity name if either side has one.
        if new_sighting.identity_name and not best_match.identity_name:
            best_match.identity_name = new_sighting.identity_name
            self._save_sighting(best_match)
        elif best_match.identity_name and not new_sighting.identity_name:
            new_sighting.identity_name = best_match.identity_name

        # Propagate the global_person_id and name to ALL sightings that share it.
        self._propagate_global_person(gpid, new_sighting.identity_name)

    def _propagate_global_person(self, gpid: str, identity_name: Optional[str]):
        """Ensure all sightings with this gpid share the same identity name."""
        if not identity_name:
            return
        updated = []
        for s in self._sightings.values():
            if s.global_person_id == gpid and s.identity_name != identity_name:
                s.identity_name = identity_name
                updated.append(s)
        if updated:
            self._save_sighting_batch(updated)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_global_persons(self) -> list[dict]:
        """Returns a list of global persons with their linked sightings.

        Each entry: {global_person_id, identity_name, sightings: [{camera_id,
        track_id, first_seen, last_seen}], cameras: [str]}.
        """
        groups: dict[str, list[Sighting]] = {}
        unlinked = []

        for s in self._sightings.values():
            if s.global_person_id:
                groups.setdefault(s.global_person_id, []).append(s)
            else:
                unlinked.append(s)

        result = []
        for gpid, sightings in groups.items():
            cameras = sorted(set(s.camera_id for s in sightings))
            name = next((s.identity_name for s in sightings if s.identity_name), None)
            result.append({
                "global_person_id": gpid,
                "identity_name": name,
                "cameras": cameras,
                "sightings": [
                    {
                        "sighting_id": s.sighting_id,
                        "camera_id": s.camera_id,
                        "track_id": s.track_id,
                        "first_seen": s.first_seen,
                        "last_seen": s.last_seen,
                        "identity_name": s.identity_name,
                    }
                    for s in sorted(sightings, key=lambda x: x.first_seen)
                ],
            })

        # Sort: most sightings first (most confident links).
        result.sort(key=lambda r: len(r["sightings"]), reverse=True)
        return result

    def clear(self):
        """Remove all sightings (useful for tests)."""
        self._sightings.clear()
        self._save_json()
        try:
            from database import init_db
            db = init_db()
            db.reid_sightings.delete_many({})
        except Exception:
            pass


# Module-level singleton (same pattern as video_gallery.py).
reid_store = ReIDStore()
