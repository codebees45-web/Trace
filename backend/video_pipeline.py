# backend/video_pipeline.py
"""
Masked-face video recognition pipeline.

Stack (per the project's architecture decision):
    detection   -> SCRFD (bundled in InsightFace's buffalo_l model pack)
    embedding   -> ArcFace, 512-d, L2-normalized
    tracking    -> lightweight IOU tracker (greedy, no external dependency)
    matching    -> cosine similarity against VideoGallery
    decision    -> multi-frame confidence-weighted voting per track

Why InsightFace: it ships SCRFD + ArcFace as a single package with ONNX
Runtime inference, runs on CPU (no GPU required for a hackathon laptop),
and both models are already trained to be robust to partial occlusion,
which matters directly for masked faces. Swapping in ByteTrack/DeepSORT
later only means replacing IOUTracker — everything downstream (embedding,
matching, voting) is agnostic to which tracker produced the track_id.

--------------------------------------------------------------------------
CHANGE (calibrated confidence): `vote_identity` used to report raw average
cosine similarity as "confidence". That number is not a probability — it's
just ArcFace's cosine similarity, which for masked-vs-unmasked face pairs
realistically tops out well under 1.0 even for a correct match. Reporting
it directly as "confidence" makes a solid match look weak (e.g. 60%) and
makes ">95% confidence" essentially unreachable no matter how good the
match actually is.

This version keeps the raw similarity around (as `raw_similarity`, for
debugging/tuning) and additionally reports a *calibrated* `confidence`:
raw similarity passed through a logistic curve fit to where your genuine
vs. impostor scores actually separate. See calibrate.py to fit
`video_calib_midpoint` / `video_calib_slope` to your own gallery + camera
setup instead of trusting the defaults below.
--------------------------------------------------------------------------
"""
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("trace.video_pipeline")

# ---------------------------------------------------------------------------
# Lazy-loaded InsightFace app — loading the ONNX models takes a couple of
# seconds and pins memory, so we do it once at first use, not at import time.
# Mirrors main.py's pattern for the Stable Diffusion pipeline.
# ---------------------------------------------------------------------------
_face_app = None
_face_app_det_size = None


def _get_face_app(det_size: int = 640, providers: Optional[list] = None):
    global _face_app, _face_app_det_size
    if _face_app is not None:
        if det_size != _face_app_det_size:
            # Cheap — just reconfigures the detector's input size, doesn't reload weights.
            _face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
            _face_app_det_size = det_size
        return _face_app

    from insightface.app import FaceAnalysis

    providers = providers or ["CPUExecutionProvider"]
    logger.info("Loading InsightFace buffalo_l (SCRFD + ArcFace)...")
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if providers[0] != "CPUExecutionProvider" else -1, det_size=(det_size, det_size))
    _face_app = app
    _face_app_det_size = det_size
    logger.info("InsightFace ready.")
    return _face_app


def is_available() -> bool:
    """Cheap check so main.py can return a clean 503 instead of an ImportError
    traceback if insightface/onnxruntime weren't installed in this environment."""
    try:
        import insightface  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Detection + embedding for a single frame
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: tuple          # (x1, y1, x2, y2) in pixel coords
    embedding: np.ndarray  # 512-d, L2-normalized
    det_score: float
    frame_idx: int
    kps: Optional[np.ndarray] = None  # 5-point landmarks (eyes, nose, mouth corners), if available


def detect_and_embed(frame_bgr, frame_idx: int, det_score_thresh: float = 0.45, det_size: int = 640) -> list[Detection]:
    app = _get_face_app(det_size=det_size)
    faces = app.get(frame_bgr)
    out = []
    for f in faces:
        if f.det_score < det_score_thresh:
            continue
        emb = f.normed_embedding  # InsightFace already L2-normalizes this
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        kps = getattr(f, "kps", None)
        out.append(Detection(bbox=(x1, y1, x2, y2), embedding=emb, det_score=float(f.det_score),
                              frame_idx=frame_idx, kps=kps))
    return out


# ---------------------------------------------------------------------------
# Tracking — greedy IOU matching frame-to-frame. Deliberately simple: no
# motion model, no Kalman filter. For short hackathon clips at a few fps
# sampling rate this is sufficient, and it has zero extra dependencies.
# ByteTrack/DeepSORT can replace this class later without touching the rest
# of the pipeline (same constructor shape: update(detections) -> tracks).
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-8)


@dataclass
class Track:
    track_id: int
    bbox: tuple
    last_seen_frame: int
    embeddings: list = field(default_factory=list)   # every Detection.embedding assigned to this track
    det_scores: list = field(default_factory=list)


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # frames a track can go undetected before being dropped
        self._tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy match: for every existing track, find the best-IOU unmatched detection.
        for track_id, track in list(self._tracks.items()):
            best_iou, best_det_idx = 0.0, None
            for i in unmatched_dets:
                iou = _iou(track.bbox, detections[i].bbox)
                if iou > best_iou:
                    best_iou, best_det_idx = iou, i
            if best_det_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_det_idx]
                track.bbox = det.bbox
                track.last_seen_frame = frame_idx
                track.embeddings.append(det.embedding)
                track.det_scores.append(det.det_score)
                unmatched_dets.remove(best_det_idx)
                matched_track_ids.add(track_id)

        # Anything left unmatched starts a new track.
        for i in unmatched_dets:
            det = detections[i]
            t = Track(track_id=self._next_id, bbox=det.bbox, last_seen_frame=frame_idx)
            t.embeddings.append(det.embedding)
            t.det_scores.append(det.det_score)
            self._tracks[self._next_id] = t
            self._next_id += 1

        # Drop stale tracks.
        for track_id in list(self._tracks.keys()):
            if frame_idx - self._tracks[track_id].last_seen_frame > self.max_age:
                del self._tracks[track_id]

        return list(self._tracks.values())

    def all_tracks_ever(self) -> dict:
        return self._tracks


# ---------------------------------------------------------------------------
# Confidence calibration — turns a raw ArcFace cosine similarity into a
# probability-like number by passing it through a logistic curve. This is
# the standard trick for any embedding-distance system that needs to show
# users a percentage: you don't chase the raw similarity number itself
# (it's bounded by how discriminative the embedding space is, and masked
# faces genuinely reduce that), you calibrate the *display* against your
# own genuine-vs-impostor score distribution.
#
# `midpoint`  — the raw similarity value that should map to 50% confidence.
#               Set this to roughly where your genuine and impostor score
#               distributions cross (see calibrate.py).
# `slope`     — how sharply confidence rises around the midpoint. Higher =
#               more decisive (near-binary); lower = more gradual.
#
# Defaults below are reasonable *starting points* for masked-face ArcFace
# matching (genuine masked-vs-unmasked similarities often cluster ~0.45–0.65,
# impostor pairs ~0.15–0.35) but you should re-fit them with calibrate.py
# using your own enrolled identities and sample videos before trusting the
# displayed percentage for anything high-stakes.
# ---------------------------------------------------------------------------

def calibrate_confidence(raw_similarity: float, midpoint: float = 0.42, slope: float = 10.0) -> float:
    x = (raw_similarity - midpoint) * slope
    # Guard against overflow for extreme inputs.
    if x > 40:
        return 1.0
    if x < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Multi-frame voting — turns a track's list of per-frame embeddings into one
# final identity decision. Each frame casts a weighted vote (weight = cosine
# similarity to its best gallery match); a person only wins if their summed
# vote clears both an absolute floor and a margin over the runner-up, which
# is what keeps a few lucky high-similarity frames from a wrong match dominating.
# ---------------------------------------------------------------------------

def vote_identity(embeddings: list[np.ndarray], gallery, min_similarity: float = 0.35,
                   min_margin: float = 0.08, calib_midpoint: float = 0.42,
                   calib_slope: float = 10.0) -> dict:
    from collections import defaultdict

    if not embeddings:
        return {"identity": "Unknown", "confidence": 0.0, "raw_similarity": 0.0,
                "votes": 0, "frames_considered": 0}

    vote_weight = defaultdict(float)
    vote_count = defaultdict(int)
    per_frame_best = []

    for emb in embeddings:
        matches = gallery.match(emb, top_k=1)
        if not matches:
            continue
        top = matches[0]
        per_frame_best.append(top["similarity"])
        if top["similarity"] >= min_similarity:
            vote_weight[top["identity"]] += top["similarity"]
            vote_count[top["identity"]] += 1

    if not vote_weight:
        avg_sim = float(np.mean(per_frame_best)) if per_frame_best else 0.0
        avg_sim = max(avg_sim, 0.0)
        return {"identity": "Unknown",
                "confidence": round(calibrate_confidence(avg_sim, calib_midpoint, calib_slope), 4),
                "raw_similarity": round(avg_sim, 4),
                "votes": 0, "frames_considered": len(embeddings)}

    ranked = sorted(vote_weight.items(), key=lambda kv: kv[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    raw_similarity = winner_score / max(vote_count[winner], 1)  # average similarity, not raw sum
    margin = (winner_score - runner_up_score) / max(winner_score, 1e-8)
    calibrated = calibrate_confidence(raw_similarity, calib_midpoint, calib_slope)

    if margin < min_margin and len(ranked) > 1:
        # Too close to call — safer to report Unknown than guess between two
        # candidates, especially in a challenge that penalizes false positives.
        return {"identity": "Unknown", "confidence": round(calibrated, 4),
                "raw_similarity": round(raw_similarity, 4),
                "votes": vote_count[winner], "frames_considered": len(embeddings)}

    return {"identity": winner, "confidence": round(calibrated, 4),
            "raw_similarity": round(raw_similarity, 4),
            "votes": vote_count[winner], "frames_considered": len(embeddings)}


# ---------------------------------------------------------------------------
# Orchestration — video in, per-person decision out.
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    gallery,
    sample_every_n_frames: int = 3,
    max_frames: Optional[int] = 900,
    det_score_thresh: float = 0.45,
    iou_threshold: float = 0.3,
    track_max_age: int = 15,
    min_similarity: float = 0.35,
    min_margin: float = 0.08,
    det_size: int = 640,
    calib_midpoint: float = 0.42,
    calib_slope: float = 10.0,
) -> dict:
    """Runs detection+embedding on sampled frames, tracks faces across them,
    and returns one voted identity decision per track.

    sample_every_n_frames > 1 trades a little temporal resolution for a lot
    of speed — running SCRFD+ArcFace on every single frame of a hackathon
    demo video is rarely necessary and eats the judging-window clock.
    """
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracker = IOUTracker(iou_threshold=iou_threshold, max_age=track_max_age)
    frame_idx = 0
    processed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every_n_frames == 0:
            dets = detect_and_embed(frame, frame_idx, det_score_thresh=det_score_thresh, det_size=det_size)
            tracker.update(dets, frame_idx)
            processed_frames += 1
            if max_frames and processed_frames >= max_frames:
                break
        frame_idx += 1

    cap.release()

    results = []
    for track in tracker.all_tracks_ever().values():
        decision = vote_identity(
            track.embeddings, gallery, min_similarity=min_similarity, min_margin=min_margin,
            calib_midpoint=calib_midpoint, calib_slope=calib_slope,
        )
        results.append({
            "track_id": track.track_id,
            "identity": decision["identity"],
            "confidence": decision["confidence"],
            "raw_similarity": decision["raw_similarity"],
            "votes": decision["votes"],
            "frames_considered": decision["frames_considered"],
            "last_bbox": {"x1": track.bbox[0], "y1": track.bbox[1], "x2": track.bbox[2], "y2": track.bbox[3]},
            "avg_det_score": round(float(np.mean(track.det_scores)), 4) if track.det_scores else 0.0,
        })

    # Highest-confidence identified person first; unresolved tracks last.
    results.sort(key=lambda r: (r["identity"] != "Unknown", r["confidence"]), reverse=True)

    elapsed = time.time() - t0
    logger.info(
        "process_video: %d tracks, %d frames sampled of %d total, %.1fs",
        len(results), processed_frames, total_frames, elapsed,
    )

    return {
        "tracks": results,
        "num_tracks": len(results),
        "frames_processed": processed_frames,
        "total_frames": total_frames,
        "video_fps": round(fps, 2),
        "processing_seconds": round(elapsed, 2),
    }