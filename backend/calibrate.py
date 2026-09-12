# backend/calibrate.py
"""
Fits video_calib_midpoint / video_calib_slope (backend/config.py) to your
own gallery and video setup, instead of trusting the generic defaults in
video_pipeline.py.

Why this exists: `vote_identity` only ever sees raw ArcFace cosine
similarity, which is not a probability. Two setups with identical pipeline
code can have very different genuine-match similarity ranges depending on
mask type, camera quality, and how many/which reference photos are
enrolled. Calibrating against defaults tuned for a different setup means
the displayed "confidence" is meaningless for yours. This script measures
YOUR actual separation and picks the logistic-curve parameters that make
the displayed number reflect it.

HOW TO USE
----------
1. Enroll your identities as usual via /enroll.
2. Collect a handful of short video clips with known ground truth:
     - "genuine" clips: person X's video, checked against X's own gallery entry
     - "impostor" clips: person Y's video, checked against X's gallery entry
   (Reusing your existing challenge/demo videos is fine — you just need to
   know the correct identity for each clip.)
3. Fill in GENUINE_PAIRS / IMPOSTOR_PAIRS below with (video_path, identity)
   tuples.
4. Run:  python calibrate.py
5. Copy the printed video_calib_midpoint / video_calib_slope into your
   .env (or config.py defaults) and restart the backend.

This does NOT change matching decisions (min_similarity / min_margin still
gate who wins the vote) — it only changes what number gets displayed as
confidence, so it's safe to re-run any time without affecting accuracy.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import video_pipeline
from video_gallery import VideoGallery

# ---------------------------------------------------------------------------
# Fill these in with your own clips before running.
# Each tuple is (path_to_video, identity_name_as_enrolled_in_gallery).
# ---------------------------------------------------------------------------
GENUINE_PAIRS: list[tuple[str, str]] = [
    # ("clips/alice_walking.mp4", "alice"),
    # ("clips/bob_hallway.mp4", "bob"),
]

IMPOSTOR_PAIRS: list[tuple[str, str]] = [
    # ("clips/bob_hallway.mp4", "alice"),   # bob's video checked against alice's gallery entry
]

SAMPLE_EVERY_N_FRAMES = 6
MAX_FRAMES = 300  # cap per clip so calibration stays fast


def _extract_embeddings(video_path: str) -> list[np.ndarray]:
    """Grabs face embeddings from sampled frames — no tracking/voting needed
    for calibration, we just want a pool of raw embeddings from the clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    embeddings = []
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
            dets = video_pipeline.detect_and_embed(frame, frame_idx)
            embeddings.extend(d.embedding for d in dets)
            processed += 1
            if processed >= MAX_FRAMES:
                break
        frame_idx += 1
    cap.release()
    return embeddings


def _score_against_identity(embeddings: list[np.ndarray], gallery: VideoGallery, identity: str) -> float:
    """Mean, across frame embeddings, of the best cosine similarity to any
    of `identity`'s enrolled gallery embeddings. Mirrors the math
    vote_identity() uses for its raw_similarity, but scoped to one
    specific claimed identity instead of whoever wins the vote — this is
    what makes impostor scoring correct."""
    entry = gallery._entries.get(identity)  # noqa: SLF001 — read-only inspection for calibration
    if entry is None:
        raise KeyError(f"'{identity}' is not enrolled in the gallery.")
    if not embeddings:
        return 0.0

    gallery_vecs = [np.asarray(e, dtype=np.float32) for e in entry.embeddings]
    per_frame_best = []
    for emb in embeddings:
        q = emb / (np.linalg.norm(emb) + 1e-8)
        best = max(float(np.dot(q, g)) for g in gallery_vecs)
        per_frame_best.append(best)
    return float(np.mean(per_frame_best))


def collect_scores(pairs: list[tuple[str, str]], gallery: VideoGallery) -> list[float]:
    scores = []
    for video_path, identity in pairs:
        embeddings = _extract_embeddings(video_path)
        if not embeddings:
            print(f"  [skip] no faces detected in {video_path}", file=sys.stderr)
            continue
        scores.append(_score_against_identity(embeddings, gallery, identity))
    return scores


def fit_logistic(genuine: list[float], impostor: list[float]) -> tuple[float, float]:
    """Grid search over (midpoint, slope) maximizing separation between
    genuine and impostor calibrated scores. Good enough for a hackathon-
    scale calibration set; swap in sklearn.linear_model.LogisticRegression
    for a more principled fit if you have >50 samples per class."""
    if not genuine or not impostor:
        raise ValueError("Need at least one genuine and one impostor sample to calibrate.")

    best = (0.42, 10.0)
    best_score = -1.0
    for midpoint in np.arange(0.10, 0.80, 0.01):
        for slope in np.arange(4.0, 30.0, 1.0):
            g_conf = [video_pipeline.calibrate_confidence(s, midpoint, slope) for s in genuine]
            i_conf = [video_pipeline.calibrate_confidence(s, midpoint, slope) for s in impostor]
            # Reward high genuine confidence, low impostor confidence, and
            # penalize spread so the curve isn't just overfit to one sample.
            score = (np.mean(g_conf) - np.mean(i_conf)) - 0.5 * (np.std(g_conf) + np.std(i_conf))
            if score > best_score:
                best_score = score
                best = (float(midpoint), float(slope))
    return best


def main():
    if not GENUINE_PAIRS or not IMPOSTOR_PAIRS:
        print(
            "GENUINE_PAIRS and IMPOSTOR_PAIRS are both empty.\n"
            "Edit calibrate.py and fill them in with (video_path, identity) "
            "tuples for a few known-correct and known-incorrect video/identity "
            "combinations, then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    gallery = VideoGallery()  # loads video_gallery.json from the current directory
    print("Scoring genuine pairs...")
    genuine = collect_scores(GENUINE_PAIRS, gallery)
    print("Scoring impostor pairs...")
    impostor = collect_scores(IMPOSTOR_PAIRS, gallery)

    print(f"\nGenuine raw similarities  (n={len(genuine)}): {sorted(round(g, 3) for g in genuine)}")
    print(f"Impostor raw similarities (n={len(impostor)}): {sorted(round(i, 3) for i in impostor)}")

    if genuine and impostor and min(genuine) <= max(impostor):
        print(
            "\nWARNING: your genuine and impostor raw-similarity ranges overlap. "
            "No calibration curve can fix that on its own — it means the "
            "underlying embeddings aren't separating these identities well "
            "enough yet. Add more/better reference photos (including masked "
            "ones) before relying on the displayed confidence for anything "
            "high-stakes.\n"
        )

    midpoint, slope = fit_logistic(genuine, impostor)
    print("\nRecommended settings (put these in backend/.env or config.py defaults):")
    print(f"  video_calib_midpoint = {midpoint}")
    print(f"  video_calib_slope    = {slope}")

    g_conf = [video_pipeline.calibrate_confidence(s, midpoint, slope) for s in genuine]
    i_conf = [video_pipeline.calibrate_confidence(s, midpoint, slope) for s in impostor]
    print("\nWith these settings:")
    print(f"  genuine confidence  -> mean {np.mean(g_conf):.3f}, min {min(g_conf):.3f}")
    print(f"  impostor confidence -> mean {np.mean(i_conf):.3f}, max {max(i_conf):.3f}")


if __name__ == "__main__":
    main()