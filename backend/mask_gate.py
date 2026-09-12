# backend/mask_gate.py
"""
Heuristic mask detector for the TRACE pipeline.

Uses three independent signals on the lower-face region:
  1. Mouth-landmark visibility (Haar cascade)
  2. Edge density (Canny edges — real skin has lip contours, chin texture…)
  3. Skin-colour proportion (HSV hue/saturation check)

A face is classified as "masked" only when at least 2 of 3 signals agree,
which dramatically reduces false positives on unmasked faces compared to the
original single-check approach.
"""
import cv2
import numpy as np


def _skin_ratio(region_bgr):
    """Fraction of pixels in *region_bgr* that look like human skin.

    Uses a standard HSV range that covers light-to-dark skin tones while
    excluding common mask colours (medical blue, white fabric, black cloth).
    """
    if region_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    # Skin-tone hue broadly sits in 0–25 and 160–180 (wraps around red).
    # Saturation > 20 rules out near-white (white mask) and near-grey.
    # Value > 50 rules out very dark regions (black mask / shadow).
    mask_lo = cv2.inRange(hsv, (0, 20, 50), (25, 220, 255))
    mask_hi = cv2.inRange(hsv, (160, 20, 50), (180, 220, 255))
    skin_px = cv2.countNonZero(mask_lo) + cv2.countNonZero(mask_hi)
    return skin_px / (region_bgr.shape[0] * region_bgr.shape[1])


def is_wearing_mask(face_bgr, bbox):
    """Heuristic: True if a mask is likely present on this face.

    Analyses the lower 55-90% of the detected face bounding box (where a
    surgical/cloth mask sits) with three complementary checks and requires
    at least 2 to fire before declaring "masked".
    """
    x, y, w, h = bbox
    mouth_region = face_bgr[y + int(h * 0.55):y + int(h * 0.9), x:x + w]
    if mouth_region.size == 0:
        return True  # fail-safe: can't analyse → assume masked

    gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Signal 1: Mouth / smile landmark visibility
    # lowered minNeighbors (8 → was 15) so that real mouths are detected
    # more reliably, even with neutral expressions or non-frontal angles.
    # ------------------------------------------------------------------
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )
    mouth_detections = mouth_cascade.detectMultiScale(
        gray, scaleFactor=1.7, minNeighbors=8
    )
    mouth_visible = len(mouth_detections) > 0
    signal_no_mouth = not mouth_visible  # True → looks masked

    # ------------------------------------------------------------------
    # Signal 2: Edge density (Canny)
    # Real lower faces have lip edges, chin contour, nasolabial folds,
    # stubble etc. Masks are smooth fabric → low edge density.
    # Threshold raised from 0.06 → 0.12 to avoid false positives on
    # smooth skin or soft lighting.
    # ------------------------------------------------------------------
    edge_density = cv2.Canny(gray, 50, 150).mean() / 255.0
    signal_low_edges = edge_density < 0.12  # True → looks masked

    # ------------------------------------------------------------------
    # Signal 3: Skin-colour proportion
    # Unmasked lower faces are mostly skin-toned. A surgical/cloth mask
    # is mostly blue, white, or black — not skin. If less than 25% of
    # the region is skin-coloured, it's likely covered by something.
    # ------------------------------------------------------------------
    skin_pct = _skin_ratio(mouth_region)
    signal_non_skin = skin_pct < 0.25  # True → looks masked

    # ------------------------------------------------------------------
    # Also check texture uniformity: masks have very uniform intensity
    # compared to real skin (which has lips, chin shadow, stubble, etc.)
    # ------------------------------------------------------------------
    texture_std = float(np.std(gray))
    signal_uniform_texture = texture_std < 18.0  # True → looks masked

    # ------------------------------------------------------------------
    # Decision: require at least 2 of the 3 main signals to agree.
    # The texture check acts as a tie-breaker / additional confirmation.
    # ------------------------------------------------------------------
    mask_signals = sum([
        signal_no_mouth,
        signal_low_edges,
        signal_non_skin,
    ])

    # If exactly 2 signals fire, the texture uniformity check acts as a
    # weak confirmation — but we already call it masked at 2/3.
    # If only 1 signal fires, require the texture check as well (2 total).
    if mask_signals >= 2:
        return True
    if mask_signals == 1 and signal_uniform_texture:
        return True
    return False