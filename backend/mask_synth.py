# backend/mask_synth.py
"""
Synthetic mask overlay for reference photos.

Problem this solves: ArcFace embeddings of an unmasked reference photo and
embeddings of the same person wearing a mask in the query video sit in
noticeably different regions of embedding space — occluding the nose/mouth
removes real signal the model relies on. The most reliable fix is enrolling
actual masked photos (see main.py's /enroll docstring), but sourcing a real
masked photo of every candidate isn't always possible.

This module draws a plausible synthetic mask over the lower half of a
detected face — using the 5-point landmarks InsightFace already returns
(eyes, nose, mouth corners) to position and size it correctly — and returns
a new image. Re-running detection+embedding on that synthetic image gives a
second embedding that's much closer to the masked-video domain than the
bare unmasked photo, without needing a second photo shoot.

This is a reasonable *supplement* to real masked photos, not a full
replacement — a synthetic mask can't capture strap indentation, real
fabric shadow/wrinkle patterns, or all mask colors/shapes. If accuracy is
still short after using both, prioritize collecting a couple of real
masked reference photos per person (or frames pulled from their own
enrollment video, if you have one) over tuning this further.
"""
import cv2
import numpy as np


def _mask_polygon(kps: np.ndarray, bbox: tuple) -> np.ndarray:
    """Builds a trapezoid covering nose-to-chin, sized from eye/mouth
    landmarks so it tracks head pose/scale instead of being a fixed
    rectangle. kps order (InsightFace 5-point): left_eye, right_eye, nose,
    mouth_left, mouth_right."""
    left_eye, right_eye, nose, mouth_l, mouth_r = kps
    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1

    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (mouth_l + mouth_r) / 2.0

    # Top edge: partway between eyes and nose (just under the nose bridge,
    # over the top of the nose) — real masks start around the nose bridge.
    top_y = eye_mid[1] + (nose[1] - eye_mid[1]) * 0.55
    # Bottom edge: below the mouth, extended toward the chin.
    bottom_y = mouth_mid[1] + (mouth_mid[1] - nose[1]) * 1.6
    bottom_y = min(bottom_y, y2 + face_h * 0.05)  # don't run off the detected box by much

    mouth_span = np.linalg.norm(mouth_r - mouth_l)
    side_margin = max(mouth_span * 0.7, face_w * 0.12)

    top_left = (nose[0] - side_margin * 0.6, top_y)
    top_right = (nose[0] + side_margin * 0.6, top_y)
    bottom_left = (mouth_mid[0] - side_margin, bottom_y)
    bottom_right = (mouth_mid[0] + side_margin, bottom_y)

    poly = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    return poly


def synthesize_masked_face(
    img_bgr: np.ndarray,
    kps: np.ndarray,
    bbox: tuple,
    mask_color: tuple = (225, 225, 225),
    add_pleats: bool = True,
) -> np.ndarray:
    """Returns a copy of img_bgr with a synthetic mask drawn over the lower
    face, positioned using `kps` (5-point landmarks) and `bbox`.

    mask_color: BGR. Default is a light surgical-mask gray/white; pass
    something like (200, 140, 60) for a blue surgical mask, or a color
    sampled from the person's actual clothing if you want variety across
    a batch of enrollment photos.
    """
    out = img_bgr.copy()
    poly = _mask_polygon(np.asarray(kps, dtype=np.float32), bbox)

    overlay = out.copy()
    cv2.fillConvexPoly(overlay, poly, mask_color)

    if add_pleats:
        # A couple of faint horizontal lines read as fabric pleats and
        # avoid a flat, obviously-synthetic block of color.
        x_min, y_min = poly[:, 0].min(), poly[:, 1].min()
        x_max, y_max = poly[:, 0].max(), poly[:, 1].max()
        pleat_color = tuple(max(0, c - 25) for c in mask_color)
        for frac in (0.35, 0.6):
            y = int(y_min + (y_max - y_min) * frac)
            cv2.line(overlay, (int(x_min), y), (int(x_max), y), pleat_color, 1, cv2.LINE_AA)

    # Blend rather than paste flat, and feather the polygon edges so the
    # detector doesn't see a hard-edged block that reads as an obvious
    # synthetic artifact.
    mask_layer = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_layer, poly, 255)
    mask_layer = cv2.GaussianBlur(mask_layer, (9, 9), 0)
    alpha = (mask_layer.astype(np.float32) / 255.0)[..., None]

    blended = (overlay.astype(np.float32) * alpha + out.astype(np.float32) * (1 - alpha))
    return blended.astype(np.uint8)