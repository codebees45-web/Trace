# backend/main.py
import base64
import json
import logging
import io
import os
import tempfile
import threading
import time
import datetime
import uuid
import collections
from contextlib import asynccontextmanager
from typing import Optional, List

import cv2
import numpy as np
import joblib
import torch
from PIL import Image
from skimage.feature import hog, local_binary_pattern

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError
from bson import ObjectId, errors as bson_errors
from starlette.concurrency import run_in_threadpool

from config import settings
from database import get_db, init_db
import models
import schemas
from mask_gate import is_wearing_mask
from records import lookup_record
import mask_synth
import video_pipeline
from video_gallery import video_gallery
from reid_store import reid_store
import cnn_backbone
from auth import (
    create_access_token,
    create_reset_token,
    decode_reset_token,
    get_current_user,
    get_optional_user,
    hash_password,
    verify_password,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("trace")

# Loaded once — used to actually locate the face (and therefore the surgical mask)
# in the frame instead of assuming it sits in a fixed part of the image.
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------------------------------------------------------
# ML pipelines — loaded once at process startup, not per-request.
# A lock serializes calls into the diffusion pipeline: diffusers pipelines
# are not thread-safe, and without this two concurrent /predict requests
# could corrupt each other's generation.
# ---------------------------------------------------------------------------
bundle = None
sd_pipe = None
_sd_lock = threading.Lock()


def _load_models():
    global bundle, sd_pipe
    logger.info("Loading classifier pipeline from %s", settings.pipeline_path)
    bundle = joblib.load(settings.pipeline_path)

    # Log model metadata if available (new training pipeline saves this)
    if "metadata" in bundle:
        meta = bundle["metadata"]
        logger.info(
            "Model metadata: %d identities, %d images, %.1f%% CV accuracy, feature_mode=%s",
            meta.get("num_identities", 0),
            meta.get("num_images", 0),
            meta.get("cv_accuracy", 0) * 100,
            meta.get("feature_mode", "unknown"),
        )

    if settings.enable_generation:
        logger.info("Loading Stable Diffusion inpainting pipeline (%s)...", settings.sd_model_id)
        from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            settings.sd_model_id, torch_dtype=torch.float32, safety_checker=None
        )
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.requires_safety_checker = False
        logger.info("Generation pipeline ready.")
    else:
        logger.warning("ENABLE_GENERATION=false — /predict will skip face reconstruction.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_for_startup()
    init_db()
    _load_models()
    logger.info("Startup complete. environment=%s generation=%s", settings.environment, settings.enable_generation)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="TRACE API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tags every request/response with a correlation ID and logs latency.

    The ID round-trips in X-Request-ID (client-supplied if present, so a
    frontend/proxy that already generates one keeps the same value end to
    end) which makes it possible to grep a single request across the
    frontend console, nginx access log, and backend log lines.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s %s %.1fms rid=%s",
        request.method, request.url.path, response.status_code, duration_ms, request_id,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on our end. Please try again."},
    )


# ---------------------------------------------------------------------------
# Feature extraction — supports both legacy HOG-only and new multi-feature
# ---------------------------------------------------------------------------

def extract_features_hog_only(face_gray):
    """Legacy HOG-only feature extraction (backward compat)."""
    return hog(
        face_gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys",
    )


def extract_multi_features(face_gray, img_size=(96, 96)):
    """Enhanced multi-feature extraction: multi-scale HOG + LBP + upper-face emphasis.

    This function MUST match exactly what train_model.py uses at training time.
    """
    hog_features = []
    scales = [(64, 64), (96, 96), (128, 128)]
    for size in scales:
        resized = cv2.resize(face_gray, size)

        # Extract HOG features
        h = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )

        # Apply spatial weighting: upper face emphasis (top 45%)
        blocks_y = (size[1] // 8) - 1
        blocks_x = (size[0] // 8) - 1
        reshaped = h.reshape((blocks_y, blocks_x, 2, 2, 9))

        split_idx = int(blocks_y * 0.45)
        reshaped[:split_idx, ...] *= 1.5

        hog_features.append(reshaped.ravel())

    hog_concat = np.concatenate(hog_features)

    # Extract LBP features
    resized_base = cv2.resize(face_gray, img_size)
    lbp_img = local_binary_pattern(resized_base, P=24, R=3, method="uniform")
    (hist, _) = np.histogram(lbp_img.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return np.concatenate([hog_concat, hist])


def extract_features_from_face(face_input):
    """Dispatches to the right feature extractor based on the loaded model.

    face_input must be a BGR color crop (as returned by cv2.imdecode / slicing
    img_bgr).  Grayscale conversion is applied *internally* only for the
    HOG/LBP branches; the CNN/ArcFace branch receives the color image as-is
    because InsightFace's ArcFace was trained on color data.
    """
    expected_dim = None
    if bundle and "scaler" in bundle and hasattr(bundle["scaler"], "n_features_in_"):
        expected_dim = bundle["scaler"].n_features_in_
    elif bundle and "model" in bundle and hasattr(bundle["model"], "n_features_in_"):
        expected_dim = bundle["model"].n_features_in_

    if expected_dim == 512:
        # ArcFace ResNet-50 path — color image, no grayscale conversion
        if settings.use_occlusion_aware_embedding:
            logger.debug("occlusion-aware embedding enabled — using masked ArcFace path")
            return cnn_backbone.extract_cnn_features_occlusion_aware(
                face_input, alpha_occluded=settings.occlusion_mask_alpha)
        return cnn_backbone.extract_cnn_features(face_input)
    elif expected_dim == 4356:
        # Multi-scale HOG + LBP path — requires grayscale
        face_gray = cv2.cvtColor(face_input, cv2.COLOR_BGR2GRAY)
        img_size = bundle.get("img_size", (96, 96)) if bundle else (96, 96)
        return extract_multi_features(face_gray, img_size)
    elif expected_dim is not None and expected_dim not in (512, 4356):
        # Legacy HOG-only path — requires grayscale
        face_gray = cv2.cvtColor(face_input, cv2.COLOR_BGR2GRAY)
        return extract_features_hog_only(face_gray)

    feature_mode = bundle.get("metadata", {}).get("feature_mode", "Deep CNN Backbone (512-d)") if bundle else "Deep CNN Backbone (512-d)"
    if feature_mode in ("cnn", "Deep CNN Backbone (512-d)", "resnet", "arcface", "cnn_resnet50_arcface"):
        # ArcFace ResNet-50 path — color image
        if settings.use_occlusion_aware_embedding:
            logger.debug("occlusion-aware embedding enabled — using masked ArcFace path")
            return cnn_backbone.extract_cnn_features_occlusion_aware(
                face_input, alpha_occluded=settings.occlusion_mask_alpha)
        return cnn_backbone.extract_cnn_features(face_input)
    elif feature_mode == "multi":
        face_gray = cv2.cvtColor(face_input, cv2.COLOR_BGR2GRAY)
        img_size = bundle.get("img_size", (96, 96)) if bundle else (96, 96)
        return extract_multi_features(face_gray, img_size)
    else:
        face_gray = cv2.cvtColor(face_input, cv2.COLOR_BGR2GRAY)
        return extract_features_hog_only(face_gray)


# ---------------------------------------------------------------------------
# Face quality assessment
# ---------------------------------------------------------------------------

def assess_face_quality(img_bgr) -> dict:
    """Evaluate image quality for face recognition.

    Returns a dict with blur_score, brightness_score, resolution_ok, overall.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Blur detection via Laplacian variance (higher = sharper)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: 0-100 is blurry, 100+ is sharp, 500+ is very sharp
    blur_score = min(1.0, laplacian_var / 200.0)

    # Brightness assessment (want ~100-170 mean)
    mean_brightness = gray.mean()
    if 80 <= mean_brightness <= 180:
        brightness_score = 1.0
    elif 50 <= mean_brightness <= 210:
        brightness_score = 0.6
    else:
        brightness_score = 0.3

    # Resolution check (minimum 64x64)
    resolution_ok = h >= 64 and w >= 64

    # Overall assessment
    avg_score = (blur_score + brightness_score + (1.0 if resolution_ok else 0.3)) / 3
    if avg_score >= 0.7:
        overall = "good"
    elif avg_score >= 0.45:
        overall = "fair"
    else:
        overall = "poor"

    return {
        "blur_score": round(blur_score, 3),
        "brightness_score": round(brightness_score, 3),
        "resolution_ok": resolution_ok,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Identity prediction — supports top-K and multi-face
# ---------------------------------------------------------------------------

def predict_identity(img_bgr, threshold: Optional[float] = None, top_k: int = 3):
    """Predict identity from a face image. Returns primary result + top-K candidates.

    Before feature extraction the function attempts to locate and crop to the
    detected face region (with a 15% margin on each side, matching the
    crop_largest_face() convention in train_model_v4.py).  Callers that
    already pass a pre-cropped face (e.g. /predict/multi-face) will simply
    fail to detect a face inside the tiny crop and fall back to using the
    input as-is — so no regressions for those paths.
    """
    threshold = settings.confidence_threshold if threshold is None else threshold

    # -- Face crop with margin (same convention as train_model_v4.crop_largest_face) --
    bbox = detect_face_bbox(img_bgr)
    if bbox is not None:
        x, y, w, h = bbox
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        h_img, w_img = img_bgr.shape[:2]
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(w_img, x + w + margin_x)
        y1 = min(h_img, y + h + margin_y)
        face_crop = img_bgr[y0:y1, x0:x1]
    else:
        # No face detected — fall back to the full image so the call still
        # succeeds (callers that already pass a pre-cropped face hit this path).
        face_crop = img_bgr

    feats = extract_features_from_face(face_crop).reshape(1, -1)
    feats_scaled = bundle["scaler"].transform(feats)
    feats_pca = bundle.get("pca").transform(feats_scaled) if bundle.get("pca") is not None else feats_scaled

    model = bundle["model"]
    top_k_matches = []

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats_pca)[0]
        # Get top-K indices
        top_indices = np.argsort(probs)[::-1][:top_k]
        for idx in top_indices:
            top_k_matches.append({
                "identity": str(bundle["label_encoder"].classes_[idx]),
                "confidence": float(probs[idx]),
            })
        best_idx = top_indices[0]
        confidence = float(probs[best_idx])
    else:
        best_idx = int(model.predict(feats_pca)[0])
        confidence = 1.0
        top_k_matches = [{"identity": str(bundle["label_encoder"].classes_[best_idx]), "confidence": confidence}]

    if confidence < threshold:
        return {
            "identity": "Unknown",
            "confidence": confidence,
            "top_k_matches": top_k_matches,
        }
    return {
        "identity": str(bundle["label_encoder"].classes_[best_idx]),
        "confidence": confidence,
        "top_k_matches": top_k_matches,
    }


def detect_face_bbox(img_bgr):
    """Returns (x, y, w, h) of the largest detected face, or None if no face found."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(int(v) for v in faces[0])


def detect_all_faces(img_bgr):
    """Returns list of (x, y, w, h) for ALL detected faces, sorted by size (largest first)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return []
    return [tuple(int(v) for v in f) for f in sorted(faces, key=lambda f: f[2] * f[3], reverse=True)]


def build_lower_face_mask(size, bbox):
    """
    Masks the lower ~55% of the detected face box (nose bridge down through
    chin/neck — i.e. where a surgical/cloth mask actually sits) plus a small
    side margin for jaw and ears. Falls back to a whole-frame heuristic only
    if no face is detected.
    """
    w_img, h_img = size
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    if bbox is None:
        mask[h_img // 2:, :] = 255
        return Image.fromarray(mask), False

    x, y, w, h = bbox
    side_margin = int(w * 0.15)
    x0 = max(0, x - side_margin)
    x1 = min(w_img, x + w + side_margin)
    y0 = min(h_img, y + int(h * 0.42))   # start just below the nose bridge
    y1 = min(h_img, y + int(h * 1.25))   # extend past the chin into the neck
    mask[y0:y1, x0:x1] = 255
    return Image.fromarray(mask), True


def generate_unmasked_face(img_bgr, seed: Optional[int] = None):
    if sd_pipe is None:
        return img_bgr
    try:
        resized_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        bbox = detect_face_bbox(resized_bgr)
        mask_pil, face_found = build_lower_face_mask((512, 512), bbox)
        if not face_found:
            logger.warning("No face detected in frame — using degraded whole-image fallback mask")

        pil_img = Image.fromarray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        with _sd_lock:
            result = sd_pipe(
                prompt="a clear photo of a human face, no mask, natural skin, photorealistic, "
                       "sharp focus, matching lighting and skin tone with the visible upper face",
                negative_prompt="mask, fabric, pattern, fabric texture, noise, static, blurry, "
                                 "distorted, deformed, extra features, low quality",
                image=pil_img,
                mask_image=mask_pil,
                num_inference_steps=settings.sd_inference_steps,
                guidance_scale=settings.sd_guidance_scale,
                generator=generator,
            ).images[0]

        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    except Exception:
        logger.exception("Generation failed, returning original image")
        return img_bgr


def img_to_base64(img_bgr, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf).decode("utf-8")


def make_thumbnail_b64(img_bgr, max_side: int = 160) -> str:
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return img_to_base64(small, quality=70)


def is_valid_image_upload(file: UploadFile) -> bool:
    valid_types = ("image/jpeg", "image/png", "image/webp", "image/jpg", "image/pjpeg", "application/octet-stream")
    if file.content_type in valid_types:
        if file.content_type == "application/octet-stream":
            ext = (file.filename or "").lower().split(".")[-1]
            return ext in ("jpg", "jpeg", "png", "webp")
        return True
    ext = (file.filename or "").lower().split(".")[-1]
    return ext in ("jpg", "jpeg", "png", "webp")


def draw_face_annotations(img_bgr, faces_data):
    """Draw bounding boxes and labels on the image for all detected faces."""
    annotated = img_bgr.copy()
    for face in faces_data:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        color = (87, 168, 227) if face["identity"] != "Unknown" else (107, 122, 201)  # amber / red in BGR
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{face['identity']} ({face['confidence']:.0%})"
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(annotated, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def run_full_pipeline(img_bgr):
    """CPU/GPU-bound work, executed off the event loop via run_in_threadpool.

    Returns a 4-tuple: (result, restored, quality, trust_score).
    trust_score is a float in [0, 1] measuring ArcFace cosine similarity
    between the original masked face crop and the SD-reconstructed face.
    It is None when generation is disabled or generation failed.
    """
    result = predict_identity(img_bgr, top_k=settings.top_k_results)

    # Only run SD inpainting if a mask is actually detected — otherwise
    # generate_unmasked_face() distorts already-clear faces for no reason.
    bbox = detect_face_bbox(img_bgr)
    should_generate = settings.enable_generation and (
        bbox is None or is_wearing_mask(img_bgr, bbox)
    )
    # bbox is None → can't tell, fail open to the old behavior (safer than
    # silently skipping generation for every undetected face)

    if should_generate:
        restored = generate_unmasked_face(img_bgr)
    else:
        restored = img_bgr

    quality = assess_face_quality(img_bgr) if settings.enable_quality_check else None

    trust_score = None
    if settings.enable_generation and should_generate and restored is not img_bgr:
        try:
            emb_masked = cnn_backbone.extract_cnn_features(img_bgr)
            emb_recon  = cnn_backbone.extract_cnn_features(restored)
            cos_sim = float(np.dot(emb_masked, emb_recon) /
                            (np.linalg.norm(emb_masked) * np.linalg.norm(emb_recon) + 1e-8))
            trust_score = round(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)), 4)
        except Exception:
            logger.warning("Could not compute reconstruction trust score", exc_info=True)

    return result, restored, quality, trust_score


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=schemas.UserOut, status_code=201)
@limiter.limit(settings.auth_rate_limit)
async def register(request: Request, payload: schemas.UserCreate, db: Database = Depends(get_db)):
    existing = db.users.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="An account with that email already exists")

    doc = {
        "email": payload.email,
        "hashed_password": hash_password(payload.password),
        "created_at": datetime.datetime.utcnow(),
    }
    try:
        res = db.users.insert_one(doc)
        doc["_id"] = res.inserted_id
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="An account with that email already exists")
    return models.User(doc)


@app.post("/auth/login", response_model=schemas.Token)
@limiter.limit(settings.auth_rate_limit)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Database = Depends(get_db),
):
    user_doc = db.users.find_one({"email": form_data.username})
    if not user_doc or not verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    token = create_access_token(subject=user_doc["email"])
    return schemas.Token(access_token=token)


@app.get("/auth/me", response_model=schemas.UserOut)
async def me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/auth/change-password", status_code=204)
@limiter.limit(settings.auth_rate_limit)
async def change_password(
    request: Request,
    payload: schemas.ChangePassword,
    current_user: models.User = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    new_hash = hash_password(payload.new_password)
    db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"hashed_password": new_hash}})


@app.post("/auth/forgot-password", status_code=202)
@limiter.limit(settings.auth_rate_limit)
async def forgot_password(
    request: Request, payload: schemas.ForgotPassword, db: Database = Depends(get_db)
):
    """
    Always returns 202 regardless of whether the email exists — leaking
    account existence through response differences is a common enumeration
    vector for this kind of endpoint.

    NOTE: no SMTP/email provider is wired up yet, so the reset link is
    logged server-side instead of emailed. Wire in an email provider (SES,
    Postmark, Resend, etc.) in production and replace the logger.info call
    below with an actual send.
    """
    user_doc = db.users.find_one({"email": payload.email})
    if user_doc is not None:
        reset_token = create_reset_token(subject=user_doc["email"])
        reset_link = f"{settings.frontend_url}/?reset_token={reset_token}"
        logger.info(
            "Password reset requested for %s (expires in %d min): %s",
            user_doc["email"], 30, reset_link,
        )
    return {"detail": "If that email exists, a reset link has been sent."}


@app.post("/auth/reset-password", status_code=204)
@limiter.limit(settings.auth_rate_limit)
async def reset_password(
    request: Request, payload: schemas.ResetPassword, db: Database = Depends(get_db)
):
    email = decode_reset_token(payload.token)
    if email is None:
        raise HTTPException(status_code=400, detail="Reset link is invalid or has expired")
    user_doc = db.users.find_one({"email": email})
    if user_doc is None:
        raise HTTPException(status_code=400, detail="Reset link is invalid or has expired")
    new_hash = hash_password(payload.new_password)
    db.users.update_one({"_id": user_doc["_id"]}, {"$set": {"hashed_password": new_hash}})


@app.delete("/auth/me", status_code=204)
async def delete_account(
    current_user: models.User = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    """Deletes the account and all scan history."""
    db.users.delete_one({"_id": ObjectId(current_user.id)})
    db.scan_history.delete_many({"user_id": current_user.id})


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(db: Database = Depends(get_db)):
    db_ok = True
    try:
        db.command("ping")
    except Exception:
        logger.exception("Health check: database ping failed")
        db_ok = False

    status_ok = db_ok and bundle is not None
    return JSONResponse(
        status_code=200 if status_ok else 503,
        content={
            "status": "ok" if status_ok else "degraded",
            "environment": settings.environment,
            "classifier_loaded": bundle is not None,
            "generation_enabled": sd_pipe is not None,
            "database_ok": db_ok,
        },
    )


@app.get("/model-info", response_model=schemas.ModelInfoResponse)
async def model_info():
    """Returns metadata about the loaded ML model."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = bundle.get("metadata", {})
    classes = list(bundle["label_encoder"].classes_)

    return schemas.ModelInfoResponse(
        num_identities=meta.get("num_identities", len(classes)),
        num_images=meta.get("num_images", 0),
        cv_accuracy=meta.get("cv_accuracy", 0.0),
        feature_dim=meta.get("feature_dim", 512),
        feature_mode=meta.get("feature_mode", "Deep CNN Backbone (512-d)"),
        training_date=meta.get("training_date", "unknown"),
        identities=classes,
    )


@app.post("/predict", response_model=schemas.PredictResponse)
@limiter.limit(settings.predict_rate_limit)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    db: Database = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_optional_user),
):
    if not is_valid_image_upload(file):
        raise HTTPException(status_code=415, detail="Please upload a JPEG, PNG, or WebP image")

    contents = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Image too large — max {settings.max_upload_mb}MB"
        )

    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    try:
        img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        # Malformed bytes can make OpenCV raise a raw C++ assertion instead of
        # returning None — catch it explicitly so a bad upload is a clean 422,
        # not an unhandled crash.
        img_bgr = None
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Couldn't read that as an image")

    result, restored, quality, trust_score = await run_in_threadpool(run_full_pipeline, img_bgr)

    saved = False
    if current_user is not None:
        try:
            quality_score = None
            if quality:
                quality_score = (quality["blur_score"] + quality["brightness_score"] + (1.0 if quality["resolution_ok"] else 0.3)) / 3
            history_doc = {
                "user_id": current_user.id,
                "identity": result["identity"],
                "confidence": result["confidence"],
                "input_thumb": make_thumbnail_b64(img_bgr),
                "generated_thumb": make_thumbnail_b64(restored),
                "face_quality_score": quality_score,
                "created_at": datetime.datetime.utcnow(),
            }
            db.scan_history.insert_one(history_doc)
            saved = True
        except Exception:
            logger.exception("Failed to save scan history for user %s", current_user.id)

    return schemas.PredictResponse(
        identity=result["identity"],
        confidence=result["confidence"],
        input_image=img_to_base64(img_bgr),
        generated_image=img_to_base64(restored),
        saved_to_history=saved,
        face_quality=schemas.FaceQuality(**quality) if quality else None,
        top_k_matches=[schemas.TopKMatch(**m) for m in result.get("top_k_matches", [])],
        reconstruction_trust_score=trust_score,
    )


@app.post("/predict/batch", response_model=schemas.BatchPredictResponse)
@limiter.limit("3/minute")
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    db: Database = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_optional_user),
):
    """Process multiple images in a single request (max 10)."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    results = []
    successful = 0
    failed = 0

    for file in files:
        try:
            if not is_valid_image_upload(file):
                results.append(schemas.BatchPredictItem(
                    filename=file.filename or "unknown",
                    identity="", confidence=0,
                    input_image="", generated_image="",
                    error="Unsupported file type",
                ))
                failed += 1
                continue

            contents = await file.read()
            if len(contents) == 0 or len(contents) > settings.max_upload_mb * 1024 * 1024:
                results.append(schemas.BatchPredictItem(
                    filename=file.filename or "unknown",
                    identity="", confidence=0,
                    input_image="", generated_image="",
                    error="File empty or too large",
                ))
                failed += 1
                continue

            img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                results.append(schemas.BatchPredictItem(
                    filename=file.filename or "unknown",
                    identity="", confidence=0,
                    input_image="", generated_image="",
                    error="Couldn't read as image",
                ))
                failed += 1
                continue

            result, restored, quality, _trust = await run_in_threadpool(run_full_pipeline, img_bgr)

            results.append(schemas.BatchPredictItem(
                filename=file.filename or "unknown",
                identity=result["identity"],
                confidence=result["confidence"],
                input_image=img_to_base64(img_bgr),
                generated_image=img_to_base64(restored),
                face_quality=schemas.FaceQuality(**quality) if quality else None,
                top_k_matches=[schemas.TopKMatch(**m) for m in result.get("top_k_matches", [])],
            ))
            successful += 1
        except Exception:
            logger.exception("Batch item failed: %s", file.filename)
            results.append(schemas.BatchPredictItem(
                filename=file.filename or "unknown",
                identity="", confidence=0,
                input_image="", generated_image="",
                error="Processing failed",
            ))
            failed += 1

    return schemas.BatchPredictResponse(
        results=results,
        total=len(files),
        successful=successful,
        failed=failed,
    )


@app.post("/predict/multi-face", response_model=schemas.MultiFaceResponse)
@limiter.limit(settings.predict_rate_limit)
async def predict_multi_face(
    request: Request,
    file: UploadFile = File(...),
):
    """Detect and identify all faces in an image."""
    if not is_valid_image_upload(file):
        raise HTTPException(status_code=415, detail="Please upload a JPEG, PNG, or WebP image")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Couldn't read that as an image")

    def _process():
        all_bboxes = detect_all_faces(img_bgr)
        faces_data = []
        for (x, y, w, h) in all_bboxes:
            # Crop face region and predict
            face_crop = img_bgr[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            result = predict_identity(face_crop, top_k=settings.top_k_results)
            faces_data.append({
                "x": x, "y": y, "w": w, "h": h,
                "identity": result["identity"],
                "confidence": result["confidence"],
                "top_k_matches": result.get("top_k_matches", []),
            })

        # Draw annotations
        annotated = draw_face_annotations(img_bgr, faces_data)
        quality = assess_face_quality(img_bgr) if settings.enable_quality_check else None
        return faces_data, annotated, quality

    faces_data, annotated, quality = await run_in_threadpool(_process)

    return schemas.MultiFaceResponse(
        faces=[schemas.FaceDetection(
            x=f["x"], y=f["y"], w=f["w"], h=f["h"],
            identity=f["identity"], confidence=f["confidence"],
            top_k_matches=[schemas.TopKMatch(**m) for m in f.get("top_k_matches", [])],
        ) for f in faces_data],
        annotated_image=img_to_base64(annotated),
        face_quality=schemas.FaceQuality(**quality) if quality else None,
    )


@app.post("/predict/explain", response_model=schemas.ExplainResponse)
@limiter.limit(settings.predict_rate_limit)
async def predict_explain(
    request: Request,
    file: UploadFile = File(...),
):
    """Predict identity and return an occlusion sensitivity heatmap showing
    which face regions drove the match decision.

    The heatmap is a 2D grid of importance scores (0-1) — each cell
    represents how much occluding that patch of the face dropped the
    cosine similarity to the matched gallery embedding. Higher values
    mean that region was more important for the match.

    This is inherently slower than /predict (~1-3 s extra on CPU) because
    it runs ~169 additional ArcFace forward passes. Use it for explainability
    audits, not for real-time inference.
    """
    if not is_valid_image_upload(file):
        raise HTTPException(status_code=415, detail="Please upload a JPEG, PNG, or WebP image")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Couldn't read that as an image")

    patch_size = 16
    stride = 8

    def _process():
        # 1. Run normal prediction
        result = predict_identity(img_bgr, top_k=1)

        # 2. Get the face crop (same logic as predict_identity)
        bbox = detect_face_bbox(img_bgr)
        if bbox is not None:
            x, y, w, h = bbox
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            h_img, w_img = img_bgr.shape[:2]
            x0 = max(0, x - margin_x)
            y0 = max(0, y - margin_y)
            x1 = min(w_img, x + w + margin_x)
            y1 = min(h_img, y + h + margin_y)
            face_crop = img_bgr[y0:y1, x0:x1]
        else:
            face_crop = img_bgr

        # 3. Get the reference embedding (the matched identity's gallery embedding,
        #    or the face's own embedding if Unknown)
        ref_emb = cnn_backbone.extract_cnn_features(face_crop)

        # 4. Compute occlusion heatmap
        heatmap = cnn_backbone.compute_occlusion_heatmap(
            face_crop, ref_emb, patch_size=patch_size, stride=stride
        )

        return result, heatmap

    result, heatmap = await run_in_threadpool(_process)

    return schemas.ExplainResponse(
        identity=result["identity"],
        confidence=result["confidence"],
        heatmap=heatmap.tolist(),
        patch_size=patch_size,
        stride=stride,
    )


@app.get("/history", response_model=list[schemas.ScanHistoryOut])
async def get_history(
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    limit: int = 50,
):
    docs = db.scan_history.find({"user_id": current_user.id}).sort("created_at", -1).limit(min(limit, 200))
    return [models.ScanHistory(doc) for doc in docs]


@app.delete("/history/{scan_id}", status_code=204)
async def delete_history_item(
    scan_id: str,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    try:
        obj_id = ObjectId(scan_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=404, detail="Not found")
    res = db.scan_history.delete_one({"_id": obj_id, "user_id": current_user.id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")


@app.get("/demo-results")
async def demo_results(db: Database = Depends(get_db)):
    doc = db.demo_results_cache.find_one({"_id": "latest_demo_results"})
    if not doc:
        if os.path.exists("demo_cache.json"):
            with open("demo_cache.json") as f:
                return json.load(f)
        raise HTTPException(
            status_code=404, detail="Demo cache not found in MongoDB — run precompute.py first"
        )
    return doc.get("results", [])


# ---------------------------------------------------------------------------
# Video pipeline endpoints — SCRFD detection + ArcFace embeddings + IOU
# tracking + multi-frame voting. Separate identity space from the legacy
# HOG/PCA /predict pipeline above: gallery photos organizers hand out for
# the masked-video challenge get enrolled here, not into gallery.json.
# ---------------------------------------------------------------------------

def _require_video_pipeline():
    if not video_pipeline.is_available():
        raise HTTPException(
            status_code=503,
            detail="Video pipeline unavailable — insightface/onnxruntime not installed. "
                   "Run: pip install insightface onnxruntime",
        )


@app.post("/enroll", response_model=schemas.EnrollResponse)
@limiter.limit(settings.video_rate_limit)
async def enroll_identity(
    request: Request,
    identity: str,
    files: List[UploadFile] = File(...),
):
    """Enroll one or more gallery photos of a person for the video pipeline.
    Each photo's largest detected face becomes one embedding in that
    person's gallery entry. Additive: calling this again for the same
    identity adds more embeddings rather than replacing them.

    IMPORTANT for matching against masked-face video: ArcFace was trained
    on full, unmasked faces, so unmasked reference photos live in a
    different visual domain than a masked query video — that gap alone
    depresses cosine similarity regardless of whether the identity is
    correct. To help close that gap automatically, each uploaded photo
    also gets a synthetic mask overlaid (see mask_synth.py) and enrolled
    as a second embedding — so `embeddings_added` will typically be close
    to 2x the number of photos with a detected face. This is a supplement,
    not a replacement: a few *real* masked (or partially occluded)
    reference photos per person, if you can get them, will still help
    more than the synthetic ones alone."""
    _require_video_pipeline()

    if not identity.strip():
        raise HTTPException(status_code=422, detail="identity must not be empty")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 photos per enrollment call")

    file_contents = []
    for f in files:
        if not is_valid_image_upload(f):
            continue
        contents = await f.read()
        if 0 < len(contents) <= settings.max_upload_mb * 1024 * 1024:
            file_contents.append(contents)

    if not file_contents:
        raise HTTPException(status_code=422, detail="No valid image files provided")

    def _process():
        embeddings_added = 0
        for file_bytes in file_contents:
            img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            dets = video_pipeline.detect_and_embed(img_bgr, frame_idx=0, det_score_thresh=0.25)
            if not dets:
                continue

            # ── Enroll ALL detected faces (important for group photos) ──
            # Previously only the largest face was enrolled, so in a 2-person
            # photo only one person would be registered. Now we enroll every
            # detected face, which is correct for a gallery intended to hold
            # multiple people. The caller is expected to pass one photo per
            # person when enrolling under a specific name.
            for det in dets:
                video_gallery.enroll(identity, det.embedding)
                embeddings_added += 1

                # Horizontal-flip augmentation — increases robustness to side profiles
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
                face_crop = img_bgr[y1:y2, x1:x2]
                if face_crop.size > 0:
                    flipped = cv2.flip(face_crop, 1)
                    # Embed the flipped crop by resizing to a full image and re-detecting
                    # (simpler: just embed the crop directly if the pipeline supports it)
                    flip_full = cv2.resize(flipped, (img_bgr.shape[1], img_bgr.shape[0]))
                    flip_dets = video_pipeline.detect_and_embed(flip_full, frame_idx=0, det_score_thresh=0.2)
                    for fd in flip_dets:
                        video_gallery.enroll(identity, fd.embedding)
                        embeddings_added += 1

                # Auto-generate a masked variant and enroll it too
                if det.kps is not None:
                    try:
                        masked_img = mask_synth.synthesize_masked_face(img_bgr, det.kps, det.bbox)
                        masked_dets = video_pipeline.detect_and_embed(masked_img, frame_idx=0, det_score_thresh=0.25)
                        for md in masked_dets:
                            video_gallery.enroll(identity, md.embedding)
                            embeddings_added += 1
                    except Exception:
                        logger.warning("Synthetic mask generation failed for one photo; continuing without it.",
                                        exc_info=True)
        return embeddings_added

    embeddings_added = await run_in_threadpool(_process)

    if embeddings_added == 0:
        raise HTTPException(status_code=422, detail="No face detected in any provided photo")

    return schemas.EnrollResponse(
        identity=identity,
        embeddings_added=embeddings_added,
        total_identities=len(video_gallery.identities()),
    )


@app.get("/enroll/identities", response_model=schemas.EnrollIdentitiesResponse)
async def enroll_identities():
    return schemas.EnrollIdentitiesResponse(identities=video_gallery.identities())


@app.post("/enroll/preview")
@limiter.limit(settings.video_rate_limit)
async def enroll_preview(
    request: Request,
    file: UploadFile = File(...),
):
    """Returns the synthetic-masked version of a single uploaded photo as a
    JPEG image — no enrollment happens here. Use this to eyeball whether
    mask_synth.py is positioning the overlay correctly (right size, right
    place on the face) for your reference photos before trusting the
    embeddings it produces in /enroll. If the mask looks obviously
    misaligned for a given photo (extreme angle, partial face, etc.),
    prefer a real masked photo of that person instead for that shot."""
    _require_video_pipeline()

    if not is_valid_image_upload(file):
        raise HTTPException(status_code=415, detail="Please upload a JPEG, PNG, or WebP image")

    contents = await file.read()
    if not (0 < len(contents) <= settings.max_upload_mb * 1024 * 1024):
        raise HTTPException(status_code=422, detail="Empty file or exceeds max upload size")

    def _process():
        img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=422, detail="Could not decode image")

        dets = video_pipeline.detect_and_embed(img_bgr, frame_idx=0, det_score_thresh=0.3)
        if not dets:
            raise HTTPException(status_code=422, detail="No face detected in the photo")

        best = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        if best.kps is None:
            raise HTTPException(
                status_code=422,
                detail="Face detected but no landmarks returned — can't position a synthetic mask "
                       "for this photo. A real masked photo would be needed instead.",
            )

        masked_img = mask_synth.synthesize_masked_face(img_bgr, best.kps, best.bbox)
        ok, buf = cv2.imencode(".jpg", masked_img)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode preview image")
        return buf.tobytes()

    jpeg_bytes = await run_in_threadpool(_process)
    return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")


@app.delete("/enroll/{identity}", status_code=204)
async def enroll_delete(identity: str):
    if not video_gallery.remove(identity):
        raise HTTPException(status_code=404, detail="Identity not found in gallery")


@app.post("/identify/video", response_model=schemas.IdentifyVideoResponse)
@limiter.limit(settings.video_rate_limit)
async def identify_video(
    request: Request,
    file: UploadFile = File(...),
):
    """Run the full masked-face video pipeline: sample frames, detect +
    embed every face with SCRFD/ArcFace, track faces across frames, and
    return a voted identity decision. If multiple people are tracked in
    the video, the highest-confidence track is reported as the primary
    identity and the rest surface as candidates."""
    _require_video_pipeline()

    if video_gallery.is_empty():
        raise HTTPException(
            status_code=422,
            detail="Gallery is empty — enroll at least one identity via /enroll first",
        )

    if file.content_type not in ("video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"):
        raise HTTPException(status_code=415, detail="Please upload an MP4, MOV, AVI, or WebM video")

    contents = await file.read()
    max_bytes = settings.max_upload_video_mb * 1024 * 1024
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Video too large — max {settings.max_upload_video_mb}MB"
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = await run_in_threadpool(
            video_pipeline.process_video,
            tmp_path,
            video_gallery,
            settings.video_sample_every_n_frames,
            settings.video_max_frames_sampled,
            settings.video_det_score_thresh,
            settings.video_track_iou_threshold,
            settings.video_track_max_age,
            settings.video_min_similarity,
            settings.video_min_margin,
            settings.video_det_size,
            settings.video_calib_midpoint,
            settings.video_calib_slope,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    tracks = result["tracks"]  # already sorted best-identified-first by process_video

    if not tracks:
        return schemas.IdentifyVideoResponse(
            identity="Unknown",
            confidence=0.0,
            frames_used=result["frames_processed"],
            detail="No faces detected in video",
            candidates=[],
            tracks=[],
            num_tracks=0,
            frames_processed=result["frames_processed"],
            total_frames=result["total_frames"],
            video_fps=result["video_fps"],
            processing_seconds=result["processing_seconds"],
        )

    top = tracks[0]
    detail = None
    if len(tracks) > 1:
        detail = f"{len(tracks)} face(s) tracked in video — showing best match"

    candidates = [
        schemas.VideoCandidate(identity=t["identity"], score=t["confidence"], raw_similarity=t["raw_similarity"])
        for t in tracks[1:6]  # runner-up tracks, capped
    ]

    return schemas.IdentifyVideoResponse(
        identity=top["identity"],
        confidence=top["confidence"],
        raw_confidence=top["raw_similarity"],
        frames_used=result["frames_processed"],
        detail=detail,
        candidates=candidates,
        tracks=[schemas.TrackResult(**t) for t in tracks],
        num_tracks=result["num_tracks"],
        frames_processed=result["frames_processed"],
        total_frames=result["total_frames"],
        video_fps=result["video_fps"],
        processing_seconds=result["processing_seconds"],
    )


# Pre-load ALL available frontal cascade files — we try each and take any hit.
_live_cascades = []
_cascade_names = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt_tree.xml",
]
for _cn in _cascade_names:
    _c = cv2.CascadeClassifier(cv2.data.haarcascades + _cn)
    if not _c.empty():
        _live_cascades.append(_c)

_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
logger.info("Live detector: loaded %d frontal cascade(s)", len(_live_cascades))


def _detect_faces_live(img_bgr):
    """Robust face detector for live camera conditions.

    Strategy:
      1. Apply histogram equalization to handle dim / warm-toned rooms.
      2. Try every frontal cascade with very permissive settings.
      3. If nothing found, try the profile cascade (tilted / side-on heads).
      4. De-duplicate overlapping boxes and return sorted largest-first.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # equalizeHist works better than CLAHE for typical indoor/webcam lighting
    # and avoids thread-safety concerns with a shared CLAHE object.
    gray_eq = cv2.equalizeHist(gray)

    all_rects = []

    # ── Try every loaded frontal cascade ────────────────────────────────────
    for cascade in _live_cascades:
        dets = cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,    # finer scan window steps
            minNeighbors=1,      # absolute minimum — accept almost any candidate
            minSize=(25, 25),    # catch faces even far from the camera
            maxSize=(0, 0),      # no upper limit
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(dets) > 0:
            all_rects.extend([tuple(int(v) for v in r) for r in dets])

    # ── Profile cascade fallback ─────────────────────────────────────────────
    if not all_rects and not _profile_cascade.empty():
        dets = _profile_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=1,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(dets) > 0:
            all_rects.extend([tuple(int(v) for v in r) for r in dets])

    if not all_rects:
        return []

    # ── Merge overlapping duplicates (from multiple cascades) ────────────────
    try:
        rects_dup = [list(r) for r in all_rects] + [list(r) for r in all_rects]
        merged, _ = cv2.groupRectangles(rects_dup, groupThreshold=1, eps=0.4)
        if len(merged) > 0:
            all_rects = [tuple(int(v) for v in r) for r in merged]
    except Exception:
        pass  # groupRectangles can occasionally fail; keep originals

    return sorted(all_rects, key=lambda r: r[2] * r[3], reverse=True)



def _process_live_frame(frame_bytes: bytes, camera_id: str, frame_id: int = 0) -> list[dict]:
    """Decode a JPEG frame, detect all faces and predict each identity.

    Recognition pipeline (in priority order):
      1. SCRFD face detection + ArcFace embedding → video_gallery match.
         This is the SAME pipeline used by enrollment (/enroll) and video
         identification, so faces enrolled via the Video Identify tab are
         actually recognised here.
      2. Haar cascade + HOG/CNN bundle — fallback when InsightFace is not
         installed or when the gallery is empty.

    Intentionally does NOT call generate_unmasked_face() — Stable Diffusion
    inference takes 1–5 s per image and would make live streaming impossible.
    Returns a list of face dicts (coords in the original received-frame space).
    """
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return []

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Primary path: InsightFace (SCRFD detection + ArcFace embedding) ────
    # Use this whenever InsightFace is available — it's more accurate than
    # the Haar cascade and produces embeddings compatible with the gallery
    # that /enroll writes to.
    if video_pipeline.is_available() and len(video_gallery.identities()) > 0:
        try:
            # Work at a sensible detection size for live frames.
            det_size = settings.live_camera_det_size
            dets = video_pipeline.detect_and_embed(
                img_bgr, frame_idx=0,
                det_score_thresh=settings.video_det_score_thresh,
                det_size=det_size,
            )
            faces = []
            for det in dets:
                x1, y1, x2, y2 = det.bbox
                # Clamp to image bounds.
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(x1 + 1, min(x2, orig_w))
                y2 = max(y1 + 1, min(y2, orig_h))
                fw, fh = x2 - x1, y2 - y1

                # Match against enrolled gallery (single-frame, no voting).
                matches = video_gallery.match(det.embedding, top_k=1)
                if matches and matches[0]["similarity"] >= settings.video_min_similarity:
                    top = matches[0]
                    conf = video_pipeline.calibrate_confidence(
                        top["similarity"],
                        midpoint=settings.video_calib_midpoint,
                        slope=settings.video_calib_slope,
                    )
                    identity = top["identity"]
                else:
                    identity = "Unknown"
                    conf = round(matches[0]["similarity"], 4) if matches else 0.0

                # Persist sighting for cross-camera tracking
                reid_store.add_sighting(
                    camera_id=camera_id,
                    track_id=frame_id,  # using frame_id as a proxy for track_id in single-frame context
                    embedding=det.embedding,
                    similarity_threshold=settings.reid_similarity_threshold,
                    identity_name=identity if identity != "Unknown" else None,
                )

                faces.append({
                    "x": x1, "y": y1, "w": fw, "h": fh,
                    "identity": identity,
                    "confidence": round(float(conf), 4),
                })
            return faces
        except Exception:
            logger.warning("InsightFace live-frame path failed; falling back to Haar.",
                           exc_info=True)

    # ── Fallback: Haar cascade + HOG/CNN bundle ─────────────────────────────
    # Used when InsightFace is unavailable or no identities are enrolled.
    proc_w = 640
    if orig_w != proc_w:
        scale = proc_w / orig_w
        proc_img = cv2.resize(img_bgr, (proc_w, int(orig_h * scale)),
                              interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        proc_img = img_bgr

    bboxes_proc = _detect_faces_live(proc_img)

    # Second pass on a half-resolution image to catch very close-up faces
    # (Haar cascades miss faces that are too large relative to the image).
    half_w = proc_w // 2
    half_img = cv2.resize(proc_img, (half_w, int(proc_img.shape[0] * 0.5)),
                          interpolation=cv2.INTER_LINEAR)
    bboxes_half = _detect_faces_live(half_img)
    bboxes_half_mapped = [(int(x * 2), int(y * 2), int(w * 2), int(h * 2))
                          for (x, y, w, h) in bboxes_half]

    all_bboxes_proc = list(bboxes_proc) + bboxes_half_mapped

    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    kept = []
    for box in sorted(all_bboxes_proc, key=lambda r: r[2] * r[3], reverse=True):
        if all(_iou(box, k) < 0.35 for k in kept):
            kept.append(box)

    faces = []
    for (x, y, fw, fh) in kept:
        ox = int(round(x / scale))
        oy = int(round(y / scale))
        ow = int(round(fw / scale))
        oh = int(round(fh / scale))
        ox = max(0, min(ox, orig_w - 1))
        oy = max(0, min(oy, orig_h - 1))
        ow = min(ow, orig_w - ox)
        oh = min(oh, orig_h - oy)

        face_crop = img_bgr[oy : oy + oh, ox : ox + ow]
        if face_crop.size == 0:
            continue
        if bundle is not None:
            result = predict_identity(face_crop, top_k=1)
        else:
            result = {"identity": "Unknown", "confidence": 0.0}
        faces.append({
            "x": ox, "y": oy, "w": ow, "h": oh,
            "identity": result["identity"],
            "confidence": round(result["confidence"], 4),
        })
    return faces



@app.websocket("/ws/live")
async def websocket_live_camera(websocket: WebSocket):
    """WebSocket endpoint for real-time live-camera face recognition.

    Protocol (binary frames from client, text JSON from server):
      client → server : bytes  — raw JPEG frame data
                         preceded by 4-byte little-endian frame_id (int32)
      server → client : UTF-8 JSON  — LiveFrameResponse

    The client prepends a 4-byte little-endian int32 frame_id to each
    message so the server can echo it back; this lets the browser
    correlate responses to the frame that triggered them and detect drops.
    """
    await websocket.accept()
    camera_id = websocket.query_params.get("camera_id") or uuid.uuid4().hex
    logger.info("Live camera WebSocket connected from %s (camera_id=%s)", websocket.client, camera_id)

    # Rolling window for backend FPS estimate.
    _frame_times: collections.deque = collections.deque(maxlen=30)
    min_interval = 1.0 / settings.live_camera_max_fps  # enforce hard rate cap
    last_frame_time = 0.0

    try:
        while True:
            # Receive next frame (binary message: 4-byte frame_id + JPEG bytes)
            data = await websocket.receive_bytes()
            if len(data) < 5:
                continue  # malformed / keepalive ping

            # Enforce max frame rate — drop frames that arrive too fast.
            now = time.perf_counter()
            if now - last_frame_time < min_interval:
                continue
            last_frame_time = now

            # First 4 bytes = frame_id (little-endian int32)
            frame_id = int.from_bytes(data[:4], "little")
            jpeg_bytes = data[4:]

            t0 = time.perf_counter()
            # Run detection + prediction in a thread so we don’t block the event loop.
            faces = await run_in_threadpool(_process_live_frame, jpeg_bytes, camera_id, frame_id)
            latency_ms = (time.perf_counter() - t0) * 1000

            _frame_times.append(time.perf_counter())
            if len(_frame_times) >= 2:
                elapsed = _frame_times[-1] - _frame_times[0]
                fps = (len(_frame_times) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                fps = 0.0

            response = schemas.LiveFrameResponse(
                faces=[schemas.LiveFaceDetection(**f) for f in faces],
                latency_ms=round(latency_ms, 2),
                frame_id=frame_id,
                fps=round(fps, 1),
            )
            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info("Live camera WebSocket disconnected")
    except Exception:
        logger.exception("Live camera WebSocket error")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@app.post("/predict/frame")
@limiter.limit("60/minute")
async def predict_frame(
    request: Request,
    file: UploadFile = File(...),
):
    """HTTP fallback for live-camera prediction.

    Identical semantics to /ws/live but over plain HTTP POST — useful when
    WebSockets are blocked by a proxy. The client should POST a JPEG and
    include an X-Frame-ID header; the response is a LiveFrameResponse JSON.
    Latency will be higher than the WebSocket path due to connection overhead.
    """
    camera_id = request.headers.get("X-Camera-ID") or request.query_params.get("camera_id") or uuid.uuid4().hex
    frame_id_str = request.headers.get("X-Frame-ID", "0")
    try:
        frame_id = int(frame_id_str)
    except ValueError:
        frame_id = 0

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Empty frame")

    t0 = time.perf_counter()
    faces = await run_in_threadpool(_process_live_frame, contents, camera_id, frame_id)
    latency_ms = (time.perf_counter() - t0) * 1000

    return schemas.LiveFrameResponse(
        faces=[schemas.LiveFaceDetection(**f) for f in faces],
        latency_ms=round(latency_ms, 2),
        frame_id=frame_id,
    )


# ---------------------------------------------------------------------------
# Cross-camera Re-Identification endpoints
# ---------------------------------------------------------------------------

@app.get("/reid/global-persons", response_model=list[schemas.GlobalPersonResponse])
async def get_global_persons():
    """List all global persons with their linked sightings across cameras.
    
    This returns the cross-camera tracking results, where the same person
    has been sighted on multiple cameras and linked via cosine similarity
    of their ArcFace embeddings.
    """
    return reid_store.get_global_persons()