import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, field_validator


class UserCreate(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserOut(BaseModel):
    id: str
    email: EmailStr
    created_at: datetime.datetime
    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChangePassword(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class ForgotPassword(BaseModel):
    email: EmailStr


class ResetPassword(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class ScanHistoryOut(BaseModel):
    id: str
    identity: str
    confidence: float
    input_thumb: str | None = None
    generated_thumb: str | None = None
    face_quality_score: float | None = None
    created_at: datetime.datetime
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# New schemas for enhanced pipeline
# ---------------------------------------------------------------------------

class FaceQuality(BaseModel):
    """Image quality assessment for uploaded face photos."""
    blur_score: float       # 0.0 (very blurry) to 1.0 (sharp)
    brightness_score: float  # 0.0 (very dark/bright) to 1.0 (good exposure)
    resolution_ok: bool      # True if image meets minimum resolution
    overall: str             # "good" | "fair" | "poor"


class TopKMatch(BaseModel):
    """A single identity candidate with confidence."""
    identity: str
    confidence: float


class PredictResponse(BaseModel):
    identity: str
    confidence: float
    input_image: str
    generated_image: str
    saved_to_history: bool
    face_detected: bool
    face_quality: Optional[FaceQuality] = None
    top_k_matches: list[TopKMatch] = []
    reconstruction_trust_score: Optional[float] = None
    """ArcFace cosine similarity [0, 1] between the masked face crop and the
    SD-reconstructed face. Higher values mean the reconstruction faithfully
    preserves the detected identity. Null when generation is disabled or
    generation failed."""


class ExplainResponse(BaseModel):
    """Response for the occlusion sensitivity heatmap endpoint."""
    identity: str
    confidence: float
    heatmap: list[list[float]]
    patch_size: int
    stride: int


class FaceDetection(BaseModel):
    """A single detected face with bounding box and identity."""
    x: int
    y: int
    w: int
    h: int
    identity: str
    confidence: float
    top_k_matches: list[TopKMatch] = []


class MultiFaceResponse(BaseModel):
    """Response when multiple faces are detected in an image."""
    faces: list[FaceDetection]
    annotated_image: str       # base64 image with bounding boxes drawn
    face_quality: Optional[FaceQuality] = None


class ModelInfoResponse(BaseModel):
    """Model metadata exposed via /model-info."""
    num_identities: int
    num_images: int
    cv_accuracy: float
    feature_dim: int
    feature_mode: str
    training_date: str
    identities: list[str]


class BatchPredictItem(BaseModel):
    """Single item in batch prediction results."""
    filename: str
    identity: str
    confidence: float
    input_image: str
    generated_image: str
    face_quality: Optional[FaceQuality] = None
    top_k_matches: list[TopKMatch] = []
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """Response for batch prediction endpoint."""
    results: list[BatchPredictItem]
    total: int
    successful: int
    failed: int


# ---------------------------------------------------------------------------
# Video pipeline schemas (SCRFD + ArcFace + tracking + voting)
# Field names below match what frontend/src/components/VideoIdentify.jsx
# already expects — do not rename without updating that component too.
# ---------------------------------------------------------------------------

class EnrollResponse(BaseModel):
    identity: str
    embeddings_added: int
    total_identities: int


class EnrollIdentitiesResponse(BaseModel):
    identities: list[str]


class VideoCandidate(BaseModel):
    identity: str
    score: float
    raw_similarity: float = 0.0


class TrackBBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TrackResult(BaseModel):
    """One tracked face across the video, with its final voted identity."""
    track_id: int
    identity: str
    confidence: float
    raw_similarity: float = 0.0
    votes: int
    frames_considered: int
    last_bbox: TrackBBox
    avg_det_score: float


class IdentifyVideoResponse(BaseModel):
    # Primary fields VideoIdentify.jsx renders directly:
    identity: str
    confidence: float
    raw_confidence: float = 0.0
    frames_used: int
    detail: Optional[str] = None
    candidates: list[VideoCandidate] = []
    # Extra detail for future multi-person UI — safe to ignore client-side.
    tracks: list[TrackResult] = []
    num_tracks: int
    frames_processed: int
    total_frames: int
    video_fps: float
    processing_seconds: float


# ---------------------------------------------------------------------------
# Live camera WebSocket pipeline schemas (/ws/live)
# Kept deliberately small — these go over the wire at 10–30 fps.
# ---------------------------------------------------------------------------

class LiveFaceDetection(BaseModel):
    """Detected face in a live video frame."""
    x: int
    y: int
    w: int
    h: int
    identity: str
    confidence: float


class LiveFrameResponse(BaseModel):
    """Per-frame result sent back over the /ws/live WebSocket."""
    faces: list[LiveFaceDetection]
    latency_ms: float        # backend processing time for this frame
    frame_id: int            # echoes back the client-supplied frame counter
    fps: float = 0.0         # rolling backend throughput estimate


# ---------------------------------------------------------------------------
# Cross-camera Re-Identification schemas
# ---------------------------------------------------------------------------

class GlobalPersonSighting(BaseModel):
    sighting_id: str
    camera_id: str
    track_id: int
    first_seen: float
    last_seen: float
    identity_name: Optional[str] = None


class GlobalPersonResponse(BaseModel):
    global_person_id: str
    identity_name: Optional[str] = None
    cameras: list[str]
    sightings: list[GlobalPersonSighting]