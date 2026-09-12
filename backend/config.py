import sys
import warnings

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Values that must never survive into a production deployment. If jwt_secret
# still equals this at startup while environment=production, the process
# refuses to boot rather than silently signing tokens with a public value.
_INSECURE_DEFAULT_SECRET = "change-me-in-.env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # "development" | "staging" | "production" — gates the strict checks below
    # and switches log formatting (see main.py).
    environment: str = "development"

    jwt_secret: str = _INSECURE_DEFAULT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    database_url: str = "sqlite:///./trace.db"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_username: Optional[str] = None
    mongodb_password: Optional[str] = None
    mongodb_db_name: str = "trace"
    allowed_origins: str = "http://localhost:5173"
    # Used only to build the password-reset link that gets logged (see
    # /auth/forgot-password) — point this at wherever the frontend is served.
    frontend_url: str = "http://localhost:5173"

    confidence_threshold: float = 0.4
    pipeline_path: str = "masked_face_pipeline.joblib"
    enable_generation: bool = True
    sd_model_id: str = "runwayml/stable-diffusion-inpainting"
    sd_inference_steps: int = 15
    sd_guidance_scale: float = 7.5

    # New feature settings
    feature_mode: str = "multi"          # "multi" (HOG+LBP) or "hog" (legacy)
    enable_quality_check: bool = True     # face quality assessment before prediction
    multi_face_detection: bool = True     # detect all faces, not just the largest
    top_k_results: int = 3               # number of top identity candidates to return

    max_upload_mb: int = 8
    predict_rate_limit: str = "10/minute"
    auth_rate_limit: str = "5/minute"
    log_level: str = "INFO"

    # Video pipeline (SCRFD detection + ArcFace embedding + IOU tracking + voting)
    max_upload_video_mb: int = 200
    video_rate_limit: str = "3/minute"
    video_sample_every_n_frames: int = 6
    video_max_frames_sampled: int = 900
    video_det_size: int = 480
    video_det_score_thresh: float = 0.45
    video_track_iou_threshold: float = 0.3
    video_track_max_age: int = 15
    video_min_similarity: float = 0.35
    video_min_margin: float = 0.08

    # Confidence calibration — maps raw ArcFace cosine similarity to the
    # displayed "confidence" percentage via a logistic curve, instead of
    # showing the raw (bounded, conservative) cosine value directly.
    # `video_calib_midpoint` should sit roughly where your genuine and
    # impostor similarity distributions cross; `video_calib_slope` controls
    # how sharply confidence rises around it. Re-fit both with
    # backend/calibrate.py against your own gallery + sample videos rather
    # than trusting these defaults for anything high-stakes.
    video_calib_midpoint: float = 0.42
    video_calib_slope: float = 10.0

    # Live camera WebSocket pipeline (/ws/live)
    # Uses the fast HOG/CNN pipeline — Stable Diffusion is always skipped.
    live_camera_det_size: int = 320        # smaller input = faster Haar cascade scan
    live_camera_skip_generation: bool = True   # never run SD inpainting on live frames
    live_camera_max_fps: int = 30          # hard cap: drop frames if backend is falling behind

    # sqlite:// URLs only — ignored for other backends.
    sqlite_busy_timeout_ms: int = 5000

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    def validate_for_startup(self) -> None:
        """Fail fast on configuration that would be unsafe in production.

        Called once from main.py's lifespan handler — not at import time —
        so test suites can import config/settings without a real .env.
        """
        problems = []
        if self.is_production and self.jwt_secret == _INSECURE_DEFAULT_SECRET:
            problems.append(
                "JWT_SECRET is still the placeholder value. Generate one with "
                "`python -c \"import secrets; print(secrets.token_urlsafe(64))\"` "
                "and set it in backend/.env."
            )
        if self.is_production and len(self.jwt_secret) < 32:
            problems.append("JWT_SECRET is too short for production (need 32+ chars).")
        if self.is_production and any(
            o.startswith("http://") and "localhost" not in o and "127.0.0.1" not in o
            for o in self.cors_origins
        ):
            warnings.warn(
                "ALLOWED_ORIGINS contains a plain http:// origin in production — "
                "browsers will block cookies/auth headers over insecure origins.",
                stacklevel=2,
            )
        if problems:
            for p in problems:
                print(f"[CONFIG ERROR] {p}", file=sys.stderr)
            sys.exit(1)


settings = Settings()