# backend/scripts/generate_masked_videos.py
"""
Masked Face Video Dataset Generator

Synthesizes surveillance-style video streams (.mp4) from static masked face images
in the datasets/images directory. Designed to create test video datasets for evaluating
and verifying the SCRFD detection + ArcFace embedding video tracking pipeline (video_pipeline.py).

Features:
- Generates realistic camera motion (gentle pan, zoom, and jitter) across frames.
- Applies ambient blurred backgrounds to frame face crops naturally in standard video resolutions (e.g. 640x480).
- Adds realistic CCTV/Live feed On-Screen Display (OSD) overlays (timestamp, camera feed label, framerate).
- Fully configurable for frame rates, resolution, duration, and input dataset selection.

Usage:
    python scripts/generate_masked_videos.py --data-dir ../datasets/images/RWMFD_part_1 --output-dir ../datasets/videos/masked_clips --num-clips 5 --frames 60
"""

import argparse
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def make_ambient_background(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Creates an aesthetically pleasing ambient background by scaling and deeply blurring the face image."""
    bg = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    # Apply heavy box/Gaussian blur for smooth dark ambient feel
    bg = cv2.GaussianBlur(bg, (99, 99), 30)
    # Darken slightly so the foreground face pops out clearly for detection
    bg = (bg.astype(np.float32) * 0.4).astype(np.uint8)
    return bg


def overlay_image(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> np.ndarray:
    """Overlays image onto canvas at (x, y), handling boundary clipping cleanly."""
    h_c, w_c = canvas.shape[:2]
    h_i, w_i = img.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_i, w_c)
    y2 = min(y + h_i, h_c)

    if x1 >= x2 or y1 >= y2:
        return canvas

    img_x1 = x1 - x
    img_y1 = y1 - y
    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)

    # Blend with subtle border smoothing if desired, or direct copy
    canvas[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]
    
    # Draw a thin sleek border around the face frame for clean visual aesthetics
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 80), 1, cv2.LINE_AA)
    return canvas


def add_osd_overlay(canvas: np.ndarray, camera_id: str, identity_label: str, frame_idx: int, fps: int) -> None:
    """Adds CCTV surveillance style text and indicators onto the video frame."""
    h, w = canvas.shape[:2]
    
    # Top-left indicator: LIVE and camera feed name
    # Red pulsing recording dot
    pulse = (frame_idx // (fps // 2)) % 2 == 0
    dot_color = (0, 0, 240) if pulse else (50, 50, 50)
    cv2.circle(canvas, (25, 25), 6, dot_color, -1, cv2.LINE_AA)
    
    cv2.putText(canvas, f"LIVE | {camera_id}", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    
    # Bottom-left text: Identity tag / timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    osd_text = f"CAM_FEED: {timestamp} | TARGET_ID: {identity_label}"
    cv2.putText(canvas, osd_text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1, cv2.LINE_AA)
    
    # Top-right frame counter
    cv2.putText(canvas, f"FRAME: {frame_idx:04d}", (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def generate_video_clip(identity_label: str, image_paths: List[str], output_path: str, width: int = 640, height: int = 480, fps: int = 15, num_frames: int = 60) -> bool:
    if not image_paths:
        return False

    # Load available face images for this identity
    images = []
    for p in image_paths:
        im = cv2.imread(p)
        if im is not None:
            images.append(im)

    if not images:
        logging.warning(f"Could not load any images for identity {identity_label}")
        return False

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        logging.error(f"Failed to open VideoWriter for {output_path}")
        return False

    # Parameters for simulated motion
    base_face_w = int(width * 0.45)
    base_face_h = int(height * 0.60)

    center_x = (width - base_face_w) // 2
    center_y = (height - base_face_h) // 2

    # Random motion frequencies
    freq_x = random.uniform(0.05, 0.15)
    freq_y = random.uniform(0.04, 0.12)
    amp_x = width * 0.12
    amp_y = height * 0.08

    logging.info(f"Generating clip {Path(output_path).name} ({num_frames} frames @ {fps} fps) for identity {identity_label}...")

    try:
        for f in range(num_frames):
            # Pick image (cycle or stay for a few frames to simulate a stable video feed)
            img_idx = (f // (fps * 2)) % len(images)
            raw_face = images[img_idx]

            # Generate background canvas
            canvas = make_ambient_background(raw_face, width, height)

            # Calculate gentle sinusoidal drift + slight zoom
            zoom = 1.0 + 0.05 * math.sin(f * 0.08)
            cur_w = int(base_face_w * zoom)
            cur_h = int(base_face_h * zoom)
            face_resized = cv2.resize(raw_face, (cur_w, cur_h), interpolation=cv2.INTER_AREA)

            dx = int(amp_x * math.sin(f * freq_x))
            dy = int(amp_y * math.cos(f * freq_y))

            pos_x = center_x + dx
            pos_y = center_y + dy

            # Overlay face onto frame
            overlay_image(canvas, face_resized, pos_x, pos_y)

            # Add surveillance OSD overlay
            add_osd_overlay(canvas, "GATE-CAM-01 (MASK_CHECK)", identity_label, f, fps)

            out.write(canvas)
        return True
    finally:
        out.release()


def main():
    default_data = Path(__file__).resolve().parent.parent.parent / "datasets" / "images" / "RWMFD_part_1"
    default_output = Path(__file__).resolve().parent.parent.parent / "datasets" / "videos" / "with_mask" / "sample_clips"

    parser = argparse.ArgumentParser(description="Generate synthetic masked surveillance video datasets.")
    parser.add_argument("--data-dir", default=str(default_data), help="Input directory containing identity subdirectories of masked face images")
    parser.add_argument("--output-dir", default=str(default_output), help="Output directory to save generated video (.mp4) clips")
    parser.add_argument("--num-clips", type=int, default=5, help="Maximum number of identity clips to generate")
    parser.add_argument("--fps", type=int, default=15, help="Framerate of output video clips")
    parser.add_argument("--frames", type=int, default=60, help="Total frame count per generated clip")
    parser.add_argument("--width", type=int, default=640, help="Video resolution width")
    parser.add_argument("--height", type=int, default=480, help="Video resolution height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logging.error(f"Input directory {data_dir} does not exist.")
        return

    # Find identity directories or standalone images (handling structures like RWMFD_part_3)
    identity_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    
    generated = 0
    if identity_dirs:
        for ident_dir in identity_dirs[: args.num_clips]:
            imgs = [str(p) for p in list(ident_dir.glob("*.jpg")) + list(ident_dir.glob("*.png")) + list(ident_dir.glob("*.jpeg")) + list(ident_dir.glob("*.webp"))]
            out_file = str(out_dir / f"masked_surveillance_feed_{ident_dir.name}.mp4")
            success = generate_video_clip(ident_dir.name, imgs, out_file, width=args.width, height=args.height, fps=args.fps, num_frames=args.frames)
            if success:
                generated += 1
    else:
        # Fallback for flat directories without identity subfolders (like RWMFD_part_3)
        logging.info("No identity subfolders found; generating clips from standalone images.")
        all_imgs = [str(p) for p in list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.webp"))]
        if all_imgs:
            chunk_size = max(1, len(all_imgs) // args.num_clips)
            for idx in range(args.num_clips):
                chunk = all_imgs[idx * chunk_size : (idx + 1) * chunk_size]
                if not chunk:
                    break
                out_file = str(out_dir / f"masked_surveillance_unconstrained_{idx+1:03d}.mp4")
                success = generate_video_clip(f"UNCLASSIFIED_{idx+1:03d}", chunk, out_file, width=args.width, height=args.height, fps=args.fps, num_frames=args.frames)
                if success:
                    generated += 1

    logging.info(f"Successfully generated {generated} masked surveillance video clips in {out_dir}.")


if __name__ == "__main__":
    main()
