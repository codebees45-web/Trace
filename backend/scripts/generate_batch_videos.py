# backend/scripts/generate_batch_videos.py
"""
Batch Surveillance Video Generator (10 by 10 Collection Engine)

Synthesizes a large-scale video dataset (minimum 100 video streams) for face tracking
and recognition benchmarking, executed in manageable batches of 10 videos per batch.

Sourcing from all available image repositories (RWMFD_part_1, part_2, part_3, and augmented),
this engine simulates 100 distinct security checkpoint video feeds with varied camera motion,
lighting environments, zoom profiles, and real-time CCTV On-Screen Displays (OSD).

Usage:
    python scripts/generate_batch_videos.py --total 100 --batch-size 10
"""

import argparse
import gc
import logging
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CAMERA_CHECKPOINTS = [
    "GATE_01_NORTH", "ENTRY_MAIN", "HALLWAY_WEST_04", "LOBBY_TURNSTILE", "SECURITY_DESK_B",
    "PERIMETER_EAST", "CORRIDOR_SOUTH", "ELEVATOR_BANK_C", "PARKING_GATE_2", "CAFETERIA_ENTRANCE",
    "SERVER_ROOM_EXT", "RECEPTION_CAM", "EMERGENCY_EXIT_N", "TURNSTILE_03", "VISITOR_GATE_01"
]


def make_ambient_background(img: np.ndarray, width: int, height: int, lighting_factor: float = 0.4) -> np.ndarray:
    """Creates a realistic ambient video canvas with custom simulated lighting."""
    bg = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (101, 101), 35)
    bg = (bg.astype(np.float32) * lighting_factor).astype(np.uint8)
    return bg


def apply_lighting_perturbation(img: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    """Applies slight brightness and contrast variation to simulate environmental camera differences."""
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def overlay_image_with_tracking_box(canvas: np.ndarray, img: np.ndarray, x: int, y: int, cam_name: str) -> np.ndarray:
    """Overlays face frame with simulated target detection brackets and security tracking aesthetics."""
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

    canvas[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]

    # Draw hi-tech tracking bracket corners (green / cyan surveillance feel)
    color = (0, 235, 140)
    thick = 2
    length = 15
    cv2.line(canvas, (x1, y1), (x1 + length, y1), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + length), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2 - length, y1), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + length), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1 + length, y2), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - length), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2 - length, y2), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - length), color, thick, cv2.LINE_AA)

    return canvas


def add_surveillance_osd(canvas: np.ndarray, camera_id: str, target_id: str, frame_idx: int, fps: int, clip_time: datetime) -> None:
    """Overlays broadcast-style timestamp, framerate telemetry, and live camera identifiers."""
    h, w = canvas.shape[:2]

    # Pulsing red recording circle
    pulse = (frame_idx // max(1, (fps // 2))) % 2 == 0
    dot_color = (0, 0, 255) if pulse else (60, 60, 60)
    cv2.circle(canvas, (25, 25), 7, dot_color, -1, cv2.LINE_AA)

    cv2.putText(canvas, f"LIVE | {camera_id}", (42, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Simulated running clock based on frame count
    cur_time = clip_time + timedelta(seconds=(frame_idx / float(fps)))
    ts_str = cur_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    osd_bottom = f"SYS_TIME: {ts_str} | TRACK_ID: {target_id}"
    cv2.putText(canvas, osd_bottom, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 160), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"FPS: {fps} | F_IDX: {frame_idx:04d}", (w - 180, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def synthesize_clip(target_label: str, image_paths: List[str], out_path: Path, width: int = 640, height: int = 480, fps: int = 15, num_frames: int = 45) -> bool:
    if not image_paths:
        return False

    loaded_images = []
    for p in image_paths:
        im = cv2.imread(str(p))
        if im is not None:
            loaded_images.append(im)

    if not loaded_images:
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not out.isOpened():
        logging.error(f"Failed to open writer for {out_path}")
        return False

    base_w = int(width * 0.44)
    base_h = int(height * 0.58)
    center_x = (width - base_w) // 2
    center_y = (height - base_h) // 2

    # Unique motion and camera simulation parameters per video clip
    cam_name = random.choice(CAMERA_CHECKPOINTS)
    freq_x = random.uniform(0.04, 0.16)
    freq_y = random.uniform(0.03, 0.14)
    amp_x = width * 0.15
    amp_y = height * 0.10
    lighting_factor = random.uniform(0.28, 0.48)
    alpha = random.uniform(0.9, 1.15)  # contrast
    beta = random.randint(-15, 15)     # brightness
    start_time = datetime.now() - timedelta(minutes=random.randint(5, 1440))

    try:
        for f in range(num_frames):
            idx = (f // max(1, int(fps * 1.5))) % len(loaded_images)
            raw = apply_lighting_perturbation(loaded_images[idx], alpha, beta)

            canvas = make_ambient_background(raw, width, height, lighting_factor)

            zoom = 1.0 + 0.06 * math.sin(f * 0.1)
            cur_w = int(base_w * zoom)
            cur_h = int(base_h * zoom)
            face_resized = cv2.resize(raw, (cur_w, cur_h), interpolation=cv2.INTER_AREA)

            pos_x = center_x + int(amp_x * math.sin(f * freq_x))
            pos_y = center_y + int(amp_y * math.cos(f * freq_y))

            overlay_image_with_tracking_box(canvas, face_resized, pos_x, pos_y, cam_name)
            add_surveillance_osd(canvas, cam_name, target_label, f, fps, start_time)

            out.write(canvas)
        return True
    finally:
        out.release()


def collect_image_sources(root_images_dir: Path) -> List[dict]:
    """Scans all dataset folders under datasets/images to build an extensive pool of identities and images."""
    sources = []
    if not root_images_dir.exists():
        logging.error(f"Images root directory not found: {root_images_dir}")
        return sources

    # Process structured subdirectories (e.g. RWMFD_part_1/0000, etc.)
    for part in root_images_dir.iterdir():
        if not part.is_dir():
            continue
        subdirs = [d for d in sorted(part.iterdir()) if d.is_dir()]
        if subdirs:
            for sd in subdirs:
                imgs = [p for p in sd.glob("*.jpg")] + [p for p in sd.glob("*.png")] + [p for p in sd.glob("*.jpeg")]
                if imgs:
                    sources.append({"label": f"{part.name}_{sd.name}", "images": imgs})
        else:
            # Handle flat folders like RWMFD_part_3
            imgs = [p for p in part.glob("*.jpg")] + [p for p in part.glob("*.png")] + [p for p in part.glob("*.jpeg")]
            if imgs:
                chunk_size = max(1, len(imgs) // 25)
                for idx in range(0, len(imgs), chunk_size):
                    chunk = imgs[idx : idx + chunk_size]
                    sources.append({"label": f"{part.name}_WILD_{idx//chunk_size+1:02d}", "images": chunk})

    logging.info(f"Collected {len(sources)} distinct candidate identity/image sources from {root_images_dir}.")
    return sources


def main():
    default_images_root = Path(__file__).resolve().parent.parent.parent / "datasets" / "images"
    default_out_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "videos" / "with_mask"

    parser = argparse.ArgumentParser(description="Generate 100+ surveillance video clips in batches of 10.")
    parser.add_argument("--images-root", default=str(default_images_root), help="Root directory containing dataset image partitions")
    parser.add_argument("--output-dir", default=str(default_out_dir), help="Base output directory for generated videos")
    parser.add_argument("--total", type=int, default=100, help="Total minimum video clips to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of videos to generate per batch (10 by 10)")
    parser.add_argument("--fps", type=int, default=15, help="Video framerate")
    parser.add_argument("--frames", type=int, default=45, help="Frames per clip (3 seconds at 15 fps)")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for repeatable generation")
    args = parser.parse_args()

    random.seed(args.seed)
    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_image_sources(images_root)
    if not sources:
        logging.error("No image sources available to synthesize videos from.")
        return

    batch_size = getattr(args, 'batch_size', 10)
    num_batches = math.ceil(args.total / batch_size)

    logging.info(f"=== STARTING LARGE-SCALE VIDEO DATASET GENERATION ===")
    logging.info(f"Target: {args.total} total videos | Execution plan: {num_batches} batches of {batch_size} (10 by 10)")

    total_created = 0
    start_all = time.time()

    for batch_idx in range(1, num_batches + 1):
        batch_folder = out_dir / f"batch_{batch_idx:02d}"
        batch_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"\n---> Starting Batch {batch_idx:02d} / {num_batches:02d} ({batch_size} clips -> {batch_folder.name}/) ...")

        batch_created = 0
        for clip_idx in range(1, batch_size + 1):
            source_idx = ((batch_idx - 1) * batch_size + (clip_idx - 1)) % len(sources)
            src = sources[source_idx]
            target_label = src["label"]
            imgs = src["images"]

            out_clip = batch_folder / f"stream_{target_label}_b{batch_idx:02d}_c{clip_idx:02d}.mp4"
            success = synthesize_clip(target_label, imgs, out_clip, width=640, height=480, fps=args.fps, num_frames=args.frames)
            if success:
                batch_created += 1
                total_created += 1
                if clip_idx % 5 == 0 or clip_idx == batch_size:
                    logging.info(f"  [Batch {batch_idx:02d}] Progress: {clip_idx}/{batch_size} clips generated.")

        gc.collect()  # Release frame memory cleanly between batches
        logging.info(f"<--- Finished Batch {batch_idx:02d}: Created {batch_created} video clips in {batch_folder.name}/")

    elapsed = time.time() - start_all
    logging.info(f"\n=== COLLECTION COMPLETE! Successfully created {total_created} video datasets across {num_batches} batches in {elapsed:.1f} seconds ===")


if __name__ == "__main__":
    main()
