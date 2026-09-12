# backend/scripts/generate_unmasked_batch_videos.py
"""
Unmasked Surveillance Video Batch Generator (Without Mask Dataset Engine)

Synthesizes surveillance-style video streams of UNMASKED individuals for evaluating
and contrasting facial recognition accuracy against masked target streams.

This engine inspects image repositories under datasets/images and dynamically runs
mask_gate.py's triple-signal heuristic detector to filter and isolate genuine
unmasked reference faces. It then generates 50+ unmasked surveillance video feeds,
structured neatly into batches of 10 (10 by 10) under datasets/videos/without_mask/.

Usage:
    python scripts/generate_unmasked_batch_videos.py --total 50 --batch-size 10
"""

import argparse
import gc
import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Ensure backend module path is accessible for importing mask_gate
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mask_gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

UNMASKED_CHECKPOINTS = [
    "CHECKPOINT_LOBBY_A", "PASSPORT_GATE_01", "BIOMETRIC_PORTAL_N", "RECEPTION_VERIFY_2",
    "VIP_ENTRY_EAST", "SECURITY_TURRET_4", "STAFF_TURNSTILE_B", "CAFETERIA_SCANNER",
    "PERIMETER_GATE_SOUTH", "ELEVATOR_BANK_D", "VISITOR_REGISTRATION", "MAIN_CONCOURSE_3"
]


def make_ambient_canvas(img: np.ndarray, width: int, height: int, dim_factor: float = 0.45) -> np.ndarray:
    bg = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (99, 99), 30)
    bg = (bg.astype(np.float32) * dim_factor).astype(np.uint8)
    return bg


def overlay_unmasked_face(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> np.ndarray:
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

    # Surveillance aesthetic: Amber / Gold detection bounding bracket for unmasked full-face verification
    color = (0, 180, 255)  # Amber / Gold in BGR
    thick = 2
    length = 18
    cv2.line(canvas, (x1, y1), (x1 + length, y1), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + length), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2 - length, y1), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + length), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1 + length, y2), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - length), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2 - length, y2), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - length), color, thick, cv2.LINE_AA)

    return canvas


def add_unmasked_osd(canvas: np.ndarray, camera_id: str, target_label: str, frame_idx: int, fps: int, clip_time: datetime) -> None:
    h, w = canvas.shape[:2]

    # Pulsing indicator
    pulse = (frame_idx // max(1, (fps // 2))) % 2 == 0
    dot_color = (0, 200, 255) if pulse else (50, 50, 50)
    cv2.circle(canvas, (25, 25), 7, dot_color, -1, cv2.LINE_AA)

    cv2.putText(canvas, f"FULL-FACE | {camera_id}", (42, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cur_time = clip_time + timedelta(seconds=(frame_idx / float(fps)))
    ts_str = cur_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    osd_bottom = f"SYS_TIME: {ts_str} | STATUS: UNMASKED_VERIFY ({target_label})"
    cv2.putText(canvas, osd_bottom, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"FPS: {fps} | F_IDX: {frame_idx:04d}", (w - 180, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def synthesize_unmasked_stream(target_label: str, image_paths: List[str], out_path: Path, width: int = 640, height: int = 480, fps: int = 15, num_frames: int = 45) -> bool:
    if not image_paths:
        return False

    loaded = []
    for p in image_paths:
        im = cv2.imread(str(p))
        if im is not None:
            loaded.append(im)

    if not loaded:
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not out.isOpened():
        logging.error(f"Failed to open video writer for {out_path}")
        return False

    base_w = int(width * 0.46)
    base_h = int(height * 0.60)
    center_x = (width - base_w) // 2
    center_y = (height - base_h) // 2

    cam_name = random.choice(UNMASKED_CHECKPOINTS)
    freq_x = random.uniform(0.05, 0.18)
    freq_y = random.uniform(0.04, 0.15)
    amp_x = width * 0.14
    amp_y = height * 0.09
    dim_factor = random.uniform(0.35, 0.50)
    alpha = random.uniform(0.95, 1.2)  # slightly higher contrast for unmasked facial features
    beta = random.randint(-10, 20)
    start_time = datetime.now() - timedelta(minutes=random.randint(5, 1440))

    try:
        for f in range(num_frames):
            idx = (f // max(1, int(fps * 1.5))) % len(loaded)
            raw = cv2.convertScaleAbs(loaded[idx], alpha=alpha, beta=beta)
            canvas = make_ambient_canvas(raw, width, height, dim_factor)

            zoom = 1.0 + 0.05 * math.cos(f * 0.09)
            cur_w = int(base_w * zoom)
            cur_h = int(base_h * zoom)
            face_resized = cv2.resize(raw, (cur_w, cur_h), interpolation=cv2.INTER_AREA)

            pos_x = center_x + int(amp_x * math.cos(f * freq_x))
            pos_y = center_y + int(amp_y * math.sin(f * freq_y))

            overlay_unmasked_face(canvas, face_resized, pos_x, pos_y)
            add_unmasked_osd(canvas, cam_name, target_label, f, fps, start_time)

            out.write(canvas)
        return True
    finally:
        out.release()


def collect_unmasked_images(root_dir: Path, target_pool_size: int = 150) -> List[dict]:
    """Scans datasets and filters via mask_gate to isolate unmasked face images."""
    unmasked_groups = []
    if not root_dir.exists():
        return unmasked_groups

    logging.info(f"Scanning image datasets under {root_dir} for unmasked reference faces via mask_gate...")
    candidates = list(root_dir.rglob("*.jpg")) + list(root_dir.rglob("*.png")) + list(root_dir.rglob("*.jpeg"))
    random.shuffle(candidates)

    found_count = 0
    group_buffer = []

    for p in candidates:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        # Check via mask_gate heuristic if face is unmasked
        if not mask_gate.is_wearing_mask(im, (0, 0, w, h)):
            group_buffer.append(p)
            found_count += 1
            if len(group_buffer) >= 3:  # Group 3 unmasked images per target feed
                group_id = len(unmasked_groups) + 1
                unmasked_groups.append({"label": f"UNMASKED_TARGET_{group_id:03d}", "images": group_buffer})
                group_buffer = []

        if found_count >= target_pool_size:
            break

    logging.info(f"Successfully collected {found_count} unmasked face images organized into {len(unmasked_groups)} tracking target profiles.")
    return unmasked_groups


def main():
    default_images = Path(__file__).resolve().parent.parent.parent / "datasets" / "images"
    default_out = Path(__file__).resolve().parent.parent.parent / "datasets" / "videos" / "without_mask"

    parser = argparse.ArgumentParser(description="Generate unmasked surveillance video clips in batches of 10.")
    parser.add_argument("--images-root", default=str(default_images), help="Root directory containing dataset partitions")
    parser.add_argument("--output-dir", default=str(default_out), help="Output base directory for unmasked videos")
    parser.add_argument("--total", type=int, default=50, help="Total minimum unmasked videos to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of clips per batch (10 by 10)")
    parser.add_argument("--fps", type=int, default=15, help="Video framerate")
    parser.add_argument("--frames", type=int, default=45, help="Total frame count per clip")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather unmasked images via automated detection filter
    sources = collect_unmasked_images(images_root, target_pool_size=max(150, args.total * 3))
    if not sources:
        logging.error("Failed to collect enough unmasked images from datasets.")
        return

    batch_size = getattr(args, "batch_size", 10)
    num_batches = math.ceil(args.total / batch_size)

    logging.info(f"=== STARTING UNMASKED SURVEILLANCE VIDEO COLLECTION ===")
    logging.info(f"Target: {args.total} unmasked feeds | Plan: {num_batches} batches of {batch_size} (10 by 10)")

    total_created = 0
    start_time = time.time()

    for b_idx in range(1, num_batches + 1):
        b_folder = out_dir / f"batch_{b_idx:02d}"
        b_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"\n---> Starting Unmasked Batch {b_idx:02d} / {num_batches:02d} ({batch_size} clips -> {b_folder.name}/) ...")

        b_count = 0
        for c_idx in range(1, batch_size + 1):
            src_idx = ((b_idx - 1) * batch_size + (c_idx - 1)) % len(sources)
            src = sources[src_idx]
            target_label = src["label"]
            imgs = src["images"]

            out_clip = b_folder / f"stream_{target_label}_b{b_idx:02d}_c{c_idx:02d}.mp4"
            success = synthesize_unmasked_stream(target_label, imgs, out_clip, width=640, height=480, fps=args.fps, num_frames=args.frames)
            if success:
                b_count += 1
                total_created += 1
                if c_idx % 5 == 0 or c_idx == batch_size:
                    logging.info(f"  [Unmasked Batch {b_idx:02d}] Progress: {c_idx}/{batch_size} clips generated.")

        gc.collect()
        logging.info(f"<--- Finished Unmasked Batch {b_idx:02d}: Created {b_count} video clips in {b_folder.name}/")

    elapsed = time.time() - start_time
    logging.info(f"\n=== UNMASKED COLLECTION COMPLETE! Successfully created {total_created} unmasked surveillance feeds across {num_batches} batches in {elapsed:.1f} seconds ===")


if __name__ == "__main__":
    main()
