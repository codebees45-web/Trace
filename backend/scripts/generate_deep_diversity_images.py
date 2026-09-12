# backend/scripts/generate_deep_diversity_images.py
"""
Deep-Diversity Surveillance Image Generation Engine (6,000+ Images)

Expands the TRACE face image repository to massive enterprise scale (10,000+ total images)
by generating 6,000 brand new, highly complex surveillance face images across existing
identities into a dedicated new partition:
datasets/images/RWMFD_part_4_synthetic/

Advanced Surveillance Transformations Applied:
1. Analog CCTV Scanline Banding (interlaced horizontal scanlines & color bleeding)
2. Outdoor Weather Haze & Rain Streaks (lens fog attenuation and raindrop light scatter)
3. RGB Chroma Sensor Noise (high-ISO digital camera multi-channel speckles)
4. Ceiling-Mounted Camera Perspective Distortion (vertical compression & angled viewpoints)
5. Synthetic Lower-Face Covering Tints (simulated surgical cyan, charcoal fabric, and white KN95 shading)

All generated images are organized cleanly by identity subfolders for automatic discovery
by all AI training scripts (train_model_v4.py and check_dataset_identities.py).

Usage:
    python scripts/generate_deep_diversity_images.py --target 6000
"""

import argparse
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_cctv_scanlines(img: np.ndarray) -> np.ndarray:
    """Simulates interlaced analog security CCTV camera scanlines and mild horizontal bleed."""
    out = img.copy()
    h, w = out.shape[:2]
    step = random.choice([2, 3])
    dark_factor = random.uniform(0.60, 0.78)
    # Darken alternating horizontal rows
    out[::step, :, :] = (out[::step, :, :].astype(np.float32) * dark_factor).astype(np.uint8)
    return out


def apply_weather_haze(img: np.ndarray) -> np.ndarray:
    """Simulates outdoor checkpoint weather attenuation (mist/fog and light rain streaks)."""
    out = img.astype(np.float32)
    h, w, c = out.shape
    # Add ambient white/grey atmospheric haze (fog)
    haze_intensity = random.uniform(0.12, 0.28)
    out = out * (1.0 - haze_intensity) + 255.0 * haze_intensity

    # Occasionally draw faint diagonal rain streaks if image is large enough
    if random.random() < 0.35 and h > 50 and w > 50:
        streak_layer = np.zeros((h, w, c), dtype=np.uint8)
        num_streaks = random.randint(4, 12)
        for _ in range(num_streaks):
            rx = random.randint(0, w - 1)
            ry = random.randint(0, int(h * 0.7))
            length = random.randint(8, int(h * 0.25) + 8)
            cv2.line(streak_layer, (rx, ry), (rx - 3, ry + length), (220, 235, 255), 1, cv2.LINE_AA)
        streak_layer = cv2.GaussianBlur(streak_layer, (3, 3), 1.0)
        out = np.clip(out + streak_layer.astype(np.float32) * 0.4, 0, 255)

    return out.astype(np.uint8)


def apply_chroma_sensor_noise(img: np.ndarray, intensity: float = 14.0) -> np.ndarray:
    """Simulates multi-channel RGB digital color sensor speckles (high ISO color noise)."""
    h, w, c = img.shape
    # Independent Gaussian noise per RGB channel
    noise = np.random.normal(0, intensity, (h, w, c)).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def apply_perspective_camera_angle(img: np.ndarray) -> np.ndarray:
    """Simulates ceiling-mounted checkpoint camera viewpoint via mild affine skew and rotation."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-16.0, 16.0)
    scale = random.uniform(0.88, 1.14)
    
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # Add slight vertical shear/compression to simulate looking down from above
    shear = random.uniform(-0.08, 0.08)
    matrix[1, 0] += shear
    
    transformed = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.5:
        transformed = cv2.flip(transformed, 1)
    return transformed


def apply_synthetic_mask_tint(img: np.ndarray) -> np.ndarray:
    """Applies a lower-face surgical or cloth fabric mask shading overlay with soft blending."""
    out = img.copy()
    h, w = out.shape[:2]
    start_y = int(h * random.uniform(0.52, 0.60))
    if start_y >= h - 2 or w < 10:
        return out
        
    mask_color = random.choice([
        (180, 220, 240),  # Surgical light blue / cyan
        (50, 50, 55),     # Dark charcoal cloth mask
        (235, 235, 240),  # White KN95 / N95 fabric
        (160, 210, 180)   # Medical mint green
    ])
    
    # Create soft alpha blending mask for realistic lower face covering
    alpha_layer = np.zeros((h, w), dtype=np.float32)
    alpha_layer[start_y:, int(w*0.1):int(w*0.9)] = random.uniform(0.40, 0.65)
    alpha_layer = cv2.GaussianBlur(alpha_layer, (31, 31), 15.0)
    
    for c in range(3):
        out[:, :, c] = (out[:, :, c].astype(np.float32) * (1.0 - alpha_layer) + mask_color[c] * alpha_layer).astype(np.uint8)
        
    return out


def generate_deep_diversity_variation(img: np.ndarray) -> np.ndarray:
    """Executes a randomized deep-diversity surveillance pipeline across 5 advanced transforms."""
    out = apply_perspective_camera_angle(img)
    
    # Randomly apply either weather haze or analog CCTV scanlines (or neither)
    choice = random.random()
    if choice < 0.35:
        out = apply_cctv_scanlines(out)
    elif choice < 0.65:
        out = apply_weather_haze(out)

    # Apply lower face mask fabric tint with 40% probability
    if random.random() < 0.40:
        out = apply_synthetic_mask_tint(out)
        
    # Apply RGB chroma sensor speckles with 60% probability
    if random.random() < 0.60:
        out = apply_chroma_sensor_noise(out, intensity=random.uniform(10.0, 22.0))
        
    return out


def load_all_source_identities(root_images_dir: Path) -> Dict[str, List[Path]]:
    """Scans all existing image dataset directories and maps identity profiles to available images."""
    catalog = {}
    if not root_images_dir.exists():
        logging.error(f"Image directory missing: {root_images_dir}")
        return catalog

    for part_dir in root_images_dir.iterdir():
        if not part_dir.is_dir() or part_dir.name == "RWMFD_part_4_synthetic":
            continue
            
        for child in part_dir.iterdir():
            if child.is_dir():
                ident_label = child.name
                imgs = list(child.glob("*.jpg")) + list(child.glob("*.png")) + list(child.glob("*.jpeg"))
                if imgs:
                    if ident_label not in catalog:
                        catalog[ident_label] = []
                    catalog[ident_label].extend(imgs)

    logging.info(f"Loaded identity catalog containing {len(catalog)} identities across existing repositories.")
    return catalog


def main():
    default_images = Path(__file__).resolve().parent.parent.parent / "datasets" / "images"
    default_out = default_images / "RWMFD_part_4_synthetic"

    parser = argparse.ArgumentParser(description="Generate 6000+ deep-diversity synthetic surveillance images.")
    parser.add_argument("--images-root", default=str(default_images), help="Input root directory containing image datasets")
    parser.add_argument("--output-dir", default=str(default_out), help="Output destination for the new deep synthetic partition")
    parser.add_argument("--target", type=int, default=6000, help="Minimum new synthetic images to generate")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed for repeatable synthesis")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_all_source_identities(images_root)
    if not catalog:
        logging.error("No identity profiles found to augment.")
        return

    identities = sorted(list(catalog.keys()))
    num_identities = len(identities)
    
    # Determine exact count per identity folder to comfortably exceed target
    imgs_per_ident = math.ceil(args.target / num_identities)
    actual_target = imgs_per_ident * num_identities

    logging.info(f"=== STARTING DEEP-DIVERSITY IMAGE DATASET SCALING ===")
    logging.info(f"Target count: {args.target} images | Identities in catalog: {num_identities} | Variations per identity: {imgs_per_ident} -> Actual total: {actual_target}")
    logging.info(f"Destination partition: {out_dir}/ (structured into {num_identities} identity subfolders)")

    created_count = 0
    start_time = time.time()
    
    for idx, ident_label in enumerate(identities, start=1):
        ident_folder = out_dir / ident_label
        ident_folder.mkdir(parents=True, exist_ok=True)
        
        src_paths = catalog[ident_label]
        for gen_idx in range(1, imgs_per_ident + 1):
            source_p = random.choice(src_paths)
            img = cv2.imread(str(source_p))
            if img is None:
                continue
                
            synthetic_img = generate_deep_diversity_variation(img)
            
            out_path = ident_folder / f"deep_synth_{ident_label}_{gen_idx:04d}_{source_p.stem[:5]}.jpg"
            success = cv2.imwrite(str(out_path), synthetic_img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(80, 96)])
            if success:
                created_count += 1

        if idx % 5 == 0 or idx == num_identities:
            elapsed = time.time() - start_time
            rate = created_count / max(0.1, elapsed)
            logging.info(f"  [Progress] Processed {idx:03d}/{num_identities} identities | Created {created_count} / {actual_target} images ({rate:.0f} imgs/sec)...")

    total_time = time.time() - start_time
    logging.info(f"\n=== DEEP-DIVERSITY IMAGE EXPANSION COMPLETE! Created {created_count} new images across {num_identities} identity folders in {total_time:.1f} seconds ===")
    
    # Compute new grand total across all image partitions
    all_imgs = list(images_root.rglob("*.jpg")) + list(images_root.rglob("*.png")) + list(images_root.rglob("*.jpeg"))
    logging.info(f"Grand total image dataset size across all partitions under datasets/images: {len(all_imgs)} images!")


if __name__ == "__main__":
    main()
