# backend/scripts/generate_massive_images.py
"""
Large-Scale Synthetic Face Image Expansion Engine (2000+ Images)

Synthesizes a high-diversity image dataset expansion (default 2,200 images) to scale
the facial recognition gallery and robustify AI model training against real-world
surveillance camera artifacts.

The engine reads existing identity structures across datasets/images (RWMFD_part_1, part_2,
augmented) and dynamically applies complex computer vision pipeline transforms:
- Sensor noise simulation (ISO Gaussian grain & camera digital noise)
- Surveillance optical blur (defocus & turnstile motion blur kernels)
- Extreme checkpoint lighting (low-light attenuation, high fluorescent contrast, daylight glare)
- Geometric re-framing (head pose tilt rotations and perspective zoom)
- CCTV tint adjustments (amber nightfall cast & cold LED lobby shading)

All generated images are neatly structured into identity subdirectories inside:
datasets/images/RWMFD_synthetic_expansion/
ensuring immediate auto-discovery by train_model_v4.py and check_dataset_identities.py.

Usage:
    python scripts/generate_massive_images.py --target 2200
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


def add_camera_noise(img: np.ndarray, intensity: float = 12.0) -> np.ndarray:
    """Simulates low-light CCTV digital sensor ISO noise."""
    noise = np.random.normal(0, intensity, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def apply_surveillance_blur(img: np.ndarray) -> np.ndarray:
    """Applies either slight motion blur (target walking) or out-of-focus blur."""
    mode = random.choice(["gauss", "motion", "none"])
    if mode == "gauss":
        return cv2.GaussianBlur(img, (3, 3), 0.8)
    elif mode == "motion":
        kernel_size = random.choice([3, 5])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        # Horizontal or angled motion kernel
        if random.random() > 0.5:
            kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size, dtype=np.float32) / kernel_size
        else:
            np.fill_diagonal(kernel, 1.0 / kernel_size)
        return cv2.filter2D(img, -1, kernel)
    return img


def adjust_lighting_and_tint(img: np.ndarray) -> np.ndarray:
    """Applies environmental checkpoint lighting (contrast/brightness) and color temperature shifts."""
    alpha = random.uniform(0.7, 1.35)  # contrast range
    beta = random.randint(-25, 25)     # brightness range
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Apply slight color tint (cool lobby fluorescent vs warm afternoon exterior)
    tint_mode = random.choice(["cool", "warm", "neutral", "neutral"])
    h, w, c = adjusted.shape
    if tint_mode == "cool" and c == 3: # Boost blue channel slightly
        b, g, r = cv2.split(adjusted)
        b = cv2.add(b, 15)
        adjusted = cv2.merge([b, g, r])
    elif tint_mode == "warm" and c == 3: # Boost red channel slightly
        b, g, r = cv2.split(adjusted)
        r = cv2.add(r, 15)
        adjusted = cv2.merge([b, g, r])

    return adjusted


def apply_geometric_transform(img: np.ndarray) -> np.ndarray:
    """Applies slight head tilt rotation and scale zoom to diversify pose training."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-14.0, 14.0)
    scale = random.uniform(0.92, 1.12)
    
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # Use border reflection or replicate so rotation leaves no ugly black voids
    transformed = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Horizontal flip (50% probability) for facial symmetry augmentation
    if random.random() < 0.5:
        transformed = cv2.flip(transformed, 1)
        
    return transformed


def generate_synthetic_variation(img: np.ndarray) -> np.ndarray:
    """Chains multiple realistic computer vision transformations into a unified pipeline."""
    out = apply_geometric_transform(img)
    out = adjust_lighting_and_tint(out)
    out = apply_surveillance_blur(out)
    if random.random() < 0.6:
        out = add_camera_noise(out, intensity=random.uniform(8.0, 18.0))
    return out


def load_identity_catalog(root_images_dir: Path) -> Dict[str, List[Path]]:
    """Scans existing dataset partitions and maps identity IDs to all their available images."""
    catalog = {}
    if not root_images_dir.exists():
        logging.error(f"Image root directory missing: {root_images_dir}")
        return catalog

    for part_dir in root_images_dir.iterdir():
        if not part_dir.is_dir() or part_dir.name == "RWMFD_synthetic_expansion":
            continue
            
        for child in part_dir.iterdir():
            if child.is_dir():
                ident_label = child.name
                imgs = list(child.glob("*.jpg")) + list(child.glob("*.png")) + list(child.glob("*.jpeg"))
                if imgs:
                    if ident_label not in catalog:
                        catalog[ident_label] = []
                    catalog[ident_label].extend(imgs)

    logging.info(f"Loaded catalog containing {len(catalog)} identity profiles across existing datasets.")
    return catalog


def main():
    default_images = Path(__file__).resolve().parent.parent.parent / "datasets" / "images"
    default_out = default_images / "RWMFD_synthetic_expansion"

    parser = argparse.ArgumentParser(description="Generate 2000+ synthetic face images organized neatly by identity.")
    parser.add_argument("--images-root", default=str(default_images), help="Input images root path")
    parser.add_argument("--output-dir", default=str(default_out), help="Target directory for generated synthetic dataset")
    parser.add_argument("--target", type=int, default=2200, help="Minimum total new synthetic images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_identity_catalog(images_root)
    if not catalog:
        logging.error("No identities found to augment and expand.")
        return

    identities = sorted(list(catalog.keys()))
    num_identities = len(identities)
    
    # Calculate how many images to generate per identity to hit and exceed the target count
    imgs_per_ident = math.ceil(args.target / num_identities)
    actual_target = imgs_per_ident * num_identities

    logging.info(f"=== STARTING MASSIVE DATASET EXPANSION ===")
    logging.info(f"Target count: {args.target} images | Identities in pool: {num_identities} | Generating {imgs_per_ident} variations per identity -> Total: {actual_target}")
    logging.info(f"Destination: {out_dir}/ (structured neatly into {num_identities} identity subdirectories)")

    created_total = 0
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
                
            synthetic_img = generate_synthetic_variation(img)
            
            out_file = ident_folder / f"synth_cam_{ident_label}_{gen_idx:03d}_{source_p.stem[:6]}.jpg"
            success = cv2.imwrite(str(out_file), synthetic_img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(82, 98)])
            if success:
                created_total += 1

        if idx % 20 == 0 or idx == num_identities:
            elapsed = time.time() - start_time
            rate = created_total / max(0.1, elapsed)
            logging.info(f"  [Progress] Processed {idx:03d}/{num_identities} identities | Created {created_total} / {actual_target} images ({rate:.0f} imgs/sec)...")

    total_time = time.time() - start_time
    logging.info(f"\n=== IMAGE EXPANSION COMPLETE! Successfully added {created_total} new images across {num_identities} identity folders in {total_time:.2f} seconds ===")
    
    # Verify overall repository image count now
    all_imgs = list(images_root.rglob("*.jpg")) + list(images_root.rglob("*.png")) + list(images_root.rglob("*.jpeg"))
    logging.info(f"Grand total image dataset size across all folders under datasets/images: {len(all_imgs)} images!")


if __name__ == "__main__":
    main()
