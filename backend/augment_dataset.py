import os
import cv2
import argparse
import logging
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def augment_image(image):
    aug_type = random.choice(['flip', 'rotate', 'brightness', 'blur', 'contrast', 'crop'])
    
    if aug_type == 'flip':
        return cv2.flip(image, 1)
    
    elif aug_type == 'rotate':
        angle = random.uniform(-10, 10)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
    elif aug_type == 'brightness':
        factor = random.uniform(0.85, 1.15)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    elif aug_type == 'blur':
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
        
    elif aug_type == 'contrast':
        alpha = random.uniform(0.8, 1.2)
        img_float = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return img_float
        
    elif aug_type == 'crop':
        h, w = image.shape[:2]
        crop_pct = random.uniform(0.85, 0.95)
        nh, nw = int(h * crop_pct), int(w * crop_pct)
        sy = random.randint(0, h - nh)
        sx = random.randint(0, w - nw)
        cropped = image[sy:sy+nh, sx:sx+nw]
        return cv2.resize(cropped, (w, h))
    return image

def main():
    default_base = Path(__file__).resolve().parent.parent / "datasets" / "images"
    parser = argparse.ArgumentParser(description="Augment RWMFD dataset to handle class imbalance.")
    parser.add_argument("--data-dirs", nargs="+", default=[
        str(default_base / "RWMFD_part_1"),
        str(default_base / "RWMFD_part_2")
    ], help="Original data directories")
    parser.add_argument("--output-dir", default=str(default_base / "RWMFD_augmented"), help="Output directory for augmented images")
    parser.add_argument("--min-per-class", type=int, default=50, help="Minimum number of images per identity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    identities = defaultdict(list)
    for data_dir in args.data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            logging.warning(f"Data directory {data_dir} does not exist. Skipping.")
            continue
            
        for identity_folder in data_path.iterdir():
            if identity_folder.is_dir():
                identity_name = identity_folder.name
                images = list(identity_folder.glob("*.jpg")) + list(identity_folder.glob("*.png"))
                identities[identity_name].extend(images)
                
    if not identities:
        logging.error("No images found in the specified data directories.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Found {len(identities)} identities. Starting augmentation...")
    
    for identity, image_paths in identities.items():
        count = len(image_paths)
        if count == 0:
            continue
            
        identity_out_dir = out_dir / identity
        
        if count >= args.min_per_class:
            logging.info(f"Identity {identity} already has {count} images (>= {args.min_per_class}). No augmentation needed.")
            continue
            
        identity_out_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is not None:
                images.append((p.stem, img))
                
        if not images:
            continue
            
        needed = args.min_per_class - count
        logging.info(f"Identity {identity}: original={count}, generating {needed} augmented images.")
        
        for i in range(needed):
            orig_name, img = random.choice(images)
            aug_img = augment_image(img)
            out_name = f"{orig_name}_aug_{i}.jpg"
            cv2.imwrite(str(identity_out_dir / out_name), aug_img)
            
    logging.info("Augmentation complete.")

if __name__ == "__main__":
    main()
