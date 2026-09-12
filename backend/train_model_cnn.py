# backend/train_model_cnn.py
"""
Deep Convolutional Neural Network (CNN) Backbone Training Pipeline

Trains the TRACE identity recognition classifier on top of deep 512-dimensional
embedding vectors extracted by the PyTorch MaskedFaceCNN backbone. Automatically
discovers and ingests all 10,140+ face images across all dataset partitions under
datasets/images/ (original RWMFD parts, synthetic expansions, and deep diversity sets).

Usage:
    python train_model_cnn.py --data-root .. --output masked_face_pipeline.joblib
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import cnn_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def discover_dataset_dirs(root: str | Path) -> list[str]:
    """Auto-discovers all dataset partitions under root or root/datasets/images."""
    root = Path(root)
    if not root.exists():
        logging.warning(f"--data-root {root} does not exist.")
        return []
        
    found = []
    for pattern in ("RWMFD_*", "RMFD_*", "*augmented*", "*_aug*", "*synthetic*"):
        found.extend(sorted(root.glob(pattern)))
        if (root / "datasets" / "images").exists():
            found.extend(sorted((root / "datasets" / "images").glob(pattern)))
        found.extend(sorted(root.rglob(pattern)))
        
    seen, uniq = set(), []
    for f in found:
        if not f.is_dir() or f.name == "images":
            continue
        rp = str(f.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(str(f))
    return uniq


def load_dataset_images(data_dirs: list[str], min_images: int = 4) -> dict[str, list[str]]:
    """Scans dataset directories for identity folders and image files."""
    catalog = {}
    for d in data_dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for child in p.iterdir():
            if child.is_dir():
                ident = child.name
                imgs = list(child.glob("*.jpg")) + list(child.glob("*.png")) + list(child.glob("*.jpeg"))
                if imgs:
                    if ident not in catalog:
                        catalog[ident] = []
                    catalog[ident].extend([str(im) for im in imgs])
                    
    # Filter minimum images per class for valid k-fold CV
    filtered = {k: v for k, v in catalog.items() if len(v) >= min_images}
    logging.info("Loaded %d valid identity profiles across %d total dataset partitions.", len(filtered), len(data_dirs))
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Train TRACE model using Deep CNN Backbone (512-d embeddings).")
    parser.add_argument("--data-root", default="..", help="Root directory for automated dataset discovery")
    parser.add_argument("--output", default="masked_face_pipeline.joblib", help="Output model pipeline path")
    parser.add_argument("--min-images", type=int, default=4, help="Minimum images per class")
    parser.add_argument("--max-per-class", type=int, default=150, help="Max images per class for balanced high-speed training")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    discovered_dirs = discover_dataset_dirs(args.data_root)
    if not discovered_dirs:
        logging.error("No dataset directories discovered under %s. Aborting.", args.data_root)
        return
        
    logging.info("=== STARTING DEEP CNN BACKBONE MODEL TRAINING ===")
    logging.info("Discovered dataset partitions: %s", discovered_dirs)
    
    data = load_dataset_images(discovered_dirs, args.min_images)
    if not data:
        logging.error("No identities met the min-images threshold.")
        return

    # Assemble dataset sample paths
    X_paths, y_labels = [], []
    for ident, paths in sorted(data.items()):
        # Sample evenly if class volume is enormous to ensure balanced classifier margins
        selected = paths if len(paths) <= args.max_per_class else np.random.choice(paths, args.max_per_class, replace=False).tolist()
        for p in selected:
            X_paths.append(p)
            y_labels.append(ident)
            
    total_samples = len(X_paths)
    logging.info("Total dataset sample pool: %d images across %d identities.", total_samples, len(data))
    
    # Extract Deep CNN 512-dimensional embeddings in batches
    logging.info("Extracting 512-d feature vectors via PyTorch Deep CNN Backbone...")
    start_time = time.time()
    X_embeddings = []
    
    batch_size = 32
    for i in range(0, total_samples, batch_size):
        chunk_paths = X_paths[i:i+batch_size]
        imgs = [cv2.imread(p) for p in chunk_paths]
        # Filter out any unreadable binaries
        valid_imgs = [im for im in imgs if im is not None]
        if not valid_imgs:
            continue
        batch_embs = cnn_backbone.extract_batch_cnn_features(valid_imgs)
        X_embeddings.append(batch_embs)
        
        if (i + batch_size) % 320 == 0 or (i + batch_size) >= total_samples:
            elapsed = time.time() - start_time
            rate = (i + len(valid_imgs)) / max(0.1, elapsed)
            logging.info("  [CNN Extraction] Progress: %d / %d images processed (%.1f imgs/sec)...", min(total_samples, i+batch_size), total_samples, rate)
            
    X_mat = np.vstack(X_embeddings)
    y_arr = np.array(y_labels[:len(X_mat)])
    
    logging.info("Extracted %d embedding vectors of feature dimension %d in %.2f seconds.", X_mat.shape[0], X_mat.shape[1], time.time() - start_time)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mat)
    
    # Train robust classification head on CNN embeddings
    logging.info("Evaluating Stratified K-Fold Cross Validation on CNN embeddings...")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    cv_accs = []
    
    clf = SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_scaled, y_enc), 1):
        clf.fit(X_scaled[tr_idx], y_enc[tr_idx])
        preds = clf.predict(X_scaled[te_idx])
        acc = accuracy_score(y_enc[te_idx], preds)
        cv_accs.append(acc)
        logging.info("  Fold %d Validation Accuracy: %.4f", fold, acc)
        
    mean_cv_acc = float(np.mean(cv_accs))
    logging.info("=== Deep CNN Backbone Mean CV Accuracy: %.4f (%.1f%%) ===", mean_cv_acc, mean_cv_acc * 100)
    
    # Train final deployment model on entire dataset
    logging.info("Fitting deployment classifier on full dataset...")
    clf.fit(X_scaled, y_enc)
    
    # Build complete model metadata for website UI synchronization
    metadata = {
        "training_date": datetime.now().isoformat(),
        "num_identities": len(le.classes_),
        "num_images": len(X_paths),  # Total training images consumed from our expanded dataset repository
        "cv_accuracy": mean_cv_acc,
        "feature_dim": X_scaled.shape[1], # 512 dimensions from CNN Backbone
        "feature_mode": "Deep CNN Backbone (512-d)",
        "backbone_architecture": "PyTorch MaskedFaceCNN (512-d)",
        "data_dirs": discovered_dirs,
    }
    
    bundle = {
        "model": clf,
        "scaler": scaler,
        "pca": None,  # No PCA needed; CNN 512-d embeddings are intrinsically dense and discriminative
        "label_encoder": le,
        "img_size": (112, 112),
        "metadata": metadata,
    }
    
    out_path = Path(args.output).resolve()
    joblib.dump(bundle, str(out_path))
    logging.info("\n=== SUCCESS! Trained Deep CNN pipeline bundle saved to %s ===", out_path)
    logging.info("Model Metadata exported for UI: %s", metadata)


if __name__ == "__main__":
    main()
