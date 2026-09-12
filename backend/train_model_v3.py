# backend/train_model_v3.py
"""
Improved training pipeline for the TRACE masked-face recognition model.

Fully compatible with main.py's inference code: same feature extraction
(extract_multi_features must match byte-for-byte) and the same bundle schema
(model / scaler / pca / label_encoder / img_size / metadata, feature_mode=
"multi"). Point PIPELINE_PATH at this script's --output and main.py needs
no other changes.

What's different from train_model.py:
  - RandomizedSearchCV hyperparameter tuning for each base classifier
    (SVC, RandomForest, GradientBoosting) instead of fixed hardcoded params
  - class_weight="balanced" on SVC/RF so any residual class imbalance
    (even after augment_dataset.py) doesn't bias predictions toward
    over-represented identities
  - PCA variance target is a CLI flag (--pca-variance) instead of a
    hardcoded 0.95, so you can sweep it
  - Metadata records which hyperparameters were selected, for reproducibility

Usage:
    python train_model_v3.py --data-dirs RWMFD_part_1 RWMFD_part_2 RWMFD_augmented \
        --output masked_face_pipeline_v3.joblib

Recommended workflow:
    1. Run augment_dataset.py first if you haven't, to balance class counts.
    2. Do one --quick run to confirm everything works end-to-end (fast,
       small hyperparameter grid).
    3. Do the real run without --quick (slower — full grid search).
    4. Compare metadata["cv_accuracy"] against your existing
       masked_face_pipeline.joblib before swapping PIPELINE_PATH in .env.
"""
import argparse
import hashlib
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage import feature
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_multi_features(face_gray, img_size=(96, 96)):
    """MUST stay identical to main.py's extract_multi_features — inference
    dispatches to this exact logic whenever metadata['feature_mode'] == 'multi'.
    """
    hog_features = []
    scales = [(64, 64), (96, 96), (128, 128)]
    for size in scales:
        resized = cv2.resize(face_gray, size)
        h = feature.hog(
            resized, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            visualize=False, feature_vector=True,
        )
        blocks_y = (size[1] // 8) - 1
        blocks_x = (size[0] // 8) - 1
        reshaped = h.reshape((blocks_y, blocks_x, 2, 2, 9))
        split_idx = int(blocks_y * 0.45)
        reshaped[:split_idx, ...] *= 1.5
        hog_features.append(reshaped.ravel())
    hog_concat = np.concatenate(hog_features)

    resized_base = cv2.resize(face_gray, img_size)
    lbp = feature.local_binary_pattern(resized_base, P=24, R=3, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return np.concatenate([hog_concat, hist])


def get_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(data_dirs, min_images):
    seen_hashes = set()
    data = defaultdict(list)
    for d in data_dirs:
        p = Path(d)
        if not p.exists():
            logging.warning(f"Data dir not found, skipping: {d}")
            continue
        for identity_dir in p.iterdir():
            if not identity_dir.is_dir():
                continue
            for img_path in list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png")):
                file_hash = get_md5(img_path)
                if file_hash in seen_hashes:
                    continue
                seen_hashes.add(file_hash)
                data[identity_dir.name].append(str(img_path))

    kept = {lbl: paths for lbl, paths in data.items() if len(paths) >= min_images}
    logging.info(f"Identities kept (>={min_images} images): {len(kept)} out of {len(data)}")
    return kept


def build_search_spaces(quick: bool):
    """RandomizedSearchCV param distributions per base classifier.

    quick=True shrinks n_iter for a fast sanity-check run; drop it for the
    real training run once you've confirmed everything works end-to-end.
    """
    svc_params = {
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel": ["rbf"],
    }
    rf_params = {
        "n_estimators": [150, 200, 300, 400],
        "max_depth": [None, 15, 25, 40],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
    }
    gb_params = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 0.9, 1.0],
    }
    n_iter = 6 if quick else 15
    return svc_params, rf_params, gb_params, n_iter


def tune_classifier(base, param_dist, X, y, cv, n_iter, name):
    logging.info(f"Tuning {name} ({n_iter} candidate settings x {cv.get_n_splits()} folds)...")
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=cv,
        scoring="accuracy", n_jobs=-1, random_state=42, refit=True,
    )
    search.fit(X, y)
    logging.info(f"{name} best CV acc: {search.best_score_:.4f} | params: {search.best_params_}")
    return search.best_estimator_


def main():
    ap = argparse.ArgumentParser(
        description="Train TRACE model with a hyperparameter-tuned ensemble (max-accuracy variant)."
    )
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--output", default="masked_face_pipeline_v3.joblib")
    ap.add_argument("--img-size", type=int, nargs=2, default=[96, 96])
    ap.add_argument("--min-images", type=int, default=6)
    ap.add_argument("--pca-variance", type=float, default=0.97,
                     help="Fraction of variance PCA should retain (try 0.95-0.99)")
    ap.add_argument("--quick", action="store_true",
                     help="Small hyperparameter grid for a fast sanity-check run")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    img_size = tuple(args.img_size)

    logging.info("Loading and deduplicating images...")
    data = load_dataset(args.data_dirs, args.min_images)
    if not data:
        logging.error("No identities met the minimum image count. Aborting.")
        return

    X_raw, y_raw = [], []
    for label, paths in data.items():
        for p in paths:
            X_raw.append(p)
            y_raw.append(label)
    logging.info(f"Total valid images: {len(X_raw)}")

    logging.info("Extracting HOG+LBP features (this can take a while)...")
    X_features, y = [], []
    for i, path in enumerate(X_raw):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X_features.append(extract_multi_features(img, img_size=img_size))
        y.append(y_raw[i])
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1}/{len(X_raw)} images.")

    X_features = np.array(X_features)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    logging.info(f"Feature matrix shape: {X_features.shape}")

    logging.info("Fitting StandardScaler and PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    pca = PCA(n_components=args.pca_variance)
    X_pca = pca.fit_transform(X_scaled)
    logging.info(f"PCA ({args.pca_variance * 100:.0f}% variance) reduced features to {X_pca.shape[1]} dimensions.")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    svc_params, rf_params, gb_params, n_iter = build_search_spaces(args.quick)

    best_svc = tune_classifier(
        SVC(probability=True, class_weight="balanced", random_state=42),
        svc_params, X_pca, y_enc, cv, n_iter, "SVC",
    )
    best_rf = tune_classifier(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        rf_params, X_pca, y_enc, cv, n_iter, "RandomForest",
    )
    best_gb = tune_classifier(
        GradientBoostingClassifier(random_state=42),
        gb_params, X_pca, y_enc, cv, n_iter, "GradientBoosting",
    )

    ensemble = VotingClassifier(
        estimators=[("svc", best_svc), ("rf", best_rf), ("gb", best_gb)],
        voting="soft",
    )

    logging.info(f"Running {args.folds}-fold Stratified Cross-Validation on the tuned ensemble...")
    cv_accuracies = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_pca, y_enc)):
        ensemble.fit(X_pca[train_idx], y_enc[train_idx])
        preds = ensemble.predict(X_pca[test_idx])
        acc = accuracy_score(y_enc[test_idx], preds)
        cv_accuracies.append(acc)
        logging.info(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    mean_cv_acc = float(np.mean(cv_accuracies))
    logging.info(f"Mean CV Accuracy: {mean_cv_acc:.4f}")

    logging.info("Training final tuned ensemble on all data...")
    ensemble.fit(X_pca, y_enc)

    final_preds = ensemble.predict(X_pca)
    logging.info("\n" + classification_report(y_enc, final_preds, target_names=label_encoder.classes_))
    logging.info(f"\nConfusion Matrix:\n{confusion_matrix(y_enc, final_preds)}")

    metadata = {
        "training_date": datetime.now().isoformat(),
        "num_identities": len(label_encoder.classes_),
        "num_images": len(X_features),
        "cv_accuracy": mean_cv_acc,
        "feature_dim": X_features.shape[1],
        "pca_components": X_pca.shape[1],
        "pca_variance_target": args.pca_variance,
        "feature_mode": "multi",  # unchanged — keeps main.py's dispatch working
        "tuned": True,
        "svc_params": best_svc.get_params(),
        "rf_params": best_rf.get_params(),
        "gb_params": best_gb.get_params(),
    }

    bundle = {
        "model": ensemble,
        "scaler": scaler,
        "pca": pca,
        "label_encoder": label_encoder,
        "img_size": img_size,
        "metadata": metadata,
    }

    logging.info(f"Saving model bundle to {args.output}...")
    joblib.dump(bundle, args.output)
    logging.info("Done.")


if __name__ == "__main__":
    main()