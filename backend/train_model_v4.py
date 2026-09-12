# backend/train_model_v4.py
"""
train_model_v4.py — builds on train_model_v3.py with three accuracy-focused
changes. Still 100% compatible with main.py's inference code (same
extract_multi_features, same bundle schema, feature_mode="multi").

What's new vs v3:

1. Dataset auto-discovery (--data-root)
   Point this at your MLproject folder and it finds every RWMFD_part_*,
   RMFD_part_*, and *augmented* directory automatically. No more manually
   listing every folder on the command line as you add more parts.

2. Train/inference-consistent face alignment (--align-faces, on by default)
   train_model.py/v3 extracted features from the RAW dataset image, but
   main.py's predict_identity() crops to the detected face bbox (via the
   same Haar cascade) BEFORE extracting features. If your dataset images
   have extra margin/background that real uploaded photos don't (or vice
   versa), your model trains on a different distribution than it sees at
   inference — this alone can silently cap real-world accuracy even when
   cross-validation accuracy looks fine. v4 runs the same Haar cascade
   crop at training time so both paths match. Falls back to the raw image
   if no face is detected (logged, so you can spot bad crops in your data).

3. Automatic PCA variance selection (--auto-pca, on by default)
   Instead of guessing one variance target, quickly cross-validates a
   plain SVC across a few candidate values and keeps the best-scoring one
   before running the full hyperparameter search on top of it.

Usage:
    python train_model_v4.py --data-root .. --output masked_face_pipeline_v4.joblib

    (--data-root .. means "look in the MLproject folder, one level up from
    backend\", and auto-discover RWMFD_part_1, RWMFD_part_2, RWMFD_part_3,
    RWMFD_augmented, etc. inside it")

You can still pass --data-dirs explicitly instead of / in addition to
--data-root if you want full manual control.
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_multi_features(face_gray, img_size=(96, 96)):
    """MUST stay identical to main.py's extract_multi_features."""
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


def crop_largest_face(gray, margin=0.15):
    """Same detection call as main.py's detect_face_bbox, plus a margin so
    we don't crop the mask/jaw off. Returns (cropped_gray, found_bool).
    """
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return gray, False
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(gray.shape[1], x + w + mx), min(gray.shape[0], y + h + my)
    return gray[y0:y1, x0:x1], True


def get_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_dataset_dirs(root):
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
        if not f.is_dir():
            continue
        rp = str(f.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(str(f))
    return uniq


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


def pick_best_pca_variance(X_scaled, y_enc, candidates, cv):
    best_var, best_score = None, -1.0
    for var in candidates:
        pca = PCA(n_components=var)
        X_p = pca.fit_transform(X_scaled)
        base = SVC(kernel="rbf", class_weight="balanced", random_state=42)
        scores = cross_val_score(base, X_p, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_score = scores.mean()
        logging.info(f"  PCA variance={var} -> {X_p.shape[1]} dims, quick SVC CV acc={mean_score:.4f}")
        if mean_score > best_score:
            best_score, best_var = mean_score, var
    logging.info(f"Selected PCA variance={best_var} (quick CV acc={best_score:.4f})")
    return best_var


def build_search_spaces(quick: bool):
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
    ap = argparse.ArgumentParser(description="Train TRACE model — auto-discovery + aligned faces + auto PCA (max-accuracy v4).")
    ap.add_argument("--data-dirs", nargs="*", default=[], help="Explicit dataset directories (optional if --data-root is set)")
    ap.add_argument("--data-root", default=None, help="Auto-discover RWMFD_part_*/augmented folders under this directory")
    ap.add_argument("--output", default="masked_face_pipeline_v4.joblib")
    ap.add_argument("--img-size", type=int, nargs=2, default=[96, 96])
    ap.add_argument("--min-images", type=int, default=6)
    ap.add_argument("--align-faces", action=argparse.BooleanOptionalAction, default=True,
                     help="Crop to detected face before feature extraction, matching main.py's inference path")
    ap.add_argument("--auto-pca", action=argparse.BooleanOptionalAction, default=True,
                     help="Try a few PCA variance targets and keep the best-scoring one")
    ap.add_argument("--pca-variance", type=float, default=0.97,
                     help="Used directly if --no-auto-pca; otherwise this is included in the candidate sweep")
    ap.add_argument("--quick", action="store_true", help="Small hyperparameter grid for a fast sanity-check run")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    img_size = tuple(args.img_size)

    data_dirs = list(args.data_dirs)
    if args.data_root:
        discovered = discover_dataset_dirs(args.data_root)
        logging.info(f"Auto-discovered {len(discovered)} dataset dir(s) under {args.data_root}: {discovered}")
        data_dirs += discovered
    if not data_dirs:
        logging.error("No dataset directories given — pass --data-dirs and/or --data-root.")
        return

    logging.info("Loading and deduplicating images...")
    data = load_dataset(data_dirs, args.min_images)
    if not data:
        logging.error("No identities met the minimum image count. Aborting.")
        return

    X_raw, y_raw = [], []
    for label, paths in data.items():
        for p in paths:
            X_raw.append(p)
            y_raw.append(label)
    logging.info(f"Total valid images: {len(X_raw)}")

    logging.info(f"Extracting HOG+LBP features (align_faces={args.align_faces})...")
    X_features, y = [], []
    faces_found, faces_missing = 0, 0
    for i, path in enumerate(X_raw):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if args.align_faces:
            img, found = crop_largest_face(img)
            faces_found += int(found)
            faces_missing += int(not found)
        X_features.append(extract_multi_features(img, img_size=img_size))
        y.append(y_raw[i])
        if (i + 1) % 200 == 0:
            logging.info(f"Processed {i + 1}/{len(X_raw)} images.")

    if args.align_faces:
        logging.info(f"Face alignment: {faces_found} cropped, {faces_missing} fell back to full image "
                      f"({faces_missing / max(1, faces_found + faces_missing) * 100:.1f}% miss rate)")
        if faces_missing / max(1, faces_found + faces_missing) > 0.3:
            logging.warning("High face-detection miss rate — check that your dataset images actually contain "
                             "a detectable frontal face, or consider --no-align-faces.")

    X_features = np.array(X_features)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    logging.info(f"Feature matrix shape: {X_features.shape}")

    logging.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    if args.auto_pca:
        candidates = sorted(set([0.90, 0.95, args.pca_variance, 0.99]))
        logging.info(f"Sweeping PCA variance candidates: {candidates}")
        pca_variance = pick_best_pca_variance(X_scaled, y_enc, candidates, cv)
    else:
        pca_variance = args.pca_variance

    pca = PCA(n_components=pca_variance)
    X_pca = pca.fit_transform(X_scaled)
    logging.info(f"PCA ({pca_variance * 100:.0f}% variance) reduced features to {X_pca.shape[1]} dimensions.")

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
        "pca_variance_target": pca_variance,
        "align_faces": args.align_faces,
        "feature_mode": "multi",  # unchanged — keeps main.py's dispatch working
        "tuned": True,
        "data_dirs": data_dirs,
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
    