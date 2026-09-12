import os
import cv2
import argparse
import logging
import hashlib
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from skimage import feature
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_multi_features(face_gray, img_size=(96, 96)):
    """
    Extracts multiple features from a grayscale face image.
    This function must be imported and used exactly as-is in inference.
    """
    hog_features = []
    scales = [(64, 64), (96, 96), (128, 128)]
    for size in scales:
        resized = cv2.resize(face_gray, size)
        
        # Extract HOG features
        h = feature.hog(
            resized, 
            orientations=9, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), 
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        
        # Apply spatial weighting: upper face emphasis (top 45%)
        blocks_y = (size[1] // 8) - 1
        blocks_x = (size[0] // 8) - 1
        reshaped = h.reshape((blocks_y, blocks_x, 2, 2, 9))
        
        split_idx = int(blocks_y * 0.45)
        reshaped[:split_idx, ...] *= 1.5
        
        hog_features.append(reshaped.ravel())

    hog_concat = np.concatenate(hog_features)

    # Extract LBP features
    resized_base = cv2.resize(face_gray, img_size)
    lbp = feature.local_binary_pattern(resized_base, P=24, R=3, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
    
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return np.concatenate([hog_concat, hist])

def get_md5(image_path):
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    default_base = Path(__file__).resolve().parent.parent / "datasets" / "images"
    parser = argparse.ArgumentParser(description="Train TRACE model with multi-feature extraction.")
    parser.add_argument("--data-dirs", nargs="+", default=[
        str(default_base / "RWMFD_part_1"),
        str(default_base / "RWMFD_part_2"),
        str(default_base / "RWMFD_part_3"),
        str(default_base / "RWMFD_augmented"),
        str(default_base / "RWMFD_synthetic_expansion"),
        str(default_base / "RWMFD_part_4_synthetic")
    ], help="List of data directories")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "masked_face_pipeline.joblib"), help="Output joblib file")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation and print report")
    parser.add_argument("--img-size", type=int, nargs=2, default=[96, 96], help="Base image size (width, height)")
    args = parser.parse_args()

    img_size = tuple(args.img_size)

    logging.info("Loading and deduplicating images...")
    seen_hashes = set()
    data = defaultdict(list)
    
    for d in args.data_dirs:
        path = Path(d)
        if not path.exists():
            continue
            
        for identity_dir in path.iterdir():
            if not identity_dir.is_dir():
                continue
                
            label = identity_dir.name
            for img_path in list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png")):
                file_hash = get_md5(img_path)
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    data[label].append(str(img_path))
                    
    # Drop identities with < 6 images
    labels_to_keep = [lbl for lbl, paths in data.items() if len(paths) >= 6]
    logging.info(f"Identities kept (>=6 images): {len(labels_to_keep)} out of {len(data)}")
    
    X_raw, y_raw = [], []
    for lbl in labels_to_keep:
        for p in data[lbl]:
            X_raw.append(p)
            y_raw.append(lbl)
            
    logging.info(f"Total valid images: {len(X_raw)}")
    if len(X_raw) == 0:
        logging.error("Not enough data to train. Exiting.")
        return

    logging.info("Extracting features (this might take a while)...")
    X_features = []
    y = []
    
    for i, path in enumerate(X_raw):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        feats = extract_multi_features(img, img_size=img_size)
        X_features.append(feats)
        y.append(y_raw[i])
        
        if (i+1) % 100 == 0:
            logging.info(f"Processed {i+1}/{len(X_raw)} images.")
            
    X_features = np.array(X_features)
    y = np.array(y)
    
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    
    logging.info(f"Feature matrix shape: {X_features.shape}")
    
    logging.info("Fitting StandardScaler and PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    logging.info(f"PCA reduced features to {X_pca.shape[1]} dimensions.")

    clf1 = SVC(kernel='rbf', probability=True, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
    clf3 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('svc', clf1), ('rf', clf2), ('gb', clf3)],
        voting='soft'
    )
    
    cv_accuracies = []
    if args.evaluate:
        logging.info("Running 5-fold Stratified Cross-Validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_pca, y_enc)):
            X_train, X_test = X_pca[train_idx], X_pca[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]
            
            ensemble.fit(X_train, y_train)
            preds = ensemble.predict(X_test)
            acc = accuracy_score(y_test, preds)
            cv_accuracies.append(acc)
            logging.info(f"Fold {fold+1} Accuracy: {acc:.4f}")
            
        mean_cv_acc = np.mean(cv_accuracies)
        logging.info(f"Mean CV Accuracy: {mean_cv_acc:.4f}")
    else:
        mean_cv_acc = 0.0

    logging.info("Training final ensemble model on all data...")
    ensemble.fit(X_pca, y_enc)
    
    if args.evaluate:
        final_preds = ensemble.predict(X_pca)
        logging.info("\n" + classification_report(y_enc, final_preds, target_names=label_encoder.classes_))
        cm = confusion_matrix(y_enc, final_preds)
        logging.info(f"\nConfusion Matrix:\n{cm}")
        
    logging.info(f"Saving model bundle to {args.output}...")
    metadata = {
        "training_date": datetime.now().isoformat(),
        "num_identities": len(label_encoder.classes_),
        "num_images": len(X_features),
        "cv_accuracy": float(mean_cv_acc),
        "feature_dim": X_features.shape[1],
        "feature_mode": "multi"
    }
    
    bundle = {
        "model": ensemble,
        "scaler": scaler,
        "pca": pca,
        "label_encoder": label_encoder,
        "img_size": img_size,
        "metadata": metadata
    }
    
    joblib.dump(bundle, args.output)
    logging.info("Done.")

if __name__ == "__main__":
    main()
