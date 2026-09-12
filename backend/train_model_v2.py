# backend/train_model_v2.py
import cv2, joblib, numpy as np, logging, torch
from pathlib import Path
from collections import defaultdict
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face_bgr):
    img = cv2.resize(face_bgr, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 128.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(t).cpu().numpy()[0]
    return emb / (np.linalg.norm(emb) + 1e-9)

def load_data(data_dirs, min_images=6):
    data = defaultdict(list)
    for d in data_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for identity_dir in p.iterdir():
            if not identity_dir.is_dir():
                continue
            for img_path in list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png")):
                data[identity_dir.name].append(str(img_path))
    return {k: v for k, v in data.items() if len(v) >= min_images}

def main(data_dirs, output):
    data = load_data(data_dirs)
    logging.info(f"Identities kept: {len(data)}")

    X, y = [], []
    for label, paths in data.items():
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            X.append(get_embedding(img))
            y.append(label)
    X = np.array(X); y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = SVC(kernel='linear', probability=True, C=1.0, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for fold, (tr, te) in enumerate(skf.split(X, y_enc)):
        clf.fit(X[tr], y_enc[tr])
        acc = accuracy_score(y_enc[te], clf.predict(X[te]))
        accs.append(acc)
        logging.info(f"Fold {fold+1}: {acc:.4f}")
    logging.info(f"Mean CV accuracy: {np.mean(accs):.4f}")

    clf.fit(X, y_enc)
    joblib.dump({
        "model": clf, "label_encoder": le,
        "embedder": "facenet_vggface2", "embedding_dim": X.shape[1]
    }, output)
    logging.info(f"Saved to {output}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--output", default="masked_face_pipeline_v2.joblib")
    args = ap.parse_args()
    main(args.data_dirs, args.output)