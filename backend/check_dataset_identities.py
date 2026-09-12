# backend/check_dataset_identities.py
"""
Sanity-checks identity folders BEFORE you train on merged datasets.

Why this exists: when you merge RWMFD_part_1 + RWMFD_part_2 + RWMFD_part_3
(or any other dataset), each defines a folder "0000", "0001", etc. These
are only safe to merge into one training label if folder "0000" in every
part really is the same person. If a mismatch slips in, you silently train
one identity label on two different people's faces, which quietly wrecks
accuracy in a way that's hard to spot just from CV numbers.

This script uses the facenet embeddings you already have installed (same
model as train_model_v2.py) purely as a verification tool — it does NOT
touch your training bundle or main.py at all. For every identity label
that appears in more than one data dir, it compares face embeddings across
sources and flags any pair with low similarity for manual review.

Usage:
    python check_dataset_identities.py --data-dirs ..\\RWMFD_part_1 ..\\RWMFD_part_2 ..\\RWMFD_part_3

    or with auto-discovery:
    python check_dataset_identities.py --data-root ..

Requires facenet-pytorch (already installed earlier in this project).
"""
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_dataset_dirs(root):
    root = Path(root)
    if not root.exists():
        return []
    found = []
    for pattern in ("RWMFD_*", "RMFD_*", "*augmented*", "*_aug*", "*synthetic*"):
        found.extend(sorted(root.glob(pattern)))
        if (root / "datasets" / "images").exists():
            found.extend(sorted((root / "datasets" / "images").glob(pattern)))
        found.extend(sorted(root.rglob(pattern)))
    seen, uniq = set(), []
    for f in found:
        if f.is_dir():
            rp = str(f.resolve())
            if rp not in seen:
                seen.add(rp)
                uniq.append(str(f))
    return uniq


def get_embedding(resnet, img_bgr):
    img = cv2.resize(img_bgr, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 128.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(t).cpu().numpy()[0]
    return emb / (np.linalg.norm(emb) + 1e-9)


def group_embedding(resnet, image_paths, max_samples=5):
    """Average embedding over up to max_samples images — a per-source
    'centroid' for this identity, cheap enough to run over a whole dataset.
    """
    embs = []
    for p in image_paths[:max_samples]:
        img = cv2.imread(p)
        if img is None:
            continue
        embs.append(get_embedding(resnet, img))
    if not embs:
        return None
    centroid = np.mean(embs, axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-9)


def main():
    ap = argparse.ArgumentParser(description="Verify identity folders match the same person across merged dataset dirs.")
    ap.add_argument("--data-dirs", nargs="*", default=[])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-samples", type=int, default=5, help="Images per (identity, source dir) to average")
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="Cosine similarity below this between sources for the same identity gets flagged")
    args = ap.parse_args()

    data_dirs = list(args.data_dirs)
    if args.data_root:
        data_dirs += discover_dataset_dirs(args.data_root)
    if not data_dirs:
        logging.error("No dataset directories given — pass --data-dirs and/or --data-root.")
        return
    logging.info(f"Checking across: {data_dirs}")

    # identity -> {source_dir: [image_paths]}
    by_identity = defaultdict(dict)
    for d in data_dirs:
        p = Path(d)
        if not p.exists():
            logging.warning(f"Skipping missing dir: {d}")
            continue
        for identity_dir in p.iterdir():
            if not identity_dir.is_dir():
                continue
            imgs = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
            if imgs:
                by_identity[identity_dir.name][str(p)] = [str(x) for x in imgs]

    shared = {lbl: sources for lbl, sources in by_identity.items() if len(sources) > 1}
    logging.info(f"{len(shared)} identity label(s) appear in more than one dataset dir — checking those.")
    if not shared:
        logging.info("Nothing to check (no identity label is shared across multiple dirs).")
        return

    logging.info("Loading facenet embedder (vggface2)...")
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    flagged = []
    for i, (label, sources) in enumerate(shared.items(), 1):
        source_names = list(sources.keys())
        centroids = {}
        for src in source_names:
            c = group_embedding(resnet, sources[src], args.max_samples)
            if c is not None:
                centroids[src] = c

        srcs = list(centroids.keys())
        for a in range(len(srcs)):
            for b in range(a + 1, len(srcs)):
                sim = float(np.dot(centroids[srcs[a]], centroids[srcs[b]]))
                status = "OK" if sim >= args.threshold else "MISMATCH?"
                logging.info(f"[{i}/{len(shared)}] identity '{label}': "
                             f"{Path(srcs[a]).name} vs {Path(srcs[b]).name} -> similarity={sim:.3f} [{status}]")
                if sim < args.threshold:
                    flagged.append((label, srcs[a], srcs[b], sim))

    logging.info("=" * 60)
    if flagged:
        logging.warning(f"{len(flagged)} possible identity mismatch(es) found — review these manually "
                         f"before training (open a couple of images from each side and compare by eye):")
        for label, src_a, src_b, sim in flagged:
            logging.warning(f"  identity '{label}': {src_a}  vs  {src_b}  (similarity={sim:.3f})")
    else:
        logging.info("No mismatches found above the similarity threshold — safe to merge these dataset dirs.")


if __name__ == "__main__":
    main()