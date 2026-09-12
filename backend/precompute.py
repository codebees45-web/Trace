"""
Regenerates demo_cache.json used by GET /demo-results.

Run manually whenever the model or demo photo list changes:
    python precompute.py
"""
import json

import cv2

import main as app_main

DEMO_PHOTOS = [
    "../datasets/images/RWMFD_part_1/0006/0071.jpg",
    "../datasets/images/RWMFD_part_1/0001/0180.jpg",
    "../datasets/images/RWMFD_part_1/0008/0023.jpg",
    "../datasets/images/RWMFD_part_1/0010/0023.jpg",
    "../datasets/images/RWMFD_part_1/0002/0084.jpg",
    "../datasets/images/RWMFD_part_2/0016/0022.jpg",
]


def main():
    app_main._load_models()  # main.py no longer loads at import time — see lifespan()

    results = []
    for path in DEMO_PHOTOS:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Skipping {path} — couldn't read file")
            continue
        identity_result = app_main.predict_identity(img_bgr, top_k=3)
        generated = app_main.generate_unmasked_face(img_bgr, seed=42)
        quality = app_main.assess_face_quality(img_bgr)
        results.append({
            "source_path": path,
            "identity": identity_result["identity"],
            "confidence": identity_result["confidence"],
            "input_image": app_main.img_to_base64(img_bgr),
            "generated_image": app_main.img_to_base64(generated),
            "face_quality": quality,
            "top_k_matches": identity_result.get("top_k_matches", []),
        })
        print(f"Done: {path} -> {identity_result['identity']} ({identity_result['confidence']:.2%})")
        if identity_result.get("top_k_matches"):
            for i, m in enumerate(identity_result["top_k_matches"][:3]):
                print(f"  #{i+1}: {m['identity']} ({m['confidence']:.2%})")

    from database import init_db
    db = init_db()
    db.demo_results_cache.update_one(
        {"_id": "latest_demo_results"},
        {"$set": {"results": results}},
        upsert=True,
    )
    print(f"Saved {len(results)} precomputed results to MongoDB demo_results_cache collection")


if __name__ == "__main__":
    main()