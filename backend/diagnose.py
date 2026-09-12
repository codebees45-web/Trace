import os
import sys

print("=" * 50)
print("DIAGNOSTIC CHECK")
print("=" * 50)

# 1. Check pipeline file exists
if os.path.exists("masked_face_pipeline.joblib"):
    print("[OK] masked_face_pipeline.joblib found")
else:
    print("[FAIL] masked_face_pipeline.joblib missing — rerun the notebook's save cell")
    sys.exit(1)

# 2. Check it loads
try:
    import joblib
    bundle = joblib.load("masked_face_pipeline.joblib")
    print(f"[OK] Pipeline loads. Model type: {type(bundle['model']).__name__}")
    print(f"     Classes: {len(bundle['label_encoder'].classes_)}")
except Exception as e:
    print(f"[FAIL] Pipeline failed to load: {e}")
    sys.exit(1)

# 3. Check recognition works on a real test image
try:
    import cv2
    test_img_path = "../RWMFD_part_1/0000/0000.jpg"
    if not os.path.exists(test_img_path):
        print(f"[FAIL] Test image not found at {test_img_path} — adjust the path")
        sys.exit(1)

    from main import predict_identity
    img = cv2.imread(test_img_path)
    result = predict_identity(img)
    print(f"[OK] Recognition works. Predicted: {result['identity']} ({result['confidence']:.1%})")
except Exception as e:
    print(f"[FAIL] Recognition failed: {e}")
    sys.exit(1)

# 4. Check generation works (slow — this is the real test)
try:
    from main import generate_unmasked_face
    print("Running generation (this takes 30-90s, please wait)...")
    restored = generate_unmasked_face(img)
    print(f"[OK] Generation ran. Output shape: {restored.shape}")
except Exception as e:
    print(f"[FAIL] Generation failed: {e}")
    sys.exit(1)

# 5. Check demo cache
if os.path.exists("demo_cache.json"):
    import json
    with open("demo_cache.json") as f:
        cache = json.load(f)
    print(f"[OK] demo_cache.json found with {len(cache)} precomputed results")
else:
    print("[WARN] demo_cache.json not found — run precompute.py before your demo")

print("=" * 50)
print("ALL CRITICAL CHECKS PASSED" if True else "")