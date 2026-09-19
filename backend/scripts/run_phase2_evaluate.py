import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import app

def evaluate_real_people():
    dataset_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "real_people"
    if not dataset_dir.exists():
        print(f"[!] Error: Directory {dataset_dir} does not exist.")
        sys.exit(1)

    identities = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not identities:
        print(f"[!] Error: No identity folders found in {dataset_dir}.")
        sys.exit(1)

    print(f"=== Starting Phase 2 Evaluation for {len(identities)} videos ===")
    
    genuine_pairs = []
    
    with TestClient(app) as client:
        for identity_dir in identities:
            identity_name = identity_dir.name
            video_path = identity_dir / "video.mp4"
            
            if not video_path.exists():
                print(f"[-] Skipping {identity_name}: 'video.mp4' not found.")
                continue
                
            print(f"[*] Processing video for {identity_name}...")
            
            with open(video_path, "rb") as f:
                response = client.post("/identify/video", files={"file": ("video.mp4", f, "video/mp4")})
            
            if response.status_code == 200:
                result = response.json()
                tracks = result.get("tracks", [])
                if tracks:
                    # Pick the track that was present for the most frames (likely the main subject)
                    main_track = max(tracks, key=lambda t: t.get("frames_considered", 0))
                    pred_identity = main_track["identity"]
                    confidence = main_track["confidence"]
                    raw_sim = main_track["raw_similarity"]
                    
                    correct = (pred_identity == identity_name)
                    marker = "[SUCCESS]" if correct else "[FAIL]"
                    
                    print(f"    {marker} Predicted: {pred_identity} | Raw Sim: {raw_sim:.4f} | Confidence: {confidence:.1%} | Ground Truth: {identity_name}")
                    
                    # Add to genuine pairs for calibration
                    genuine_pairs.append((str(video_path).replace("\\", "/"), identity_name))
                else:
                    print(f"    [!] No faces tracked in video.")
            else:
                print(f"    [!] API Error: {response.status_code} - {response.text}")
                
    print("\n=== Evaluation Complete ===")
    
    # Generate the configuration for calibrate.py
    print("\n" + "="*60)
    print("COPY THE FOLLOWING INTO backend/calibrate.py:")
    print("="*60)
    
    print("GENUINE_PAIRS: list[tuple[str, str]] = [")
    for path, ident in genuine_pairs:
        print(f'    ("{path}", "{ident}"),')
    print("]\n")
    
    print("IMPOSTOR_PAIRS: list[tuple[str, str]] = [")
    if len(genuine_pairs) >= 2:
        # Create a simple impostor pair by checking person A against person B
        path_a = genuine_pairs[0][0]
        ident_b = genuine_pairs[-1][1]
        print(f'    # Generated impostor example (please verify this makes sense):')
        print(f'    ("{path_a}", "{ident_b}"),')
    else:
        print("    # (Add at least one impostor pair here: someone else's video vs a different identity)")
    print("]")
    print("="*60)
    print("Next: Run `python calibrate.py` to get your new thresholds!")

if __name__ == "__main__":
    evaluate_real_people()
