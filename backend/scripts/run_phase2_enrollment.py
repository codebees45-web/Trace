import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend to path so imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import app

def enroll_real_people():
    dataset_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "real_people"
    if not dataset_dir.exists():
        print(f"[!] Error: Directory {dataset_dir} does not exist.")
        print("    Please complete Step 1: Collect photos and videos of real people and place them in datasets/real_people/<name>/photos/")
        sys.exit(1)

    identities = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not identities:
        print(f"[!] Error: No identity folders found in {dataset_dir}.")
        sys.exit(1)

    print(f"=== Starting Phase 2 Enrollment for {len(identities)} identities ===")
    
    with TestClient(app) as client:
        for identity_dir in identities:
            identity_name = identity_dir.name
            photos_dir = identity_dir / "photos"
            
            if not photos_dir.exists():
                print(f"[-] Skipping {identity_name}: 'photos/' folder not found.")
                continue
                
            photo_paths = [p for p in photos_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
            if not photo_paths:
                print(f"[-] Skipping {identity_name}: no valid images found in photos/")
                continue
            
            print(f"[*] Enrolling {identity_name} ({len(photo_paths)} photos)...")
            
            files = []
            for path in photo_paths:
                files.append(("files", (path.name, open(path, "rb"), "image/jpeg")))
                
            params = {"identity": identity_name}
            
            response = client.post("/enroll", params=params, files=files)
            
            for _, file_tuple in files:
                file_tuple[1].close()
                
            if response.status_code == 200:
                result = response.json()
                print(f"    [+] Success! Added {result['embeddings_added']} embeddings.")
            else:
                print(f"    [!] Failed: {response.status_code} - {response.text}")
                
    print("\n=== Enrollment Complete ===")
    print("Run `/enroll/identities` on the API (or check video_gallery.json) to verify.")

if __name__ == "__main__":
    enroll_real_people()
