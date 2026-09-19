import os
import shutil
import cv2
from pathlib import Path

def setup_mock_data():
    base_dir = Path("c:/Users/HP/Downloads/Trace-main/Trace-main/datasets/real_people")
    artifacts_dir = Path(r"C:\Users\HP\.gemini\antigravity-ide\brain\7281c56b-e130-4664-884e-3f058fbe8d16")
    
    people = {
        "alice": {
            "clean": "mock_alice_clean_1789613093313.jpg",
            "masked": "mock_alice_masked_1789613306845.jpg"
        },
        "bob": {
            "clean": "mock_bob_clean_1789613115826.jpg",
            "masked": "mock_bob_masked_1789613358289.jpg"
        }
    }
    
    for name, data in people.items():
        person_dir = base_dir / name
        photos_dir = person_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy clean photo
        clean_src = artifacts_dir / data["clean"]
        clean_dst = photos_dir / "ref_01.jpg"
        shutil.copy2(clean_src, clean_dst)
        print(f"Copied {name} reference photo.")
        
        # Create video from masked photo
        masked_src = artifacts_dir / data["masked"]
        video_dst = person_dir / "video.mp4"
        
        img = cv2.imread(str(masked_src))
        if img is None:
            print(f"Failed to read {masked_src}")
            continue
            
        height, width, layers = img.shape
        # Use mp4v codec for mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_dst), fourcc, 30.0, (width, height))
        
        print(f"Generating video for {name}...")
        for _ in range(150): # 5 seconds at 30 fps
            video.write(img)
            
        cv2.destroyAllWindows()
        video.release()
        print(f"Finished video for {name}.")

if __name__ == "__main__":
    setup_mock_data()
