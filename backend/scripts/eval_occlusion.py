import os
import cv2
from pathlib import Path
from main import predict_identity
from config import settings

def evaluate_accuracy(dataset_path: str):
    print(f"Evaluating Match Accuracy (use_occlusion_aware_embedding={settings.use_occlusion_aware_embedding})")
    print(f"Dataset: {dataset_path}\n")
    
    correct = 0
    total = 0
    
    data_dir = Path(dataset_path)
    if not data_dir.exists():
        print(f"Error: {dataset_path} not found.")
        return
        
    for ident_dir in sorted(data_dir.iterdir()):
        if not ident_dir.is_dir():
            continue
            
        true_identity = ident_dir.name
        
        for img_path in ident_dir.glob("*.jpg"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
                
            result = predict_identity(img_bgr)
            predicted_identity = result["identity"]
            
            total += 1
            if predicted_identity == true_identity:
                correct += 1
                
    if total > 0:
        acc = correct / total
        print(f"Total Images: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {acc:.2%}")
        
        if settings.use_occlusion_aware_embedding:
            print("\nNote: This is inference-only evaluation. The classifier head (joblib) was")
            print("still trained on standard features. If accuracy drops, the classifier")
            print("may need to be retrained on the occlusion-aware features to see a benefit.")
    else:
        print("No images found to evaluate.")

if __name__ == "__main__":
    evaluate_accuracy("../RWMFD_part_1")
