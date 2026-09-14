import numpy as np
from cnn_backbone import extract_cnn_features, extract_cnn_features_occlusion_aware, make_occlusion_weight_mask

def test_occlusion_aware_embedding_differs_from_standard():
    # Create a synthetic 112x112 color face crop
    face_bgr = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    
    # 1. Standard extraction
    std_emb = extract_cnn_features(face_bgr)
    
    # 2. Occlusion-aware extraction
    occ_emb = extract_cnn_features_occlusion_aware(face_bgr, alpha_occluded=0.15)
    
    # Assert shapes are identical (512-d)
    assert std_emb.shape == (512,)
    assert occ_emb.shape == (512,)
    
    # Assert the embeddings are actually different
    assert not np.allclose(std_emb, occ_emb, rtol=1e-5, atol=1e-5), "Occlusion mask had no effect on embedding!"

    # Quick check that make_occlusion_weight_mask output is broadcastable and dims the lower half
    mask = make_occlusion_weight_mask(face_bgr, 0.15)
    assert mask.shape == (112, 1)
    assert mask[0, 0] == 1.0  # Top of head is 1.0
    assert mask[-1, 0] == 0.15  # Chin is 0.15
